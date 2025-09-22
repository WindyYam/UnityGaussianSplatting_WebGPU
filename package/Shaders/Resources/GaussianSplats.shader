// SPDX-License-Identifier: MIT
Shader "Gaussian Splatting/Render Splats"
{
	Properties
	{
		_SrcBlend("Src Blend", Float) = 8 // OneMinusDstAlpha
		_DstBlend("Dst Blend", Float) = 1 // One
		_ZWrite("ZWrite", Float) = 0  // Off
	}
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }

        Pass
        {
            ZWrite [_ZWrite]
            Blend [_SrcBlend] [_DstBlend]
            Cull Off
            
CGPROGRAM
#pragma vertex vert
#pragma fragment frag
#pragma require compute
#pragma use_dxc

#include "GaussianSplatting.hlsl"

StructuredBuffer<uint> _OrderBuffer;

struct v2f
{
    half4 col : COLOR0;
    float2 pos : TEXCOORD0;
    uint idx : TEXCOORD1;
    float2 vel : TEXCOORD2; // NDC motion delta (current - previous)
    float4 vertex : SV_POSITION;
};

// reason for using a separate uniform buffer is DX12
// stale uniform variable bug in Unity 2022.3/6000.0 at least,
// "IN-99220 - DX12 stale render state issue with a sequence of compute shader & DrawProcedural calls"
cbuffer SplatGlobalUniforms // match struct SplatGlobalUniforms in C#
{
	uint sgu_transparencyMode;
	uint sgu_frameOffset;
}

StructuredBuffer<SplatViewData> _SplatViewData;
StructuredBuffer<SplatViewData> _PrevSplatViewData; // previous frame per-splat view data
ByteAddressBuffer _SplatSelectedBits;
uint _SplatBitsValid;

v2f vert (uint vtxID : SV_VertexID, uint instID : SV_InstanceID)
{
    v2f o = (v2f)0;
	if (sgu_transparencyMode == 0)
		instID = _OrderBuffer[instID];
	o.idx = instID + sgu_frameOffset;
	SplatViewData view = _SplatViewData[instID];
	float4 centerClipPos = view.pos;
	bool behindCam = centerClipPos.w <= 0;
	if (behindCam)
	{
		o.vertex = asfloat(0x7fc00000); // NaN discards the primitive
	}
	else
	{
		o.col.r = f16tof32(view.color.x >> 16);
		o.col.g = f16tof32(view.color.x);
		o.col.b = f16tof32(view.color.y >> 16);
		o.col.a = f16tof32(view.color.y);

		uint idx = vtxID;
		float2 quadPos = float2(idx&1, (idx>>1)&1) * 2.0 - 1.0;
		quadPos *= 2;

		o.pos = quadPos;

		float2 deltaScreenPos = (quadPos.x * view.axis1 + quadPos.y * view.axis2) * 2 / _ScreenParams.xy;
		o.vertex = centerClipPos;
		o.vertex.xy += deltaScreenPos * centerClipPos.w;

		// motion (NDC delta) -- reconstruct previous vertex clip pos similarly
		o.vel = 0;
		if (sgu_transparencyMode != 0)
		{
			SplatViewData prevView = _PrevSplatViewData[instID];
			if (prevView.pos.w > 0)
			{
				float4 prevClipPos = prevView.pos;
				float2 prevDeltaScreenPos = (quadPos.x * prevView.axis1 + quadPos.y * prevView.axis2) * 2 / _ScreenParams.xy;
				prevClipPos.xy += prevDeltaScreenPos * prevClipPos.w;
				float2 ndcCurr = o.vertex.xy / max(o.vertex.w, 1e-6);
				float2 ndcPrev = prevClipPos.xy / max(prevClipPos.w, 1e-6);
				o.vel = ndcCurr - ndcPrev; // current - previous
			}
		}

		// selection check
		if (_SplatBitsValid)
		{
			uint wordIdx = instID / 32;
			uint bitIdx = instID & 31;
			uint selVal = _SplatSelectedBits.Load(wordIdx * 4);
			if (selVal & (1 << bitIdx))
			{
				o.col.a = -1;				
			}
		}
	}
	FlipProjectionIfBackbuffer(o.vertex);
    return o;
}

// Hash Functions for GPU Rendering
// https://jcgt.org/published/0009/03/02/
uint3 pcg3d16(uint3 v)
{
    v = v * 12829u + 47989u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

	v >>= 16u;
    return v;
}

struct FragOut { half4 col : SV_Target0; half2 motion : SV_Target1; };

FragOut frag (v2f i)
{
    FragOut o; o.col = 0; o.motion = 0;
    float power = -dot(i.pos, i.pos);
    half alpha = exp(power);
    if (i.col.a >= 0)
    {
        alpha = saturate(alpha * i.col.a);
    }
    else
    {
        half3 selectedColor = half3(1,0,1);
        if (alpha > 7.0/255.0)
        {
            if (alpha < 10.0/255.0)
            {
                alpha = 1;
                i.col.rgb = selectedColor;
            }
            alpha = saturate(alpha + 0.3);
        }
        i.col.rgb = lerp(i.col.rgb, selectedColor, 0.5);
    }
    if (alpha < 1.0/255.0)
        discard;

    if (sgu_transparencyMode == 0)
    {
        i.col.rgb *= alpha; // premultiply
    }
    else if(sgu_transparencyMode == 1)
    {
        uint3 coord = uint3(i.vertex.x, i.vertex.y, i.idx);
        uint3 hash = pcg3d16(coord);
        half cutoff = (hash.x & 0xFFFF) / 65535.0;
        if (alpha <= cutoff)
            discard;
        alpha = 1;
        o.motion = half2(i.vel);
    }
    else
    {
        // Halftone ordered dither (4x4 Bayer) for stable stochastic-like transparency.
        // Use screen-space integer pixel coordinates and a small per-instance offset
        // to decorrelate neighboring splats. This is deterministic across frames,
        // reducing temporal flicker compared to random sampling.
        uint2 p = uint2(i.vertex.xy); // truncate to integer pixel coords
        uint bx = p.x & 3u;
        uint by = p.y & 3u;

        // Apply a small per-splat offset to the dither tile to reduce repeating artifacts.
        uint inst = i.idx;
        bx = (bx + (inst & 3u)) & 3u;
        by = (by + ((inst >> 2) & 3u)) & 3u;

        // 4x4 Bayer matrix values 0..15
        const uint bayer4[16] = {
            0u, 8u, 2u, 10u,
            12u,4u,14u,6u,
            3u,11u,1u,9u,
            15u,7u,13u,5u
        };
        // Add a shift to the LUT index based on instance and frame offset
        uint shift = (sgu_frameOffset) & 15u;
        uint t = bayer4[(bx + (by << 2) + shift) & 15u];
        half cutoff = (t + 0.5) / 16.0;

        // Binary decision: keep pixel if alpha exceeds threshold, otherwise discard.
        if (alpha <= cutoff)
            discard;
        alpha = 1;
        o.motion = half2(i.vel);
    }

    o.col = half4(i.col.rgb, alpha);
    return o;
}
ENDCG
        }
    }
}
