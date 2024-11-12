/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * Temporal accumulation render pass.
 *
 * There are entry points for each of the three supported accumulation modes.
 * Note that for the compensated summation mode the shader _must_ be compiled
 * for precise floating-point operations (no reordering).
 *
 * In all modes, the shader writes the current accumulated average to the
 * output texture. The intermediate buffers are internal to the pass.
 */

import Utils.Math.MathHelpers;

cbuffer PerFrameCB
{
    uint2 gResolution;
    uint  gAccumCount;
    uint  gMaxAccum;
    float gMaxDistance;
    float gCosMaxAngle;
    bool  gAccumulate;
}

// Input data to accumulate and accumulated output.
#define INPUT_FORMAT_FLOAT 0
#define INPUT_FORMAT_UINT 1
#define INPUT_FORMAT_SINT 2

#if _INPUT_FORMAT == INPUT_FORMAT_FLOAT
typedef float4 InputDataType;
#elif _INPUT_FORMAT == INPUT_FORMAT_UINT
typedef uint4 InputDataType;
#elif _INPUT_FORMAT == INPUT_FORMAT_SINT
typedef int4 InputDataType;
#else
#error "Invalid format"
#endif

#ifndef _USE_MOTION_VECS
#define _USE_MOTION_VECS 0
#endif
static const bool kUseMotionVectors = (bool)_USE_MOTION_VECS;
#ifndef _USE_GBUFFER
#define _USE_GBUFFER 0
#endif
static const bool kUseGbuffer = (bool)_USE_GBUFFER;

Texture2D<InputDataType> gCurFrame;
RWTexture2D<float4>      gOutputFrame;

Texture2D<float2>        gMotionVectors;
Texture2D<float4>        gCurPosW;
Texture2D<float2>        gCurNormalW;
Texture2D<float4>        gLastPosW;
Texture2D<float2>        gLastNormalW;

// frame data, format depends on accumulation mode.
RWTexture2D<uint>   gSampleCount;
RWTexture2D<float4> gFrameSum;  // If mode is Single or SingleKahan
RWTexture2D<float4> gFrameCorr; // If mode is SingleKahan
RWTexture2D<uint4>  gFrameSumLo; // If mode is Double
RWTexture2D<uint4>  gFrameSumHi; // If mode is Double

// Last frame data, format depends on accumulation mode.
Texture2D<uint>   gLastSampleCount;
Texture2D<float4> gLastFrameSum;  // If mode is Single or SingleKahan
Texture2D<float4> gLastFrameCorr; // If mode is SingleKahan
Texture2D<uint4>  gLastFrameSumLo; // If mode is Double
Texture2D<uint4>  gLastFrameSumHi; // If mode is Double

bool reproject(uint2 pixelPos, out int2 p, out uint accumCount)
{
    p = pixelPos;
    accumCount = gAccumCount;

    if (kUseMotionVectors)
    {
        p = int2((float2(pixelPos) + 0.5) + float2(gResolution) * gMotionVectors[pixelPos]);
        if (any(p < 0) || any(p >= gResolution))
        {
            accumCount = 0;
            return false;
        }

        accumCount = gLastSampleCount[p];
    }

    if (kUseGbuffer)
    {
        float3 posW = gCurPosW[pixelPos].xyz;
        float3 normalW = oct_to_ndir_snorm(gCurNormalW[pixelPos]);
        float3 prevPosW = gLastPosW[p].xyz;
        float3 prevNormalW = oct_to_ndir_snorm(gLastNormalW[p]);

        if (length(posW - prevPosW) > gMaxDistance || dot(normalW, prevNormalW) < gCosMaxAngle)
        {
            accumCount = 0;
            return false;
        }
    }

    return true;

}

/**
 * Single precision standard summation.
 */
[numthreads(16, 16, 1)]
void accumulateSingle(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate)
    {
        float4 lastSum = 0;
        uint accumCount;
        int2 prevPixel;
        if (reproject(pixelPos, prevPixel, accumCount))
        {
            lastSum = gLastFrameSum[prevPixel];
        }

        if (gMaxAccum > 0)
        {
            // Exponential weighted moving average mode.
            // Blend the accumulated frame with the current frame.
            output = lerp(lastSum, curColor, 1.0 / (min(accumCount, gMaxAccum) + 1));

            gFrameSum[pixelPos] = output;
        }
        else
        {
            // High precision mode with no frame limit.
            // Fetch previous sum and compute the new sum.
            float4 sum = lastSum + curColor;

            output = sum / (accumCount + 1);

            gFrameSum[pixelPos] = sum;
        }

        if (kUseMotionVectors)
            gSampleCount[pixelPos] = accumCount + 1;
    }
    else
    {
        output = curColor;
    }

    gOutputFrame[pixelPos] = output;
}

/**
 * Single precision compensated summation.
 */
[numthreads(16, 16, 1)]
void accumulateSingleCompensated(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate)
    {
        float4 sum = 0;
        float4 c = 0;
        uint accumCount;
        int2 prevPixel;
        if (reproject(pixelPos, prevPixel, accumCount)) {
            // Fetch the previous sum and running compensation term.
            sum = gLastFrameSum[prevPixel];
            // c measures how large (+) or small (-) the current sum is compared to what it should be.
            c = gLastFrameCorr[prevPixel];
        }

        // Adjust current value to minimize the running error.
        // Compute the new sum by adding the adjusted current value.
        float4 y = curColor - c;
        // The value we'll see in 'sum' on the next iteration.
        float4 sumNext = sum + y;
        output = sumNext / (accumCount + 1);

        gFrameSum[pixelPos] = sumNext;
        // Store new correction term.
        gFrameCorr[pixelPos] = (sumNext - sum) - y;

        if (kUseMotionVectors)
            gSampleCount[pixelPos] = accumCount + 1;
    }
    else
    {
        output = curColor;
    }

    gOutputFrame[pixelPos] = output;
}

/**
 * Double precision standard summation.
 */
[numthreads(16, 16, 1)]
void accumulateDouble(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gResolution))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;
    const float4 curColor = gCurFrame[pixelPos];

    float4 output;
    if (gAccumulate)
    {
        // Fetch the previous sum in double precision.
        // There is no 'double' resource format, so the bits are stored in two uint4 textures.
        uint4 sumLo = 0;
        uint4 sumHi = 0;
        uint accumCount;
        int2 prevPixel;
        if (reproject(pixelPos, prevPixel, accumCount)) {
            sumLo = gLastFrameSumLo[prevPixel];
            sumHi = gLastFrameSumHi[prevPixel];
        }

        double sum[4];

        if (gMaxAccum > 0)
        {
            double curWeight = 1.0 / (min(accumCount, gMaxAccum) + 1);

            // Exponential weighted moving average mode.
            for (int i = 0; i < 4; i++)
            {
                sum[i] = asdouble(sumLo[i], sumHi[i]);
                sum[i] = sum[i] * (1.0 - curWeight) + (double)curColor[i] * curWeight;
                asuint(sum[i], sumLo[i], sumHi[i]);
                output[i] = (float)sum[i];
            }
        }
        else
        {
            double curWeight = 1.0 / (gMaxAccum + 1);

            // High precision mode with no frame limit.
            for (int i = 0; i < 4; i++)
            {
                sum[i] = asdouble(sumLo[i], sumHi[i]);
                sum[i] += (double)curColor[i];
                asuint(sum[i], sumLo[i], sumHi[i]);
                output[i] = (float)(sum[i] * curWeight);
            }
        }

        gFrameSumLo[pixelPos] = sumLo;
        gFrameSumHi[pixelPos] = sumHi;

        if (kUseMotionVectors)
            gSampleCount[pixelPos] = accumCount + 1;
    }
    else
    {
        output = curColor;
    }

    gOutputFrame[pixelPos] = output;
}
