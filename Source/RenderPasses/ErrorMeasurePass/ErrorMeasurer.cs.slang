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
Texture2D<float4> gReference;
Texture2D<float4> gSource;
Texture2D<float4> gWorldPosition;
RWTexture2D<float4> gResult;

cbuffer PerFrameCB
{
    uint2 gResolution;
    uint gIgnoreBackground;
    uint gComputeDiffSqr;
    uint gComputeAverage;
    uint gComputePercentage;
    float gDifferenceOffset;
};

[numthreads(16, 16, 1)]
void main(uint3 DTid: SV_DispatchThreadID)
{
    const uint2 pixel = DTid.xy;
    if (any(pixel >= gResolution))
        return;

    // Determine if we should include this pixel or not.
    const float4 worldPos = gWorldPosition[pixel];
    const bool isForeground = worldPos.w != 0.0f; // We're using the w-component to identify valid pixels.

    const bool isPixelValid = !gIgnoreBackground || isForeground;

    float3 src = float3(0.0f);
    float3 ref = float3(0.0f);
    float3 diff = float3(0.0f);
    if (isPixelValid) {
        src = gSource[pixel].rgb;
        ref = gReference[pixel].rgb;
        diff = abs((src - ref) + gDifferenceOffset);
    }

    // Compute error based on the current options.
    if (gComputePercentage != 0) {
        if (ref.r > 0) diff.r /= ref.r;
        if (ref.g > 0) diff.g /= ref.g;
        if (ref.b > 0) diff.b /= ref.b;
    }
    if (gComputeDiffSqr != 0) diff *= diff;
    if (gComputeAverage != 0) diff.rgb = (diff.r + diff.g + diff.b) / 3.f;

    gResult[pixel] = float4(diff, 0.0f);
}
