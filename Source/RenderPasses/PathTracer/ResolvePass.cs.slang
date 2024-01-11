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
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.RTXDI.RTXDI;
import LoadShadingData;
import PathTracer;
import PathState;
import ColorType;
import PhotonMap;
import Params;
import PathReservoir;

/** This pass combines reservoir samples with the light trace image, and writes the final color
    to the output buffer.
*/
struct ResolvePass
{
    // Resources
    PathTracerParams params;                        ///< Runtime parameters.

    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> lastVbuffer;           ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<float3> lastViewDir;                  ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<float2> motionVecs;                   ///< Motion vectors.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.

    Texture2D<uint> sampleCount;

    RWByteAddressBuffer lightImage;                 ///< Input light traced image. Only valid when kUseBPT == true.

    LightVertexCache lightVertexCache;              ///< Light vertex cache.
    PhotonMap photonMap;                            ///< Photon map.
    RWStructuredBuffer<PathReservoir> pathReservoirs; ///< Path reservoirs.
    StructuredBuffer<PathReservoir> lastPathReservoirs; ///< Path reservoirs.


    /** Entry point for the resolve pass.
        \param[in] pixel Pixel index.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;

        if (kLightTraceOnly)
        {
            outputColor[pixel] = float4(lightImage.Load<float3>(sizeof(float3) * (pixel.y * params.frameDim.x + pixel.x)), 1.f);
            return;
        }

        PathReservoir pixelReservoir;
        const uint idx = pixel.y * params.frameDim.x + pixel.x;
        if ((params.spatialReusePasses & 1) == 0)
        {
            pixelReservoir = pathReservoirs[idx];
        }
        else
        {
            pixelReservoir = lastPathReservoirs[idx];
        }

        float3 L = pixelReservoir.getRadiance();

        if (kDebugMIS)
        {
            L = pixelReservoir.sampleF / pixelReservoir.M;
        }

        if (kUseBPT)
        {
            // Add light traced contribution without resampling.
            L += lightImage.Load<float3>(sizeof(float3) * (pixel.y * params.frameDim.x + pixel.x));
        }

        // Write final sample color.
        outputColor[pixel] = float4(L, 1.f);
    }
};

cbuffer CB
{
    ResolvePass gResolvePass;
}

[numthreads(8, 4, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    gResolvePass.execute(id.xy);
}
