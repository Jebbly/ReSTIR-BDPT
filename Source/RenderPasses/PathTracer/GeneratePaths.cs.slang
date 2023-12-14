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

/** Helper struct for generating paths in screen space.

    The dispatch size is one thread group per screen tile. A warp is assumed to be 32 threads.
    Within a thread group, the threads are linearly indexed and mapped to pixels in Morton order.

    Output sample buffer
    --------------------

    For each pixel that belongs to the background, and hence does not need to be path traced,
    we directly evaluate the background color and write all samples to the output sample buffers.

    The output sample buffers are organized by tiles in scanline order. Within tiles,
    the pixels are enumerated in Morton order with all samples for a pixel stored consecutively.

    When the number of samples/pixel is not fixed, we additionally write a 2D lookup table,
    for each pixel storing the tile-local offset to where the first sample is stored.
    Based on this information, subsequent passes can easily find the location of a given sample.
*/
struct PathGenerator
{
    // Resources
    PathTracerParams params;                        ///< Runtime parameters.

    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.

    Texture2D<uint> sampleCount;

    RWByteAddressBuffer lightImage;                 ///< Input light traced image. Only valid when kUseBPT == true.

    LightVertexCache lightVertexCache;              ///< Light vertex cache.
    PhotonMap photonMap;                            ///< Photon map.
    RWStructuredBuffer<PathReservoir> pathReservoirs; ///< Path reservoirs.

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;

    /** Entry point for path generator.
        \param[in] tileID Tile ID in x and y on screen.
        \param[in] threadIdx Thread index within the tile.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;

        // Process each pixel.

        // Compute the primary ray.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];

        // Load the primary hit from the V-buffer.
        const HitInfo hit = HitInfo(vbuffer[pixel]);
        bool hitSurface = hit.isValid();

        // If we don't hit any surface then the background will be evaluated and written out directly.
        if (!hitSurface)
        {
            // Evaluate background color for the current pixel.
            float3 color = float3(0.f);
            if (kUseEnvLight)
            {
                color = gScene.envMap.eval(cameraRay.dir);
            }

            outputColor[pixel] = float4(color, 1.f);
        }

        // Prepare per-pixel surface data for RTXDI.
        if (kUseRTXDI)
        {
            bool validSurface = false;
            if (hitSurface)
            {
                let lod = ExplicitLodTextureSampler(0.f);
                ShadingData sd = loadShadingData(hit, cameraRay.origin, cameraRay.dir, lod);

                // Create material instance and query its properties.
                let hints = getMaterialInstanceHints(hit, true /* primary hit */);
                let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
                let bsdfProperties = mi.getProperties(sd);

                // Check for BSDF lobes that RTXDI can sample.
                uint lobeTypes = mi.getLobeTypes(sd);
                validSurface = (lobeTypes & (uint)LobeType::NonDeltaReflection) != 0;

                if (validSurface)
                {
                    // RTXDI uses a simple material model with only diffuse and specular reflection lobes.
                    // We query the BSDF for the diffuse albedo and specular reflectance.
                    gRTXDI.setSurfaceData(pixel, sd.computeRayOrigin(), bsdfProperties.guideNormal, bsdfProperties.diffuseReflectionAlbedo, bsdfProperties.specularReflectance, bsdfProperties.roughness);
                }
            }
            if (!validSurface) gRTXDI.setInvalidSurfaceData(pixel);
        }
    }
};

cbuffer CB
{
    PathGenerator gPathGenerator;
}

[numthreads(8, 4, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    gPathGenerator.execute(id.xy);
}
