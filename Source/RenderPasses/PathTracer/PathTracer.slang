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
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.EmissiveLightSampler;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import Rendering.RTXDI.RTXDI;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDData;
import RenderPasses.Shared.Denoising.NRDHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import LoadShadingData;
import ColorType;
import NRDHelpers;
import PathTracerHelpers;
__exported import PathState;
__exported import Params;

/** Path tracer.

    This implements the high-level path tracing logic and is shared among
    different scheduling implementations.
*/
struct PathTracer
{
    /** Interface for querying visibility in the scene.
        This is used in `handleHit`.
    */
    interface IVisibilityQuery
    {
        /** Trace a visibility ray against the scene.
            \param[in] ray Ray.
            \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
        */
        [mutating] bool traceVisibilityRay(const Ray ray);
    };

    PathTracerParams params;                        ///< Runtime parameters.

    // Samplers
    LightSampler lightSampler;                      ///< Light sampler.

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<uint> sampleCount;                    ///< Optional input sample count buffer. Only valid when kSamplesPerPixel == 0.
    Texture2D<uint> sampleOffset;                   ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.
    RWByteAddressBuffer lightImage;                 ///< Input light traced image. Only valid when kUseBPT == true. Added to the output when kSamplesPerPixel == 1.

    // Outputs
    RWStructuredBuffer<ColorType> sampleColor;      ///< Output per-sample color if kSamplesPerPixel != 1.
    RWStructuredBuffer<GuideData> sampleGuideData;  ///< Output per-sample guide data.
    NRDBuffers outputNRD;                           ///< Output NRD data.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Additional specialization.
    static const bool kOutputGuideData   = OUTPUT_GUIDE_DATA;
    static const bool kUseBPT            = USE_BPT;
    #ifdef LIGHT_TRACE_PASS
    static const bool kTraceFromLight    = true;
    #else
    static const bool kTraceFromLight    = false;
    #endif

    float getImagePlaneDist() { return params.frameDim.y * (gScene.camera.data.focalLength / gScene.camera.data.frameHeight); }
    float getMisVmWeightFactor() { return 0; }

    /** Set guiding data when background is hit.
    */
    static void setBackgroundGuideData(inout GuideData guideData, const float3 dir, const float3 Le)
    {
        if (kOutputGuideData)
        {
            guideData.setGuideNormal(-dir);
            // Compress dynamic range similar to UE4.
            const float3 compressedColor = pow(Le / (Le + 1.f), 0.454545f);
            guideData.setAlbedo(compressedColor);
            guideData.setSpecularAlbedo(float3(0.f));
            guideData.setIndirectAlbedo(float3(0.f));
            guideData.setReflectionPos(float3(0.f));
        }
    }

    /** Set guiding data when primary surface is hit.
    */
    static void setPrimarySurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (kOutputGuideData)
        {
            guideData.setGuideNormal(bsdfProperties.guideNormal);
            // Use sum of reflection/transmission albedo as they are denoised together.
            guideData.setAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            guideData.setReflectionPos(sd.posW);

            const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;

            // We use an approximate shading model with a single specular lobe.
            // TODO: Generalize this to arbitrary materials.
            if (sd.mtl.getMaterialType() == MaterialType::Hair)
            {
                // Use a simple heuristic for hair material.
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(float3(0.04f), ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
            else
            {
                // Use pre-integrated FG term for specular
                const float3 preintegratedSpecular = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
                guideData.setSpecularAlbedo(preintegratedSpecular);
            }
        }
    }

    /** Set guiding data when indirect surface is hit.
    */
    static void setIndirectSurfaceGuideData(inout GuideData guideData, const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        if (any(bsdfProperties.emission > 0.f))
        {
            float3 indirectBaseColor = pow(bsdfProperties.emission / (bsdfProperties.emission + 1.0f), 0.454545f);
            guideData.setIndirectAlbedo(indirectBaseColor);
        }
        else
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            guideData.setIndirectAlbedo(bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
        }
        guideData.setReflectionPos(sd.posW);
    }

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;

        const uint2 pixel = path.getPixel();

        if (kTraceFromLight)
        {
            // Create sample generator.
            path.sg = SampleGenerator(pixel, -params.seed);

            LightSample ls;
            lightSampler.generateLightSample(params.sceneSphere, path.sg, ls);

            path.thp = ls.Li;
            path.origin = ls.origin;
            path.dir = ls.dir;

            path.setVertexIndex(0);
        }
        else
        {
            path.thp = float3(1.f);

            // Create primary ray.
            Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
            if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
            path.origin = cameraRay.origin;
            path.dir = cameraRay.dir;

            // Create sample generator.
            const uint maxSpp = kSamplesPerPixel > 0 ? kSamplesPerPixel : kMaxSamplesPerPixel;
            path.sg = SampleGenerator(pixel, params.seed * maxSpp + path.getSampleIdx());

            // Load the primary hit info from the V-buffer.
            const HitInfo hit = HitInfo(vbuffer[pixel]);

            // If invalid, the path is still active and treated as a miss.
            if (hit.isValid())
            {
                path.setHit(hit);
                path.setVertexIndex(1);
            }
        }
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Compute a set of coherence hints for a given path state.
        These hints predict the code path taken in the next call to handleHit() with the given path state.
        They are used by reordering schedulers to extract coherence.
        \param[in] path Path state.
        \param[in] isDeltaSpecular Hit material is pure delta specular.
        \param[in] isEmissive Hit material is emissive.
        \return Returns a bit mask of coherence hints. MSB is highest priority hint. Currently 4 bits are used.
    */
    uint getCoherenceHints(const PathState path, const bool isDeltaSpecular, const bool isEmissive)
    {
        bool terminatedByRussianRoulette = {}; ///< handleHit() will be terminated early due to russian roulette.
        bool terminated = {};                  ///< handleHit() will be terminated early.
        bool samplesLight = {};                ///< handleHit() will sample a light.
        bool computesEmissive = {};            ///< handleHit() will compute emission.

        // Determines if handleHit() will return early due to russian roulette.
        if (kUseRussianRoulette)
        {
            // This is under the assumption that the first random number drawn from the sample generator is used for russian roulette.
            // Keep in sync with handleHit().
            PathState tmpPath = path;
            terminatedByRussianRoulette = terminatePathByRussianRoulette(tmpPath, sampleNext1D(tmpPath.sg));
        }

        // Determines if handleHit() will return early.
        terminated = hasFinishedSurfaceBounces(path);
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isDeltaSpecular)
            terminated = true;

        // Determines if handleHit() will sample a light.
        samplesLight = true;
        if (isDeltaSpecular)
            samplesLight = false;
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
            samplesLight = false;

        // Determines if handleHit() will compute emission.
        computesEmissive = isEmissive && path.isLightSamplable();
        if (kUseRTXDI && path.getVertexIndex() == 2 && !isDeltaSpecular)
            computesEmissive = false;

        // MSB = highest priority hint.
        return
            select(computesEmissive, 0x1, 0x0) |
            select(samplesLight, 0x2, 0x0) |
            select(terminated, 0x4, 0x0) |
            select(terminatedByRussianRoulette, 0x8, 0x0);
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    /** Add radiance to the path contribution.
        \param[in,out] path Path state.
        \param[in] radiance Vertex radiance.
    */
    void addToPathContribution(inout PathState path, const float3 radiance)
    {
        path.L += path.thp * radiance;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;
        bool valid = mi.sample(sd, path.sg, result, kUseBSDFSampling);
        if (valid) valid = generateScatterRay(result, sd, mi, path);

        // Ignore valid on purpose for now.
        if (!kTraceFromLight && kOutputNRDData)
        {
            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool hasDeltaTransmissionLobe = (lobeTypes & (uint)LobeType::DeltaTransmission) != 0;
            const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

            if (path.getVertexIndex() == 1)
            {
                path.setDiffusePrimaryHit(result.isLobe(LobeType::Diffuse));
                path.setSpecularPrimaryHit(result.isLobe(LobeType::Specular));

                if (kOutputNRDAdditionalData)
                {
                    // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
                    path.setDeltaOnlyPath(result.isLobe(LobeType::DeltaReflection) || result.isLobe(LobeType::DeltaTransmission));

                    path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType::DeltaReflection));
                    path.setDeltaTransmissionPath(result.isLobe(LobeType::DeltaTransmission));
                }
            }

            if (path.getVertexIndex() > 1)
            {
                if (hasNonDeltaLobes) path.setDeltaOnlyPath(false);

                if (kOutputNRDAdditionalData && path.isDeltaTransmissionPath() && path.isDeltaOnlyPath() && hasDeltaTransmissionLobe)
                {
                    if (result.isLobe(LobeType::DeltaReflection) && !isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
                    {
                        path.setDeltaTransmissionPath(false);
                    }
                }
            }
        }

        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (path.hit.getType() == HitType::Curve);
        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        path.dir = bs.wo;
        updatePathThroughput(path, bs.weight);
        path.pdf = bs.pdf;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = mi.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > params.specularRoughnessThreshold;

            if (isDiffuse)
            {
                path.incrementBounces(BounceType::Diffuse);
            }
            else
            {
                path.incrementBounces(BounceType::Specular);
                path.setSpecular();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            path.incrementBounces(BounceType::Transmission);
            path.setTransmission();

            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.frame.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }

            if (kTraceFromLight && !bs.isLobe(LobeType::DiffuseTransmission))
            {
                float eta = mi.getProperties(sd).eta;
                path.thp *= 1.f / (eta*eta);
            }
        }

        // Save the normal used for NEE. This is needed for MIS.
        path.normal = sd.getOrientedFaceNormal();

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);

        return valid;
    }

    /** Connect a light sub-path to the camera to form a full path. Splats the contribution to lightImage.
     */
    void connectToCamera<VisibilityQuery : IVisibilityQuery>( const ShadingData sd, const IMaterialInstance mi, inout PathState path, inout VisibilityQuery vq )
    {
        const uint bounces = path.getBounces(BounceType::Diffuse) + path.getBounces(BounceType::Specular) + path.getBounces(BounceType::Transmission);
        if (params.debugLightVertexCount != -1 && params.debugLightVertexCount != bounces + 2)
            return;

        float3 toCamera = gScene.camera.getPosition() - sd.posW;
        const float dist = length(toCamera);
        toCamera /= dist;

        // Reject sample if not in the hemisphere of a BSDF lobe.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasReflection   = lobeTypes & uint(LobeType::Reflection);
        const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
        float NdotL = dot(sd.getOrientedFaceNormal(), toCamera);
        if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
            return;

        if (gScene.camera.data.focalLength == 0.0f)
        {
            // TODO: ortho camera
            return;
        }

        const float cosCamera = -dot(toCamera, normalize(gScene.camera.data.cameraW));

        float4 clip = mul(gScene.camera.data.viewProjMat, float4(sd.posW, 1));
        clip.y = -clip.y;
        const int2 pixel = int2(((clip.xy / clip.w) * .5 + .5) * params.frameDim);
        if (any(pixel < 0) || any(pixel >= params.frameDim) || cosCamera <= 0)
            return;

        const float imagePointToCameraDist = getImagePlaneDist() / cosCamera;
        const float imageToSolidAngleFactor = (imagePointToCameraDist * imagePointToCameraDist) / cosCamera;

        const float invDist2 = 1.f / (dist * dist);

        // Compute MIS weight
        const float cameraPdfA = imageToSolidAngleFactor * abs(dot(sd.faceN, toCamera)) * invDist2;
        ShadingData revSd = sd;
        revSd.V = toCamera;
        const float bsdfRevPdfW = mi.evalPdf(revSd, sd.V, kUseBSDFSampling);
        const float wLight = evalMIS(1, cameraPdfA / params.lightSubpathCount) * (
            getMisVmWeightFactor() + path.dVCM +
            path.dVC * evalMIS(1, bsdfRevPdfW));
        const float misWeight = (kLightTraceOnly || params.debugLightVertexCount == -1) ? 1.f : (1.f / (wLight + 1.f));

        // Compute full path contribution
        const float3 f = path.thp * mi.eval(sd, toCamera, path.sg) * getShadingNormalCorrection(sd, toCamera) * (misWeight * imageToSolidAngleFactor * invDist2 / params.lightSubpathCount);
        if (!any(f > 0))
            return;

        // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
        const float3 origin = sd.computeRayOrigin(NdotL >= 0.f);

        // Test visibility by tracing a shadow ray.
        bool V = vq.traceVisibilityRay(Ray(origin, toCamera, 0, dist));
        if (!V)
            return;

        uint addr = sizeof(float3) * (pixel.y * params.frameDim.x + pixel.x);
        lightImage.InterlockedAddF32(addr + 0, f.r);
        lightImage.InterlockedAddF32(addr + 4, f.g);
        lightImage.InterlockedAddF32(addr + 8, f.b);
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = (kUseBPT && !kLightTraceOnly) ? params.contProb : luminance(path.thp);
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            path.terminate();
            return true;
        }
        path.thp /= 1.f - prob;
        return false;
    }

    /** Helper to create a texture sampler instance.
        The method for computing texture level-of-detail depends on the configuration.
        \param[in] path Path state.
        \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
        \return Texture sampler instance.
    */
    ITextureSampler createTextureSampler(const PathState path, bool isPrimaryHit, bool isTriangleHit)
    {
        if (kPrimaryLodMode == TexLODMode::RayDiffs && isPrimaryHit && isTriangleHit)
        {
            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return ExplicitGradientTextureSampler(ddx, ddy);
        }
        else
        {
            float lod = isPrimaryHit ? 0.f : params.lodBias;
            return ExplicitLodTextureSampler(lod);
        }
    }

    /** Helper to compute the MIS weight when an emissive surface is hit.
        \param[in] path Path state.
        \param[in] lightPdf PDF of sampling the current vertex using NEE.
        \param[in] lightEmitPdf PDF of sampling a light ray from the current vertex.
        \return MIS weight.
    */
    float computeHitMIS(const PathState path, float lightPdf, float lightEmitPdf)
    {
        if (kUseBPT)
        {
            return evalMIS(1, path.pdf, 1, lightPdf);
        }
        else
        {
            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            return evalMIS(1, path.pdf, 1, lightPdf);
        }
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, lod);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
            updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(vp, length(sd.posW - path.origin)));
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create BSDF instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        float3 attenuatedEmission = 0.f;
        if (kTraceFromLight)
        {
            // Store light sub-path vertex

        }
        else
        {
            // Compute emission at hit

            // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
            bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
            if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
            {
                sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
            }

            // Optionally disable emission inside volumes.
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
            {
                bsdfProperties.emission = float3(0.f);
            }

            // Check if the scatter event is samplable by the light sampling technique.
            const bool isLightSamplable = path.isLightSamplable();

            // Add emitted radiance.
            // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
            bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

            // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
            // Skip any additional contribution on the secondary hit unless it comes from a scatter event
            // that RTXDI cannot handle, such as transmission or delta scattering events.
            if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEmissive = false;

            if (kUseBPT && params.debugLightVertexCount > 0) computeEmissive = false;

            if (computeEmissive && any(bsdfProperties.emission > 0.f))
            {
                float misWeight = 1.f;
                if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
                {
                    // If NEE and MIS are enabled, and we've already sampled emissive lights,
                    // then we need to evaluate the MIS weight here to account for the remaining contribution.
                    // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                    // Prepare hit point struct with data needed for emissive light PDF evaluation.
                    TriangleHit triangleHit = path.hit.getTriangleHit();
                    TriangleLightHit hit;
                    hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                    hit.posW = sd.posW;
                    hit.normalW = sd.getOrientedFaceNormal();

                    // Evaluate PDF at the hit, had it been generated with light sampling.
                    // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                    bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                    float lightPdf = lightSampler.getEmissiveSelectionProbability() * lightSampler.emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                    const float cosTheta = abs(dot(sd.V, sd.faceN));
                    misWeight = computeHitMIS(path, lightPdf, lightPdf * cosTheta * M_1_PI);
                }

                if (kUseBPT && params.debugLightVertexCount == 0) misWeight = 1.f;

                // Accumulate emitted radiance weighted by path throughput and MIS weight.
                addToPathContribution(path, misWeight * bsdfProperties.emission);

                attenuatedEmission = path.thp * misWeight * bsdfProperties.emission;
            }
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.frame.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.frame.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeRayOrigin();
        }

        // Determine if BSDF has non-delta lobes.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        if (kTraceFromLight && hasNonDeltaLobes)
        {
            connectToCamera(sd, mi, path, vq);
        }
        else
        {
            // Check if we should apply NEE.
            const bool applyNEE = kUseNEE && hasNonDeltaLobes;

            // Check if sample from RTXDI should be applied instead of NEE.
            const bool applyRTXDI = kUseRTXDI && isPrimaryHit && hasNonDeltaLobes;

            // TODO: Support multiple shadow rays.
            path.setLightSampled(false, false);
            if (applyNEE || applyRTXDI)
            {
                LightSample ls = {};
                bool validSample = false;

                if (applyRTXDI)
                {
                    // Query final sample from RTXDI.
                    validSample = gRTXDI.getFinalSample(path.getPixel(), ls.dir, ls.distance, ls.Li);
                    ls.origin = path.origin;
                }
                else
                {
                    // Determine if upper/lower hemispheres need to be sampled.
                    bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
                    if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                    bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);

                    // Sample a light.
                    validSample = lightSampler.generateLightSample(sd, sampleUpperHemisphere, sampleLowerHemisphere, params.sceneSphere, path.sg, ls);
                    path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
                }

                if (validSample)
                {
                    float misWeight = 1.f;
                    // Apply MIS weight.
                    if (kUseMIS && !applyRTXDI && ls.lightType != (uint)LightSampleType::Analytic)
                    {
                        float scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling);
                        misWeight = evalMIS(1, ls.pdf, 1, scatterPdf);
                    }

                    if (kUseBPT && params.debugLightVertexCount == 1) misWeight = 1.f;

                    ls.Li *= misWeight;

                    float3 weight = mi.eval(sd, ls.dir, path.sg);
                    float3 Lr = weight * ls.Li;
                    if (any(Lr > 0.f))
                    {
                        Ray ray = ls.getVisibilityRay();

                        if (isCurvePolyTubeHit)
                        {
                            // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                            // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                            if (dot(sd.frame.N, ray.dir) < 0.f)
                            {
                                ray.origin = ray.origin - sd.frame.N * sd.curveRadius * 2.1f;
                            }
                        }

                        logTraceRay(PixelStatsRayType::Visibility);
                        bool visible = vq.traceVisibilityRay(ray);
                        if (visible) addToPathContribution(path, Lr);
                    }
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg))) return;
        }

        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        if (kTraceFromLight)
        {
            // Terminate caustics paths.
            if (kDisableCaustics && path.isSpecular()) valid = false;
        }
        else
        {
            // Output guide data.
            if (path.getVertexIndex() == 1)
            {
                setPrimarySurfaceGuideData(path.guideData, sd, bsdfProperties);
            }
            if (path.getVertexIndex() == 2 && (path.getBounces(BounceType::Specular) == 1 || path.getBounces(BounceType::Transmission) == 1))
            {
                setIndirectSurfaceGuideData(path.guideData, sd, bsdfProperties);
            }

            if (kOutputNRDData)
            {
                const uint2 pixel = path.getPixel();
                const uint outSampleIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

                setNRDPrimaryHitEmission(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, attenuatedEmission);
                setNRDPrimaryHitReflectance(outputNRD, kUseNRDDemodulation, path, pixel, isPrimaryHit, sd, bsdfProperties);

                setNRDSampleHitDist(outputNRD, path, outSampleIdx);
                setNRDSampleEmission(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, attenuatedEmission, wasDeltaOnlyPathBeforeScattering);
                setNRDSampleReflectance(outputNRD, kUseNRDDemodulation, path, outSampleIdx, isPrimaryHit, sd, bsdfProperties, lobeTypes, wasDeltaOnlyPathBeforeScattering);
                setNRDPrimaryHitSeparatedRadiance(outputNRD, kUseNRDDemodulation, path, isPrimaryHit, outSampleIdx);
            }

            // Check if this is the last path vertex.
            const bool isLastVertex = hasFinishedSurfaceBounces(path);

            // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
            if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

            // Terminate caustics paths.
            if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;
        }

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        if (!kTraceFromLight)
        {
            // Upon miss:
            // - Compute MIS weight if previous path vertex sampled a light
            // - Evaluate environment map
            // - Write guiding data
            // - Terminate the path

            // Check if the scatter event is samplable by the light sampling technique.
            const bool isLightSamplable = path.isLightSamplable();

            // Add env radiance.
            bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

            // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
            // Skip any additional contribution on the secondary hit unless it comes from a scatter event
            // that RTXDI cannot handle, such as transmission, delta or volume scattering events.
            if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEnv = false;

            if (kUseBPT && params.debugLightVertexCount > 0) computeEnv = false;

            float3 emitterRadiance = 0.f;

            if (computeEnv)
            {
                logPathVertex();

                float misWeight = 1.f;
                if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
                {
                    // If NEE and MIS are enabled, and we've already sampled the env map,
                    // then we need to evaluate the MIS weight here to account for the remaining contribution.

                    // Evaluate PDF, had it been generated with light sampling.
                    float lightPdf = lightSampler.getEnvMapSelectionProbability() * lightSampler.envMapSampler.evalPdf(path.dir);

                    misWeight = computeHitMIS(path, lightPdf, lightPdf * getSceneSpherePdf(params.sceneSphere));
                }

                if (kUseBPT && params.debugLightVertexCount == 0) misWeight = 1;

                float3 Le = lightSampler.envMapSampler.eval(path.dir);
                emitterRadiance = misWeight * Le;
                addToPathContribution(path, emitterRadiance);

                if (kOutputGuideData && path.getVertexIndex() == 2
                    && (path.getBounces(BounceType::Specular) == 1
                    || path.getBounces(BounceType::Transmission) == 1))
                {
                    // Compress dynamic range similar to UE4.
                    float3 compressedColor = pow(Le / (Le + 1.0f), 0.454545f);
                    path.guideData.setIndirectAlbedo(compressedColor);
                    path.guideData.setReflectionPos(path.dir * kEnvMapDepth);
                }
            }

            if (kOutputGuideData && path.getVertexIndex() == 1)
            {
                path.guideData.setGuideNormal(-path.dir);
            }

            if (kOutputNRDData)
            {
                const uint outSampleIdx = params.getSampleOffset(path.getPixel(), sampleOffset) + path.getSampleIdx();
                setNRDSampleHitDist(outputNRD, path, outSampleIdx);
            }

        #if defined(DELTA_REFLECTION_PASS)
            if (path.isDeltaReflectionPrimaryHit())
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
            }
            else
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
            }
        #elif defined(DELTA_TRANSMISSION_PASS)
            if (path.isDeltaTransmissionPath())
            {
                writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, path.thp * emitterRadiance, -path.dir, 0.f, kNRDInvalidPathLength, path.origin + path.dir * kNRDInvalidPathLength);
            }
            else
            {
                writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, 0.f);
            }
        #endif
        }

        path.terminate();
    }

    void writeLightImageOutput(const uint2 pixel)
    {
        if (!kTraceFromLight && kUseBPT && kSamplesPerPixel == 1)
        {
            float3 L = lightImage.Load<float3>(sizeof(float3) * (pixel.y * params.frameDim.x + pixel.x));
            if (!kLightTraceOnly)
                L += outputColor[pixel].rgb;
            assert(!any(isnan(L)));
            outputColor[pixel] = float4(L, 1.f);
        }
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(const PathState path)
    {
        if (kTraceFromLight)
            return;

        assert(!any(isnan(path.L)));

        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        const uint outIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

        if (kSamplesPerPixel == 1)
        {
            // Write color directly to frame buffer.
            outputColor[pixel] = float4(path.L, 1.f);
        }
        else
        {
            // Write color to per-sample buffer.
            sampleColor[outIdx].set(path.L);
        }

        if (kOutputGuideData)
        {
            sampleGuideData[outIdx] = path.guideData;
        }

        if (kOutputNRDData)
        {
            // TODO: Optimize this for 1 SPP. It doesn't have to go through resolve pass like the color above.
            NRDRadiance data = {};

            if (path.isDiffusePrimaryHit()) data.setPathType(NRDPathType::Diffuse);
            else if (path.isSpecularPrimaryHit()) data.setPathType(NRDPathType::Specular);
            else if (path.isDeltaReflectionPrimaryHit()) data.setPathType(NRDPathType::DeltaReflection);
            else if (path.isDeltaTransmissionPath()) data.setPathType(NRDPathType::DeltaTransmission);
            else data.setPathType(NRDPathType::Residual);

            data.setRadiance(path.L);

            outputNRD.sampleRadiance[outIdx] = data;
        }
    }
};
