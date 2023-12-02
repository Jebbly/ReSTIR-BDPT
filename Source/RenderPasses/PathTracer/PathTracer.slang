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
import Utils.Math.MathHelpers;
import LoadShadingData;
import ColorType;
import NRDHelpers;
import BidirLightHelpers;
import PhotonMap;
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

    PathTracerParams params;                               ///< Runtime parameters.

    // Samplers
    LightSampler lightSampler;                             ///< Light sampler.

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;                      ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                             ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<uint> sampleCount;                           ///< Optional input sample count buffer. Only valid when kSamplesPerPixel == 0.
    Texture2D<uint> sampleOffset;                          ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.
    RWByteAddressBuffer lightImage;                        ///< Input light traced image. Only valid when kUseBPT == true. Added to the output when kSamplesPerPixel == 1.

    // Outputs
    RWStructuredBuffer<ColorType> sampleColor;             ///< Output per-sample color if kSamplesPerPixel != 1.
    RWStructuredBuffer<GuideData> sampleGuideData;         ///< Output per-sample guide data.
    NRDBuffers outputNRD;                                  ///< Output NRD data.

    RWTexture2D<float4> outputColor;                       ///< Output color buffer if kSamplesPerPixel == 1.

    LightVertexCache lightVertexCache;                     ///< Light vertex cache.
    PhotonMap photonMap;                                   ///< Photon map.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseCurves         = USE_CURVES;
    static const bool kUseHairMaterial   = USE_HAIR_MATERIAL;

    // Additional specialization.
    static const bool kOutputGuideData = OUTPUT_GUIDE_DATA;
    #ifdef LIGHT_TRACE_PASS
    static const bool kTraceFromLight = true;
    #else
    static const bool kTraceFromLight = false;
    #endif

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n Number of samples taken from the sampling strategy.
        \param[in] p Pdf for the sampling strategy.
        \return Partial MIS weight
    */
    float evalMIS(float n, float p)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            return n * p;
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            return (n * p) * (n * p);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            return pow(n * p, kMISPowerExponent);
        }
        default:
            return 0.f;
        }
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        float q0 = evalMIS(n0, p0);
        float q1 = evalMIS(n1, p1);
        float qs = q0 + q1;
        return qs > 0 ? q0 / qs : 0.f;
    }


    float getEtaVCM() { return (params.mergeRadius * params.mergeRadius) * M_PI * params.lightSubpathCount; }
    float getMisVmWeightFactor() { return kUseVM ? evalMIS(1.f, getEtaVCM()) : 0.f; }
    float getMisVcWeightFactor() { return evalMIS(1.f, 1.f / getEtaVCM()); }

    float getImageToSolidAngleFactor(const float3 dir, out float cosCamera)
    {
        cosCamera = dot(dir, normalize(gScene.camera.data.cameraW));
        const float imagePlaneDist = params.frameDim.y * (gScene.camera.data.focalLength / gScene.camera.data.frameHeight);
        const float imagePointToCameraDist = imagePlaneDist / cosCamera;
        return (imagePointToCameraDist * imagePointToCameraDist) / cosCamera;
    }

    /**
     * Returns the shading normal and eta corrections for light tracing.
     */
    float getAdjointCorrection(const ShadingData sd, const IMaterialInstance mi, const float3 wo) {
        float cosOut = dot(sd.faceN, wo);
        float cosIn  = dot(sd.faceN, sd.V);

        float factor = 1.f;

        // shading normal correction.
        const float num   = abs(dot(sd.frame.N, sd.V) * cosOut);
        const float denom = abs(dot(sd.frame.N, wo) * cosIn);
        if (denom >= kMinCosTheta)
            factor = num / denom;

        if (cosIn * cosOut < 0.f)
        {
            // eta factor correction.
            float eta = mi.getProperties(sd).eta;
            factor *= 1.f / (eta * eta);
        }

        return factor;
    }

    float getContinuationProbability()
    {
        if (!kUseRussianRoulette || (kUseBPT && kLightTraceOnly))
            return 1.f;
        return params.contProb;
    }

    /** Helper to compute the MIS weight when an emissive surface is hit via BSDF sampling.
        \param[in] path Path state.
        \param[in] lightPdfW Pdf w.r.t. solid angle at the previous vertex of sampling the current vertex using NEE.
        \param[in] lightEmitPdfW Pdf w.r.t. of sampling a light sub-path towards the current vertex.
        \param[in] lightPdfA Pdf w.r.t area.
        \return MIS weight.
    */
    float evalScatterMIS(const PathState path, float lightPdfW, float lightEmitPdfW, float lightPdfA)
    {
        if (kUseBPT)
        {
            const float wCamera = evalMIS(1.f, lightPdfA) * path.dVCM + evalMIS(1.f, lightEmitPdfW) * path.dVC;
            return 1.f / (1.f + wCamera);
        }
        else
        {
            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume the bsdf pdf (in dVC) is nonzero since we shouldn't have got here otherwise.
            return evalMIS(1, path.dVC, 1, lightPdfW);
        }
    }

    /** Helper to compute the MIS weight for next event estimation.
        \param[in] path Path state.
        \param[in] bsdfPdf PDF of sampling the light vertex using BSDF sampling. Solid-angle w.r.t the shading point.
        \param[in] lightPdf PDF of sampling the light. Solid-angle w.r.t the shading point.
        \param[in] lightEmitPdf PDF of sampling a light ray from the light to the current vertex. Solid-angle w.r.t. the shading point
        \return MIS weight.
    */
    float evalDirectMIS(const PathState path, float bsdfFwdPdf, float bsdfRevPdf, float cosOut, LightSample ls)
    {
        if (kUseBPT)
        {
            const float wLight = evalMIS(1.f, bsdfFwdPdf / ls.pdf);
            const float wCamera = evalMIS(1.f, ls.emitPdf * abs(cosOut) / (ls.pdf * ls.cosLight)) * (
                getMisVmWeightFactor() + path.dVCM +
                path.dVC * evalMIS(1.f, bsdfRevPdf));

            return 1.f / (wLight + 1.f + wCamera);
        }
        else
        {
            return evalMIS(1.f, ls.pdf, 1.f, bsdfFwdPdf);
        }
    }

    /** Helper to compute the MIS weight for pure light tracing.
        \param[in] path Path state.
        \param[in] bsdfRevPdf BSDF sampling probability density towards the previous light sub-path vertex.
        \param[in] cameraPdfA
        \return MIS weight.
    */
    float evalLightTraceMIS(const PathState path, float bsdfRevPdf, float cameraPdfA)
    {
        const float wLight = evalMIS(1.f, cameraPdfA / params.lightSubpathCount) * (
            getMisVmWeightFactor() + path.dVCM +
            path.dVC * evalMIS(1.f, bsdfRevPdf));
        return 1.f / (wLight + 1.f);
    }

    float evalRevScatterPdf(const ShadingData sd, const IMaterialInstance mi, const float3 wo)
    {
        /*
        VertexData v;
        v.coneTexLODValue = 0.f;
        v.curveRadius = sd.curveRadius;
        v.faceNormalW = sd.faceN;
        v.normalW = sd.frame.N;
        v.posW = sd.posW;
        v.tangentW = float4(sd.frame.T, sign(dot(sd.frame.B, cross(sd.frame.N, sd.frame.T))));
        v.texC = sd.uv;
        let lod = ExplicitLodTextureSampler(params.lodBias);
        ShadingData revSd = gScene.materials.prepareShadingData(v, sd.materialID, wo, lod);
        /*/
        ShadingData revSd = sd;
        revSd.V = wo;
        revSd.frontFacing = dot(revSd.faceN, revSd.V) >= 0.f;
        //*/
        return mi.evalPdf(revSd, sd.V, kUseBSDFSampling);
    }

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
            (surfaceBounces > params.maxSurfaceBounces) ||
            (diffuseBounces > params.maxDiffuseBounces) ||
            (specularBounces > params.maxSpecularBounces) ||
            (transmissionBounces > params.maxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces      = min(params.maxDiffuseBounces,      path.getBounces(BounceType::Diffuse));
        uint specularBounces     = min(params.maxSpecularBounces,     path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(params.maxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces      = min(params.maxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
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

            path.throughput = ls.contrib;
            path.sampleWeight = ls.weight;
            path.origin = ls.origin;
            path.dir = ls.dir;
            path.pathPdf = ls.emitPdf;

            path.setVertexIndex(0);
            path.setFiniteLight(ls.getIsFinite());

            path.dVCM = evalMIS(1.f, ls.pdf / ls.emitPdf);
            path.dVC = ls.getIsDelta() ? 0 : evalMIS(1.f, ls.cosLight / ls.emitPdf);
            path.dVM = path.dVC * getMisVcWeightFactor();
        }
        else
        {
            path.throughput = float3(1.f);
            path.sample.contribution = 0.f;
            path.sampleWeight = 0.f;
            path.prefixThroughput = 0.f;

            // Create primary ray.
            Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
            if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
            path.origin = cameraRay.origin;
            path.dir = cameraRay.dir;
            path.pathPdf = 1.f;
            path.dVC = 0.f;

            if (kUseBPT)
            {
                float cosCamera;
                const float cameraPdfW = getImageToSolidAngleFactor(path.dir, cosCamera);
                path.dVCM = evalMIS(1.f, params.lightSubpathCount / cameraPdfW);
                path.dVC = 0.f;
                path.dVM = 0.f;
            }

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
        if (!kTraceFromLight)
        {
            printSetPixel(path.getPixel());
            logSetPixel(path.getPixel());
        }
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

    /** Add radiance to the path contribution.
        \param[in,out] path Path state.
        \param[in] contribution Unweighted path contribution.
        \param[in] integrationWeight Integration weight, excluding MIS. Typically this is the reciprocal of the path's sampling /pdf
        \param[in] misWeight MIS weight.
        \param[in] lightVertices number of light vertices in the path.
    */
    void addToPathContribution( inout PathState path, const float3 contribution, const float integrationWeight, float misWeight, const uint lightVertices )
    {
        if (!any(contribution > 0) || any(isnan(contribution)) || any(isinf(contribution)))
            return;

        if (kDebugBPT)
        {
            const uint viewVertices = path.getVertexIndex() + 1;
            if ((params.debugPathLength != -1 && (viewVertices + lightVertices) - 1 != params.debugPathLength) ||
                (params.debugLightVertexCount != -1 && lightVertices != params.debugLightVertexCount))
                return;

            if (!kDebugMIS && params.debugLightVertexCount != -1)
            {
                if (kUseVM && lightVertices > 1)
                    misWeight = 0.5f;
                else
                    misWeight = 1.f;
            }
        }

        if (kDebugMIS)
        {
            path.sample.contribution += misWeight;
            path.sampleWeight = 1.f;
            return;
        }

        // Stream the path through this pixel's reservoir
        float W = luminance(contribution) * misWeight * integrationWeight;
        if (!(W > 0) || isnan(W) || isinf(W))
            return;

        path.sampleWeight += W;
        if (sampleNext1D(path.sg) * path.sampleWeight >= W)
            return;

        path.sample.contribution = contribution;
        path.sample.pathLength = path.getVertexIndex() + lightVertices;
        if (any(path.prefixThroughput > 0))
        {
            path.sample.rcv.contribution = contribution / path.prefixThroughput;
        }
    }

    /** Atomically add radiance to the light image buffer.
        \param[in] pixel pixel index.
        \param[in] contribution Unweighted path contribution.
        \param[in] integrationWeight Integration weight, excluding MIS. Typically this is the reciprocal of the path's sampling pdf
        \param[in] misWeight sample MIS weight.
        \param[in] lightVertices number of light vertices in the path.
    */
    void addToLightImage(const uint2 pixel, const float3 contribution, const float integrationWeight, float misWeight, const uint lightVertices )
    {
        if (kDebugBPT)
        {
            const uint viewVertices = 1;
            if ((params.debugPathLength != -1 && (viewVertices + lightVertices) - 1 != params.debugPathLength) ||
                (params.debugLightVertexCount != -1 && lightVertices != params.debugLightVertexCount))
                return;

            if (!kDebugMIS && params.debugLightVertexCount != -1)
                misWeight = 1.f;
        }

        const float3 L = kDebugMIS ? misWeight : contribution * (integrationWeight * misWeight);
        if (!any(L > 0) || any(isnan(L)))
            return;

        const uint addr = sizeof(float3) * (pixel.y * params.frameDim.x + pixel.x);
        lightImage.InterlockedAddF32(addr + 0, L.r);
        lightImage.InterlockedAddF32(addr + 4, L.g);
        lightImage.InterlockedAddF32(addr + 8, L.b);
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
        if (valid)
        {
            if (kTraceFromLight) result.weight *= getAdjointCorrection(sd, mi, result.wo);
            valid = generateScatterRay(result, sd, mi, path);
        }

        const bool isDeltaLobe = result.isLobe(LobeType::DeltaReflection) || result.isLobe(LobeType::DeltaTransmission);

        // Update MIS weight quantities
        if (kUseBPT && valid)
        {
            const float cosOut = abs(dot(path.dir, path.normal));

            if (isDeltaLobe)
            {
                path.dVCM = 0.f;
                path.dVC *= evalMIS(1.f, cosOut);
                path.dVM *= evalMIS(1.f, cosOut);
            }
            else
            {
                float contProb = getContinuationProbability();
                float fwdPdf = result.pdf * contProb;
                float revPdf;
                if (kTraceFromLight && path.getVertexIndex() == 1 && path.isAnalyticLight())
                    revPdf = 0.f;
                else
                    revPdf = evalRevScatterPdf(sd, mi, path.dir) * contProb;

                path.dVC  = evalMIS(1.f, cosOut / fwdPdf) * (path.dVC * evalMIS(1.f, revPdf) + path.dVCM + getMisVmWeightFactor());
                path.dVM  = evalMIS(1.f, cosOut / fwdPdf) * (path.dVM * evalMIS(1.f, revPdf) + path.dVCM + getMisVcWeightFactor() + 1.f);
                path.dVCM = evalMIS(1.f, 1.f / fwdPdf);
            }
        }

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

                // Mark path as delta-only if it followed delta lobe on the primary hit, even though there might have been non-delta lobes.
                path.setDeltaOnlyPath(isDeltaLobe);

                if (kOutputNRDAdditionalData)
                {
                    path.setDeltaReflectionPrimaryHit(result.isLobe(LobeType::DeltaReflection));
                    path.setDeltaTransmissionPath(result.isLobe(LobeType::DeltaTransmission));
                }
            }

            if (path.getVertexIndex() > 1)
            {
                if (!isDeltaLobe) path.setDeltaOnlyPath(false);

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

        float3 f = bs.weight;
        if (bs.pdf > 0.f)
        {
            path.pathPdf *= bs.pdf;
            f *= bs.pdf; // multiply out pdf
        }
        path.throughput *= f;

        if (!kUseBPT)
        {
            // hijack dVC to store the pdf in dVC for calculating MIS for NEE
            path.dVC = bs.pdf * getContinuationProbability();
        }

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
        }

        // Save the normal used for NEE. This is needed for MIS.
        path.normal = sd.getOrientedFaceNormal();

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.throughput > 0.f);

        return valid;
    }

    /**
     * Evaluates the indirect illumination by merging with a random light subpath vertex.
     * This function samples a light vertex uniformly from lightVertices.
     * @param[in] sd Shading data.
     * @param[in] mi Material instance.
     * @param[in,out] path PathState object.
     */
    void mergeWithLightSubpath(const ShadingData sd, const IMaterialInstance mi, inout PathState path, const LightSubpathVertex lightVertex, const float weight)
    {
        if ((path.getVertexIndex() + lightVertex.vertexIndex) - 1 > params.maxSurfaceBounces)
            return;

        if (length(sd.posW - lightVertex.getPosW()) >= params.mergeRadius)
            return;

        const float3 dir = -lightVertex.rayDirection;
        const float cosCamera = dot(dir, sd.getOrientedFaceNormal());

        // Reject sample if not in the hemisphere of a BSDF lobe.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasReflection   = lobeTypes & uint(LobeType::Reflection);
        const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
        if ((cosCamera <= kMinCosTheta && !hasTransmission) || (cosCamera >= -kMinCosTheta && !hasReflection))
            return;

        // evaluate bsdf at camera vertex
        const float3 f = path.throughput * mi.eval(sd, dir, path.sg) * lightVertex.L;
        if (!any(f > 0)) return;

        const float sampleWeight = weight / (lightVertex.pathPdf * getEtaVCM());

        const float cameraBsdfFwdPdfW = mi.evalPdf(sd, dir, kUseBSDFSampling);
        const float cameraBsdfRevPdfW = evalRevScatterPdf(sd, mi, dir);

        // evaluate MIS weight
        const float wLight = lightVertex.dVCM * getMisVcWeightFactor() + lightVertex.dVM * evalMIS(1.f, cameraBsdfFwdPdfW);
        const float wCamera = path.dVCM * getMisVcWeightFactor() + path.dVM * evalMIS(1.f, cameraBsdfRevPdfW);
        const float misWeight = 1.f / (wLight + 1.f + wCamera);

        addToPathContribution(path, f, sampleWeight, misWeight, lightVertex.vertexIndex + 1);
    }

    /**
     * Evaluates the indirect illumination by merging with a random light subpath vertex.
     * This function samples a light vertex uniformly from lightVertices.
     * @param[in] sd Shading data.
     * @param[in] mi Material instance.
     * @param[in,out] path PathState object.
     */
    void mergeWithLightSubpaths(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        const int3 o = int3( lerp(
            float3( 0,  0,  0),
            float3(-1, -1, -1),
            float3(frac(sd.posW / (2.f * params.mergeRadius)) < 0.5f) ));

        uint cell = -1;
        float cellW = 0.f;

        // Iterate the 8 neighbor grid cells and merge with the photons in them.
        for (int x = 0; x < 2; x++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int z = 0; z < 2; z++)
                {
                    const uint cellIndex = photonMap.getCellIndex(sd.posW, params.mergeRadius, params.cellCount, int3(x, y, z) + o);
                    if (cellIndex == -1)
                        continue;

                    const uint count  = photonMap.getCellPhotonCount(cellIndex);
                    if (count == 0)
                        continue;

                    if (kUseFastVM)
                    {
                        // pick one photon to merge with, with probability proportional to its count
                        cellW += count;
                        if (sampleNext1D(path.sg)*cellW < count)
                            cell = cellIndex;
                    }
                    else
                    {
                        // merge with all photons
                        const LightSubpathVertex lightVertex = lightVertexCache[photonMap.getCellOffset(cellIndex)];
                        mergeWithLightSubpath(sd, mi, path, lightVertex, float(count));
                    }
                }
            }
        }

        if (kUseFastVM)
        {
            if (cell != -1)
            {
                const LightSubpathVertex lightVertex = lightVertexCache[photonMap.getCellOffset(cell)];
                mergeWithLightSubpath(sd, mi, path, lightVertex, cellW);
            }
        }
    }

    /**
     * Evaluates the indirect illumination by connecting to a random light subpath vertex.
     * This function samples a light vertex uniformly from lightVertices.
     * @param[in] sd Shading data.
     * @param[in] mi Material instance.
     * @param[in,out] path PathState object.
     * @param[in,out] vq Visibility query.
     */
    void connectToLightSubpath<VisibilityQuery : IVisibilityQuery>(const ShadingData sd, const IMaterialInstance mi, inout PathState path, inout VisibilityQuery vq)
    {
        const uint numVertices = lightVertexCache.size();
        if (numVertices == 0)
            return;

        const LightSubpathVertex lightVertex = lightVertexCache[min(uint(sampleNext1D(path.sg) * numVertices), numVertices - 1u)];

        if (path.getVertexIndex() + lightVertex.vertexIndex > params.maxSurfaceBounces)
            return;

        let lod = ExplicitLodTextureSampler(0.f);
        const ShadingData lightSd = loadShadingData(HitInfo(lightVertex.packedHit), lightVertex.rayOrigin, lightVertex.rayDirection, lod);

        float3 toLight = lightSd.posW - sd.posW;
        const float dist2 = max(dot(toLight, toLight), kMinLightDistSqr);
        toLight = normalize(toLight);

        const float cosLight = -dot(toLight, lightSd.getOrientedFaceNormal());
        const float cosCamera = dot(toLight, sd.getOrientedFaceNormal());

        float3 contribution = path.throughput * lightVertex.L;

        // evaluate bsdf at light vertex
        float lightBsdfFwdPdfW, lightBsdfRevPdfW;
        {
            let lightMi = gScene.materials.getMaterialInstance(lightSd, lod);
            // Reject sample if not in the hemisphere of a BSDF lobe.
            const uint lobeTypes = lightMi.getLobeTypes(lightSd);
            const bool hasReflection   = lobeTypes & uint(LobeType::Reflection);
            const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
            if ((cosLight <= kMinCosTheta && !hasTransmission) || (cosLight >= -kMinCosTheta && !hasReflection))
                return;
            contribution *= lightMi.eval(lightSd, -toLight, path.sg) * getAdjointCorrection(lightSd, lightMi, -toLight) / abs(cosLight); // divide by cosine term since its handled in the integration weight

            lightBsdfFwdPdfW = lightMi.evalPdf(lightSd, -toLight, kUseBSDFSampling);
            lightBsdfRevPdfW = evalRevScatterPdf(lightSd, lightMi, -toLight);
        }
        if (!any(contribution > 0)) return;

        // evaluate bsdf at camera vertex
        float cameraBsdfFwdPdfW, cameraBsdfRevPdfW;
        {
            // Reject sample if not in the hemisphere of a BSDF lobe.
            const uint lobeTypes = mi.getLobeTypes(sd);
            const bool hasReflection   = lobeTypes & uint(LobeType::Reflection);
            const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
            if ((cosCamera <= kMinCosTheta && !hasTransmission) || (cosCamera >= -kMinCosTheta && !hasReflection))
                return;
            contribution *= mi.eval(sd, toLight, path.sg);

            cameraBsdfFwdPdfW = mi.evalPdf(sd, toLight, kUseBSDFSampling);
            cameraBsdfRevPdfW = evalRevScatterPdf(sd, mi, toLight);
        }
        if (!any(contribution > 0)) return;

        // Test visibility by tracing a shadow ray.
        Ray ray = {};
        // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
        ray.origin = computeRayOrigin(sd, toLight);
        ray.tMin = 0.f;
        ray.dir = computeRayOrigin(lightSd, -toLight) - ray.origin;
        ray.tMax = sqrt(max(dot(ray.dir, ray.dir), kMinLightDistSqr));
        ray.dir /= ray.tMax;
        logTraceRay(PixelStatsRayType::Visibility);
        bool V = vq.traceVisibilityRay(ray);
        if (!V) return;

        const float invDist2 = 1.f / dist2;

        const float wLight  = evalMIS(1.f, cameraBsdfFwdPdfW * abs(cosLight) * invDist2) * (getMisVmWeightFactor() + lightVertex.dVCM + lightVertex.dVC * evalMIS(1.f, lightBsdfRevPdfW));
        const float wCamera = evalMIS(1.f, lightBsdfFwdPdfW * abs(cosCamera) * invDist2) * (getMisVmWeightFactor() + path.dVCM + path.dVC * evalMIS(1.f, cameraBsdfRevPdfW));
        const float misWeight = 1.f / (wLight + 1.f + wCamera);

        float sampleWeight = (1.f / (lightVertex.pathPdf * path.pathPdf)) * abs(cosLight) * invDist2;
        sampleWeight *= numVertices / float(params.lightSubpathCount);

        addToPathContribution(path, contribution, sampleWeight, misWeight, lightVertex.vertexIndex+1);
    }

    /** Connect a light sub-path to the camera to form a full path. Splats the contribution to lightImage.
     */
    void connectToCamera<VisibilityQuery : IVisibilityQuery>(const ShadingData sd, const IMaterialInstance mi, inout PathState path, inout VisibilityQuery vq)
    {
        if (gScene.camera.data.focalLength == 0.0f)
        {
            // TODO: ortho camera
            return;
        }

        float3 toCamera = gScene.camera.getPosition() - sd.posW;
        const float dist2 = max(dot(toCamera, toCamera), kMinLightDistSqr);
        toCamera = normalize(toCamera);

        // Reject sample if not in the hemisphere of a BSDF lobe.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasReflection   = lobeTypes & uint(LobeType::Reflection);
        const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
        float NdotL = dot(sd.getOrientedFaceNormal(), toCamera);
        if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
            return;

        float cosCamera;
        const float imageToSolidAngleFactor = getImageToSolidAngleFactor(-toCamera, cosCamera);

        float4 clip = mul(gScene.camera.data.viewProjMat, float4(sd.posW, 1));
        clip.y = -clip.y;
        const int2 pixel = int2(((clip.xy / clip.w) * .5 + .5) * params.frameDim);
        if (any(pixel < 0) || any(pixel >= params.frameDim) || cosCamera <= 0)
            return;

        // Compute full path contribution. Note we divide the cosine term out of the BSDF since it's handled in the integration weight.
        const float cosOut = abs(dot(sd.faceN, toCamera));
        float3 contribution = path.throughput * (mi.eval(sd, toCamera, path.sg)/cosOut) * getAdjointCorrection(sd, mi, toCamera);
        if (!any(contribution > 0))
            return;

        const float invDist2 = 1.f / dist2;
        const float imageToSurfaceFactor = imageToSolidAngleFactor * cosOut * invDist2;

        // Compute MIS weight
        float bsdfRevPdfW;
        if (path.getVertexIndex() == 1 && path.isAnalyticLight())
        {
            bsdfRevPdfW = 0.f;
        }
        else
        {
            bsdfRevPdfW = evalRevScatterPdf(sd, mi, toCamera) * getContinuationProbability();
        }
        const float misWeight = kLightTraceOnly ? 1.f : evalLightTraceMIS(path, bsdfRevPdfW, imageToSurfaceFactor);

        // Test visibility by tracing a shadow ray.
        logTraceRay(PixelStatsRayType::Visibility);
        bool V = vq.traceVisibilityRay(Ray(computeRayOrigin(sd, toCamera), toCamera, 0, sqrt(dist2)));
        if (!V)
            return;

        float sampleWeight = (imageToSolidAngleFactor * cosOut * invDist2) / (params.lightSubpathCount * path.pathPdf);

        addToLightImage(pixel, contribution, sampleWeight, misWeight, path.getVertexIndex()+1);
    }

    /** Appends the current path vertex to the lightVertices buffer.
    */
    void storeLightVertex(const ShadingData sd, const PathState path)
    {
        LightSubpathVertex v = {};
        v.packedHit = path.hit.data;
        v.L = path.throughput;
        v.vertexIndex = path.getVertexIndex();
        v.rayOrigin = path.origin;
        v.dVCM = path.dVCM;
        v.rayDirection = path.dir;
        v.dVC = path.dVC;
        v.dVM = path.dVM;
        v.pathPdf = path.pathPdf;
        v.hitDist = length(sd.posW - path.origin);

        const uint i = lightVertexCache.addVertex(v);
        if (kUseVM) photonMap.addPhoton(i, sd.posW, params.mergeRadius, params.cellCount);
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
        const float contProb = getContinuationProbability();
        if (u >= contProb)
        {
            path.terminate();
            return true;
        }
        path.pathPdf *= contProb;
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

        const float rr_u = sampleNext1D(path.sg); // needs to be computed before path.sg is modified

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, lod);

        const float hitDist = length(sd.posW - path.origin);

        if (kUseBPT)
        {
            if (!kTraceFromLight || !isPrimaryHit || path.isFiniteLight())
                path.dVCM *= evalMIS(1.f, max(kMinLightDistSqr, hitDist * hitDist));
            const float invCos = 1.f / abs(dot(sd.V, sd.faceN));
            path.dVCM *= evalMIS(1.f, invCos);
            path.dVC  *= evalMIS(1.f, invCos);
            path.dVM  *= evalMIS(1.f, invCos);
        }

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
            path.throughput *= HomogeneousVolumeSampler::evalTransmittance(vp, hitDist);
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create BSDF instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        float3 attenuatedEmission = 0.f;
        if (!kTraceFromLight)
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

            if (computeEmissive && any(bsdfProperties.emission > 0.f))
            {
                float misWeight = 1.f;
                if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && ((path.isLightSampled() && isLightSamplable) || (kUseBPT && !path.isDeltaOnlyPath())))
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
                    float p = lightSampler.getEmissiveSelectionProbability();
                    float lightPdfW = p * lightSampler.emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);
                    float lightPdfA = p * lightSampler.emissiveSampler.evalTriangleSelectionPdf(hit.triangleIndex) / gScene.lightCollection.getTriangle(hit.triangleIndex).area;

                    misWeight = evalScatterMIS(path, lightPdfW, lightPdfA * (abs(dot(sd.V, sd.faceN)) * M_1_PI), lightPdfA);
                }

                // Accumulate emitted radiance weighted by path throughput and MIS weight.
                addToPathContribution(path, path.throughput * bsdfProperties.emission, 1.f / path.pathPdf, misWeight, 0);

                attenuatedEmission = path.throughput * bsdfProperties.emission * (misWeight / path.pathPdf);
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

        // Path connection strategies.
        if (hasNonDeltaLobes)
        {
            if (kTraceFromLight)
            {
                storeLightVertex(sd, path);
                connectToCamera(sd, mi, path, vq);
            }
            else
            {
                // sample direct illumination by connecting to a light source
                {
                    // Check if sample from RTXDI should be applied instead of NEE.
                    const bool applyRTXDI = kUseRTXDI && isPrimaryHit;

                    // TODO: Support multiple shadow rays.
                    path.setLightSampled(false, false);
                    if (kUseBPT || kUseNEE || applyRTXDI)
                    {
                        LightSample ls = {};
                        bool validSample = false;

                        if (applyRTXDI)
                        {
                            // Query final sample from RTXDI.
                            validSample = gRTXDI.getFinalSample(path.getPixel(), ls.dir, ls.distance, ls.contrib);
                            ls.origin = path.origin;
                            ls.weight = 1.f;
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
                            if (kUseMIS && !applyRTXDI && (!ls.getIsAnalytic() || kUseBPT))
                            {
                                float contProb = getContinuationProbability();
                                float scatterPdf;
                                if (ls.getIsAnalytic())
                                    scatterPdf = 0;
                                else
                                    scatterPdf = mi.evalPdf(sd, ls.dir, kUseBSDFSampling) * contProb;
                                float revScatterPdf = evalRevScatterPdf(sd, mi, ls.dir) * contProb;
                                misWeight = evalDirectMIS(path, scatterPdf, revScatterPdf, abs(dot(sd.faceN, ls.dir)), ls);
                            }

                            float3 contribution = path.throughput * mi.eval(sd, ls.dir, path.sg) * ls.contrib;
                            if (any(contribution > 0.f) && misWeight > 0.f)
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
                                if (visible) addToPathContribution(path, contribution, ls.weight / path.pathPdf, misWeight, 1);
                            }
                        }
                    }
                }

                // sample indirect illumination by connecting to a light subpath vertex
                if (kUseBPT)
                {
                    connectToLightSubpath(sd, mi, path, vq);

                    if (kUseVM)
                    {
                        mergeWithLightSubpaths(sd, mi, path);
                    }
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, rr_u)) return;
        }

        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        // Determine whether to terminate, and output guide data.
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

            float3 pathRadiance = 0.f;

            if (computeEnv)
            {
                logPathVertex();

                float misWeight = 1.f;
                if (kUseNEE && kUseMIS && ((path.isLightSampled() && isLightSamplable) || (kUseBPT && !path.isDeltaOnlyPath())))
                {
                    // If NEE and MIS are enabled, and we've already sampled the env map,
                    // then we need to evaluate the MIS weight here to account for the remaining contribution.

                    // Evaluate PDF, had it been generated with light sampling.
                    float lightPdf = lightSampler.getEnvMapSelectionProbability() * lightSampler.envMapSampler.evalPdf(path.dir);

                    misWeight = evalScatterMIS(path, lightPdf, lightPdf * getSceneSpherePdf(params.sceneSphere), lightPdf);
                }

                float3 Le = lightSampler.envMapSampler.eval(path.dir);
                addToPathContribution(path, path.throughput * Le, 1.f / path.pathPdf, misWeight, 0);
                pathRadiance = path.throughput * Le * (misWeight / path.pathPdf);

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
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, pathRadiance, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
            }
            else
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
            }
        #elif defined(DELTA_TRANSMISSION_PASS)
            if (path.isDeltaTransmissionPath())
            {
                writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, pathRadiance, -path.dir, 0.f, kNRDInvalidPathLength, path.origin + path.dir * kNRDInvalidPathLength);
            }
            else
            {
                writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, path.getPixel(), 0.f, 0.f, -path.dir, 0.f, kNRDInvalidPathLength, 0.f);
            }
        #endif
        }

        path.terminate();
    }

    /** Atomically write path contribution to light image buffer.
        The light image is copied to the final image during camera tracing.
    */
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

        const float3 L = path.getSampleRadiance();

        assert(!any(isnan(L)));

        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        const uint outIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();


        if (kSamplesPerPixel == 1)
        {
            // Write color directly to frame buffer.
            outputColor[pixel] = float4(L, 1.f);
        }
        else
        {
            // Write color to per-sample buffer.
            sampleColor[outIdx].set(L);
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

            data.setRadiance(L);

            outputNRD.sampleRadiance[outIdx] = data;
        }
    }
};
