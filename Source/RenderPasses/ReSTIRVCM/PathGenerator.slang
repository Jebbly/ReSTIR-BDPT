#include "Utils/Math/MathConstants.slangh"
#include "BitfieldMacros.slangh"
import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import LoadShadingData;
import HashMap;
import PhotonMap;
__exported import Params;
__exported import PathReservoir;

cbuffer CB {
    uint gRandomSeed;
    uint gSwapReservoirs;
};

// Static parameters
extension PathGenerator {
    static const bool  gUseEnvLight                   = USE_ENV_LIGHT;
    static const bool  gUseEmissiveLights             = USE_EMISSIVE_LIGHTS;
    static const bool  gUseViewDir                    = USE_VIEW_DIR;
    static const bool  gSampleDirectLight             = USE_NEE;
    static const bool  gBidirectional                 = USE_BIDIRECTIONAL;
    static const bool  gShiftLightPathsToPixelCenters = SHIFT_LIGHT_PATHS_TO_CENTER;
    static const bool  gVertexMerging                 = USE_VERTEX_MERGING;
    static const bool  gDynamicMergeRadius            = DYNAMIC_MERGE_RADIUS;
    static const bool  gProgressivePhotonMapping      = USE_PPM_ONLY;
    static const bool  gLightTraceOnly                = LIGHT_TRACE_ONLY;
    static const bool  gUseResampling                 = USE_RESAMPLING;
    static const bool  gUseReconnectionMIS            = USE_RECONNECTION_MIS;
    static const bool  gUseCausticReservoirs          = CAUSTIC_RESERVOIRS;
    static const bool  gUseCausticMotionVectors       = CAUSTIC_MOTION_VECTORS;
    static const bool  gShiftSuffixes                 = SHIFT_SUFFIXES;
    static const bool  gUseBsdfImportanceSampling     = USE_BSDF_IMPORTANCE_SAMPLING;
    static const bool  gDebugBPT                      = DEBUG_BPT;
    static const bool  gDebugHeatmap                  = DEBUG_HEATMAP;
    static const float gMISPowerExponent              = MIS_POWER_EXPONENT;
    static const bool  gDisableCameraConnection       = DISABLE_CAMERA_CONNECTION;
    static const bool  gDisableEarlyReconnection      = DISABLE_EARLY_RECONNECTION;
    static const bool  gDisableLVC                    = DISABLE_LVC;
}

struct LightTraceReservoir {
    PathSample path;
    float W;
};

struct PathGenerator {
    PathGeneratorParams mParams;

    EnvMapSampler mEnvMapSampler;
    EmissiveLightSampler mEmissiveSampler;

    LightVertexCache mLightVertexCache;
    PhotonMap mPhotonMap;

    RWByteAddressBuffer mAtomicRadiance;
    RWTexture2D<float4> mOutputRadiance;
    RWByteAddressBuffer mOutputCounterData;

    RWStructuredBuffer<PathReservoir> mPathReservoirs0;
    RWStructuredBuffer<PathReservoir> mPathReservoirs1;
    RWStructuredBuffer<PathReservoir> mCausticReservoirs;
    HashMap<LightTraceReservoir> mLightTraceReservoirs;
    HashMap<PathReservoir> mCausticReservoirMap;

    Texture2D<PackedHitInfo> mVbuffer;
    Texture2D<float3> mViewDir;
    RWTexture2D<float2> mMotionVectors;
    RWByteAddressBuffer mCausticMotionVectorMutex;

    RWStructuredBuffer<PathReservoir> mLastReservoirs;
    RWStructuredBuffer<PathReservoir> mLastCausticReservoirs;
    Texture2D<float3> mLastViewDir;
    Texture2D<PackedHitInfo> mLastVbuffer;
}
ParameterBlock<PathGenerator> gPathGenerator;

///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Path vertex interface //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

struct ReflectanceEvalData {
    float3 mReflectance = 0;
    float mFwdPdfW = 0;
    float mRevPdfW = 0;
};
struct DirectionSampleData {
    float3 mDirection = 0;
    float3 mReflectance = 0;
    float  mFwdPdfW = 0;
    float  mRevPdfW = 0;
    uint   mLobeType = 0;

    bool IsLobe(const LobeType lobe) { return (mLobeType & uint(lobe)) != 0; }

    [mutating]
    void SetReflectance(const ReflectanceEvalData r) {
        mReflectance = r.mReflectance;
        mFwdPdfW = r.mFwdPdfW;
        mRevPdfW = r.mRevPdfW;
    }
};

struct PathVertex {
    PackedPathVertex  mVertex;
    ShadingData       mShadingData;
    IMaterialInstance mMaterial;

    __init(const PackedPathVertex v) {
        mVertex = v;
        if (v.IsFinite()) {
            HitInfo hit = HitInfo(v.mPackedHit);

            let lod = ExplicitLodTextureSampler(0.f);
            mShadingData = loadShadingData(hit, v.mRayOrigin, v.mRayDirection, lod);

            let hints = getMaterialInstanceHints(hit, false);
            mMaterial = gScene.materials.getMaterialInstance(mShadingData, lod, hints);
        }
    }

    PathVertex Reversed(const float3 dir) {
        PathVertex vertex;
        vertex.mVertex = mVertex;
        vertex.mVertex.mRayDirection = -vertex.mVertex.mRayDirection;

        let lod = ExplicitLodTextureSampler(0.f);

        //*
        VertexData v;
        v.posW = mShadingData.posW;
        v.normalW = mShadingData.frame.N;
        v.tangentW = float4(mShadingData.frame.T, sign(dot(mShadingData.frame.B, cross(mShadingData.frame.N, mShadingData.frame.T))));
        v.texC = mShadingData.uv;
        v.faceNormalW = mShadingData.faceN;
        v.curveRadius = mShadingData.curveRadius;
        v.coneTexLODValue = 0.f;
        vertex.mShadingData = gScene.materials.prepareShadingData(v, mShadingData.materialID, dir, lod);
        /*/
        vertex.mShadingData = mShadingData;
        vertex.mShadingData.V = dir;
        vertex.mShadingData.frontFacing = dot(vertex.mShadingData.faceN, dir) > 0.f;
        //*/

        HitInfo hit = HitInfo(mVertex.mPackedHit);

        let hints = getMaterialInstanceHints(hit, false);
        vertex.mMaterial = gScene.materials.getMaterialInstance(vertex.mShadingData, lod, hints);

        return vertex;
    }

    PackedPathVertex Pack() { return mVertex; }

    bool IsFinite() { return mVertex.IsFinite(); }

    float3 GetPosition() { return mShadingData.posW; }
    // points away from vertex
    float3 GetDirIn() { return mShadingData.V; }

    bool HasFaceNormal() { return IsFinite(); }
    float3 GetFaceNormal() { return mShadingData.getOrientedFaceNormal(); }
    float GetCosN(const float3 dir) {
        if (HasFaceNormal())
            return abs(dot(mShadingData.faceN, dir));
        else
            return 1;
    }

    bool HasLobe(const LobeType l) { return (mMaterial.getLobeTypes(mShadingData) & uint(l)) != 0; }

    bool UpperHemisphereOnly() { return HasFaceNormal() && !HasLobe(LobeType::NonDeltaTransmission); }

    float3 GetEmission() {
        if (IsFinite()) {
            if (mShadingData.frontFacing) {
                return mMaterial.getProperties(mShadingData).emission;
            } else {
                return 0;
            }
        } else {
            return gScene.envMap.eval(mVertex.mRayDirection, 0);
        }
    }

    float3 ComputeRayOrigin(const float3 dir) {
        return computeRayOrigin(mShadingData.posW, dot(dir, mShadingData.faceN) >= 0.f ? mShadingData.faceN : -mShadingData.faceN);
    }

    bool IsConnectable() {
        if (!IsFinite())
            return false;

        if (HasLobe(LobeType::Diffuse))
            return true;

        if (HasLobe(LobeType::Specular) && mMaterial.getProperties(mShadingData).roughness >= gPathGenerator.mParams.mReconnectionRoughness)
            return true;

        return false;
    }

    float GetContinuationProbability() {
        if (!IsConnectable())
            return 1.f;

        return 1.f - gPathGenerator.mParams.mTerminationProbability * mMaterial.getProperties(mShadingData).roughness;
    }

    float GetAdjointCorrection(const float3 dirOut, const bool etaCorrection = false) {
        if (!HasFaceNormal()) return 1;

        let properties = mMaterial.getProperties(mShadingData);

        float factor = 1.f;

        {
            const float cosOut = dot(GetFaceNormal(), dirOut);
            const float cosIn  = dot(GetFaceNormal(), GetDirIn());

            const float3 shadingNormal = properties.guideNormal;
            const float cosInShading  = dot(shadingNormal, GetDirIn());
            const float cosOutShading = dot(shadingNormal, dirOut);

            // Account for asymmetry in the BSDF due to the use of shading normals (Veach 5.3)
            const float num   = abs(cosInShading * cosOut);
            const float denom = abs(cosOutShading * cosIn);
            if (denom >= kMinCosTheta)
                factor *= num / denom;
        }

        if (etaCorrection) {
            // Account for asymmetry in the BSDF due to refraction (Veach 5.2)
            factor /= sqr(properties.eta);
        }

        return factor;
    }

    ReflectanceEvalData EvaluateReflectance(inout SampleGenerator sg, const float3 dirOut, const bool adjoint) {
        ReflectanceEvalData r = {};
        r.mReflectance = mMaterial.eval(mShadingData, dirOut, sg);
        if (adjoint) r.mReflectance *= GetAdjointCorrection(dirOut);

        r.mFwdPdfW = mMaterial.evalPdf(mShadingData, dirOut, gPathGenerator.gUseBsdfImportanceSampling);

        const PathVertex revVertex = Reversed(dirOut);
        r.mRevPdfW = revVertex.mMaterial.evalPdf(revVertex.mShadingData, GetDirIn(), gPathGenerator.gUseBsdfImportanceSampling);

        const float contProb = GetContinuationProbability();
        r.mRevPdfW *= contProb;
        r.mFwdPdfW *= contProb;

        return r;
    }
    DirectionSampleData SampleDirection(inout SampleGenerator sg, const bool adjoint) {
        DirectionSampleData r = {};
        BSDFSample s;
        if (!mMaterial.sample(mShadingData, sg, s, gPathGenerator.gUseBsdfImportanceSampling))
            return r;

        r.mDirection = s.wo;
        r.mReflectance = s.weight;
        r.mLobeType = s.lobeType;
        if (adjoint) r.mReflectance *= GetAdjointCorrection(r.mDirection, s.isLobe(LobeType::DeltaTransmission) || s.isLobe(LobeType::SpecularTransmission));

        r.mFwdPdfW = s.pdf;

        // cancel divide by pdf performed by material sampling.
        if (s.pdf > 0) r.mReflectance *= s.pdf;

        if (r.IsLobe(LobeType::Delta)) {
            // rev  pdf should be the same as the fwdf pdf for delta lobes
            r.mRevPdfW = r.mFwdPdfW;
        } else {
            const PathVertex revVertex = Reversed(r.mDirection);
            r.mRevPdfW = revVertex.mMaterial.evalPdf(revVertex.mShadingData, GetDirIn(), gPathGenerator.gUseBsdfImportanceSampling);

            // get unweighted BSDF
            // TODO: modify the material sampling code to return the unweighted BSDF
            r.mReflectance = EvaluateReflectance(sg, r.mDirection, adjoint).mReflectance;
        }

        const float contProb = GetContinuationProbability();
        r.mFwdPdfW *= contProb;
        r.mRevPdfW *= contProb;

        return r;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// Helper functions ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

extension PathGenerator {
    void DebugSetPixel(const uint2 id) {
        printSetPixel(id);
        #if DEBUG_HEATMAP
        gPixelDebugPixel = id;
        #endif
    }

    void IncrementCounter(const DebugCounterType type, const uint increment = 1) {
        #if DEBUG_HEATMAP
        if ((uint(type) & uint(mParams.mDebugCounter)) != 0) {
            uint prevValue;
            mOutputCounterData.InterlockedAdd((1 + gPixelDebugPixel.y * mParams.mOutputDim.x + gPixelDebugPixel.x) * sizeof(uint), increment, prevValue);
            mOutputCounterData.InterlockedMax(0, prevValue + increment);
        }
        #endif
    }

    PackedPathVertex TraceRay(const Ray ray) {
        IncrementCounter(DebugCounterType::eRays);
        HitInfo hit;
        float dist;
        traceSceneRay<true>(ray, hit, dist, 0, 0xFFFFFFFF);
        return PackedPathVertex(ray, hit, dist);
    }
    bool Occluded(const Ray ray) {
        IncrementCounter(DebugCounterType::eOcclusionRays);
        return !traceSceneVisibilityRay<true>(ray, 0, 0xFFFFFFFF);
    }

    float Mis(float a) {
        if (gMISPowerExponent == 0.f)
            return a > 0 ? 1 : 0;
        else if (gMISPowerExponent == 1.f)
            return abs(a);
        else if (gMISPowerExponent == 2.f)
            return a * a;
        else
            return pow(abs(a), gMISPowerExponent);
    }
    float Mis(float a, float b) {
        a = Mis(a);
        return SafeDivide(a, a + Mis(b));
    }

    void StableONB(const float3 n, out float3 t, out float3 b) {
        const float3 u = (1 - abs(n.y) < kMinCosTheta) ? float3(1, 0, 0) : float3(0, 1, 0);
        t = normalize(cross(n, u));
        b = normalize(cross(n, t));
    }

    bool DebugDisableMIS() { return (gDebugBPT && mParams.mDebugLightVertices != -1); }
    bool DebugDisablePath(const uint bounces, const uint lightVertices, const bool merged = false) {
        if (!gDebugBPT)
            return false;
        return
            (merged && bool(mParams.mDebugDisableMerging))
            || (mParams.mDebugTotalVertices != -1 && mParams.mDebugTotalVertices != bounces + 2)
            || (mParams.mDebugLightVertices != -1 && mParams.mDebugLightVertices != lightVertices);
    }
    float GetMisVcWeightFactor() { return Mis(1.f / (sqr(mParams.mMergeRadius) * M_PI * mParams.mLightSubpathCount)); }
    float GetMisVmWeightFactor() { return gVertexMerging ? Mis((sqr(mParams.mMergeRadius) * M_PI * mParams.mLightSubpathCount)) : 0; }

    float GetImageToSolidAngleFactor(const float3 dir, out float cosCamera) {
        cosCamera = dot(dir, normalize(gScene.camera.data.cameraW));
        const float imagePlaneDist = mParams.mOutputDim.y * (gScene.camera.data.focalLength / gScene.camera.data.frameHeight);
        const float imagePointToCameraDist = imagePlaneDist / cosCamera;
        return (imagePointToCameraDist * imagePointToCameraDist) / cosCamera;
    }
    float GetMergeRadius(const float3 pos, const bool discretize = false) {
        if (gDynamicMergeRadius) {
            const float d = length(pos - gScene.camera.getPosition());
            const float pixelSize = tan(2.0f * atan(0.5f * gScene.camera.data.frameHeight / gScene.camera.data.focalLength)) / mParams.mOutputDim.y;
            float r = mParams.mDynamicMergeRadius * d * pixelSize;
            //if (discretize) {
            //    r = ldexp(1, int(ceil(log2(r))));
            //}
            return r;
        }
        return mParams.mMergeRadius;
    }

    int2 GetPixelCoord(const float3 pos) {
        float4 clip = mul(gScene.camera.data.viewProjMatNoJitter, float4(pos, 1));
        clip.y = -clip.y;
        return int2(((clip.xy / clip.w) * .5 + .5) * mParams.mOutputDim);
    }
    int2 GetPrevPixelCoord(const float3 pos) {
        float4 clip = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(pos, 1));
        clip.y = -clip.y;
        return int2(((clip.xy / clip.w) * .5 + .5) * mParams.mOutputDim);
    }

    // IO helpers

    LightSubpathVertex FindPhoton(inout SampleGenerator sg, const PathVertex vertex, out float weight) {
        // merge with a random vertex from the 8 nearest grid cells
        uint cell = -1;
        weight = 0.f;

        const float3 pos = vertex.GetPosition();
        const float mergeRadius = GetMergeRadius(pos, true);

        // Iterate the 8 neighbor grid cells and merge with the vertices in them.
        // pick one random grid cell to merge with, with probability proportional to its count
        const int3 o = int3(lerp(
                            float3(0, 0, 0),
                            float3(-1, -1, -1),
                            float3(frac(pos / (2.f * mergeRadius)) < 0.5f)));

        for (int x = 0; x < 2; x++) {
            for (int y = 0; y < 2; y++) {
                for (int z = 0; z < 2; z++) {
                    const uint cellIndex = mPhotonMap.getCellIndex(pos, mergeRadius, mParams.mPhotonCellCount, int3(x, y, z) + o);
                    if (cellIndex == -1)
                        continue;

                    const uint count = mPhotonMap.getCellPhotonCount(cellIndex);
                    if (count == 0)
                        continue;

                    weight += count;
                    if (sampleNext1D(sg) * weight < count)
                        cell = cellIndex;
                }
            }
        }

        if (cell == -1) {
            weight = 0;
            return {};
        }

        return mLightVertexCache[mPhotonMap.getCellOffset(cell)];
    }

    PackedPathVertex LoadPrimaryVertex(const uint2 id) {
        if (any(id > mParams.mOutputDim))
            return {};

        Ray cameraRay = gScene.camera.computeRayPinhole(id, mParams.mOutputDim);
        if (gUseViewDir) cameraRay.dir = -mViewDir[id];
        return PackedPathVertex(cameraRay, HitInfo(mVbuffer[id]), cameraRay.tMax);
    }
    PackedPathVertex LoadPrevPrimaryVertex(const uint2 id) {
        if (any(id > mParams.mOutputDim))
            return {};

        Ray cameraRay = gScene.camera.computeRayPinholePrevFrame(id, mParams.mOutputDim);
        if (gUseViewDir) cameraRay.dir = -mLastViewDir[id];
        return PackedPathVertex(cameraRay, HitInfo(mLastVbuffer[id]), cameraRay.tMax);
    }

    PathReservoir LoadPrevReservoir(const uint2 index) {
        if (!gUseResampling) return {};
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        return mLastReservoirs[flatIndex];
    }
    PathReservoir LoadInputReservoir(const uint2 index) {
        if (!gUseResampling) return {};
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        if (gSwapReservoirs != 0)
            return mPathReservoirs1[flatIndex];
        else
            return mPathReservoirs0[flatIndex];
    }
    PathReservoir LoadOutputReservoir(const uint2 index) {
        if (!gUseResampling) return {};
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        if (gSwapReservoirs != 0)
            return mPathReservoirs0[flatIndex];
        else
            return mPathReservoirs1[flatIndex];
    }
    void StoreReservoir(const uint2 index, const PathReservoir r) {
        if (!gUseResampling) return;
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        if (gSwapReservoirs != 0)
            mPathReservoirs0[flatIndex] = r;
        else
            mPathReservoirs1[flatIndex] = r;
    }

    PathReservoir LoadPrevCausticReservoir(const uint2 pixel) {
        if (!gUseResampling || !gUseCausticReservoirs) return {};
        const uint flatIndex = pixel.y * mParams.mOutputDim.x + pixel.x;
        return mLastCausticReservoirs[flatIndex];
    }
    PathReservoir LoadCausticReservoir(const uint2 pixel) {
        if (!gUseResampling || !gUseCausticReservoirs) return {};
        const uint flatIndex = pixel.y * mParams.mOutputDim.x + pixel.x;
        return mCausticReservoirs[flatIndex];
    }
    void StoreCausticReservoir(const uint2 pixel, const PathReservoir r) {
        if (!gUseResampling || !gUseCausticReservoirs) return;
        const uint flatIndex = pixel.y * mParams.mOutputDim.x + pixel.x;
        mCausticReservoirs[flatIndex] = r;
    }

    // ---------------------------------------------------------------------------------------------------------
    // ------------------------------------------- Emission sampling -------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    float3 SampleCosHemisphere(float2 u) {
        const float2 d = sample_disk_concentric(u);
        return float3(d, sqrt(max(0.f, 1.f - dot(d, d))));
    }
    float CosHemispherePdfW(float cosDir) {
        return cosDir * M_1_PI;
    }
    static const float gConcentricDiscPdfA = M_1_PI;

    void GetLightSamplePdf(const PathVertex vertex, const float3 prevPos, const float3 prevNormal, const bool upperOnly, out float emissionPdfW, out float directPdfW, out float directPdfA) {
        emissionPdfW = 0;
        directPdfW = 0;
        directPdfA = 0;

        if (vertex.IsFinite()) {
            if (gUseEmissiveLights) {
                HitInfo hit = HitInfo(vertex.mVertex.mPackedHit);
                if (hit.getType() == HitType::Triangle) {
                    const float cosIn = dot(vertex.GetFaceNormal(), vertex.GetDirIn());
                    if (cosIn > kMinCosTheta) {
                        const TriangleHit triHit = hit.getTriangleHit();
                        const uint triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
                        if (triangleIndex != LightCollection::kInvalidIndex) {
                            directPdfA = 1.f / gScene.lightCollection.getTriangle(triangleIndex).area;
                            directPdfA *= mEmissiveSampler.evalTriangleSelectionPdf(prevPos, prevNormal, upperOnly, triangleIndex);

                            if (gUseEnvLight) directPdfA *= 1 - mParams.mEnvMapProb;

                            emissionPdfW = directPdfA * CosHemispherePdfW(cosIn);
                            directPdfW   = directPdfA * sqr(vertex.mVertex.mRayDistance) / cosIn;
                        }
                    }
                }
            }
        } else {
            if (gUseEnvLight) {
                directPdfA = mEnvMapSampler.evalPdf(vertex.mVertex.mRayDirection);

                if (gUseEmissiveLights) directPdfA *= mParams.mEnvMapProb;

                emissionPdfW = directPdfA * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
                directPdfW = directPdfA;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// MIS weight helpers ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

extension RecursiveMisState {
    __init(const PathVertex vertex) {
        if (gPathGenerator.gBidirectional) {
            float cosAtCamera;
            const float cameraPdfW = gPathGenerator.GetImageToSolidAngleFactor(-vertex.GetDirIn(), cosAtCamera);
            dVCM = gPathGenerator.Mis(gPathGenerator.mParams.mLightSubpathCount / cameraPdfW);
            dVC = 0;
            dVM = 0;

            const float cosIn = vertex.GetCosN(vertex.GetDirIn());
            const float3 d = gScene.camera.getPosition() - vertex.GetPosition();
            dVCM *= gPathGenerator.Mis(dot(d,d) / cosIn);

            if (gPathGenerator.gDisableCameraConnection)
                dVCM = 0;
        }
    }

    __init(const bool finiteEmitter, const float directPdfA, const float emissionPdfW, const float cosLight) {
        dVCM = gPathGenerator.Mis(directPdfA / emissionPdfW);
        dVC = finiteEmitter ? gPathGenerator.Mis(cosLight / emissionPdfW) : 0;
        dVM = dVC * gPathGenerator.GetMisVcWeightFactor();
    }

    [mutating]
    void ProcessDirection(const DirectionSampleData r, const float cosOut, const bool connectable) {
        // update mis quantities
        if (gPathGenerator.gBidirectional) {
            if (connectable && !r.IsLobe(LobeType::Delta)) {
                // SmallVCM tech. rep. (34)-(36)
                const float invPdf = 1.f / r.mFwdPdfW;
                dVC = gPathGenerator.Mis(cosOut * invPdf) * (dVC * gPathGenerator.Mis(r.mRevPdfW) + dVCM + gPathGenerator.GetMisVmWeightFactor());
                dVM = gPathGenerator.Mis(cosOut * invPdf) * (dVM * gPathGenerator.Mis(r.mRevPdfW) + dVCM * gPathGenerator.GetMisVcWeightFactor() + 1);
                dVCM = gPathGenerator.Mis(invPdf);
            } else {
                // SmallVCM tech. rep. (53)-(55)
                dVC *= gPathGenerator.Mis(cosOut);
                dVM *= gPathGenerator.Mis(cosOut);
                dVCM = 0.f;
            }
        }
    }

    [mutating]
    void ProcessRay(const PackedPathVertex newVertex, const float cosIn, const bool prevFinite = true) {
        // update mis quantities
        if (gPathGenerator.gBidirectional && newVertex.IsFinite()) {
            if (prevFinite)
                dVCM *= gPathGenerator.Mis(sqr(newVertex.mRayDistance));
            const float invCosIn = 1 / gPathGenerator.Mis(cosIn);
            dVCM *= invCosIn;
            dVC  *= invCosIn;
            dVM  *= invCosIn;
        }
    }


    float EvalDirectLightMIS(const float directPdfW, const float emissionPdfW, const ReflectanceEvalData f, const float cosOut, const float cosLight) {
        float misWeight = 1;
        if (gPathGenerator.DebugDisableMIS()) return misWeight;

        if (gPathGenerator.gBidirectional) {
            const float wLight  = gPathGenerator.Mis(f.mFwdPdfW / directPdfW);
            const float wCamera = gPathGenerator.Mis(emissionPdfW * abs(cosOut) / (directPdfW * cosLight)) * (gPathGenerator.GetMisVmWeightFactor() + dVCM + dVC * gPathGenerator.Mis(f.mRevPdfW));
            misWeight = 1 / (wLight + 1 + wCamera);
        } else {
            misWeight = gPathGenerator.Mis(directPdfW, f.mFwdPdfW);
        }
        return misWeight;
    }

    float EvalVertexConnectionMIS(const RecursiveMisState v, const ReflectanceEvalData r, const ReflectanceEvalData f, const float cosCam, const float cosLight, const float invDist2) {
        float misWeight = 1;
        if (gPathGenerator.DebugDisableMIS()) return misWeight;

        const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
        const float lightBsdfDirPdfA  = r.mFwdPdfW * cosCam   * invDist2;
        const float wLight  = gPathGenerator.Mis(cameraBsdfDirPdfA) * (gPathGenerator.GetMisVmWeightFactor() + v.dVCM + v.dVC * gPathGenerator.Mis(r.mRevPdfW));
        const float wCamera = gPathGenerator.Mis(lightBsdfDirPdfA)  * (gPathGenerator.GetMisVmWeightFactor() +   dVCM +   dVC * gPathGenerator.Mis(f.mRevPdfW));
        misWeight = 1 / (wLight + 1 + wCamera);
        return misWeight;
    }

    float EvalVertexMergingMIS(const RecursiveMisState v, const ReflectanceEvalData f) {
        float misWeight = 1;
        if (gPathGenerator.DebugDisableMIS() || gPathGenerator.gProgressivePhotonMapping) return misWeight;

        const float wLight  = v.dVCM * gPathGenerator.GetMisVcWeightFactor() + v.dVM * gPathGenerator.Mis(f.mFwdPdfW);
        const float wCamera =   dVCM * gPathGenerator.GetMisVcWeightFactor() +   dVM * gPathGenerator.Mis(f.mRevPdfW);
        misWeight = 1 / (wLight + 1 + wCamera);

        return misWeight;
    }

    float EvalLightTraceMIS(const float scatterRevPdfW, const float cameraPdfA) {
        float misWeight = 1;
        if (gPathGenerator.DebugDisableMIS() || gPathGenerator.gLightTraceOnly) return misWeight;

        const float wLight = gPathGenerator.Mis(cameraPdfA / gPathGenerator.mParams.mLightSubpathCount) * (gPathGenerator.GetMisVmWeightFactor() + dVCM + dVC * gPathGenerator.Mis(scatterRevPdfW));
        misWeight = 1 / (wLight + 1);

        return misWeight;
    }

    float EvalCameraTraceMIS(const float scatterFwdPdfW, const float directPdfA, const float directPdfW, const float emissionPdfW) {
        float misWeight = 1;
        if (gPathGenerator.DebugDisableMIS()) return misWeight;

        if (gPathGenerator.gBidirectional) {
            const float wCamera = gPathGenerator.Mis(directPdfA) * dVCM + gPathGenerator.Mis(emissionPdfW) * dVC;
            misWeight = 1 / (1 + wCamera);
        } else if (gPathGenerator.gSampleDirectLight)
            misWeight = scatterFwdPdfW > 0 ? gPathGenerator.Mis(scatterFwdPdfW, directPdfW) : 1;

        return misWeight;
    }
}

extension ReconnectionMisState {
    property float sumProd_eta { get { return 0; } }

    [mutating]
    void Update(const RecursiveMisState mis, float pdfFwd, float pdfRev, float cosIn, float cosOut, float dist, uint idx) {
        bool isDelta = mis.dVCM == 0;
        if (idx == 0) {
            invFwdG_r = (dist*dist) / cosIn;
            sumProd_dVCM = 0;
            prod_dVCM    = 0;
            //sumProd_eta  = 0;
            last_invPdf  = 1;
            prod_ratio = gPathGenerator.Mis(cosOut / cosIn);
        }
        if (idx == 1)
            prod_dVCM = isDelta ? 0 : 1;

        // update sum products

        if (idx > 1) {
            float r = gPathGenerator.Mis(pdfRev * last_invPdf);
            prod_ratio *= gPathGenerator.Mis(last_invPdf);
            prod_dVCM  *= r;
            //sumProd_eta  *= r;
            sumProd_dVCM *= r;
            if (!isDelta) {
                //sumProd_eta  += r;
                sumProd_dVCM += end_dVCM;
            }
        }
        if (idx > 0) {
            if (!isDelta)
                prod_ratio *= gPathGenerator.Mis(pdfRev);
        }

        // update last_invPdf for next bounce

        float r = cosOut / cosIn;
        if (pdfFwd > 0)
            r /= pdfFwd;
        if (idx > 0) {
            last_invPdf = r;
        }

        end_coeff = gPathGenerator.Mis(r);
        end_dVCM = mis.dVCM;
    }

    void Apply(inout RecursiveMisState mis_r, float pdfFwd_r, float pdfRev_r, uint N) {
        float invPdf = 1 / pdfFwd_r;
        float ratio_r = gPathGenerator.Mis(invPdf) * prod_ratio;

        if (N == 0) {
            // vertex after rcv is last vertex
            mis_r.dVC = ratio_r * (
                gPathGenerator.GetMisVmWeightFactor()
                +
                mis_r.dVCM
                +
                mis_r.dVC * gPathGenerator.Mis(pdfRev_r));
            mis_r.dVCM = gPathGenerator.Mis(invPdf * invFwdG_r);
        } else {
            mis_r.dVC = end_coeff * (
                gPathGenerator.GetMisVmWeightFactor() * (1 + ratio_r + sumProd_eta)
                +
                mis_r.dVCM * ratio_r + gPathGenerator.Mis(invPdf * invFwdG_r) * prod_dVCM + sumProd_dVCM
                +
                mis_r.dVC * ratio_r * gPathGenerator.Mis(pdfRev_r));

            mis_r.dVCM = end_dVCM;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Path tracing implementation ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

SampleGenerator MakeSampleGenerator(const uint2 id, const uint seed) { return SampleGenerator(id, seed); }

extension PathGenerator {
    struct CachedVertexData {
        float3            position       = 0;
        float             scatterFwdPdfW = 0;
        float3            normal         = 0;
        float             scatterRevPdfW = 0;
        RecursiveMisState mis = {};
        uint              flags = 0;

        property bool upperHemisphereOnly {
            get { return (bool)BF_GET(flags, 0, 1); }
            set { BF_SET(flags, (uint)newValue, 0, 1); }
        }
        property bool connectable {
            get { return (bool)BF_GET(flags, 1, 1); }
            set { BF_SET(flags, (uint)newValue, 1, 1); }
        }

        bool HasNormal() { return all(normal != 0); }
    };

    struct PathState<let bShift : bool> {
        SampleGenerator sg;     // sample generator for everything but BSDF sampling
        SampleGenerator sgBsdf; // sample generator for BSDF sampling

        float3 throughput = 0;
        float  pdfW = 0;

        PathId id;
        uint packedBounces = 0;
        uint packedDiffuseBounces = 0;

        CachedVertexData lastVertex = {};

        RecursiveMisState mis = {};
        ReconnectionMisState rcvMis = {};

        PackedPathVertex rcv = {};
        float3 throughputAtRcv = 0;
        float  pdfAtRcv = 0;
        float3 rcvDirOut;
        float  prefixPdf = 0;
        float  rcvCos = 0;
        float  rcvDist = 0;

        PathId        lightSubpathId;
        PackedHitInfo lightSubpathVertex = {};

        property uint bounces {
            get { return BF_GET(packedBounces,    0, 15); }
            set { BF_SET(packedBounces, newValue, 0, 15); }
        }
        property uint prefixBounces {
            get { return BF_GET(packedBounces,    15, 15); }
            set { BF_SET(packedBounces, newValue, 15, 15); }
        }
        property uint diffuseBounces {
            get { return BF_GET(packedDiffuseBounces, 0, 16); }
            set { BF_SET(packedDiffuseBounces, newValue, 0, 16); }
        }
        property uint rcvDiffuseBounces {
            get { return BF_GET(packedDiffuseBounces, 16, 16); }
            set { BF_SET(packedDiffuseBounces, newValue, 16, 16); }
        }
        property bool isLightPath {
            get { return (bool)BF_GET(packedBounces,    30, 1); }
            set { BF_SET(packedBounces, (uint)newValue, 30, 1); }
        }
        property bool deltaRcv {
            get { return (bool)BF_GET(packedBounces,    31, 1); }
            set { BF_SET(packedBounces, (uint)newValue, 31, 1); }
        }
        property bool rcvFound {
            get { return prefixPdf > 0; }
        }

        [mutating]
        void Terminate() {
            throughput = 0;
        }
    };

    LightSubpathVertex PackLightSubpathVertex<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex) {
        LightSubpathVertex v = {};
        v.mLightSubpath = LightSubpathData::Create(
            s.lightSubpathVertex,
            s.id,
            s.bounces + 2,
            s.diffuseBounces,
            s.pdfW
        );
        v.mVertex = vertex.Pack();
        v.mThroughput = s.throughput;
        v.mis = s.mis;
        return v;
    }

    // Merges a canonical sample into the pixel's reservoir
    void ProcessSample(inout PathReservoir r, inout SampleGenerator sg, PathSample ps, float weight) {
        if (gUseResampling) {
            if (!PathSample::gIntegrandMIS) weight *= ps.mMisWeight;
            r.Merge(sampleNext1D(sg), ps, weight * ps.GetTargetPdf(), 0);
        } else if (ps.IsValid() && weight > 0) {
            print("Found path. contrib:", ps.mContribution);
            print("\tLight subpath vertices:", ps.mSuffix.mLightSubpath.mVertexCount);
            print("\tMerge:", ps.HasFlag(PathFlags::eVertexMerging));
            print("\tPrefix bounces:", ps.mPrefixBounces);
            r.mSample.mContribution += ps.mContribution * (ps.mMisWeight * weight);
        }
    }


    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------- Connection stategies ------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathSample ConnectToLight       <let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out float integrationWeight) {
        integrationWeight = 0;

        PathSample result = {};

        if (DebugDisablePath(s.bounces + 1, 1)) {
            if (bShift) print("Shift failed: disabled", 0);
            return result;
        }

        // Sample emissive path vertex

        const PathId lightPathId = bShift ? s.lightSubpathId : PathId(s.sg.next(), uint2(s.sg.next(), s.sg.next()));
        SampleGenerator sg = MakeSampleGenerator(lightPathId.mId, lightPathId.mSeed);

        float directPdfA = 0;
        float directPdfW = 0;
        float emissionPdfW = 0;
        PackedPathVertex l = {};
        float cosLight = 1;
        float3 le = 0;
        {
            const bool useEnvMap = sampleNext1D(sg) < mParams.mEnvMapProb;
            const float2 uv = sampleNext2D(sg);

            if (gUseEnvLight && (!gUseEmissiveLights || useEnvMap)) {
                EnvMapSample envSample;
                if (!mEnvMapSampler.sample(uv, envSample)) {
                    if (bShift) print("Shift failed: invalid sample", 0);
                    return result;
                }

                if (gUseEmissiveLights) envSample.pdf *= mParams.mEnvMapProb;

                le = envSample.Le;
                if (!any(le > 0)) {
                    if (bShift) print("Shift failed: zero le", 0);
                    return result;
                }

                l = PackedPathVertex(Ray(vertex.ComputeRayOrigin(envSample.dir), envSample.dir), HitInfo(0), FLT_MAX);

                directPdfA = envSample.pdf;
                directPdfW = envSample.pdf;
                emissionPdfW = directPdfW * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && !(gUseEnvLight && useEnvMap)) {
                const bool upperOnly = vertex.UpperHemisphereOnly();
                const float3 vertexNormal = upperOnly ? vertex.GetFaceNormal() : 0;
                float selectionPdf;
                uint triangleIndex;
                float3 barycentrics;
                if (bShift) {
                    const TriangleHit triHit = HitInfo(s.lightSubpathVertex).getTriangleHit();
                    triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
                    if (triangleIndex != LightCollection::kInvalidIndex) {
                        selectionPdf = mEmissiveSampler.evalTriangleSelectionPdf(vertex.GetPosition(), vertexNormal, upperOnly, triangleIndex);
                        barycentrics = float3(1 - triHit.barycentrics.x - triHit.barycentrics.y, triHit.barycentrics);
                    }
                } else {
                    triangleIndex = mEmissiveSampler.sampleLight(vertex.GetPosition(), vertexNormal, upperOnly, sg, selectionPdf);
                    barycentrics = sample_triangle(uv);
                }

                if (triangleIndex == LightCollection::kInvalidIndex) {
                    if (bShift) print("Shift failed: invalid triangle", triangleIndex);
                    return result;
                }

                if (gUseEnvLight) selectionPdf *= 1 - mParams.mEnvMapProb;

                let tri = gScene.lightCollection.getTriangle(triangleIndex);

                le = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(barycentrics));
                if (!any(le > 0)) {
                    if (bShift) print("Shift failed: zero le", le);
                    return result;
                }

                directPdfA = selectionPdf / tri.area;

                const float3 triPos = computeRayOrigin(tri.getPosition(barycentrics), tri.normal);
                const float3 rayOrigin = vertex.ComputeRayOrigin(triPos - vertex.GetPosition());
                float3 toLight = triPos - rayOrigin;
                const float dist = length(toLight);
                toLight /= dist;

                cosLight = -dot(tri.normal, toLight);
                if (cosLight < kMinCosTheta) {
                    if (bShift) print("Shift failed: cosLight < kMinCosTheta", cosLight);
                    return result;
                }

                const MeshLightData meshData = gScene.lightCollection.getMeshData(triangleIndex);

                TriangleHit triHit = {};
                triHit.instanceID     = { meshData.instanceID };
                triHit.primitiveIndex = triangleIndex - meshData.triangleOffset;
                triHit.barycentrics = barycentrics.yz;

                l = PackedPathVertex(Ray(rayOrigin, toLight), HitInfo(triHit), dist);

                directPdfW   = directPdfA * sqr(dist) / cosLight;
                emissionPdfW = directPdfA * CosHemispherePdfW(cosLight);
            } else {
                return result;
            }
        }

        const float cosOut = vertex.GetCosN(l.mRayDirection);
        if (cosOut < kMinCosTheta)
            return result;

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, l.mRayDirection, false);

        if (gBidirectional && gDisableCameraConnection && !s.isLightPath) {
            if (s.bounces == 0)
                emissionPdfW = 0;
            else if (s.bounces == 1)
                f.mRevPdfW = 0;
        }

        const float3 contrib = s.throughput * f.mReflectance * le;
        const float samplePdf = s.pdfW * directPdfW;
        if (!any(contrib > 0) || !(samplePdf > 0))
            return result;

        if (Occluded(Ray(l.mRayOrigin, l.mRayDirection, 0, l.mRayDistance)))
            return result;

        // divide by pdf
        integrationWeight = 1.f / samplePdf;

        // compute mis weight
        const float misWeight = s.mis.EvalDirectLightMIS(directPdfW, emissionPdfW, f, cosOut, cosLight);
        MisEvalData misEvalArgs = {};
        misEvalArgs.reconnectionMis = s.rcvMis;
        misEvalArgs.directPdfW = directPdfW;
        misEvalArgs.emissionPdfW = emissionPdfW;
        misEvalArgs.cameraScatterFwdPdfW = f.mFwdPdfW;
        misEvalArgs.cameraScatterRevPdfW = f.mRevPdfW;
        misEvalArgs.cosCam = cosOut;
        misEvalArgs.cosLight = cosLight;

        float3 suffixContrib;
        float3 suffixDir;
        float suffixPdf = directPdfW;
        const bool hasRcv = s.rcvFound;
        if (hasRcv) {
            // use rcv found earlier along the camera subpath
            if (s.bounces == s.prefixBounces + 1) {
                // the light vertex is the 2nd vertex in the suffix
                suffixContrib = le;
                suffixDir = l.mRayDirection;
                suffixPdf = directPdfW;
            } else {
                // the light vertex is after the 2nd suffix vertex
                suffixContrib = SafeDivide(contrib, s.throughputAtRcv);
                suffixDir = s.rcvDirOut;
                suffixPdf = directPdfW * (s.pdfW / s.pdfAtRcv);
            }
        } else {
            // light vertex is the first vertex in the suffix
            suffixContrib = le;
            suffixDir = 0;
            suffixPdf = directPdfA;
        }

        const LightSubpathData lightSubpath = LightSubpathData::Create(
            l.mPackedHit,
            lightPathId,
            1,
            0,
            directPdfA
        );

        const PathSuffix suffix = PathSuffix::Create(
            suffixContrib,
            suffixPdf,
            hasRcv ? s.rcv : l,
            suffixDir,
            hasRcv ? s.rcvDist : l.mRayDistance,
            hasRcv ? s.rcvCos : cosLight,
            misEvalArgs,
            lightSubpath
        );

        result = PathSample::Create(
            contrib,
            misWeight,
            s.bounces + 1,
            hasRcv ? s.prefixPdf         : s.pdfW,
            hasRcv ? s.prefixBounces     : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eNone
        );

        return result;
    }
    PathSample ConnectToLightSubpath<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, out float integrationWeight) {
        integrationWeight = 0;

        PathSample result = {};

        const uint bounces = s.bounces + v.mBounces + 2;
        if (bounces > mParams.mMaxBounces || s.diffuseBounces + v.mDiffuseBounces + 2 > mParams.mMaxDiffuseBounces || DebugDisablePath(bounces, v.mLightSubpath.mVertexCount))
            return result;

        // evaluate the bsdf at the light subpath vertex
        float cosLight, dist;
        float3 toLight;
        Ray visibilityRay;
        ReflectanceEvalData r;
        {
            const PathVertex lightVertex = PathVertex(v.mVertex);
            toLight = lightVertex.GetPosition() - vertex.GetPosition();
            dist = length(toLight);
            toLight /= dist;

            cosLight = lightVertex.GetCosN(toLight);
            if (cosLight < kMinCosTheta)
                return result;

            r = lightVertex.EvaluateReflectance(s.sg, -toLight, true);
            r.mReflectance /= cosLight;

            if (gBidirectional && gDisableCameraConnection && !s.isLightPath && s.bounces == 0)
                r.mFwdPdfW = 0;

            const float3 rayOrigin = vertex.ComputeRayOrigin(toLight);
            visibilityRay = Ray(rayOrigin, lightVertex.ComputeRayOrigin(-toLight) - rayOrigin, 0, 0);
            visibilityRay.tMax = length(visibilityRay.dir);
            visibilityRay.dir /= visibilityRay.tMax;
        }
        if (!any(r.mReflectance > 0)) {
            if (bShift) print("Subpath reconnection failed: !any(r.mReflectance > 0)", r.mReflectance);
            return result;
        }

        // evaluate the bsdf at the camera subpath vertex
        const float cosCam = vertex.GetCosN(toLight);
        if (cosCam < kMinCosTheta)
            return result;

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, toLight, false);
        if (!any(f.mReflectance > 0)) {
            if (bShift) print("Subpath reconnection failed: !any(f.mReflectance > 0)", f.mReflectance);
            return result;
        }

        if (gBidirectional && gDisableCameraConnection && !s.isLightPath && s.bounces == 1)
            f.mRevPdfW = 0;

        const float3 contrib = (s.throughput * v.mThroughput) * (f.mReflectance * r.mReflectance);

        if (!any(contrib > 0) || any(contrib < 0)) {
            if (bShift) print("Subpath reconnection failed: any(contrib > 0) || any(contrib < 0)", contrib);
            return result;
        }

        if (Occluded(visibilityRay)) {
            if (bShift) print("Subpath reconnection failed: Occluded", 0);
            return result;
        }

        float lightSubpathPdfW = v.mLightSubpath.mPdf * (sqr(dist) / cosLight);

        integrationWeight = 1 / (s.pdfW * lightSubpathPdfW);

        const float invDist2 = 1 / sqr(dist);

        const float misWeight = s.mis.EvalVertexConnectionMIS(v.mis, r, f, cosCam, cosLight, invDist2);
        MisEvalData misEvalArgs = {};
        misEvalArgs.reconnectionMis = s.rcvMis;
        misEvalArgs.lightSubpathMis = v.mis;
        misEvalArgs.lightScatterFwdPdfW = r.mFwdPdfW;
        misEvalArgs.lightScatterRevPdfW = r.mRevPdfW;
        misEvalArgs.cameraScatterFwdPdfW = f.mFwdPdfW;
        misEvalArgs.cameraScatterRevPdfW = f.mRevPdfW;
        misEvalArgs.cosCam = cosCam;
        misEvalArgs.cosLight = cosLight;
        misEvalArgs.invDist2 = invDist2;

        float3 suffixContrib;
        float3 suffixDir;
        float suffixPdf;
        const bool hasRcv = s.rcvFound;
        if (hasRcv) {
            // use rcv found earlier along the camera subpath
            if (s.bounces == s.prefixBounces + 1) {
                // the light vertex is the 2nd vertex in the suffix
                suffixContrib = v.mThroughput * r.mReflectance;
                suffixDir = toLight;
                suffixPdf = lightSubpathPdfW;
            } else {
                // the light vertex is after the 2nd suffix vertex
                suffixContrib = SafeDivide(contrib, s.throughputAtRcv);
                suffixDir = s.rcvDirOut;
                suffixPdf = lightSubpathPdfW * (s.pdfW / s.pdfAtRcv);
            }
        } else {
            // use the light subpath vertex as the rcv
            suffixContrib = v.mThroughput;
            suffixDir = v.GetDirIn();
            suffixPdf = v.mLightSubpath.mPdf;
        }

        const PathSuffix suffix = PathSuffix::Create(
            suffixContrib,
            suffixPdf,
            hasRcv ? s.rcv : v.mVertex,
            suffixDir,
            hasRcv ? s.rcvDist : dist,
            hasRcv ? s.rcvCos : cosLight,
            misEvalArgs,
            v.mLightSubpath
        );

        result = PathSample::Create(
            contrib,
            misWeight,
            bounces,
            hasRcv ? s.prefixPdf         : s.pdfW,
            hasRcv ? s.prefixBounces     : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eNone);

        return result;
    }
    PathSample MergeWithLightVertex <let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, out float integrationWeight) {
        integrationWeight = 0;

        PathSample result = {};

        const float mergeRadius = GetMergeRadius(vertex.GetPosition());

        const uint bounces = s.bounces + v.mBounces + 1;
        if (bounces > mParams.mMaxBounces || s.diffuseBounces + v.mDiffuseBounces + 1 > mParams.mMaxDiffuseBounces || DebugDisablePath(bounces, v.mLightSubpath.mVertexCount, true))
            return result;

        const PathVertex photon = PathVertex(v.mVertex);
        const float3 toPhoton = photon.GetPosition() - vertex.GetPosition();
        if (dot(toPhoton, toPhoton) >= sqr(mergeRadius))
            return result;

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, v.GetDirIn(), false);
        if (vertex.HasFaceNormal()) {
            const float3 guideNormal = vertex.mMaterial.getProperties(vertex.mShadingData).guideNormal;
            const float cosOut = abs(dot(guideNormal, v.GetDirIn()));
            if (cosOut < kMinCosTheta)
                return result;
            f.mReflectance /= cosOut; // divide out dot(N, wo)
        }

        if (gBidirectional && gDisableCameraConnection && !s.isLightPath && s.bounces == 1)
            f.mRevPdfW = 0;

        const float3 contrib = s.throughput * v.mThroughput * f.mReflectance;
        if (all(contrib <= 0))
            return result;

        integrationWeight = 1 / (s.pdfW * v.mLightSubpath.mPdf * (sqr(mergeRadius) * M_PI * mParams.mLightSubpathCount));

        const float misWeight = s.mis.EvalVertexMergingMIS(v.mis, f);
        MisEvalData misEvalArgs = {};
        misEvalArgs.reconnectionMis = s.rcvMis;
        misEvalArgs.lightSubpathMis = v.mis;
        misEvalArgs.cameraScatterFwdPdfW = f.mFwdPdfW;
        misEvalArgs.cameraScatterRevPdfW = f.mRevPdfW;

        float3 suffixContrib;
        float3 suffixDir;
        float suffixPdf = v.mLightSubpath.mPdf;
        const bool hasRcv = !bShift && s.rcvFound;
        if (hasRcv) {
            // use rcv found earlier along the camera subpath
            if (s.bounces == s.prefixBounces + 1) {
                // the light vertex is the 2nd vertex in the suffix
                suffixContrib = v.mThroughput * f.mReflectance;
                suffixDir = v.GetDirIn();
            } else {
                // the light vertex is after the 2nd suffix vertex
                suffixContrib = SafeDivide(contrib, s.throughputAtRcv);
                suffixDir = s.rcvDirOut;
                suffixPdf *= (s.pdfW / s.pdfAtRcv);
            }
        } else {
            // use the light subpath vertex as the rcv
            suffixContrib = v.mThroughput;
            suffixDir = v.GetDirIn();
        }

        const PathSuffix suffix = PathSuffix::Create(
            suffixContrib,
            suffixPdf,
            hasRcv ? s.rcv     : v.mVertex,
            suffixDir,
            hasRcv ? s.rcvDist : 0,
            hasRcv ? s.rcvCos  : 0,
            misEvalArgs,
            v.mLightSubpath
        );

        result = PathSample::Create(
            contrib,
            misWeight,
            bounces,
            hasRcv ? s.prefixPdf         : s.pdfW,
            hasRcv ? s.prefixBounces     : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eVertexMerging);

        return result;
    }
    PathSample ConnectToCamera      <let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out int2 pixelCoord, out float integrationWeight, const bool checkVisibility = true) {
        integrationWeight = 0;
        pixelCoord = -1;
        PathSample result = {};

        if (DebugDisablePath(s.bounces + 1, s.bounces + 2))
            return result;

        const float3 rayOrigin = vertex.ComputeRayOrigin(gScene.camera.getPosition() - vertex.GetPosition());

        float3 toCamera = gScene.camera.getPosition() - rayOrigin;
        const float dist = length(toCamera);
        toCamera /= dist;

        float cosAtCamera;
        const float imageToSolidAngleFactor = GetImageToSolidAngleFactor(-toCamera, cosAtCamera);

        pixelCoord = GetPixelCoord(vertex.GetPosition());
        if (any(pixelCoord < 0) || any(pixelCoord >= mParams.mOutputDim) || cosAtCamera <= 0)
            return result;

        const float cosOut = vertex.GetCosN(toCamera);
        if (cosOut < kMinCosTheta)
            return result;

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, toCamera, true);
        f.mReflectance /= cosOut;

        /*{
            const uint2 cached = gPixelDebugPixel;
            printSetPixel(pixelCoord);
            print(bShift ? "shifting" : "canonical", uint4(s.bounces, s.deltaRcv, s.id.mId));
            print("   s.throughput", luminance(s.throughput));
            print("   f.mReflectance", luminance(f.mReflectance));
            printSetPixel(cached);
        }*/

        // Compute pdf conversion factor from image plane area to surface area
        const float cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);

        const float cosIn = vertex.GetCosN(vertex.GetDirIn());

        const float3 contrib = s.throughput * f.mReflectance;
        if (!any(contrib > 0))
            return result;

        const float samplePdf = s.pdfW * (1 / cameraPdfA);
        if (!(samplePdf > 0))
            return result;
        if (checkVisibility) {
            if (Occluded(Ray(rayOrigin, toCamera, 0, dist)))
                return result;
        }

        integrationWeight = 1.f / samplePdf;

        if (!gUseResampling) integrationWeight /= mParams.mLightSubpathCount;

        const float misWeight = s.mis.EvalLightTraceMIS(f.mRevPdfW, cameraPdfA);
        MisEvalData misEvalArgs = {};
        misEvalArgs.reconnectionMis = s.rcvMis;
        misEvalArgs.lightSubpathMis = s.deltaRcv ? s.mis : s.lastVertex.mis;

        PathSuffix suffix = {};
        if (s.deltaRcv) {
            // if the rcv is unconnectable, this is a caustic light path and we store the primary vertex as the rcv instead.
            suffix = PathSuffix::Create(
                s.throughput,
                1,
                vertex.Pack(),
                vertex.GetDirIn(),
                dist,
                cosIn,
                misEvalArgs,
                LightSubpathData::Create(
                    s.lightSubpathVertex,
                    s.id,
                    s.bounces + 2,
                    s.diffuseBounces + 1,
                    s.pdfW
                ),
            );
        } else {
            suffix = PathSuffix::Create(
                s.throughputAtRcv,
                1,
                s.rcv,
                s.rcvDirOut,
                s.rcvDist,
                s.rcvCos,
                misEvalArgs,
                LightSubpathData::Create(
                    s.lightSubpathVertex,
                    s.id,
                    s.bounces + 1,
                    s.diffuseBounces + 1,
                    s.pdfAtRcv
                )
            );
        }

        result = PathSample::Create(
            contrib,
            misWeight,
            s.bounces + 1,
            1,
            0,
            0,
            s.id,
            suffix,
            s.deltaRcv ? PathFlags::eCausticLightPath : PathFlags::eNonCausticLightPath);

        return result;
    }

    PathSample EvalLightContribution<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out float integrationWeight) {
        integrationWeight = 0;
        PathSample result = {};

        if (DebugDisablePath(s.bounces, 0))
            return result;

        const float3 le = vertex.GetEmission();
        const float3 contrib = s.throughput * le;
        if (!any(contrib > 0))
            return result;

        // calculate mis weight
        float emissionPdfW = 0;
        float directPdfW = 0;
        float directPdfA = 0;
        if (gBidirectional || gSampleDirectLight)
            GetLightSamplePdf(vertex, s.lastVertex.position, s.lastVertex.normal, s.lastVertex.upperHemisphereOnly, emissionPdfW, directPdfW, directPdfA);

        // divide by pdf
        integrationWeight = 1 / s.pdfW;

        float misWeight = s.mis.EvalCameraTraceMIS(s.lastVertex.scatterFwdPdfW, directPdfA, directPdfW, emissionPdfW);
        MisEvalData misEvalArgs = {};
        misEvalArgs.reconnectionMis = s.rcvMis;
        misEvalArgs.cameraScatterFwdPdfW = s.lastVertex.scatterFwdPdfW;
        misEvalArgs.directPdfA = directPdfA;
        misEvalArgs.directPdfW = directPdfW;
        misEvalArgs.emissionPdfW = emissionPdfW;

        PathSuffix suffix = {};
        const bool hasRcv = s.rcvFound;
        if (hasRcv) {
            // finalize the existing rcv
            float suffixPdf;
            float3 suffixContrib;
            if (s.bounces == s.prefixBounces + 1) {
                // rcv is the last vertex
                suffixPdf = 1;
                suffixContrib = le;
            } else {
                suffix.mDirOut = s.rcvDirOut;
                suffixPdf = s.pdfW / s.pdfAtRcv;
                suffixContrib = SafeDivide(contrib, s.throughputAtRcv);
            }

            suffix = PathSuffix::Create(
                suffixContrib,
                suffixPdf,
                s.rcv,
                s.rcvDirOut,
                s.rcvDist,
                s.rcvCos,
                misEvalArgs,
                LightSubpathData::Create(
                    {},
                    {},
                    0,
                    0,
                    0
                ) );
        }

        result = PathSample::Create(
            contrib,
            misWeight,
            s.bounces,
            hasRcv ? s.prefixPdf : s.pdfW,
            hasRcv ? s.prefixBounces : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eNone );
        return result;

    }

    // ---------------------------------------------------------------------------------------------------------
    // -------------------------------------------- Path generation --------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathState<bShift> InitializeCameraPath<let bShift : bool>(const PathId id, const PathVertex vertex) {
        PathState<bShift> s = {};
        s.id = id;
        s.isLightPath = false;

        s.sg     = MakeSampleGenerator(s.id.mId, s.id.mSeed);
        s.sgBsdf = MakeSampleGenerator(s.id.mId, -s.id.mSeed);

        s.bounces = 0;
        s.diffuseBounces = 0;
        s.prefixBounces = 0;
        s.rcvDiffuseBounces = 0;
        s.throughput = 1;
        s.throughputAtRcv = 0;
        s.prefixPdf = 0;
        s.rcvDirOut = 0;
        s.pdfW = 1;
        s.mis = RecursiveMisState(vertex);
        return s;
    }
    PathState<bShift> InitializeLightPath<let bShift : bool>(const PathId id, out PathVertex vertex, PackedHitInfo firstVertex = {}) {
        PathState<bShift> s = {};
        s.id = id;
        s.isLightPath = true;

        s.throughput = 0;
        s.pdfW = 0;
        s.throughputAtRcv = 0;
        s.rcvDirOut = 0;

        s.sg     = MakeSampleGenerator(s.id.mId, s.id.mSeed);
        s.sgBsdf = MakeSampleGenerator(s.id.mId, -s.id.mSeed);

        vertex.mVertex = {};

        float cosLight = 0;
        float directPdfA = 0;
        float emissionPdfW = 0;
        Ray ray = {};
        PackedPathVertex l = {};
        {
            const bool useEnvMap = sampleNext1D(s.sg) < mParams.mEnvMapProb;
            const float4 sampleU = sampleNext4D(s.sg);

            if (gUseEnvLight && (!gUseEmissiveLights || useEnvMap || (bShift && !HitInfo(firstVertex).isValid()))) {
                EnvMapSample envSample;
                if (!mEnvMapSampler.sample(sampleU.xy, envSample))
                    return s;

                if (gUseEmissiveLights) envSample.pdf *= mParams.mEnvMapProb;

                float3 t, b;
                StableONB(envSample.dir, t, b);
                const float2 uv = sample_disk_concentric(sampleU.zw);

                ray = Ray(mParams.mSceneSphere.xyz + mParams.mSceneSphere.w * (t * uv.x + b * uv.y + envSample.dir), -envSample.dir);
                l = PackedPathVertex(ray, HitInfo(0), FLT_MAX);
                s.throughput = envSample.Le;
                cosLight = 1;

                directPdfA = envSample.pdf;
                emissionPdfW = directPdfA * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && !(gUseEnvLight && useEnvMap)) {
                float selectionPdf;
                uint triangleIndex;
                float3 barycentrics;
                if (bShift) {
                    TriangleHit triHit = HitInfo(firstVertex).getTriangleHit();
                    triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
                    if (triangleIndex != LightCollection::kInvalidIndex) {
                        selectionPdf = mEmissiveSampler.evalTriangleSelectionPdf(triangleIndex);
                        barycentrics = float3(1 - triHit.barycentrics.x - triHit.barycentrics.y, triHit.barycentrics);
                    }
                } else {
                    triangleIndex = mEmissiveSampler.sampleLight(s.sg, selectionPdf);
                    barycentrics = sample_triangle(sampleU.xy);
                }

                if (triangleIndex == LightCollection::kInvalidIndex)
                    return s;

                if (gUseEnvLight) selectionPdf *= 1 - mParams.mEnvMapProb;

                let tri = gScene.lightCollection.getTriangle(triangleIndex);
                directPdfA = selectionPdf / tri.area;

                float dirPdfW;
                const float3 localDir = sample_cosine_hemisphere_concentric(sampleU.zw, dirPdfW);
                cosLight = localDir.z;

                if (cosLight < kMinCosTheta)
                    return s;

                float3 t, b;
                StableONB(tri.normal, t, b);
                const float3 rayDir = normalize(localDir.x * t + localDir.y * b + localDir.z * tri.normal);

                const float3 rayOrigin = computeRayOrigin(tri.getPosition(barycentrics), tri.normal);
                ray = Ray(rayOrigin, rayDir);

                const MeshLightData meshData = gScene.lightCollection.getMeshData(triangleIndex);

                TriangleHit triHit = {};
                triHit.instanceID = { meshData.instanceID };
                triHit.primitiveIndex = triangleIndex - meshData.triangleOffset;
                triHit.barycentrics = barycentrics.yz;
                l = PackedPathVertex(ray, HitInfo(triHit), FLT_MAX);

                s.throughput = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(barycentrics)) * cosLight;

                emissionPdfW = directPdfA * dirPdfW;
            } else {
                return s;
            }
        }

        const bool finiteEmitter = l.IsFinite();

        s.lightSubpathVertex = l.mPackedHit;
        s.lightSubpathId = s.id;

        s.pdfW = emissionPdfW;
        s.bounces = 0;
        s.diffuseBounces = 0;
        s.prefixBounces = 0;
        s.rcvDiffuseBounces = 0;

        s.mis = RecursiveMisState(finiteEmitter, directPdfA, emissionPdfW, cosLight);
        s.lastVertex.mis = s.mis;

        // set initial vertex as rcv
        s.rcv = l;
        s.rcv.mRayDirection = -s.rcv.mRayDirection;
        s.throughputAtRcv = s.throughput / cosLight;
        s.pdfAtRcv = directPdfA;
        s.rcvCos = cosLight;

        vertex = PathVertex(TraceRay(ray));
        if (vertex.IsFinite()) {
            const float cosIn = vertex.GetCosN(ray.dir);
            if (cosIn < kMinCosTheta) {
                s.Terminate();
                return s;
            }

            s.mis.ProcessRay(vertex.mVertex, cosIn, finiteEmitter);

            s.rcvDist = vertex.mVertex.mRayDistance;
        } else
            s.Terminate();

        return s;
    }

    void UpdateReconnectionData<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const bool connectable, const float lastPdfW) {
        if (gDisableEarlyReconnection)
            return;

        float cosIn = vertex.GetCosN(vertex.GetDirIn());

        // set vertex as rcv if one hasn't been found yet.
        if (!s.rcvFound && connectable) {
            s.rcv = vertex.Pack();
            if (!vertex.IsFinite()) s.rcv.mRayDistance = FLT_MAX;
            s.rcvCos = cosIn;
            s.rcvDist = s.rcv.mRayDistance;
            s.prefixBounces = s.bounces - 1;
            s.rcvDiffuseBounces = s.diffuseBounces - 1;
            s.prefixPdf = lastPdfW;
            // throughputAtRcv and rcvDirOut are assigned next iteration after a direction is sampled
        }

        // update rcvDirOut and throughputAtRcv if this vertex was selected as the rcv last iteration
        if (s.rcvFound && s.bounces == s.prefixBounces + 2) {
            if (s.lastVertex.connectable) {
                s.rcvDirOut = vertex.mVertex.mRayDirection;
                s.throughputAtRcv = s.throughput;
                s.pdfAtRcv = s.pdfW;
            } else {
                s.prefixPdf = 0;
                s.prefixBounces = 0;
            }
        }

        if (s.rcvFound && s.bounces > s.prefixBounces) {
            const float cosOut = s.lastVertex.HasNormal() ? abs(dot(s.lastVertex.normal, vertex.GetDirIn())) : 1;
            s.rcvMis.Update( s.mis,
                s.lastVertex.scatterFwdPdfW, s.lastVertex.scatterRevPdfW,
                cosIn, cosOut, vertex.mVertex.mRayDistance,
                s.bounces - (s.prefixBounces + 1) );
        }
    }

    // Sample the BSDF and trace a ray to find the next vertex.
    // Updates throughput, pdfW, bounces, MIS quantities, and reconnection data
    // returns sampled LobeType
    uint AdvanceVertex<let bShift : bool, let bSetNext : bool>(inout PathState<bShift> s, inout PathVertex vertex, const PathVertex nextVertex) {
        // russian roulette
        const float contProb = vertex.GetContinuationProbability();
        const float rrU = sampleNext1D(s.sg);
        if (!bShift && contProb < 1 && rrU > contProb) {
            s.Terminate();
            return 0;
        }

        // sample direction
        DirectionSampleData r = vertex.SampleDirection(s.sgBsdf, s.isLightPath);

        if (bSetNext) {
            r.mDirection = nextVertex.mVertex.mRayDirection;
            r.SetReflectance(vertex.EvaluateReflectance(s.sg, r.mDirection, s.isLightPath));
            r.mLobeType = vertex.mMaterial.getLobeTypes(vertex.mShadingData) & ~uint(LobeType::Delta);
        }

        if (gBidirectional && gDisableCameraConnection && !s.isLightPath && s.bounces == 1)
            r.mRevPdfW = 0;

        // cache this vertex
        s.lastVertex.position            = vertex.GetPosition();
        s.lastVertex.normal              = vertex.HasFaceNormal() ? vertex.GetFaceNormal() : 0;
        s.lastVertex.upperHemisphereOnly = vertex.UpperHemisphereOnly();
        s.lastVertex.scatterFwdPdfW      = r.IsLobe(LobeType::Delta) ? 0 : r.mFwdPdfW;
        s.lastVertex.scatterRevPdfW      = r.mRevPdfW;
        s.lastVertex.mis = s.mis;
        const float lastPdfW = s.pdfW;

        // modify throughput and pdfW
        s.throughput *= r.mReflectance;
        if (r.mFwdPdfW > 0)
            s.pdfW *= r.mFwdPdfW;

        const float cosOut = vertex.GetCosN(r.mDirection);

        if (cosOut < kMinCosTheta || !any(s.throughput > 0) || any(s.throughput != s.throughput) || !(s.pdfW > 0)) {
            s.Terminate();
            return 0;
        }

        const bool connectable = r.IsLobe(LobeType::Diffuse) || (r.IsLobe(LobeType::Specular) && vertex.mMaterial.getProperties(vertex.mShadingData).roughness >= mParams.mReconnectionRoughness);
        s.lastVertex.connectable = connectable;

        // diffuse bounce limit
        if (connectable && s.diffuseBounces == mParams.mMaxDiffuseBounces) {
            s.Terminate();
            return 0;
        }

        // update mis quantities
        s.mis.ProcessDirection(r, cosOut, connectable);

        // trace ray to find next vertex
        if (bSetNext) {
            vertex = nextVertex;
        } else {
            vertex = PathVertex(TraceRay(Ray(vertex.ComputeRayOrigin(r.mDirection), r.mDirection)));
        }
        s.bounces++;
        if (connectable)
            s.diffuseBounces++;

        const float cosIn = vertex.GetCosN(vertex.GetDirIn());

        if (cosIn < kMinCosTheta) {
            s.Terminate();
            return 0;
        }

        s.mis.ProcessRay(vertex.mVertex, cosIn);

        if (!s.isLightPath) {
            const bool canReconnect = connectable && vertex.mVertex.mRayDistance >= mParams.mReconnectionDistance && (vertex.IsConnectable() || any(vertex.GetEmission() > 0));
            UpdateReconnectionData(s, vertex, canReconnect, lastPdfW);
        }

        return r.mLobeType;
    }

    uint SampleNextVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex) {
        PathVertex tmp;
        return AdvanceVertex<bShift, false>(s, vertex, tmp);
    }

    // Does everything SampleNextVertex does, but discards the sampled direction and instead sets the next vertex explicitly
    void SetNextVertex<let bShift : bool>( inout PathState<bShift> s, inout PathVertex vertex, const PathVertex nextVertex ) {
        AdvanceVertex<bShift, true>(s, vertex, nextVertex);
    }
}

// ---------------------------------------------------------------------------------------------------------
// ----------------------------------------------- Top-level -----------------------------------------------
// ---------------------------------------------------------------------------------------------------------

extension PathGenerator {
    // Sample a light subpath. Updates the LVC/photon map. Connects connectable vertices to the camera directly.
    void SampleLightPath(const PathId id) {
        if (mParams.mMaxBounces == 0 || mParams.mMaxDiffuseBounces == 0)
            return;

        if (gDisableLVC) mLightVertexCache.lightVertexCount[id.mId.y * mParams.mOutputDim.x + id.mId.x] = 0;

        PathVertex vertex;
        PathState<false> s = InitializeLightPath<false>(id, vertex);

        while (any(s.throughput > 0) && s.pdfW > 0 && vertex.IsFinite() && s.bounces + 1 <= mParams.mMaxBounces && s.diffuseBounces + 1 <= mParams.mMaxDiffuseBounces) {
            const bool connectable = vertex.IsConnectable();
            if (connectable) {
                // store vertex for connection and merging
                if (!gLightTraceOnly) {
                    const LightSubpathVertex v = PackLightSubpathVertex(s, vertex);

                    if (gVertexMerging || s.bounces + 2 <= mParams.mMaxBounces) {
                        uint i;
                        if (gDisableLVC) {
                            i = id.mId.y * mParams.mOutputDim.x + id.mId.x;

                            const uint b = mLightVertexCache.lightVertexCount[i];
                            mLightVertexCache.lightVertexCount[i] = b + 1;

                            i += b * mParams.mOutputDim.x * mParams.mOutputDim.y;
                            mLightVertexCache[i] = v;
                        } else {
                            i = mLightVertexCache.Insert(v);
                        }

                        if (gVertexMerging) {
                            const float3 pos = vertex.GetPosition();
                            mPhotonMap.addPhoton(i, pos, GetMergeRadius(pos, true), mParams.mPhotonCellCount, sampleNext1D(s.sg));
                        }
                    }
                }

                // form a full path by connecting the vertex to the camera
                if (!gProgressivePhotonMapping && !gDisableCameraConnection) {
                    float weight = 0;
                    int2 pixelCoord;
                    const PathSample ps = ConnectToCamera(s, vertex, pixelCoord, weight);
                    if (ps.IsValid() && weight > 0 && all(pixelCoord >= 0) && all(pixelCoord < mParams.mOutputDim)) {
                        DebugSetPixel(pixelCoord);
                        if (!gUseResampling) {
                            print("Found path. contrib:", ps.mContribution);
                            print("\tLight subpath vertices:", ps.mSuffix.mLightSubpath.mVertexCount);
                            print("\tMerge:", ps.HasFlag(PathFlags::eVertexMerging));
                            print("\tPrefix bounces:", ps.mPrefixBounces);
                        }
                        DebugSetPixel(-1);

                        const uint flatId = pixelCoord.y * mParams.mOutputDim.x + pixelCoord.x;
                        if (gUseResampling) {
                            // store path reservoirs for each pixel in a hash map
                            // these reservoirs are merged into each pixel's reservoir in a later pass
                            LightTraceReservoir r = {};
                            r.path = ps;
                            r.W = weight;
                            mLightTraceReservoirs.Insert(flatId, r);
                        } else {
                            // splat radiance onto output directly, instead of using reservoirs
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3)                  , ps.mContribution.r * (weight * ps.mMisWeight));
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)  , ps.mContribution.g * (weight * ps.mMisWeight));
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)*2, ps.mContribution.b * (weight * ps.mMisWeight));
                        }
                    }
                }
            }

            if (s.bounces + 1 == mParams.mMaxBounces)
                break;

            // for light paths: rcv is always the 2nd to last vertex
            s.rcv = vertex.Pack();
            s.rcvDirOut = vertex.GetDirIn();
            s.throughputAtRcv = s.throughput;
            s.pdfAtRcv = s.pdfW;
            s.prefixPdf = s.pdfW;
            s.prefixBounces = 0;
            s.rcvDiffuseBounces = 0;

            SampleNextVertex(s, vertex);

            s.rcvCos = abs(dot(s.lastVertex.normal, vertex.mVertex.mRayDirection));
            s.rcvDist = vertex.mVertex.mRayDistance;
            s.deltaRcv = !s.lastVertex.connectable;
        }
    }

    // Samples a canonical camera path by connecting to lights, light subpath vertices, and merging with photons.
    PathReservoir SampleCanonicalPaths(const PathId id) {
        PathVertex vertex = PathVertex(LoadPrimaryVertex(id.mId));

        // directly output primary radiance
        mOutputRadiance[id.mId] = float4(vertex.GetEmission(), 1);

        if (gLightTraceOnly || !vertex.IsFinite())
            return {};

        PathReservoir r = {};
        r.mM = mParams.mCanonicalSpp;

        for (uint i = 0; i < mParams.mCanonicalSpp; i++) {
            if (i > 0) vertex = PathVertex(LoadPrimaryVertex(id.mId));

            PathState<false> s = InitializeCameraPath<false>(PathId(id.mSeed + i, id.mId), vertex);

            while (s.bounces < mParams.mMaxBounces && any(s.throughput > 0) && s.pdfW > 0 && vertex.IsFinite()) {
                if (vertex.IsConnectable() && s.diffuseBounces < mParams.mMaxDiffuseBounces) {
                    if ((gBidirectional || gSampleDirectLight) && !gProgressivePhotonMapping) {
                        // connect to a random light
                        float weight = 0;
                        const PathSample ps = ConnectToLight(s, vertex, weight);
                        ProcessSample(r, s.sg, ps, weight);
                    }

                    if (gBidirectional) {
                        if (gBidirectional && !gProgressivePhotonMapping) {
                            if (gDisableLVC) {
                                uint i = id.mId.y * mParams.mOutputDim.x + id.mId.x;
                                const uint count = mLightVertexCache.lightVertexCount[i];
                                //*
                                // pick a random light vertex and weight it by 'count'
                                uint b = min(uint(sampleNext1D(s.sg) * count), count - 1);
                                /*/
                                // connect to light subpath vertices
                                for (uint b = 0; b < count; b++)
                                //*/
                                {
                                    const LightSubpathVertex v = mLightVertexCache[i + b*mParams.mOutputDim.x * mParams.mOutputDim.y];
                                    float weight = 0;
                                    PathSample ps = ConnectToLightSubpath(s, vertex, v, weight);
                                    ProcessSample(r, s.sg, ps, weight * count);
                                }
                            } else {
                                // connect to a random light subpath vertex in the LVC
                                const uint count = mLightVertexCache.size();
                                if (count > 0) {
                                    const LightSubpathVertex v = mLightVertexCache[min(uint(sampleNext1D(s.sg) * count), count - 1)];

                                    float weight = 0;
                                    PathSample ps = ConnectToLightSubpath(s, vertex, v, weight);
                                    ProcessSample(r, s.sg, ps, weight * (count / (float)mParams.mLightSubpathCount));
                                }
                            }
                        }

                        if (gVertexMerging) {
                            // merge with the photon in the selected cell
                            float cellW = 0.f;
                            const LightSubpathVertex v = FindPhoton(s.sg, vertex, cellW);
                            if (cellW > 0) {
                                float weight = 0;
                                const PathSample ps = MergeWithLightVertex(s, vertex, v, weight);
                                ProcessSample(r, s.sg, ps, weight * cellW);
                            }

                            if (gProgressivePhotonMapping)
                                break;
                        }
                    }
                }

                // sample direction and trace ray
                SampleNextVertex(s, vertex);

                // add emission from new vertex
                {
                    float weight = 0;
                    const PathSample ps = EvalLightContribution(s, vertex, weight);
                    if (ps.IsValid())
                        ProcessSample(r, s.sg, ps, weight);
                }
            }
        }

        r.mW /= mParams.mCanonicalSpp;

        if (!gUseResampling) {
            mOutputRadiance[id.mId].rgb += r.mSample.mContribution / mParams.mCanonicalSpp;
        }

        r.FinalizeMerge();
        return r;
    }
}
