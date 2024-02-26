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
    HashMap<PathReservoir> mLightTraceReservoirs;
    HashMap<PathReservoir> mCausticReservoirMap;

    Texture2D<PackedHitInfo> mVbuffer;
    Texture2D<float3> mViewDir;
    RWTexture2D<float2> mMotionVectors;
    RWByteAddressBuffer mCausticMotionVectorMutex;

    RWStructuredBuffer<PathReservoir> mLastReservoirs;
    RWStructuredBuffer<PathReservoir> mLastCausticReservoirs;
    Texture2D<float3> mLastViewDir;
    Texture2D<PackedHitInfo> mLastVbuffer;

    // Static parameters

    static const bool  gUseEnvLight                 = USE_ENV_LIGHT;
    static const bool  gUseEmissiveLights           = USE_EMISSIVE_LIGHTS;
    static const bool  gUseViewDir                  = USE_VIEW_DIR;
    static const bool  gSampleDirectLight           = USE_NEE;
    static const bool  gBidirectional               = USE_BIDIRECTIONAL;
    static const bool  gVertexMerging               = USE_VERTEX_MERGING;
    static const bool  gDynamicMergeRadius          = DYNAMIC_MERGE_RADIUS;
    static const bool  gProgressivePhotonMapping    = USE_PPM_ONLY;
    static const bool  gAllowEarlyReconnection      = false;
    static const bool  gLightTraceOnly              = LIGHT_TRACE_ONLY;
    static const bool  gUseResampling               = USE_RESAMPLING;
    static const bool  gUseCausticReservoirs        = CAUSTIC_RESERVOIRS;
    static const bool  gUseCausticMotionVectors     = CAUSTIC_MOTION_VECTORS;
    static const bool  gValidateSuffixes            = VALIDATE_SUFFIXES;
    static const bool  gUseBsdfImportanceSampling   = USE_BSDF_IMPORTANCE_SAMPLING;
    static const bool  gWavefrontTechniqueSelection = WAVEFRONT_TECHNIQUE_SELECTION;
    static const bool  gDebugBPT                    = DEBUG_BPT;
    static const bool  gDebugHeatmap                = DEBUG_HEATMAP;
    static const float gMISPowerExponent            = MIS_POWER_EXPONENT;
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
    float mFwdPdfW = 0;
    float mRevPdfW = 0;
    bool mDeltaLobe = false;

    [mutating]
    void SetReflectance(const ReflectanceEvalData r) {
        mReflectance = r.mReflectance;
        mFwdPdfW = r.mFwdPdfW;
        mRevPdfW = r.mRevPdfW;
    }
};

struct PathVertex {
    PackedPathVertex mVertex;
    ShadingData mShadingData;
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

    float3 GetEmission() {
        if (IsFinite()) {
            if (dot(mShadingData.faceN, mVertex.mRayDirection) > 0) {
                return 0; // triangle lights are one-sided
            } else {
                return mMaterial.getProperties(mShadingData).emission;
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
        r.mDeltaLobe = s.isLobe(LobeType::Delta);
        if (adjoint) r.mReflectance *= GetAdjointCorrection(r.mDirection, s.isLobe(LobeType::DeltaTransmission) || s.isLobe(LobeType::SpecularTransmission));

        // cancel divide by pdf performed by material sampling.
        // TODO: modify the material sampling code to return the unweighted BSDF
        if (s.pdf > 0) r.mReflectance *= s.pdf;
        //r.mReflectance = EvaluateReflectance(sg, r.mDirection, adjoint).mReflectance;

        r.mFwdPdfW = s.pdf;

        if (r.mDeltaLobe) {
            // Reverse pdf cannot be evaluated, use forward pdf instead, as they should be the same.
            r.mRevPdfW = r.mFwdPdfW;
        } else {
            const PathVertex revVertex = Reversed(r.mDirection);
            r.mRevPdfW = revVertex.mMaterial.evalPdf(revVertex.mShadingData, GetDirIn(), gPathGenerator.gUseBsdfImportanceSampling);
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
                        directPdfA = 1.f / gScene.lightCollection.getTriangle(triangleIndex).area;
                        directPdfA *= mEmissiveSampler.evalTriangleSelectionPdf(prevPos, prevNormal, upperOnly, triangleIndex);

                        if (gUseEnvLight) directPdfA *= 1 - mParams.mEnvMapProb;

                        emissionPdfW = directPdfA * CosHemispherePdfW(cosIn);
                        directPdfW   = directPdfA * sqr(vertex.mVertex.mRayDistance) / cosIn;
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
/////////////////////////////// Path tracing implementation ///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

extension PathGenerator {
    struct PathState<let bShift : bool> {
        float3 throughput = 0;
        float pdfW = 0;

        SampleGenerator sg;     // sample generator for everything but BSDF sampling
        SampleGenerator sgBsdf; // sample generator for BSDF sampling

        PackedPathVertex rcv = {};
        float3 throughputAtRcv = 0;
        float pdfAtRcv = 0;
        float3 rcvDirOut;
        float prefixPdf = 0;

        float rcvCos = 0;
        float rcvDist = 0;

        float dVCM = 0;
        float dVC = 0;
        float dVM = 0;

        uint packedBounces = 0;
        uint packedDiffuseBounces = 0;

        PathId id;
        PathId lightSubpathId;
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
            s.diffuseBounces
        );
        v.mVertex = vertex.Pack();
        v.mThroughput = s.throughput;
        v.mSubpathPdf = s.pdfW;
        v.dVCM = s.dVCM;
        v.dVC  = s.dVC;
        v.dVM  = s.dVM;
        return v;
    }

    // Merges a canonical sample into the pixel's reservoir
    void ProcessSample(inout PathReservoir r, inout SampleGenerator sg, const PathSample ps, const float weight) {
        if (gUseResampling) {
            r.Merge(sampleNext1D(sg), ps, weight * ps.GetTargetPdf(), 0);
        } else if (ps.IsValid() && weight > 0) {
            print("Found path. contrib:", ps.mContribution);
            print("\tLight subpath vertices:", ps.mSuffix.mLightSubpath.mVertexCount);
            print("\tMerge:", ps.HasFlag(PathFlags::eVertexMerging));
            print("\tPrefix bounces:", ps.mPrefixBounces);
            r.mSample.mContribution += ps.mContribution * weight;
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------- Connection stategies ------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathSample ConnectToLight       <let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const bool upperOnly, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;

        PathSample result = {};

        if (DebugDisablePath(s.bounces + 1, 1))
            return result;

        // Sample emissive path vertex

        const PathId lightPathId = bShift ? s.lightSubpathId : PathId(s.sg.next(), uint2(s.sg.next(), s.sg.next()));
        SampleGenerator sg = SampleGenerator(lightPathId.mId, lightPathId.mSeed);

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
                if (!mEnvMapSampler.sample(uv, envSample))
                    return result;

                if (gUseEmissiveLights) envSample.pdf *= mParams.mEnvMapProb;

                le = envSample.Le;
                if (!any(le > 0))
                    return result;

                l = PackedPathVertex(Ray(vertex.ComputeRayOrigin(envSample.dir), envSample.dir), HitInfo(0), FLT_MAX);

                directPdfA = envSample.pdf;
                directPdfW = envSample.pdf;
                emissionPdfW = directPdfW * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && !(gUseEnvLight && useEnvMap)) {
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

                if (triangleIndex == LightCollection::kInvalidIndex)
                    return result;

                if (gUseEnvLight) selectionPdf *= 1 - mParams.mEnvMapProb;

                let tri = gScene.lightCollection.getTriangle(triangleIndex);

                le = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(barycentrics));
                if (!any(le > 0))
                    return result;

                directPdfA = selectionPdf / tri.area;

                const float3 triPos = computeRayOrigin(tri.getPosition(barycentrics), tri.normal);
                const float3 rayOrigin = vertex.ComputeRayOrigin(triPos - vertex.GetPosition());
                float3 toLight = triPos - rayOrigin;
                const float dist = length(toLight);
                toLight /= dist;

                cosLight = -dot(tri.normal, toLight);
                if (cosLight < kMinCosTheta)
                    return result;

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

        let f = vertex.EvaluateReflectance(s.sg, l.mRayDirection, false);

        const float3 contrib = s.throughput * f.mReflectance * le;
        const float samplePdf = s.pdfW * directPdfW;
        if (!any(contrib > 0) || !(samplePdf > 0))
            return result;

        if (Occluded(Ray(l.mRayOrigin, l.mRayDirection, 0, l.mRayDistance)))
            return result;

        // divide by pdf
        integrationWeight = 1.f / samplePdf;

        // compute mis weight
        if (DebugDisableMIS()) {
            misWeight = 1;
        } else {
            if (gBidirectional) {
                const float wLight = Mis(f.mFwdPdfW / directPdfW);
                const float wCamera = Mis(emissionPdfW * abs(cosOut) / (directPdfW * cosLight)) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
                misWeight = 1 / (wLight + 1 + wCamera);
            } else {
                misWeight = Mis(directPdfW, f.mFwdPdfW);
            }
        }

        float3 suffixContrib;
        float3 suffixDir;
        float suffixPdf;
        const bool hasRcv = gAllowEarlyReconnection && s.rcvFound;
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
            0
        );

        const PathSuffix suffix = PathSuffix::Create(
            suffixContrib,
            suffixPdf,
            hasRcv ? s.rcv : l,
            suffixDir,
            hasRcv ? s.rcvDist : l.mRayDistance,
            hasRcv ? s.rcvCos : cosLight,
            lightSubpath
        );

        result = PathSample::Create(
            contrib,
            s.bounces + 1,
            hasRcv ? s.prefixPdf         : s.pdfW,
            hasRcv ? s.prefixBounces     : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eNone);

        return result;
    }
    PathSample ConnectToLightSubpath<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, const float subpathWeight, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;

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

        let f = vertex.EvaluateReflectance(s.sg, toLight, false);
        if (!any(f.mReflectance > 0)) {
            if (bShift) print("Subpath reconnection failed: !any(f.mReflectance > 0)", f.mReflectance);
            return result;
        }

        const float3 contrib = (s.throughput * v.mThroughput) * (f.mReflectance * r.mReflectance) / cosLight;

        if (!any(contrib > 0) || any(contrib < 0)) {
            if (bShift) print("Subpath reconnection failed: any(contrib > 0) || any(contrib < 0)", contrib);
            return result;
        }

        if (Occluded(visibilityRay)) {
            if (bShift) print("Subpath reconnection failed: Occluded", 0);
            return result;
        }

        const float invDist2 = 1 / sqr(dist);

        integrationWeight = subpathWeight * (cosLight * invDist2) / (s.pdfW * v.mSubpathPdf);

        // mis weight
        if (DebugDisableMIS()) {
            misWeight = 1;
        } else {
            const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
            const float lightBsdfDirPdfA  = r.mFwdPdfW * cosCam   * invDist2;
            const float wLight = Mis(cameraBsdfDirPdfA) * (GetMisVmWeightFactor() + v.dVCM + v.dVC * Mis(r.mRevPdfW));
            const float wCamera = Mis(lightBsdfDirPdfA) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
            misWeight = 1 / (wLight + 1.f + wCamera);
        }

        float3 suffixContrib;
        float3 suffixDir;
        float suffixPdf = v.mSubpathPdf;
        const bool hasRcv = gAllowEarlyReconnection && s.rcvFound;
        if (hasRcv) {
            // use rcv found earlier along the camera subpath
            if (s.bounces == s.prefixBounces + 1) {
                // the light vertex is the 2nd vertex in the suffix
                suffixContrib = v.mThroughput * (r.mReflectance / cosLight);
                suffixDir = toLight;
                suffixPdf *= sqr(dist) / cosLight;
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
            hasRcv ? s.rcvDist : dist,
            hasRcv ? s.rcvCos  : cosLight,
            v.mLightSubpath
        );

        result = PathSample::Create(
            contrib,
            bounces,
            hasRcv ? s.prefixPdf         : s.pdfW,
            hasRcv ? s.prefixBounces     : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eNone);

        return result;
    }
    PathSample MergeWithLightVertex <let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, const float subpathWeight, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;

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

        const float3 contrib = s.throughput * v.mThroughput * f.mReflectance;
        if (all(contrib <= 0))
            return result;

        integrationWeight = subpathWeight / (s.pdfW * v.mSubpathPdf * (sqr(mergeRadius) * M_PI * mParams.mLightSubpathCount));

        const float wLight  = v.dVCM * GetMisVcWeightFactor() + v.dVM * Mis(f.mFwdPdfW);
        const float wCamera = s.dVCM * GetMisVcWeightFactor() + s.dVM * Mis(f.mRevPdfW);
        misWeight = 1;
        if (DebugDisableMIS() || gProgressivePhotonMapping || bShift)
            misWeight = 1;
        else
            misWeight = 1.f / (wLight + 1.f + wCamera);

        float3 suffixContrib;
        float3 suffixDir;
        float suffixPdf = v.mSubpathPdf;
        const bool hasRcv = gAllowEarlyReconnection && !bShift && s.rcvFound;
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
            v.mLightSubpath
        );

        result = PathSample::Create(
            contrib,
            bounces,
            hasRcv ? s.prefixPdf         : s.pdfW,
            hasRcv ? s.prefixBounces     : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.id,
            suffix,
            PathFlags::eVertexMerging);

        return result;
    }
    PathSample ConnectToCamera      <let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out int2 pixelCoord, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;
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
        if (any(pixelCoord < 0) || any(pixelCoord >= mParams.mOutputDim) || cosAtCamera <= 0) {
            if (bShift) print("Caustic shift failed: offscreen", pixelCoord);
            return result;
        }

        const float cosOut = vertex.GetCosN(toCamera);
        if (cosOut < kMinCosTheta) {
            if (bShift) print("Caustic shift failed: cosOut < kMinCosTheta", cosOut);
            return result;
        }

        let f = vertex.EvaluateReflectance(s.sg, toCamera, true);

        // Compute pdf conversion factor from image plane area to surface area
        const float cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);

        const float cosIn = vertex.GetCosN(vertex.GetDirIn());

        const float3 bsdf = vertex.Reversed(toCamera).EvaluateReflectance(s.sg, vertex.GetDirIn(), false).mReflectance / cosIn;
        float3 contrib = s.throughput * bsdf;
        if (!any(contrib > 0)) {
            if (bShift) print("Caustic shift failed: zero contrib", 0);
            return result;
        }

        const float samplePdf = s.pdfW * (1 / cameraPdfA) * mParams.mLightSubpathCount;
        if (!(samplePdf > 0)) {
            if (bShift) print("Caustic shift failed: Invalid samplePdf", samplePdf);
            return result;
        }
        if (Occluded(Ray(rayOrigin, toCamera, 0, dist))) {
            if (bShift) print("Caustic shift failed: Occluded", samplePdf);
            return result;
        }

        integrationWeight = 1.f / samplePdf;

        const float wLight = Mis(cameraPdfA / mParams.mLightSubpathCount) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
        misWeight = (gLightTraceOnly || DebugDisableMIS()) ? 1.f : (1.f / (wLight + 1.f));

        PathSuffix suffix = {};
        if (s.deltaRcv || bShift) {
            // if the rcv is unconnectable, this is a caustic light path and we store the primary vertex as the rcv instead.

            suffix = PathSuffix::Create(
                s.throughput,
                s.pdfW,
                vertex.Pack(),
                vertex.GetDirIn(),
                dist,
                cosIn,
                LightSubpathData::Create(
                    s.lightSubpathVertex,
                    s.id,
                    s.bounces + 2,
                    s.diffuseBounces + 1
                )
            );
        } else {
            suffix = PathSuffix::Create(
                s.throughputAtRcv,
                s.prefixPdf,
                s.rcv,
                s.rcvDirOut,
                s.rcvDist,
                s.rcvCos,
                LightSubpathData::Create(
                    s.lightSubpathVertex,
                    s.id,
                    s.bounces + 1,
                    s.diffuseBounces + 1
                )
            );
        }

        result = PathSample::Create(
            contrib,
            s.bounces + 1,
            1,
            0,
            0,
            s.id,
            suffix,
            s.deltaRcv ? PathFlags::eCausticLightPath : PathFlags::eNone);

        return result;
    }

    PathSample ConnectToSuffix(inout PathState<true> s, inout PathVertex vertex, const PathSample basePath, const LightSubpathVertex lightSubpathVertex) {
        PathSample result = {};

        const bool lightSubpathOnly = gBidirectional && basePath.mSuffix.mLightSubpath.mVertexCount > 1 && !basePath.HasFlag(PathFlags::eCausticLightPath) &&
            basePath.mBounces == (basePath.mPrefixBounces + basePath.mSuffix.mLightSubpath.mVertexCount) - (gVertexMerging && basePath.HasFlag(PathFlags::eVertexMerging) ? 1 : 0);
        const bool mergeSuffix = lightSubpathOnly && !gLightTraceOnly && basePath.HasFlag(PathFlags::eVertexMerging);

        PathSuffix newSuffix = basePath.mSuffix;

        // if the suffix is just the light subpath, we can connect to it directly to get the new suffix
        if (gValidateSuffixes && lightSubpathOnly) {
            float w, m;
            if (mergeSuffix)
                newSuffix =  MergeWithLightVertex(s, vertex, lightSubpathVertex, 1, w, m).mSuffix;
            else
                newSuffix = ConnectToLightSubpath(s, vertex, lightSubpathVertex, 1, w, m).mSuffix;
        }

        PathVertex suffixVertex = PathVertex(newSuffix.mVertex);

        // For caustic light paths, reconnection is always impossible, but we simply copy the path if it's close enough to the base pixel.
        if (!gUseCausticReservoirs && !gValidateSuffixes && basePath.HasFlag(PathFlags::eCausticLightPath)) {
            const int2 offsetPixel = GetPixelCoord(vertex.GetPosition());
            const int2 basePixel   = GetPixelCoord(suffixVertex.GetPosition());
            if (mParams.mCausticReuseRadius > 0) {
                const float dist = length(offsetPixel - basePixel);
                if (dist > mParams.mCausticReuseRadius) {
                    print("Shift failed: Caustic light path", dist);
                    return result;
                }
            } else {
                if (any(offsetPixel != basePixel)) {
                    print("Shift failed: Caustic light path", offsetPixel != basePixel);
                    return result;
                }
            }
            return basePath;
        }

        // update suffix vertex with new ray direction and compute ray between prefix and suffix
        Ray visibilityRay;
        if (mergeSuffix) {
            // Apply merge radius
            const float3 toPhoton = suffixVertex.GetPosition() - vertex.GetPosition();
            const float dist2 = dot(toPhoton, toPhoton);
            if (dist2 >= sqr(GetMergeRadius(vertex.GetPosition()))) {
                print("Shift failed: merge radius", dist2);
                return result;
            }

            visibilityRay.dir = newSuffix.mDirOut;
        } else {
            if (newSuffix.mVertex.IsFinite()) {
                // update ray direction
                newSuffix.mVertex.mRayDirection = suffixVertex.GetPosition() - vertex.GetPosition();
                // recompute origin and distance
                newSuffix.mVertex.mRayOrigin = vertex.ComputeRayOrigin(newSuffix.mVertex.mRayDirection);
                newSuffix.mVertex.mRayDirection = suffixVertex.ComputeRayOrigin(-newSuffix.mVertex.mRayDirection) - newSuffix.mVertex.mRayOrigin;
                newSuffix.mVertex.mRayDistance = length(newSuffix.mVertex.mRayDirection);
                newSuffix.mVertex.mRayDirection /= newSuffix.mVertex.mRayDistance;

                newSuffix.mCos  = suffixVertex.GetCosN(newSuffix.mVertex.mRayDirection);
                newSuffix.mDist = newSuffix.mVertex.mRayDistance;

                if (newSuffix.mDist < mParams.mReconnectionDistance) {
                    print("Shift failed: reconnection distance", newSuffix.mDist);
                    return result;
                }
                if (abs(newSuffix.mCos) < kMinCosTheta) {
                    print("Shift failed: reconnection cos", newSuffix.mCos);
                    return result;
                }

                visibilityRay = Ray(newSuffix.mVertex.mRayOrigin, newSuffix.mVertex.mRayDirection, 0, newSuffix.mVertex.mRayDistance);
            } else {
                newSuffix.mVertex.mRayOrigin = vertex.ComputeRayOrigin(newSuffix.mVertex.mRayDirection);

                visibilityRay = Ray(newSuffix.mVertex.mRayOrigin, newSuffix.mVertex.mRayDirection);
            }

            suffixVertex = PathVertex(newSuffix.mVertex);
        }

        s.prefixPdf         = s.pdfW;
        s.prefixBounces     = s.bounces;
        s.rcvDiffuseBounces = s.diffuseBounces;
        s.rcv       = newSuffix.mVertex;
        s.rcvCos    = newSuffix.mCos;
        s.rcvDist   = newSuffix.mDist;
        s.rcvDirOut = newSuffix.mDirOut;

        // evaluate reflectance at last prefix vertex
        {
            let r = vertex.EvaluateReflectance(s.sg, visibilityRay.dir, false);
            s.throughput *= r.mReflectance;
            if (!any(s.throughput > 0)) {
                print("Shift failed: Zero prefix vertex reflectance", s.throughput);
                return result;
            }

            if (mergeSuffix) {
                // for photons, divide out cosOut as it's in the light subpath PDF
                if (vertex.HasFaceNormal()) {
                    const float3 guideNormal = vertex.mMaterial.getProperties(vertex.mShadingData).guideNormal;
                    const float cosOut = abs(dot(guideNormal, newSuffix.mDirOut));
                    if (cosOut < kMinCosTheta) {
                        print("Shift failed: prefix cosOut < kMinCosTheta", cosOut);
                        return result;
                    }
                    s.throughput /= cosOut;
                }
            } else {
                if (gValidateSuffixes) {
                    s.pdfW *= r.mFwdPdfW;
                    if (!(s.pdfW > 0)) {
                        print("Shift failed: Zero prefix vertex pdf", s.pdfW);
                    }
                    // sample direction to advance sample generator
                    vertex.SampleDirection(s.sgBsdf, false);

                    // TODO: update MIS quantities
                }
            }
        }

        float3 prefixContrib = s.throughput;

        if (!mergeSuffix) {
            // evaluate contribution from the rest of the camera subpath
            if (Occluded(visibilityRay)) {
                print("Shift failed: suffix occluded", 0);
                return result;
            }
            s.bounces++;
            s.diffuseBounces++;

            if (basePath.mPrefixBounces + 1 == basePath.mBounces) {
                // suffix is just a single vertex
                if (gValidateSuffixes) {
                    newSuffix.mContribution = suffixVertex.GetEmission();
                    if (!any(newSuffix.mContribution > 0)) {
                        print("Shift failed: Zero emission", newSuffix.mContribution);
                        return result;
                    }
                } else if (suffixVertex.HasFaceNormal() && dot(suffixVertex.mShadingData.faceN, -newSuffix.mVertex.mRayDirection) < kMinCosTheta) {
                    // surface lights are one-sided
                    print("Shift failed: backfacing light", newSuffix.mCos);
                    return result;
                }
            } else {
                // connect to suffix
                if (!suffixVertex.IsConnectable()) {
                    print("Shift failed: suffix unconnectable", 0);
                    return result;
                }

                // Evaluate reflectance at suffix vertex
                let r = suffixVertex.EvaluateReflectance(s.sg, newSuffix.mDirOut, false);
                s.throughput *= r.mReflectance;
                if (!any(s.throughput > 0)) {
                    print("Shift failed: zero suffix reflectance", s.throughput);
                    return result;
                }

                prefixContrib = s.throughput;

                // Trace the rest of the camera subpath
                if (gValidateSuffixes && !lightSubpathOnly) {
                    s.throughputAtRcv = s.throughput;
                    s.pdfW *= r.mFwdPdfW;
                    s.pdfAtRcv = s.pdfW;

                    // sample direction to advance sample generator
                    suffixVertex.SampleDirection(s.sgBsdf, false);

                    // TODO: update MIS quantities

                    // Trace first ray in suffix
                    suffixVertex = PathVertex( TraceRay(Ray(suffixVertex.ComputeRayOrigin(newSuffix.mDirOut), newSuffix.mDirOut)) );
                    s.bounces++;
                    s.diffuseBounces++;

                    // trace rest of subpath
                    const uint cameraSubpathBounces = basePath.mBounces - newSuffix.mLightSubpath.mVertexCount;
                    while (s.bounces < cameraSubpathBounces && any(s.throughput > 0) && s.pdfW > 0 && suffixVertex.IsFinite()) {
                        float bsdfPdfW;
                        bool deltaLobe;
                        SampleNextVertex(s, suffixVertex, bsdfPdfW, deltaLobe);
                    }
                    if (s.bounces != cameraSubpathBounces) {
                        print("Suffix shift failed: Camera subpath length mismatch", uint2(s.bounces, cameraSubpathBounces));
                        return result;
                    }

                    // Apply connection strategy based on number of light subpath vertices

                    if (newSuffix.mLightSubpath.mVertexCount == 0) {
                        // Pure camera path
                        newSuffix.mContribution = SafeDivide(s.throughput, s.throughputAtRcv) * suffixVertex.GetEmission();
                        newSuffix.mPdf = s.pdfW / s.pdfAtRcv;
                    } else {
                        if (!suffixVertex.IsConnectable()) {
                            print("Suffix shift failed: Unconnectable suffix vertex", 0);
                            return result;
                        }

                        float w, m;
                        if (newSuffix.mLightSubpath.mVertexCount == 1) {
                            // NEE path, formed by connecting a camera subpath to a light directly
                            const bool upperOnly = suffixVertex.HasFaceNormal() && !suffixVertex.HasLobe(LobeType::NonDeltaTransmission);
                            result = ConnectToLight(s, suffixVertex, upperOnly, w, m);
                        } else {
                            // Bidirectional path, formed by connecting a camera subpath to a light subpath
                            if (basePath.HasFlag(PathFlags::eVertexMerging))
                                result = MergeWithLightVertex(s, vertex, lightSubpathVertex, 1, w, m);
                            else
                                result = ConnectToLightSubpath(s, vertex, lightSubpathVertex, 1, w, m);
                        }
                        return result;
                    }
                }
            }
        }

        result = basePath;
        result.mContribution = prefixContrib * newSuffix.mContribution;
        result.mPrefixPdf = s.prefixPdf;
        result.mSuffix = newSuffix;

        return result;
    }

    // ---------------------------------------------------------------------------------------------------------
    // -------------------------------------------- Path generation --------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathState<bShift> InitializeCameraPath<let bShift : bool>(const PathId id, const PathVertex vertex) {
        PathState<bShift> s = {};
        s.id = id;
        s.isLightPath = false;

        s.sg     = SampleGenerator(s.id.mId, s.id.mSeed);
        s.sgBsdf = SampleGenerator(s.id.mId, -s.id.mSeed);

        s.bounces = 0;
        s.diffuseBounces = 0;
        s.prefixBounces = 0;
        s.rcvDiffuseBounces = 0;
        s.throughput = 1;
        s.throughputAtRcv = 0;
        s.prefixPdf = 0;
        s.rcvDirOut = 0;
        s.pdfW = 1;
        if (gBidirectional) {
            float cosAtCamera;
            const float cameraPdfW = GetImageToSolidAngleFactor(-vertex.GetDirIn(), cosAtCamera);
            s.dVCM = Mis(mParams.mLightSubpathCount / cameraPdfW);
            s.dVC = 0;
            s.dVM = 0;

            const float cosIn = vertex.GetCosN(vertex.GetDirIn());
            const float3 d = gScene.camera.getPosition() - vertex.GetPosition();
            s.dVCM *= Mis(dot(d,d) / cosIn);
        }
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

        s.sg     = SampleGenerator(s.id.mId, s.id.mSeed);
        s.sgBsdf = SampleGenerator(s.id.mId, -s.id.mSeed);

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
        s.dVCM = Mis(directPdfA / emissionPdfW);
        s.dVC = finiteEmitter ? Mis(cosLight / emissionPdfW) : 0;
        s.dVM = s.dVC * GetMisVcWeightFactor();

        // set initial vertex as rcv
        s.rcv = l;
        s.rcv.mRayDirection = -s.rcv.mRayDirection;
        s.throughputAtRcv = s.throughput;
        s.rcvCos = cosLight;

        vertex = PathVertex(TraceRay(ray));
        if (vertex.IsFinite()) {
            const float cosIn = vertex.GetCosN(ray.dir);
            if (cosIn < kMinCosTheta) {
                s.Terminate();
                return s;
            }

            if (finiteEmitter)
                s.dVCM *= Mis(sqr(vertex.mVertex.mRayDistance));
            s.dVCM /= Mis(cosIn);
            s.dVC  /= Mis(cosIn);
            s.dVM  /= Mis(cosIn);

            s.rcvDist = vertex.mVertex.mRayDistance;
        } else
            s.Terminate();

        return s;
    }

    // Sample the BSDF and trace a ray to find the next vertex. Updates path throughput, pdfW, bounces, and MIS quantities
    void SampleNextVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex, out float bsdfPdfW, out bool deltaLobe) {
        bsdfPdfW = 0;
        deltaLobe = false;

        bool connectable = vertex.IsConnectable();

        // russian roulette
        const float contProb = vertex.GetContinuationProbability();
        const float rrU = sampleNext1D(s.sg);
        if (!bShift && contProb < 1 && rrU > contProb) {
            s.Terminate();
            return;
        }

        // sample direction

        //SampleGenerator tmp = SampleGenerator(s.id, s.sgBsdf.next());
        let r = vertex.SampleDirection(s.sgBsdf, s.isLightPath);
        const float cosOut = vertex.GetCosN(r.mDirection);
        bsdfPdfW  = r.mFwdPdfW;
        deltaLobe = r.mDeltaLobe;

        // modify throughput and pdfW
        s.throughput *= r.mReflectance;
        if (r.mFwdPdfW > 0)
            s.pdfW *= r.mFwdPdfW;
        else
            connectable = false;

        if (cosOut < kMinCosTheta || all(s.throughput <= 0) || any(s.throughput != s.throughput)) {
            s.Terminate();
            return;
        }

        // diffuse bounce limit
        if (connectable && s.diffuseBounces == mParams.mMaxDiffuseBounces) {
            s.Terminate();
            return;
        }

        // update mis quantities
        if (gBidirectional) {
            if (connectable && !r.mDeltaLobe) {
                // SmallVCM tech. rep. (34)-(36)
                const float invPdf = 1.f / r.mFwdPdfW;
                s.dVC = Mis(cosOut * invPdf) * (s.dVC * Mis(r.mRevPdfW) + s.dVCM + GetMisVmWeightFactor());
                s.dVM = Mis(cosOut * invPdf) * (s.dVM * Mis(r.mRevPdfW) + s.dVCM * GetMisVcWeightFactor() + 1);
                s.dVCM = Mis(invPdf);
            } else {
                // SmallVCM tech. rep. (53)-(55)
                s.dVC *= Mis(cosOut);
                s.dVM *= Mis(cosOut);
                s.dVCM = 0.f;
            }
        }

        // trace ray to find next vertex
        vertex = PathVertex(TraceRay(Ray(vertex.ComputeRayOrigin(r.mDirection), r.mDirection)));
        s.bounces++;
        if (connectable)
            s.diffuseBounces++;

        const float cosIn = vertex.GetCosN(vertex.GetDirIn());

        if (cosIn < kMinCosTheta) {
            s.Terminate();
            return;
        }

        // update mis quantities
        if (gBidirectional && vertex.IsFinite()) {
            s.dVCM *= Mis(sqr(vertex.mVertex.mRayDistance) / cosIn);
            s.dVC  /= Mis(cosIn);
            s.dVM  /= Mis(cosIn);
        }
    }

    PathState<true> ShiftLightSubpath(const PathSample basePath, out PathVertex vertex) {
        PathState<true> s = InitializeLightPath<true>(basePath.mSuffix.mLightSubpath.mId, vertex, basePath.mSuffix.mLightSubpath.mFirstVertex);

        while (any(s.throughput > 0) && s.pdfW > 0 && s.bounces + 2 < basePath.mSuffix.mLightSubpath.mVertexCount && vertex.IsFinite()) {
            const bool connectable = vertex.IsConnectable();

            // for light paths: rcv is always the 2nd to last vertex
            s.rcv = vertex.Pack();
            s.rcvDirOut = vertex.GetDirIn();
            s.throughputAtRcv = s.throughput;
            s.prefixPdf = s.pdfW;
            s.prefixBounces = 0;
            s.rcvDiffuseBounces = 0;

            const float3 prevN = vertex.GetFaceNormal();

            float bsdfPdfW;
            bool deltaLobe;
            SampleNextVertex(s, vertex, bsdfPdfW, deltaLobe);

            s.rcvCos = abs(dot(prevN, vertex.mVertex.mRayDirection));
            s.rcvDist = vertex.mVertex.mRayDistance;
            s.deltaRcv = !connectable || deltaLobe;
        }

        return s;
    }

    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- Top-level -----------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    // Samples a canonical camera by connecting to lights, light subpath vertices, and merging with photons.
    PathReservoir SampleCameraPath(const PathId id) {
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

            while (s.bounces + 1 <= mParams.mMaxBounces && any(s.throughput > 0) && vertex.IsFinite()) {
                const float3 prevThroughput = s.throughput;
                const float prevPdf = s.pdfW;
                const float3 prevPos = vertex.GetPosition();
                const float3 prevNormal = vertex.HasFaceNormal() ? vertex.GetFaceNormal() : 0;
                bool prevReconnectable = vertex.IsConnectable();
                const bool upperOnly = vertex.HasFaceNormal() && !vertex.HasLobe(LobeType::NonDeltaTransmission);

                if (prevReconnectable && s.diffuseBounces < mParams.mMaxDiffuseBounces) {
                    bool doDI = (gBidirectional || gSampleDirectLight) && !gProgressivePhotonMapping;
                    bool doVC = gBidirectional && !gProgressivePhotonMapping;
                    bool doVM = gBidirectional && gVertexMerging;
                    float techniqueW = 1;

                    // Select a random connection technique to evaluate
                    if (gBidirectional && gWavefrontTechniqueSelection && !gProgressivePhotonMapping) {
                        bool doDI = false;
                        bool doVC = false;
                        bool doVM = false;

                        float vcRnd = sampleNext1D(s.sg);

                        // Force all threads in a warp to select the same technique
                        vcRnd = WaveReadLaneFirst(vcRnd);

                        // pick between direct light, vertex connection, and vertex merging.
                        if (vcRnd < mParams.mDirectLightProb) {
                            doDI = true;
                            techniqueW = 1 / mParams.mDirectLightProb;
                        } else if (gVertexMerging && vcRnd < (mParams.mDirectLightProb + mParams.mMergeProb)) {
                            doVM = true;
                            techniqueW = 1 / mParams.mMergeProb;
                        } else {
                            doVC = true;
                            techniqueW = 1 / (1 - (mParams.mDirectLightProb + (gVertexMerging ? mParams.mMergeProb : 0)));
                        }
                    }

                    if (doDI) {
                        float weight = 0;
                        float misWeight = 0;
                        const PathSample ps = ConnectToLight(s, vertex, upperOnly, weight, misWeight);
                        ProcessSample(r, s.sg, ps, (weight * misWeight) * techniqueW);
                    }

                    if (gBidirectional) {
                        if (doVC) {
                            // connect to a random light subpath vertex in the LVC
                            const uint count = mLightVertexCache.size();
                            if (count > 0) {
                                const LightSubpathVertex v = mLightVertexCache[min(uint(sampleNext1D(s.sg) * count), count - 1)];

                                float weight = 0;
                                float misWeight = 0;
                                PathSample ps = ConnectToLightSubpath(s, vertex, v, count / (float)mParams.mLightSubpathCount, weight, misWeight);
                                ProcessSample(r, s.sg, ps, (weight * misWeight) * techniqueW);
                            }
                        }

                        if (gVertexMerging && doVM) {
                            // merge with the photon in the selected cell
                            float cellW = 0.f;
                            const LightSubpathVertex v = FindPhoton(s.sg, vertex, cellW);
                            if (cellW > 0) {
                                float weight = 0;
                                float misWeight = 0;
                                const PathSample ps = MergeWithLightVertex(s, vertex, v, cellW, weight, misWeight);
                                ProcessSample(r, s.sg, ps, (weight * misWeight) * techniqueW);
                            }

                            if (gProgressivePhotonMapping)
                                break;
                        }
                    }
                }

                {
                    float bsdfPdfW = 0;
                    bool deltaLobe = false;
                    SampleNextVertex(s, vertex, bsdfPdfW, deltaLobe);

                    const float3 le = vertex.GetEmission();

                    // set vertex as rcv if one hasn't been found yet.
                    if (!s.rcvFound && prevReconnectable && !deltaLobe && vertex.mVertex.mRayDistance >= mParams.mReconnectionDistance && (any(le > 0) || vertex.IsConnectable())) {
                        s.rcv = vertex.Pack();
                        if (!vertex.IsFinite()) s.rcv.mRayDistance = FLT_MAX;
                        s.rcvCos = vertex.HasFaceNormal() ? abs(dot(vertex.GetDirIn(), vertex.GetFaceNormal())) : 1;
                        s.rcvDist = s.rcv.mRayDistance;
                        s.prefixBounces     = s.bounces - 1;
                        s.rcvDiffuseBounces = s.diffuseBounces - 1;
                        s.prefixPdf = prevPdf;
                        // throughputAtRcv and rcvDirOut are assigned next iteration after a direction is sampled
                    }

                    // update rcvDirOut and throughputAtRcv if this vertex was selected as the rcv last iteration
                    if (s.rcvFound && s.bounces == s.prefixBounces + 2) {
                        s.rcvDirOut = vertex.mVertex.mRayDirection;
                        s.throughputAtRcv = s.throughput;
                        s.pdfAtRcv = s.pdfW;
                    }

                    // if vertex is emissive, form a path ending at vertex
                    if (any(le > 0) && !DebugDisablePath(s.bounces, 0)) {
                        const float3 contrib = s.throughput * le;

                        // calculate mis weight
                        float emissionPdfW = 0;
                        float directPdfW = 0;
                        float directPdfA = 0;
                        if (gBidirectional || gSampleDirectLight) {
                            GetLightSamplePdf(vertex, prevPos, prevNormal, upperOnly, emissionPdfW, directPdfW, directPdfA);
                        }

                        // divide by pdf
                        float integrationWeight = 1 / s.pdfW;

                        float misWeight = 1;
                        if (!DebugDisableMIS()) {
                            if (gBidirectional) {
                                const float wCamera = Mis(directPdfA) * s.dVCM + Mis(emissionPdfW) * s.dVC;
                                misWeight = 1 / (1 + wCamera);
                            } else if (gSampleDirectLight)
                                misWeight = prevReconnectable && bsdfPdfW > 0 ? Mis(bsdfPdfW, directPdfW) : 1;
                        }

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
                                {}
                            );
                        }

                        const PathSample ps = PathSample::Create(
                            contrib,
                            s.bounces,
                            hasRcv ? s.prefixPdf         : s.pdfW,
                            hasRcv ? s.prefixBounces     : s.bounces,
                            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
                            s.id,
                            suffix,
                            PathFlags::eNone );

                        ProcessSample(r, s.sg, ps, misWeight * integrationWeight);
                    }
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

    // Sample a light subpath. Updates the LVC/photon map. Connects connectable vertices to the camera directly.
    void SampleLightPath(const PathId id) {
        if (mParams.mMaxBounces == 0 || mParams.mMaxDiffuseBounces == 0)
            return;

        PathVertex vertex;
        PathState<false> s = InitializeLightPath<false>(id, vertex);

        while (any(s.throughput > 0) && s.pdfW > 0 && vertex.IsFinite() && s.bounces + 1 <= mParams.mMaxBounces && s.diffuseBounces + 1 <= mParams.mMaxDiffuseBounces) {
            const bool connectable = vertex.IsConnectable();
            if (connectable) {
                // store vertex for connection and merging
                if (!gLightTraceOnly) {
                    const LightSubpathVertex v = PackLightSubpathVertex(s, vertex);

                    if (gVertexMerging || s.bounces + 2 <= mParams.mMaxBounces) {
                        const uint i = mLightVertexCache.Insert(v);

                        if (gVertexMerging) {
                            const float3 pos = vertex.GetPosition();
                            mPhotonMap.addPhoton(i, pos, GetMergeRadius(pos, true), mParams.mPhotonCellCount, sampleNext1D(s.sg));
                        }
                    }
                }

                // form a full path by connecting the vertex to the camera
                if (!gProgressivePhotonMapping) {
                    float weight = 0;
                    float misWeight = 0;
                    int2 pixelCoord;
                    const PathSample ps = ConnectToCamera(s, vertex, pixelCoord, weight, misWeight);
                    if (weight > 0 && misWeight > 0 && any(ps.mContribution > 0) && all(pixelCoord >= 0) && all(pixelCoord < mParams.mOutputDim)) {
                        DebugSetPixel(pixelCoord);
                        if (!gUseResampling && ps.IsValid() && weight > 0) {
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
                            mLightTraceReservoirs.Insert(flatId, PathReservoir(ps, (weight * misWeight), 1));
                        } else {
                            // splat radiance onto output directly, instead of using reservoirs
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3)                  , ps.mContribution.r * (weight * misWeight));
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)  , ps.mContribution.g * (weight * misWeight));
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)*2, ps.mContribution.b * (weight * misWeight));
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
            s.prefixPdf = s.pdfW;
            s.prefixBounces = 0;
            s.rcvDiffuseBounces = 0;

            const float3 prevN = vertex.GetFaceNormal();

            float bsdfPdfW;
            bool deltaLobe;
            SampleNextVertex(s, vertex, bsdfPdfW, deltaLobe);

            s.rcvCos = abs(dot(prevN, vertex.mVertex.mRayDirection));
            s.rcvDist = vertex.mVertex.mRayDistance;
            s.deltaRcv = !connectable || deltaLobe;
        }
    }

    // Shift a path to a new primary vertex
    PathSample ShiftPath(const PathSample basePath, const PackedPathVertex newPrimaryHit, out float jacobian) {
        jacobian = 0;
        PathSample result = {};

        if (DebugDisablePath(basePath.mBounces, basePath.mSuffix.mLightSubpath.mVertexCount, basePath.HasFlag(PathFlags::eVertexMerging)))
            return result;

        // retrace the light subpath
        LightSubpathVertex newLightSubpathVertex = {};
        if (gValidateSuffixes && gBidirectional && basePath.mSuffix.mLightSubpath.mVertexCount > 1) {
            PathVertex vertex;
            PathState<true> s = ShiftLightSubpath(basePath, vertex);
            if (!any(s.throughput > 0) || !(s.pdfW > 0) || s.bounces + 2 != basePath.mSuffix.mLightSubpath.mVertexCount || !vertex.IsFinite()) {
                print("Shift failed: Light subpath shift failed", s.throughput);
                return result;
            }

            // store light subpath vertex for reconnection
            newLightSubpathVertex = PackLightSubpathVertex(s, vertex);

            // Connect caustic paths to the camera directly
            if (!gUseCausticReservoirs && basePath.HasFlag(PathFlags::eCausticLightPath)) {
                int2 offsetPixel;
                float w, m;
                result = ConnectToCamera(s, vertex, offsetPixel, w, m);
                if (!result.IsValid()) {
                    result = {};
                    return result;
                }

                // reject path if it's not in the same pixel (or close)
                const float2 d = GetPixelCoord(PathVertex(newPrimaryHit).GetPosition()) - offsetPixel;
                const float d2 = dot(d, d);
                const float r = gPathGenerator.mParams.mCausticReuseRadius;
                if ((r == 0 && any(d != 0.f)) || (r > 0 && d2 > r * r)) {
                    print("Shift failed: Caustic path not in pixel", offsetPixel);
                    result = {};
                    return result;
                }

                jacobian = basePath.mSuffix.mPdf / result.mSuffix.mPdf;
                return result;
            }
        }

        // retrace the prefix from the new primary hit
        {
            PathVertex vertex = PathVertex(newPrimaryHit);

            // initialize path state
            PathState<true> s = InitializeCameraPath<true>(basePath.mId, vertex);
            s.lightSubpathVertex = basePath.mSuffix.mLightSubpath.mFirstVertex;
            s.lightSubpathId     = basePath.mSuffix.mLightSubpath.mId;

            // trace the new prefix
            while (s.bounces < basePath.mPrefixBounces && any(s.throughput > 0)) {
                float bsdfPdfW;
                bool deltaLobe;
                SampleNextVertex(s, vertex, bsdfPdfW, deltaLobe);
            }
            if (s.bounces != basePath.mPrefixBounces || s.diffuseBounces != basePath.mPrefixDiffuseBounces || !any(s.throughput > 0)) {
                print("Shift failed: Prefix length mismatch:", uint2(s.bounces, basePath.mPrefixBounces));
                return result;
            }

            if (basePath.HasSuffix()) {
                if (!vertex.IsConnectable()) {
                    print("Shift failed: Non reconnectable prefix", 0);
                    return result;
                }

                // reconnect to suffix
                result = ConnectToSuffix(s, vertex, basePath, newLightSubpathVertex);
            } else {
                // pure random replay path, no suffix
                const float3 le = vertex.GetEmission();
                if (!any(le > 0)) {
                    print("Shift failed: Zero le:", le);
                    return result;
                }
                result = basePath;
                result.mContribution = s.throughput * le;
                result.mPrefixPdf = s.pdfW;
            }
        }

        if (!result.IsValid() || !(result.mPrefixPdf > 0)) {
            print("Shift failed: Invalid result", 0);
            return {};
        }

        // random replay jacobian
        jacobian = basePath.mPrefixPdf / result.mPrefixPdf;

        if (basePath.HasSuffix() && basePath.mSuffix.mVertex.IsFinite()) {
            // reconnection jacobian
            if (!gUseCausticReservoirs || !basePath.HasFlag(PathFlags::eCausticLightPath))
                jacobian *= (result.mSuffix.mCos / basePath.mSuffix.mCos) * sqr(basePath.mSuffix.mDist / result.mSuffix.mDist);

            // suffix jacobian
            if (gValidateSuffixes) jacobian *= basePath.mSuffix.mPdf / result.mSuffix.mPdf;
        }

        return result;
    }

    // Shift a pure light path
    PathSample ShiftCausticPath(const PathSample basePath, out float jacobian, out int2 pixelCoord) {
        jacobian = 0;
        pixelCoord = -1;

        PathVertex vertex;
        PathState<true> s = ShiftLightSubpath(basePath, vertex);
        if (!any(s.throughput > 0) || !(s.pdfW > 0) || s.bounces + 2 != basePath.mSuffix.mLightSubpath.mVertexCount || !vertex.IsFinite()) {
            print("Caustic shift failed: Path length mismatch", uint2(s.bounces + 2, basePath.mSuffix.mLightSubpath.mVertexCount));
            return {};
        }

        float integrationWeight, misWeight;
        PathSample result = ConnectToCamera(s, vertex, pixelCoord, integrationWeight, misWeight);

        if (!result.IsValid()) {
            print("Caustic shift failed: invalid result", 0);
            return {};
        }

        if (!result.HasFlag(PathFlags::eCausticLightPath)) {
            print("Caustic shift failed: Result not caustic", 0);
            return {};
        }

        // random replay jacobian
        jacobian = basePath.mSuffix.mPdf / result.mSuffix.mPdf;

        return result;
    }
};
