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
    RWStructuredBuffer<PathReservoir> mLastCausticReservoirs;
    HashMap<PathReservoir> mLightTraceReservoirs;
    HashMap<PathReservoir> mCausticReservoirMap;

    Texture2D<PackedHitInfo> mVbuffer;
    Texture2D<float3> mViewDir;
    Texture2D<float2> mMotionVectors;
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
    static const bool  gRetraceSuffix               = RETRACE_SUFFIX;
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
////////////////////////////// Path generator implementation //////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

extension PathGenerator {
    // ---------------------------------------------------------------------------------------------------------
    // ------------------------------------------- Helper functions --------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    static uint debugPixel;

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

    int2 GetPixelCoord(const float3 pos) {
        float4 clip = mul(gScene.camera.data.viewProjMat, float4(pos, 1));
        clip.y = -clip.y;
        return int2(((clip.xy / clip.w) * .5 + .5) * mParams.mOutputDim);
    }

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
    // Merges a canonical sample into the pixel's reservoir
    void ProcessSample(inout PathReservoir r, inout SampleGenerator sg, const PathSample ps, const float weight, const float M = 0) {
        if (gUseResampling) {
            if (r.Merge(sampleNext1D(sg), ps, weight * ps.GetTargetPdf(), M)) {
                print("Selected path. contrib:", ps.mContribution);
            }
        } else if (ps.IsValid() && weight > 0) {
            print("Found path. contrib:", ps.mContribution);
            print("\tLight subpath vertices:", ps.mNumLightSubpathVertices);
            print("\tMerge:", ps.HasFlag(PathFlags::eMergeAtRcv));
            print("\tPrefix bounces:", ps.mPrefixBounces);
            mOutputRadiance[ps.mRngIdx].rgb += ps.mContribution * weight;
        }
    }

    // ---------------------------------------------------------------------------------------------------------
    // ---------------------------------------------- Path state -----------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    struct PathState<let bShift : bool> {
        float3 throughput = 0;
        float pdfW = 0;

        SampleGenerator sg;     // sample generator for everything but BSDF sampling
        SampleGenerator sgBsdf; // sample generator for BSDF sampling

        PackedPathVertex rcv = {};
        float3 throughputAtRcv = 0;
        float pdfAtRcv = 0;
        float3 rcvDirOut;
        float rcvCos = 0;
        float rcvDist = 0;

        float dVCM = 0;
        float dVC = 0;
        float dVM = 0;

        uint seed;
        uint packedId;
        uint packedBounces = 0;
        uint packedDiffuseBounces = 0;

        PackedHitInfo firstVertex = {};

        property uint bounces {
            get { return BF_GET(packedBounces,    0, 15); }
            set { BF_SET(packedBounces, newValue, 0, 15); }
        }
        property uint rcvBounces {
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
            get { return pdfAtRcv > 0; }
        }
        property uint2 id {
            get { return uint2(packedId & 0xFFFF, packedId >> 16); }
            set { packedId = (newValue.y << 16) | newValue.x; }
        }

        [mutating]
        void Terminate() {
            throughput = 0;
        }
    };

    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------- Connection stategies ------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathSample ConnectToLight<let bShift : bool>( inout PathState<bShift> s, const PathVertex vertex, const bool upperOnly, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;

        PathSample result = {};

        if (DebugDisablePath(s.bounces + 1, 1))
            return result;

        // Sample emissive path vertex

        float directPdfW = 0;
        float emissionPdfW = 0;
        PackedPathVertex l = {};
        float cosLight = 1;
        float3 le = 0;
        {
            const bool useEnvMap = sampleNext1D(s.sg) < mParams.mEnvMapProb;
            const float2 uv = sampleNext2D(s.sg);

            if (gUseEnvLight && (!gUseEmissiveLights || useEnvMap)) {
                EnvMapSample envSample;
                if (!mEnvMapSampler.sample(uv, envSample))
                    return result;

                if (gUseEmissiveLights) envSample.pdf *= mParams.mEnvMapProb;

                le = envSample.Le;
                if (!any(le > 0))
                    return result;

                l = PackedPathVertex(Ray(vertex.ComputeRayOrigin(envSample.dir), envSample.dir), HitInfo(0), FLT_MAX);

                directPdfW = envSample.pdf;
                emissionPdfW = directPdfW * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && !(gUseEnvLight && useEnvMap)) {
                const float3 vertexNormal = upperOnly ? vertex.GetFaceNormal() : 0;
                float selectionPdf;
                uint triangleIndex = mEmissiveSampler.sampleLight(vertex.GetPosition(), vertexNormal, upperOnly, s.sg, selectionPdf);
                if (triangleIndex == LightCollection::kInvalidIndex)
                    return result;

                if (gUseEnvLight) selectionPdf *= 1 - mParams.mEnvMapProb;

                let tri = gScene.lightCollection.getTriangle(triangleIndex);
                const float3 barycentrics = sample_triangle(uv);

                le = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(barycentrics));
                if (!any(le > 0))
                    return result;

                const float directPdfA = selectionPdf / tri.area;

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

        // divide by pdf
        integrationWeight = 1.f / samplePdf;

        const bool hasRcv = gAllowEarlyReconnection && !bShift && s.rcvFound;
        PathSuffix suffix = {};
        if (hasRcv) {
            // finalize the existing rcv
            suffix.mVertex = s.rcv;
            suffix.mCos  = s.rcvCos;
            suffix.mDist = s.rcvDist;
            if (s.rcvBounces == s.bounces - 1) {
                suffix.mDirOut = l.mRayDirection;
                suffix.mContribution = le;
            } else {
                suffix.mDirOut = s.rcvDirOut;
                suffix.mContribution = SafeDivide(contrib, s.throughputAtRcv);
            }
            suffix.mNumVertices = (s.bounces - s.rcvBounces) + 1;
        } else {
            // set sampled light vertex as the rcv
            suffix.mVertex = l;
            suffix.mCos = cosLight;
            suffix.mDist = l.mRayDistance;
            suffix.mNumVertices = 1;
            suffix.mDirOut = 0;
            suffix.mContribution = le;
        }

        result = PathSample::Create(
            contrib,
            hasRcv ? s.pdfAtRcv          : s.pdfW,
            hasRcv ? s.rcvBounces        : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            1,
            s.seed,
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
        if (bounces > mParams.mMaxBounces || s.diffuseBounces + v.mDiffuseBounces + 2 > mParams.mMaxDiffuseBounces || DebugDisablePath(bounces, v.mBounces + 2))
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
        if (all(r.mReflectance <= 0)) {
            //print("all(r.mReflectance <= 0)", 0);
            return result;
        }

        // evaluate the bsdf at the camera subpath vertex
        const float cosCam = vertex.GetCosN(toLight);
        if (cosCam < kMinCosTheta)
            return result;

        let f = vertex.EvaluateReflectance(s.sg, toLight, false);
        if (all(f.mReflectance <= 0)) {
            //print("all(f.mReflectance <= 0)", 0);
            return result;
        }

        const float3 contrib = (s.throughput * v.mThroughput) * (f.mReflectance * r.mReflectance) / cosLight;

        if (!any(contrib > 0) || any(contrib < 0)) {
            //print("contrib <= 0", 0);
            return result;
        }

        if (Occluded(visibilityRay)) {
            return result;
        }

        // Convert pdfs to area measure
        const float invDist2 = 1 / sqr(dist);
        const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
        const float lightBsdfDirPdfA  = r.mFwdPdfW * cosCam   * invDist2;

        // mis weight
        if (DebugDisableMIS()) {
            misWeight = 1;
        } else {
            const float wLight = Mis(cameraBsdfDirPdfA) * (GetMisVmWeightFactor() + v.dVCM + v.dVC * Mis(r.mRevPdfW));
            const float wCamera = Mis(lightBsdfDirPdfA) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
            misWeight = 1 / (wLight + 1.f + wCamera);
        }

        integrationWeight = subpathWeight * v.mIntegrationWeight * (cosLight * invDist2) / s.pdfW;

        const bool hasRcv = gAllowEarlyReconnection && !bShift && s.rcvFound;
        PathSuffix suffix = {};
        if (hasRcv) {
            // finalize the existing rcv
            suffix.mVertex = s.rcv;
            suffix.mCos = s.rcvCos;
            suffix.mDist = s.rcvDist;
            if (s.rcvBounces == s.bounces - 1) {
                suffix.mDirOut = toLight;
                suffix.mContribution = v.mThroughput * (r.mReflectance / cosLight);
            } else {
                suffix.mDirOut = s.rcvDirOut;
                suffix.mContribution = SafeDivide(contrib, s.throughputAtRcv);
            }
            suffix.mNumVertices = (s.bounces - s.rcvBounces) + v.mBounces + 2;
        } else {
            // use the light subpath vertex as the rcv
            suffix.mVertex = v.mVertex;
            suffix.mContribution = v.mThroughput;
            suffix.mDirOut = v.GetDirIn();
            suffix.mCos = cosLight;
            suffix.mDist = dist;
            suffix.mNumVertices = v.mBounces + 2;
        }

        result = PathSample::Create(
            contrib,
            hasRcv ? s.pdfAtRcv          : s.pdfW,
            hasRcv ? s.rcvBounces        : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            v.mBounces + 1,
            s.seed,
            s.id,
            suffix,
            PathFlags::eNone);

        if (!gUseResampling) {
            print("Connected to path id", v.mRngIdx);
        }

        return result;
    }
    PathSample MergeWithLightVertex<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, const float subpathWeight, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;

        PathSample result = {};

        const float mergeRadius = GetMergeRadius(vertex.GetPosition());

        const uint bounces = s.bounces + v.mBounces + 1;
        if (bounces > mParams.mMaxBounces || s.diffuseBounces + v.mDiffuseBounces + 1 > mParams.mMaxDiffuseBounces || DebugDisablePath(bounces, v.mBounces + 2, true))
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

        const float wLight  = v.dVCM * GetMisVcWeightFactor() + v.dVM * Mis(f.mFwdPdfW);
        const float wCamera = s.dVCM * GetMisVcWeightFactor() + s.dVM * Mis(f.mRevPdfW);
        misWeight = 1;
        if (DebugDisableMIS() || gProgressivePhotonMapping || bShift)
            misWeight = 1;
        else
            misWeight = 1.f / (wLight + 1.f + wCamera);

        integrationWeight = subpathWeight * v.mIntegrationWeight / (s.pdfW * (sqr(mergeRadius) * M_PI * mParams.mLightSubpathCount));

        const bool hasRcv = gAllowEarlyReconnection && !bShift && s.rcvFound;
        PathSuffix suffix = {};
        if (hasRcv) {
            // finalize the existing rcv
            suffix.mVertex = s.rcv;
            suffix.mCos = s.rcvCos;
            suffix.mDist = s.rcvDist;
            if (s.rcvBounces == s.bounces - 1) {
                suffix.mDirOut = v.GetDirIn();
                suffix.mContribution = v.mThroughput * f.mReflectance;
            } else {
                suffix.mDirOut = s.rcvDirOut;
                suffix.mContribution = SafeDivide(contrib, s.throughputAtRcv);
            }
            suffix.mNumVertices = (s.bounces - s.rcvBounces) + v.mBounces + 2;
        } else {
            // store photon as reconnection vertex
            suffix.mVertex = v.mVertex;
            suffix.mContribution = v.mThroughput;
            suffix.mDirOut = v.GetDirIn();
            suffix.mNumVertices = v.mBounces + 2;
        }

        result = PathSample::Create(
            contrib,
            hasRcv ? s.pdfAtRcv          : s.pdfW,
            hasRcv ? s.rcvBounces        : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            v.mBounces + 2,
            s.seed,
            s.id,
            suffix,
            hasRcv ? PathFlags::eNone : PathFlags::eMergeAtRcv);

        if (!gUseResampling) {
            print("Merged with path id", v.mRngIdx);
            print("Photon throughput", v.mThroughput);
            print("bounces, subpath bounces", uint2(s.bounces, v.mBounces));
        }

        return result;
    }
    PathSample ConnectToCamera<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out int2 pixelCoord, out float integrationWeight, out float misWeight) {
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

        float3 contrib = s.throughput * vertex.Reversed(toCamera).EvaluateReflectance(s.sg, vertex.GetDirIn(), false).mReflectance / cosIn;
        if (!any(contrib > 0)) {
            if (bShift) print("Caustic shift failed: zero contrib", contrib);
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

        const float wLight = Mis(cameraPdfA / mParams.mLightSubpathCount) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
        misWeight = (gLightTraceOnly || DebugDisableMIS()) ? 1.f : (1.f / (wLight + 1.f));
        integrationWeight = 1.f / samplePdf;

        PathSuffix suffix = {};
        if (s.deltaRcv || bShift) {
            // if the rcv is unconnectable, this is a caustic light path and we store the primary vertex as the rcv instead.
            suffix.mVertex = vertex.Pack();
            suffix.mContribution = s.throughput;
            suffix.mDirOut = vertex.GetDirIn();
            suffix.mCos = cosIn;
            suffix.mDist = dist;
            suffix.mNumVertices = s.bounces + 2;
            suffix.mPdf = s.pdfW;

            // for caustic paths, the mis weight is in the path contribution instead of the resampling/integration weights
            contrib *= misWeight;
            misWeight = 1;
        } else {
            suffix.mVertex = s.rcv;
            suffix.mContribution = s.throughputAtRcv;
            suffix.mDirOut = s.rcvDirOut;
            suffix.mCos    = s.rcvCos;
            suffix.mDist   = s.rcvDist;
            suffix.mNumVertices = s.bounces + 1;
        }

        result = PathSample::Create(
            contrib,
            1,
            0,
            0,
            s.bounces + 2,
            s.seed,
            s.id,
            suffix,
            s.deltaRcv ? PathFlags::eCausticLightPath : PathFlags::eNone);

        result.mLightSubpathVertex = s.firstVertex;

        return result;
    }
    PathSample ConnectToSuffix(inout PathState<true> s, const PathVertex vertex, const PathSample basePath, out float jacobian) {
        jacobian = 0;

        if (!vertex.IsConnectable()) {
            print("Shift failed: Non reconnectable prefix", 0);
            return {};
        }

        float3 contribution = s.throughput * basePath.mSuffix.mContribution;
        if (!any(contribution > 0)) {
            print("Reconnection failed: zero contribution", contribution);
            return {};
        }

        float newCos, newDist;

        const PathVertex suffixVertex = PathVertex(basePath.mSuffix.mVertex);

        if (!gUseCausticReservoirs && basePath.HasFlag(PathFlags::eCausticLightPath)) {
            // For caustic light paths, reconnection is always impossible, but we copy the path anyways.
            const int2 offsetPixel = GetPixelCoord(vertex.GetPosition());
            const int2 basePixel   = GetPixelCoord(suffixVertex.GetPosition());
            if (mParams.mCausticReuseRadius > 0) {
                const float dist = length(offsetPixel - basePixel);
                if (dist > mParams.mCausticReuseRadius) {
                    print("Shift failed: Caustic light path", dist);
                    return {};
                }
            } else {
                if (any(offsetPixel != basePixel)) {
                    print("Shift failed: Caustic light path", offsetPixel != basePixel);
                    return {};
                }
            }

            newCos = vertex.GetCosN(basePath.mSuffix.mDirOut);
            newDist = length(vertex.GetPosition() - gScene.camera.getPosition());
        } else {
            Ray visibilityRay;

            // compute visibility ray and reflectance between the prefix and the suffix
            if (basePath.mSuffix.mVertex.IsFinite()) {
                // update the ray origin and direction
                PackedPathVertex newRcv = basePath.mSuffix.mVertex;
                newRcv.mRayDirection = suffixVertex.GetPosition() - vertex.GetPosition();
                newRcv.mRayOrigin = vertex.ComputeRayOrigin(newRcv.mRayDirection);

                // nudge both ends of ray away from surfaces
                newRcv.mRayDirection = suffixVertex.ComputeRayOrigin(-newRcv.mRayDirection) - newRcv.mRayOrigin;
                newRcv.mRayDistance = length(newRcv.mRayDirection);
                newRcv.mRayDirection /= newRcv.mRayDistance;

                if (newRcv.mRayDistance < mParams.mReconnectionDistance) {
                    print("Shift failed: Non reconnectable prefix (distance)", newRcv.mRayDistance);
                    return {};
                }

                const float cosCamera = vertex.GetCosN(newRcv.mRayDirection);
                if (cosCamera < kMinCosTheta) {
                    print("Shift failed: Non reconnectable prefix (cosCamera)", cosCamera);
                    return {};
                }

                newCos = suffixVertex.HasFaceNormal() ? dot(suffixVertex.GetFaceNormal(), -newRcv.mRayDirection) : 1;
                newDist = newRcv.mRayDistance;
                visibilityRay = Ray(newRcv.mRayOrigin, newRcv.mRayDirection, 0, newRcv.mRayDistance);

                if (basePath.mSuffix.mNumVertices == 1) {
                    // triangle lights are one-sided
                    if (newCos < kMinCosTheta) {
                        print("Reconnection failed: newCos < kMinCosTheta", newCos);
                        return {};
                    }
                } else {
                    if (abs(newCos) < kMinCosTheta) {
                        print("Reconnection failed: abs(newCos) <= kMinCosTheta", newCos);
                        return {};
                    }

                    const PathVertex rcv = PathVertex(newRcv);
                    if (!rcv.IsConnectable()) {
                        print("Reconnection failed: rcv unconnectable", 0);
                        return {};
                    }

                    // evaluate reflectance at reconnection vertex
                    contribution *= rcv.EvaluateReflectance(s.sg, basePath.mSuffix.mDirOut, false).mReflectance;
                    if (!any(contribution > 0)) {
                        print("Reconnection failed: zero rcv reflectance", contribution);
                        return {};
                    }
                }
            } else {
                visibilityRay = Ray(vertex.ComputeRayOrigin(basePath.mSuffix.mVertex.mRayDirection), basePath.mSuffix.mVertex.mRayDirection);
                newCos = 1;
                newDist = FLT_MAX;
            }

            // evaluate reflectance at vertex
            contribution *= vertex.EvaluateReflectance(s.sg, visibilityRay.dir, false).mReflectance;
            if (!any(contribution > 0)) {
                print("Reconnection failed: zero vertex reflectance", 0);
                return {};
            }

            if (Occluded(visibilityRay)) {
                print("Reconnection failed: suffix occluded", 0);
                return {};
            }
        }

        PathSample result = basePath;
        result.mContribution = contribution;
        result.mPrefixPdf = s.pdfW;
        result.mSuffix.mCos  = newCos;
        result.mSuffix.mDist = newDist;

        // random replay jacobian
        jacobian = basePath.mPrefixPdf / result.mPrefixPdf;

        // reconnection jacobian
        if (basePath.mSuffix.mVertex.IsFinite() && !basePath.HasFlag(PathFlags::eCausticLightPath)) {
            jacobian *= (result.mSuffix.mCos / basePath.mSuffix.mCos) * sqr(basePath.mSuffix.mDist / result.mSuffix.mDist);
        }

        return result;
    }
    PathSample MergeWithSuffix(inout PathState<true> s, const PathVertex vertex, const float mergeRadius, const PathSample basePath, out float jacobian) {
        PathSample result = {};
        jacobian = 0;

        const PathVertex photon = PathVertex(basePath.mSuffix.mVertex);
        const float3 toPhoton = photon.GetPosition() - vertex.GetPosition();
        if (dot(toPhoton, toPhoton) >= sqr(mergeRadius))
            return result;

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, basePath.mSuffix.mDirOut, false);
        if (vertex.HasFaceNormal()) {
            const float3 guideNormal = vertex.mMaterial.getProperties(vertex.mShadingData).guideNormal;
            const float cosOut = abs(dot(guideNormal, basePath.mSuffix.mDirOut));
            if (cosOut < kMinCosTheta)
                return result;
            f.mReflectance /= cosOut; // divide out dot(N, wo)
        }
        const float3 contrib = s.throughput * f.mReflectance * basePath.mSuffix.mContribution;
        if (all(contrib <= 0))
            return result;

        result = basePath;
        result.mContribution = contrib;
        result.mPrefixPdf = s.pdfW;

        // random replay jacobian
        jacobian = basePath.mPrefixPdf / result.mPrefixPdf;

        return result;
    }

    // ---------------------------------------------------------------------------------------------------------
    // -------------------------------------------- Path generation --------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathState<bShift> InitializeCameraPath<let bShift : bool>(const uint seed, const uint2 id, const PathVertex vertex) {
        PathState<bShift> s = {};
        s.seed = seed;
        s.id = id;
        s.isLightPath = false;

        s.sg     = SampleGenerator(s.id, s.seed);
        s.sgBsdf = SampleGenerator(s.id, -s.seed);

        s.bounces = 0;
        s.diffuseBounces = 0;
        s.rcvBounces = 0;
        s.rcvDiffuseBounces = 0;
        s.throughput = 1;
        s.throughputAtRcv = 0;
        s.pdfAtRcv = 0;
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
    PathState<bShift> InitializeLightPath<let bShift : bool>(const uint seed, const uint2 id, out PathVertex vertex, PackedHitInfo firstVertex = {}) {
        PathState<bShift> s = {};
        s.seed = seed;
        s.id = id;
        s.isLightPath = true;

        s.throughput = 0;
        s.pdfW = 0;
        s.throughputAtRcv = 0;
        s.rcvDirOut = 0;

        s.sg     = SampleGenerator(s.id, s.seed);
        s.sgBsdf = SampleGenerator(s.id, -s.seed);

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
                branchlessONB(envSample.dir, t, b);
                const float2 uv = sample_disk_concentric(sampleU.zw);

                ray = Ray(mParams.mSceneSphere.xyz + mParams.mSceneSphere.w * (t * uv.x + b * uv.y + envSample.dir), -envSample.dir);
                l = PackedPathVertex(ray, HitInfo(0), FLT_MAX);
                s.throughput = envSample.Le;
                cosLight = 1;

                directPdfA = envSample.pdf;
                emissionPdfW = directPdfA * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && !(gUseEnvLight && useEnvMap)) {
                float selectionPdf;
                uint triangleIndex = mEmissiveSampler.sampleLight(s.sg, selectionPdf);

                if (bShift) {
                    TriangleHit triHit = HitInfo(firstVertex).getTriangleHit();
                    triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
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
                branchlessONB(tri.normal, t, b);
                const float3 rayDir = normalize(localDir.x * t + localDir.y * b + localDir.z * tri.normal);

                const float3 barycentrics = sample_triangle(sampleU.xy);
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

        s.firstVertex = l.mPackedHit;

        s.pdfW = emissionPdfW;
        s.bounces = 0;
        s.diffuseBounces = 0;
        s.rcvBounces = 0;
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

    // Sample the BSDF and trace a ray to find the next vertex.
    void SampleNextVertex<let bShift : bool>( inout PathState<bShift> s, inout PathVertex vertex, out float bsdfPdfW, out bool deltaLobe) {
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

    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- Top-level -----------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    // Samples a canonical camera by connecting to lights, light subpath vertices, and merging with photons.
    PathReservoir SampleCameraPath(const uint seed, const uint2 id) {
        PathVertex vertex = PathVertex(LoadPrimaryVertex(id));

        // directly output primary radiance
        mOutputRadiance[id] = float4(vertex.GetEmission(), 1);

        if (gLightTraceOnly || !vertex.IsFinite())
            return {};

        PathState<false> s = InitializeCameraPath<false>(seed, id, vertex);

        PathReservoir r = {};
        r.mM = 1;

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
                    s.rcvBounces = s.bounces - 1;
                    s.rcvDiffuseBounces = s.diffuseBounces - 1;
                    s.pdfAtRcv = prevPdf;
                    // throughputAtRcv and rcvDirOut are assigned next iteration after a direction is sampled
                }

                // update rcvDirOut and throughputAtRcv if this vertex was selected as the rcv last iteration
                if (s.rcvFound && s.rcvBounces == s.bounces - 2) {
                    s.rcvDirOut = vertex.mVertex.mRayDirection;
                    s.throughputAtRcv = s.throughput;
                }

                // if vertex is emissive, form a path ending at vertex
                if (any(le > 0) && !DebugDisablePath(s.bounces, 0)) {
                    // calculate mis weight
                    float emissionPdfW = 0;
                    float directPdfW = 0;
                    float directPdfA = 0;
                    if (gBidirectional || gSampleDirectLight) {
                        GetLightSamplePdf(vertex, prevPos, prevNormal, upperOnly, emissionPdfW, directPdfW, directPdfA);
                    }

                    float weight = 1;
                    if (!DebugDisableMIS()) {
                        if (gBidirectional) {
                            const float wCamera = Mis(directPdfA) * s.dVCM + Mis(emissionPdfW) * s.dVC;
                            weight = 1 / (1 + wCamera);
                        } else if (gSampleDirectLight)
                            weight = prevReconnectable && bsdfPdfW > 0 ? Mis(bsdfPdfW, directPdfW) : 1;
                    }

                    // divide by pdf
                    weight /= s.pdfW;

                    // finalize reconnection vertex
                    const bool hasRcv = s.rcvFound;
                    PathSuffix suffix = {};
                    if (hasRcv) {
                        // finalize the existing rcv
                        suffix.mVertex = s.rcv;
                        suffix.mCos = s.rcvCos;
                        suffix.mDist = s.rcvDist;
                        suffix.mNumVertices = s.bounces - s.rcvBounces;
                        if (suffix.mNumVertices == 1) {
                            suffix.mContribution = le; // rcv is the last vertex
                        } else {
                            suffix.mDirOut = s.rcvDirOut;
                            suffix.mContribution = le * SafeDivide(s.throughput, s.throughputAtRcv);
                        }
                    } else
                        suffix.mNumVertices = 0;

                    const PathSample ps = PathSample::Create(
                        s.throughput * le,
                        hasRcv ? s.pdfAtRcv : s.pdfW,
                        hasRcv ? s.rcvBounces : s.bounces,
                        hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
                        0,
                        s.seed,
                        s.id,
                        suffix,
                        PathFlags::eNone );

                    ProcessSample(r, s.sg, ps, weight);
                }
            }
        }

        r.FinalizeMerge();
        return r;
    }

    // Sample a light subpath. Updates the LVC/photon map. Connects connectable vertices to the camera directly.
    void SampleLightPath(const uint seed, const uint2 id) {
        if (mParams.mMaxBounces == 0 || mParams.mMaxDiffuseBounces == 0)
            return;

        PathVertex vertex;
        PathState<false> s = InitializeLightPath<false>(seed, id, vertex);

        while (any(s.throughput > 0) && s.pdfW > 0 && s.diffuseBounces < mParams.mMaxDiffuseBounces && vertex.IsFinite()) {
            const bool connectable = vertex.IsConnectable();
            if (connectable) {
                // store vertex for connection and merging
                if (!gLightTraceOnly) {
                    LightSubpathVertex v;
                    v.mVertex = vertex.Pack();
                    v.mThroughput = s.throughput;
                    v.dVC  = s.dVC;
                    v.dVM  = s.dVM;
                    v.dVCM = s.dVCM;
                    v.mIntegrationWeight = 1/s.pdfW;
                    v.mRngIdx  = s.id;
                    v.mBounces = s.bounces;
                    v.mDiffuseBounces = s.diffuseBounces;

                    const uint i = mLightVertexCache.addVertex(v);
                    if (gVertexMerging) {
                        const float3 pos = vertex.GetPosition();
                        const float mergeRadius = GetMergeRadius(pos, true);
                        mPhotonMap.addPhoton(i, pos, mergeRadius, mParams.mPhotonCellCount, sampleNext1D(s.sg));
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
                            print("\tLight subpath vertices:", ps.mNumLightSubpathVertices);
                            print("\tMerge:", ps.HasFlag(PathFlags::eMergeAtRcv));
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

            if (s.bounces + 1 >= mParams.mMaxBounces)
                break;

            // for light paths: rcv is always the 2nd to last vertex
            s.rcv = vertex.Pack();
            s.rcvDirOut = vertex.GetDirIn();
            s.throughputAtRcv = s.throughput;
            s.rcvBounces = 0;
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

        PathVertex vertex = PathVertex(newPrimaryHit);

        // initialize path state
        PathState<true> s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex);

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

        if (basePath.mSuffix.mNumVertices == 0) {
            // pure random replay path, no suffix
            const float3 le = vertex.GetEmission();
            if (!any(le > 0)) {
                print("Shift failed: Zero le:", le);
                return result;
            }
            result = basePath;
            result.mContribution = s.throughput * le;
            result.mPrefixPdf = s.pdfW;
        } else {
            // connect to or merge with the suffix
            if (!vertex.IsConnectable()) {
                print("Shift failed: Unconnectable prefix", 0);
                return result;
            }

            if (basePath.HasFlag(PathFlags::eMergeAtRcv) && (gBidirectional && gVertexMerging && !gLightTraceOnly)) {
                result = MergeWithSuffix(s, vertex, GetMergeRadius(vertex.GetPosition()), basePath, jacobian);
            } else {
                result = ConnectToSuffix(s, vertex, basePath, jacobian);
            }
        }

        if (!result.IsValid()) {
            jacobian = 0;
            return {};
        }

        return result;
    }

    // Shift a pure light path
    PathSample ShiftCausticPath(const PathSample basePath, out float jacobian, out int2 pixelCoord) {
        jacobian = 0;
        pixelCoord = -1;

        PathVertex vertex;
        PathState<true> s = InitializeLightPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, basePath.mLightSubpathVertex);

        while (any(s.throughput > 0) && s.pdfW > 0 && s.bounces + 2 < basePath.mNumLightSubpathVertices && vertex.IsFinite()) {
            const bool connectable = vertex.IsConnectable();

            // for light paths: rcv is always the 2nd to last vertex
            s.rcv = vertex.Pack();
            s.rcvDirOut = vertex.GetDirIn();
            s.throughputAtRcv = s.throughput;
            s.rcvBounces = 0;
            s.rcvDiffuseBounces = 0;

            const float3 prevN = vertex.GetFaceNormal();

            float bsdfPdfW;
            bool deltaLobe;
            SampleNextVertex(s, vertex, bsdfPdfW, deltaLobe);

            s.rcvCos = abs(dot(prevN, vertex.mVertex.mRayDirection));
            s.rcvDist = vertex.mVertex.mRayDistance;
            s.deltaRcv = !connectable || deltaLobe;
        }

        if (s.bounces + 2 != basePath.mNumLightSubpathVertices) {
            print("Caustic shift failed: Path length mismatch", uint2(s.bounces + 2, basePath.mNumLightSubpathVertices));
            return {};
        }


        float integrationWeight, misWeight;
        PathSample result = ConnectToCamera(s, vertex, pixelCoord, integrationWeight, misWeight);

        if (!result.IsValid()) {
            print("Caustic shift failed: invalid result", 0);
            print("base pos:", PathVertex(basePath.mSuffix.mVertex).GetPosition());
            print("vertex pos:", vertex.GetPosition());
            return {};
        }

        if (!result.HasFlag(PathFlags::eCausticLightPath)) {
            print("Shift failed: Result not caustic", 0);
            return {};
        }


        // random replay jacobian
        jacobian = basePath.mSuffix.mPdf / result.mSuffix.mPdf;

        return result;
    }
};
