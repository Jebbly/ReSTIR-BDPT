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
__exported import PathVertex;

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

    RWStructuredBuffer<PathReservoir> mPathReservoirs0;
    RWStructuredBuffer<PathReservoir> mPathReservoirs1;
    HashMap<PathReservoir> mLightTraceReservoirs;

    Texture2D<PackedHitInfo> mVbuffer;
    Texture2D<float3> mViewDir;
    Texture2D<float3> mLastViewDir;
    Texture2D<PackedHitInfo> mLastVbuffer;
    Texture2D<float2> mMotionVectors;

    // Static parameters

    static const bool gUseEnvLight = USE_ENV_LIGHT;
    static const bool gUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool gUseViewDir = USE_VIEW_DIR;
    static const bool gSampleDirectLight = USE_NEE;
    static const bool gBidirectional = USE_BIDIRECTIONAL;
    static const bool gVertexMerging = USE_VERTEX_MERGING;
    static const bool gDynamicMergeRadius = DYNAMIC_MERGE_RADIUS;
    static const bool gProgressivePhotonMapping = USE_PPM_ONLY;
    static const bool gAllowEarlyReconnection = true;
    static const bool gLightTraceOnly = LIGHT_TRACE_ONLY;
    static const bool gUseResampling = USE_RESAMPLING;
    static const bool gRetraceSuffix = RETRACE_SUFFIX;
    static const bool gUseBsdfImportanceSampling = USE_BSDF_IMPORTANCE_SAMPLING;
    static const bool gWavefrontTechniqueSelection = WAVEFRONT_TECHNIQUE_SELECTION;
    static const bool gDebugBPT = DEBUG_BPT;
    static const float gMISPowerExponent = MIS_POWER_EXPONENT;
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

    PackedPathVertex TraceRay(const Ray ray) {
        HitInfo hit;
        float dist;
        traceSceneRay<true>(ray, hit, dist, 0, 0xFFFFFFFF);
        return PackedPathVertex(ray, hit, dist);
    }
    bool Occluded(const Ray ray) {
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
    bool DebugDisablePath(const uint bounces, const uint lightVertices) {
        return gDebugBPT && (
                (mParams.mDebugTotalVertices != -1 && mParams.mDebugTotalVertices != bounces + 2) ||
                (mParams.mDebugLightVertices != -1 && mParams.mDebugLightVertices != lightVertices));
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

    // Sample the BSDF and trace a ray to find the next vertex. Returns a path if the new vertex is emissive.
    void SampleNextVertex<let bShift : bool>( inout PathState<bShift> s, inout PathVertex vertex, out float bsdfPdfW, out bool deltaLobe) {
        bsdfPdfW = 0;
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
                s.dVCM = 0.f;
                s.dVC *= Mis(cosOut);
                s.dVM *= Mis(cosOut);
            }
        }


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

                const float cosLight = -dot(tri.normal, toLight);
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
        ReconnectionVertex rcv = {};
        if (hasRcv) {
            // finalize the existing rcv
            rcv.mVertex = s.rcv;
            rcv.mCos    = s.rcvCos;
            rcv.mDist   = s.rcvDist;
            if (s.rcvBounces == s.bounces - 1) {
                rcv.mDirOut = l.mRayDirection;
                rcv.mRadiance = le;
            } else {
                rcv.mDirOut = s.rcvDirOut;
                rcv.mRadiance = float3(
                    SafeDivide(contrib.r, s.throughputAtRcv.r),
                    SafeDivide(contrib.g, s.throughputAtRcv.g),
                    SafeDivide(contrib.b, s.throughputAtRcv.b));
            }
        } else {
            // set sampled light vertex as the rcv
            rcv.mVertex = l;
            rcv.mRadiance = le;
            rcv.mDirOut = 0;
            rcv.mCos = cosLight;
            rcv.mDist = l.mRayDistance;
        }

        result = PathSample(
            contrib,
            hasRcv ? s.pdfAtRcv : s.pdfW,
            s.bounces + 1,
            hasRcv ? s.rcvBounces : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.seed,
            s.id,
            rcv,
            PathShiftType::eReconnection);

        return result;
    }
    PathSample ConnectToLightSubpath<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, out float integrationWeight, out float misWeight) {
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

        integrationWeight = cosLight * invDist2 / s.pdfW;

        const bool hasRcv = gAllowEarlyReconnection && !bShift && s.rcvFound;
        ReconnectionVertex rcv = {};
        if (hasRcv) {
            // finalize the existing rcv
            rcv.mVertex = s.rcv;
            rcv.mCos  = s.rcvCos;
            rcv.mDist = s.rcvDist;
            if (s.rcvBounces == s.bounces - 1) {
                rcv.mDirOut = toLight;
                rcv.mRadiance = v.mThroughput * (r.mReflectance / cosLight);
            } else {
                rcv.mDirOut = s.rcvDirOut;
                rcv.mRadiance = float3(
                    SafeDivide(contrib.r, s.throughputAtRcv.r),
                    SafeDivide(contrib.g, s.throughputAtRcv.g),
                    SafeDivide(contrib.b, s.throughputAtRcv.b));
            }
        } else {
            // use the light subpath vertex as the rcv
            rcv.mVertex = v.mVertex;
            rcv.mRadiance = v.mThroughput;
            rcv.mDirOut = v.GetDirIn();
            rcv.mCos = cosLight;
            rcv.mDist = dist;
        }

        result = PathSample(
            contrib,
            hasRcv ? s.pdfAtRcv : s.pdfW,
            bounces,
            hasRcv ? s.rcvBounces : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.seed, s.id,
            rcv,
            PathShiftType::eReconnection);

        return result;
    }
    PathSample MergeWithLightVertex<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, const float mergeRadius, out float integrationWeight, out float misWeight) {
        integrationWeight = 0;
        misWeight = 0;

        PathSample result = {};

        const uint bounces = s.bounces + v.mBounces + 1;
        if (bounces > mParams.mMaxBounces || s.diffuseBounces + v.mDiffuseBounces + 1 > mParams.mMaxDiffuseBounces || DebugDisablePath(bounces, v.mBounces + 2))
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

        integrationWeight = 1.f / (s.pdfW * (sqr(mergeRadius) * M_PI * mParams.mLightSubpathCount));

        const bool hasRcv = gAllowEarlyReconnection && !bShift && s.rcvFound;
        ReconnectionVertex rcv = {};
        if (hasRcv) {
            // finalize the existing rcv
            rcv.mVertex = s.rcv;
            rcv.mCos  = s.rcvCos;
            rcv.mDist = s.rcvDist;
            if (s.rcvBounces == s.bounces - 1) {
                rcv.mDirOut = v.GetDirIn();
                rcv.mRadiance = v.mThroughput * f.mReflectance;
            } else {
                rcv.mDirOut = s.rcvDirOut;
                rcv.mRadiance = float3(
                    SafeDivide(contrib.r, s.throughputAtRcv.r),
                    SafeDivide(contrib.g, s.throughputAtRcv.g),
                    SafeDivide(contrib.b, s.throughputAtRcv.b));
            }
        } else {
            // store photon as reconnection vertex
            rcv.mVertex   = v.mVertex;
            rcv.mRadiance = v.mThroughput;
            rcv.mDirOut   = v.GetDirIn();
        }

        result = PathSample(
            contrib,
            hasRcv ? s.pdfAtRcv : s.pdfW,
            bounces,
            hasRcv ? s.rcvBounces : s.bounces,
            hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
            s.seed,
            s.id,
            rcv,
            hasRcv ? PathShiftType::eReconnection : PathShiftType::eMerge);

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

        float4 clip = mul(gScene.camera.data.viewProjMat, float4(vertex.GetPosition(), 1));
        clip.y = -clip.y;
        pixelCoord = int2(((clip.xy / clip.w) * .5 + .5) * mParams.mOutputDim);
        if (any(pixelCoord < 0) || any(pixelCoord >= mParams.mOutputDim) || cosAtCamera <= 0)
            return result;

        const float cosOut = vertex.GetCosN(toCamera);
        if (cosOut < kMinCosTheta)
            return result;

        let f = vertex.EvaluateReflectance(s.sg, toCamera, true);

        // Compute pdf conversion factor from image plane area to surface area
        const float cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);

        const float3 contrib = s.throughput * vertex.Reversed(toCamera).EvaluateReflectance(s.sg, vertex.GetDirIn(), false).mReflectance / vertex.GetCosN(vertex.GetDirIn());
        if (!any(contrib > 0))
            return result;

        const float samplePdf = s.pdfW * (1 / cameraPdfA) * mParams.mLightSubpathCount;
        if (!(samplePdf > 0))
            return result;

        const float wLight = Mis(cameraPdfA / mParams.mLightSubpathCount) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
        misWeight = (gLightTraceOnly || DebugDisableMIS()) ? 1.f : (1.f / (wLight + 1.f));

        if (Occluded(Ray(rayOrigin, toCamera, 0, dist)))
            return result;

        integrationWeight = 1.f / samplePdf;

        ReconnectionVertex rcv = {};
        rcv.mVertex   = s.rcv;
        rcv.mRadiance = s.throughputAtRcv;
        rcv.mDirOut   = s.rcvDirOut;
        rcv.mDist     = s.rcvDist;
        rcv.mCos      = s.rcvCos;

        result = PathSample(
            contrib,
            1,
            s.bounces + 1,
            0,
            0,
            s.seed, s.id,
            rcv,
            PathShiftType::eReconnection);

        result.mIsCausticPath = s.deltaRcv;

        return result;
    }
    PathSample ConnectToReconnectionVertex(inout PathState<true> s, const PathVertex vertex, const PathSample basePath) {
        if (!vertex.IsConnectable())
            return {};

        ReconnectionVertex newRcv = basePath.mReconnectionVertex;
        const bool rcvFinite = basePath.mReconnectionVertex.mVertex.IsFinite();

        // compute ray to rcv
        if (rcvFinite) {
            const PathVertex baseRcv = PathVertex(basePath.mReconnectionVertex.mVertex);
            newRcv.mVertex.mRayDirection = baseRcv.GetPosition() - vertex.GetPosition();
            newRcv.mVertex.mRayOrigin = vertex.ComputeRayOrigin(newRcv.mVertex.mRayDirection);

            // nudge both ends of ray away from surfaces
            newRcv.mVertex.mRayDirection = baseRcv.ComputeRayOrigin(-newRcv.mVertex.mRayDirection) - newRcv.mVertex.mRayOrigin;
            newRcv.mVertex.mRayDistance = length(newRcv.mVertex.mRayDirection);
            newRcv.mVertex.mRayDirection /= newRcv.mVertex.mRayDistance;

            newRcv.mDist = newRcv.mVertex.mRayDistance;
            newRcv.mCos = baseRcv.HasFaceNormal() ? dot(baseRcv.GetFaceNormal(), -newRcv.mVertex.mRayDirection) : 1;

            if (newRcv.mDist < mParams.mReconnectionDistance)
                return {};

            const float cosCamera = vertex.GetCosN(newRcv.mVertex.mRayDirection);
            if (cosCamera < kMinCosTheta)
                return {};
        } else
            newRcv.mVertex.mRayOrigin = vertex.ComputeRayOrigin(newRcv.mVertex.mRayDirection);

        float3 le = s.throughput * basePath.mReconnectionVertex.mRadiance;
        if (!any(le > 0)) {
            print("Reconnection failed: zero le", le);
            return {};
        }

        le *= vertex.EvaluateReflectance(s.sg, newRcv.mVertex.mRayDirection, false).mReflectance;
        if (!any(le > 0)) {
            print("Reconnection failed: zero vertex reflectance", 0);
            return {};
        }

        // Evaluate reflectance if reconnection vertex is not the last vertex
        if (basePath.mReplayBounces + 1 < basePath.mBounces) {
            const PathVertex rcv = PathVertex(newRcv.mVertex);

            if (!rcv.IsConnectable()) {
                return {};
            }
            if (abs(newRcv.mCos) < kMinCosTheta) {
                print("Reconnection failed: abs(cosLight) <= kMinCosTheta", newRcv.mCos);
                return {};
            }
            le *= rcv.EvaluateReflectance(s.sg, basePath.mReconnectionVertex.mDirOut, false).mReflectance;
            if (!any(le > 0)) {
                print("Reconnection failed: zero rcv reflectance", le);
                return {};
            }
        } else {
            // triangle lights are one-sided
            if (rcvFinite && newRcv.mCos < kMinCosTheta) {
                print("Reconnection failed: backfacing light", newRcv.mCos);
                return {};
            }
        }

        if (Occluded(Ray(newRcv.mVertex.mRayOrigin, newRcv.mVertex.mRayDirection, 0, newRcv.mVertex.mRayDistance))) {
            print("Reconnection failed: occluded", 0);
            return {};
        }

        PathSample r = basePath;
        r.mRadiance = le;
        r.mReplayPdfW = s.pdfW;
        if (rcvFinite) {
            r.mReconnectionVertex.mCos = abs(newRcv.mCos);
            r.mReconnectionVertex.mDist = newRcv.mDist;
        }
        return r;
    }
    PathSample MergeWithReconnectionVertex(inout PathState<true> s, const PathVertex vertex, const float mergeRadius, const PathSample basePath) {
        PathSample result = {};

        const PathVertex photon = PathVertex(basePath.mReconnectionVertex.mVertex);
        const float3 toPhoton = photon.GetPosition() - vertex.GetPosition();
        if (dot(toPhoton, toPhoton) >= sqr(mergeRadius))
            return result;

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, basePath.mReconnectionVertex.mDirOut, false);
        if (vertex.HasFaceNormal()) {
            const float3 guideNormal = vertex.mMaterial.getProperties(vertex.mShadingData).guideNormal;
            const float cosOut = abs(dot(guideNormal, basePath.mReconnectionVertex.mDirOut));
            if (cosOut < kMinCosTheta)
                return result;
            f.mReflectance /= cosOut; // divide out dot(N, wo)
        }
        const float3 contrib = s.throughput * basePath.mReconnectionVertex.mRadiance * f.mReflectance;
        if (all(contrib <= 0))
            return result;

        result = basePath;
        result.mRadiance = contrib;

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
    PathState<bShift> InitializeLightPath<let bShift : bool>(const uint seed, const uint2 id, out PathVertex vertex) {
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

        float cosLight = 0;
        float directPdfA = 0;
        float emissionPdfW = 0;
        Ray ray = {};
        PackedPathVertex l = {};
        vertex.mVertex = {};
        {
            const bool useEnvMap = sampleNext1D(s.sg) < mParams.mEnvMapProb;
            const float4 sampleU = sampleNext4D(s.sg);

            if (gUseEnvLight && (!gUseEmissiveLights || useEnvMap)) {
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
                const uint triangleIndex = mEmissiveSampler.sampleLight(s.sg, selectionPdf);
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
    // Merges a canonical sample into the pixel's reservoir
    void ProcessSample(inout PathReservoir r, inout SampleGenerator sg, const PathSample ps, const float weight) {
        if (gUseResampling) {
            if (r.Merge(sampleNext1D(sg), ps, weight * ps.GetTargetPdf(), 1)) {
                print("Selected path (bounces, replayBounces, shiftType)", uint3(ps.mBounces, ps.mReplayBounces, uint(ps.mShiftType)));
            }
            r.mM = 1;
        } else if (any(ps.mRadiance > 0) && weight > 0) {
            print("Found path (radiance, bounces)", float4(ps.mRadiance, ps.mBounces));
            mOutputRadiance[ps.mRngIdx].rgb += ps.mRadiance * weight;
        }
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
    // ----------------------------------------------- Top-level -----------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathReservoir SampleCameraPath(const uint seed, const uint2 id) {
        PathVertex vertex = PathVertex(LoadPrimaryVertex(id));

        // directly output primary radiance
        mOutputRadiance[id] = float4(vertex.GetEmission(), 1);

        if (gLightTraceOnly || !vertex.IsFinite())
            return {};

        PathState<false> s = InitializeCameraPath<false>(seed, id, vertex);

        PathReservoir r = {};

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
                            PathSample ps = ConnectToLightSubpath(s, vertex, v, weight, misWeight);
                            weight *= v.mIntegrationWeight * (count / (float)mParams.mLightSubpathCount);
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
                            const PathSample ps = MergeWithLightVertex(s, vertex, v, GetMergeRadius(vertex.GetPosition()), weight, misWeight);
                            weight *= v.mIntegrationWeight * cellW;
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
                    ReconnectionVertex rcv = {};
                    if (hasRcv) {
                        // finalize the existing rcv
                        rcv.mVertex = s.rcv;
                        rcv.mCos = s.rcvCos;
                        rcv.mDist = s.rcvDist;
                        if (s.rcvBounces == s.bounces - 1) {
                            rcv.mRadiance = le; // rcv is the last vertex
                        } else {
                            rcv.mDirOut = s.rcvDirOut;
                            rcv.mRadiance = le * float3(
                                                     SafeDivide(s.throughput.r, s.throughputAtRcv.r),
                                                     SafeDivide(s.throughput.g, s.throughputAtRcv.g),
                                                     SafeDivide(s.throughput.b, s.throughputAtRcv.b) );
                        }
                    }

                    const PathSample ps = PathSample(
                        s.throughput * le,
                        hasRcv ? s.pdfAtRcv : s.pdfW,
                        s.bounces,
                        hasRcv ? s.rcvBounces : s.bounces,
                        hasRcv ? s.rcvDiffuseBounces : s.diffuseBounces,
                        s.seed,
                        s.id,
                        rcv,
                        hasRcv ? PathShiftType::eReconnection : PathShiftType::eReplay );

                    ProcessSample(r, s.sg, ps, weight);
                }

            }
        }

        r.FinalizeMerge();
        return r;
    }

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
                    if (weight > 0 && misWeight > 0 && any(ps.mRadiance > 0) && all(pixelCoord >= 0) && all(pixelCoord < mParams.mOutputDim)) {
                        const uint flatId = pixelCoord.y * mParams.mOutputDim.x + pixelCoord.x;
                        if (gUseResampling) {
                            // store path reservoirs for each pixel in a hash map
                            // these reservoirs are merged into each pixel's reservoir in a later pass
                            mLightTraceReservoirs.Insert(flatId, PathReservoir(ps, (weight * misWeight), 1));
                        } else {
                            // splat radiance onto output directly, instead of using reservoirs
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3)                  , ps.mRadiance.r * (weight * misWeight));
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)  , ps.mRadiance.g * (weight * misWeight));
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)*2, ps.mRadiance.b * (weight * misWeight));
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

    PathSample ShiftPath(const PathSample basePath, const PackedPathVertex dstVertex, out float jacobian) {
        jacobian = 0;

        if (!dstVertex.IsFinite() || basePath.mIsCausticPath)
            return {};

        // initialize path state

        PathVertex vertex = PathVertex(dstVertex);
        PathState<true> s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex);

        // trace the new prefix
        while (s.bounces < basePath.mReplayBounces && any(s.throughput > 0)) {
            float bsdfPdfW;
            bool deltaLobe;
            SampleNextVertex(s, vertex, bsdfPdfW, deltaLobe);
        }

        PathSample result = {};
        if (s.bounces != basePath.mReplayBounces || s.diffuseBounces != basePath.mReplayDiffuseBounces || !any(s.throughput > 0))
            return result;

        if (basePath.mShiftType == PathShiftType::eReplay) {
            result = basePath;
            result.mReplayPdfW = s.pdfW;
            result.mRadiance = s.throughput * vertex.GetEmission();
        } else {
            // reconnection strategies
            if (!vertex.IsFinite() || !any(s.throughput > 0)) {
                //print("Shift failed: Prefix mismatch. Replay bounces:", basePath.mReplayBounces);
                return result;
            }

            switch (basePath.mShiftType) {
            default:
                return result;

            case PathShiftType::eReconnection:
                result = ConnectToReconnectionVertex(s, vertex, basePath);
                break;

            case PathShiftType::eMerge: {
                if (gBidirectional && gVertexMerging && !gLightTraceOnly)
                    result = MergeWithReconnectionVertex(s, vertex, GetMergeRadius(vertex.GetPosition()), basePath);
                break;
            }
            }
        }

        if (!result.IsValid())
            return {};

        jacobian = basePath.mReplayPdfW / result.mReplayPdfW;

        if (basePath.mShiftType == PathShiftType::eReconnection) {
            if (basePath.mReconnectionVertex.mVertex.IsFinite())
                jacobian *= (result.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / result.mReconnectionVertex.mDist);
        }

        return result;
    }
};
