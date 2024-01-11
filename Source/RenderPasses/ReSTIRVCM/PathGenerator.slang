#include "Utils/Math/MathConstants.slangh"
#include "BitfieldMacros.slangh"
import Scene.RaytracingInline;
import Rendering.Lights.EmissiveLightSamplerInterface;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Random;
import PhotonMap;
import HashMap;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
__exported import Params;
__exported import PathReservoir;

float SafeDivide(float a, float b) { return abs(b) > 0 ? a/b : 0; }

struct PathGenerator {
    PathGeneratorParams               mParams;

    EnvMapSampler                     mEnvMapSampler;
    EmissiveLightSampler              mEmissiveSampler;

    LightVertexCache                  mLightVertexCache;
    PhotonMap                         mPhotonMap;

    RWByteAddressBuffer               mAtomicRadiance;
    RWTexture2D<float4>               mOutputRadiance;

    RWStructuredBuffer<PathReservoir> mPathReservoirs0;
    RWStructuredBuffer<PathReservoir> mPathReservoirs1;
    HashMap<PathReservoir>            mLightTraceReservoirs;

    Texture2D<PackedHitInfo>          mVbuffer;
    Texture2D<float3>                 mViewDir;
    Texture2D<float3>                 mLastViewDir;
    Texture2D<PackedHitInfo>          mLastVbuffer;
    Texture2D<float2>                 mMotionVectors;

    // Static parameters

    static const bool  gUseEnvLight                 = USE_ENV_LIGHT;
    static const bool  gUseEmissiveLights           = USE_EMISSIVE_LIGHTS;
    static const bool  gUseViewDir                  = USE_VIEW_DIR;
    static const bool  gSampleDirectLight           = USE_NEE;
    static const bool  gBidirectional               = USE_BIDIRECTIONAL;
    static const bool  gVertexMerging               = USE_VERTEX_MERGING;
    static const bool  gProgressivePhotonMapping    = USE_PPM_ONLY;
    static const bool  gLightTraceOnly              = LIGHT_TRACE_ONLY;
    static const bool  gUseLightTraceReservoirs     = LIGHT_TRACE_RESERVOIRS;
    static const bool  gWavefrontTechniqueSelection = WAVEFRONT_TECHNIQUE_SELECTION;
    static const bool  gDebugBPT                    = DEBUG_BPT;
    static const float gMISPowerExponent            = MIS_POWER_EXPONENT;

    #define gDebugDisableMIS (gDebugBPT && mParams.mDebugTotalVertices != -1)

    // ---------------------------------------------------------------------------------------------------------
    // ------------------------------------------- Helper functions --------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    float Mis(float a) {
        //if      (gMISPowerExponent == 1.f)
        //    return abs(a);
        //else if (gMISPowerExponent == 2.f)
            return a * a;
        //else
        //    return pow(abs(a), gMISPowerExponent);
    }
    float Mis(float a, float b) {
        a = Mis(a);
        return SafeDivide(a, a + Mis(b));
    }

    float GetEtaVCM() { return sqr(mParams.mMergeRadius) * M_PI * mParams.mLightSubpathCount; }
    float GetMisVcWeightFactor() { return gVertexMerging ? Mis(1.f / GetEtaVCM()) : 0; }
    float GetMisVmWeightFactor() { return gVertexMerging ? Mis(GetEtaVCM()) : 0; }

    float GetImageToSolidAngleFactor(const float3 dir, out float cosCamera) {
        cosCamera = dot(dir, normalize(gScene.camera.data.cameraW));
        const float imagePlaneDist = mParams.mOutputDim.y * (gScene.camera.data.focalLength / gScene.camera.data.frameHeight);
        const float imagePointToCameraDist = imagePlaneDist / cosCamera;
        return (imagePointToCameraDist * imagePointToCameraDist) / cosCamera;
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
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        if ((mParams.mPathReservoirIndex&1) == 0)
            return mPathReservoirs0[flatIndex];
        else
            return mPathReservoirs1[flatIndex];
    }
    PathReservoir LoadOutputReservoir(const uint2 index) {
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        if ((mParams.mPathReservoirIndex&1) == 0)
            return mPathReservoirs1[flatIndex];
        else
            return mPathReservoirs0[flatIndex];
    }
    void StoreReservoir(const uint2 index, const PathReservoir r) {
        const uint flatIndex = index.y * mParams.mOutputDim.x + index.x;
        if ((mParams.mPathReservoirIndex&1) == 0)
            mPathReservoirs1[flatIndex] = r;
        else
            mPathReservoirs0[flatIndex] = r;
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
                    const TriangleHit triHit = hit.getTriangleHit();
                    const uint triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
                    directPdfA = 1.f / gScene.lightCollection.getTriangle(triangleIndex).area;
                    directPdfA *= mEmissiveSampler.evalTriangleSelectionPdf(prevPos, prevNormal, upperOnly, triangleIndex);

                    if (gUseEnvLight) directPdfA *= 1 - mParams.mEnvMapProb;

                    const float cosIn = vertex.GetCosN(vertex.GetDirIn());
                    emissionPdfW = directPdfA * CosHemispherePdfW(cosIn);
                    directPdfW   = directPdfA * sqr(vertex.mVertex.mRayDistance) / cosIn;
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

        property uint bounces {
            get { return BF_GET(packedBounces, 0, 16); }
            set { BF_SET(packedBounces, newValue, 0, 16); }
        }
        property uint rcvBounces {
            get { return BF_GET(packedBounces, 16, 15); }
            set { BF_SET(packedBounces, newValue, 16, 15); }
        }
        property bool isLightPath {
            get { return (bool)BF_GET(packedBounces, 31, 1); }
            set { BF_SET(packedBounces, (uint)newValue, 31, 1); }
        }
        property bool rcvFound {
            get { return pdfAtRcv > 0; }
        }
        property uint2 id {
            get { return uint2(packedId & 0xFFFF, packedId >> 16); }
            set { packedId = (newValue.y << 16) | newValue.x; }
        }
    };

    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------- Connection stategies ------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    // Sample the BSDF and trace a ray to find the next vertex. Returns a path if the new vertex is emissive.
    PathSample SampleNextVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex, out float weight) {
        const float prevPdf = s.pdfW;
        const bool nonDelta = !vertex.IsDelta();
        const float3 prevPos    = vertex.GetPosition();
        const float3 prevNormal = vertex.GetFaceNormal();
        const bool upperOnly = vertex.HasFaceNormal() && !vertex.HasLobe(LobeType::NonDeltaTransmission);
        const bool prevReconnectable = nonDelta && vertex.GetRoughness() >= mParams.mReconnectionRoughness;

        weight = 0;

        // russian roullette
        const float contProb = vertex.GetContinuationProbability();
        if (!bShift && contProb < 1 && (sampleNext1D(s.sg) > contProb)) {
            s.throughput = 0;
            return {};
        }

        // sample bsdf
        float bsdfPdfW;
        {
            SampleGenerator tmp = SampleGenerator(s.id, s.sgBsdf.next());
            let r = vertex.SampleDirection(tmp, s.isLightPath);
            const float cosOut = vertex.GetCosN(r.mDirection);

            // modify throughput and pdfW
            s.throughput *= r.mReflectance;
            if (r.mFwdPdfW > 0) s.pdfW *= r.mFwdPdfW;
            if (all(s.throughput <= 0)) {
                return {};
            }
            bsdfPdfW = r.mFwdPdfW;

            // update mis quantities
            if (gBidirectional) {
                if (r.mDeltaLobe) {
                    s.dVCM = 0.f;
                    s.dVC *= Mis(cosOut);
                    s.dVM *= Mis(cosOut);
                } else {
                    const float invPdf = 1.f / r.mFwdPdfW;
                    s.dVCM = Mis(invPdf);
                    s.dVC = Mis(cosOut * invPdf) * (s.dVC * Mis(r.mRevPdfW) + s.dVCM + GetMisVmWeightFactor());
                    s.dVM = Mis(cosOut * invPdf) * (s.dVM * Mis(r.mRevPdfW) + s.dVCM * GetMisVcWeightFactor() + 1);
                }
            }

            if (s.isLightPath) {
                // for light paths: rcv is always the 2nd to last vertex
                s.rcvCos = cosOut;
            } else if (!bShift && s.rcvFound && s.rcvBounces == s.bounces-1) {
                // for camera paths: update rcvDirOut and throughputAtRcv if this vertex was selected as the rcv last iteration
                // TODO: clear rcv instead if we sample a non-diffuse lobe
                s.rcvDirOut = r.mDirection;
                s.throughputAtRcv = s.throughput;
            }

            vertex = PathVertex(TraceRay(Ray(vertex.ComputeRayOrigin(r.mDirection), r.mDirection)));
            s.bounces++;
        }

        const float cosIn = vertex.GetCosN(vertex.GetDirIn());

        // update mis quantities
        if (gBidirectional && vertex.IsFinite()) {
            s.dVCM *= Mis(sqr(vertex.mVertex.mRayDistance) / cosIn);
            s.dVC  /= Mis(cosIn);
            s.dVM  /= Mis(cosIn);
        }

        if (s.isLightPath) {
            s.rcvDist = vertex.mVertex.mRayDistance;
        } else {
            const float3 le = vertex.GetEmission();

            if (!bShift && !s.rcvFound) {
                // set last vertex as rcv if one hasn't been found yet.
                const bool reconnectable = vertex.mVertex.mRayDistance >= mParams.mReconnectionDistance && (any(le > 0) || (vertex.IsFinite() && vertex.GetRoughness() >= mParams.mReconnectionRoughness));
                if (prevReconnectable && reconnectable) {
                    s.rcv = vertex.Pack();
                    s.rcvCos = cosIn;
                    s.rcvDist = vertex.mVertex.mRayDistance;
                    s.rcvBounces = s.bounces - 1;
                    s.pdfAtRcv = prevPdf;
                    // rcvDirOut and throughputAtRcv are assigned next iteration after a direction is sampled
                }
            }

            // if vertex is emissive, form a path ending at vertex
            if (any(le > 0)) {
                // calculate mis weight
                float emissionPdfW, directPdfW, directPdfA;
                if (gBidirectional || gSampleDirectLight) {
                    GetLightSamplePdf(vertex, prevPos, prevNormal, upperOnly, emissionPdfW, directPdfW, directPdfA);
                } else {
                    emissionPdfW = 0;
                    directPdfW = 0;
                    directPdfA = 0;
                }

                float misWeight = 1;
                if (nonDelta && !gDebugDisableMIS) {
                    if (gBidirectional) {
                        const float wCamera = Mis(directPdfA) * s.dVCM + Mis(emissionPdfW) * s.dVC;
                        misWeight = 1 / (1 + wCamera);
                    } else if (gSampleDirectLight)
                        misWeight = Mis(bsdfPdfW, directPdfW);
                }

                // divide by pdf
                weight = misWeight / s.pdfW;

                // finalize reconnection vertex
                const bool hasRcv = !bShift && s.rcvFound;
                ReconnectionVertex rcv = {};
                if (hasRcv) {
                    // finalize the existing rcv
                    rcv.mVertex = s.rcv;
                    rcv.mCos    = s.rcvCos;
                    rcv.mDist   = s.rcvDist;
                    if (s.rcvBounces == s.bounces - 1) {
                        rcv.mRadiance = le; // rcv is the last vertex
                    } else {
                        rcv.mDirOut = s.rcvDirOut;
                        rcv.mRadiance = le * float3(
                            SafeDivide(s.throughput.r, s.throughputAtRcv.r),
                            SafeDivide(s.throughput.g, s.throughputAtRcv.g),
                            SafeDivide(s.throughput.b, s.throughputAtRcv.b));
                    }
                }

                return PathSample(
                    s.throughput * le,
                    hasRcv ? s.pdfAtRcv : s.pdfW,
                    s.bounces,
                    hasRcv ? s.rcvBounces : s.bounces,
                    s.seed,
                    s.id,
                    rcv,
                    hasRcv ? PathShiftType::eReconnection : PathShiftType::eReplay,
                    0);
            }
        }

        return {};
    }
    PathSample ConnectToLight<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out float weight) {
        weight = 0;

        PathSample result = {};

        // Sample emissive path vertex

        float directPdfW = 0;
        float emissionPdfW = 0;
        PackedPathVertex l = {};
        float cosLight = 1;
        float3 le = 0;
        {
            const float3 vertexNormal = vertex.GetFaceNormal();
            const bool upperOnly = vertex.HasFaceNormal() && !vertex.HasLobe(LobeType::NonDeltaTransmission);

            const bool useEnvMap = sampleNext1D(s.sg) < mParams.mEnvMapProb;
            const float2 uv = sampleNext2D(s.sg);

            if (gUseEnvLight && (!gUseEmissiveLights || useEnvMap)) {
                EnvMapSample envSample;
                if (!mEnvMapSampler.sample(uv, envSample))
                    return result;

                if (gUseEmissiveLights) envSample.pdf *= mParams.mEnvMapProb;

                l = PackedPathVertex(Ray(vertex.ComputeRayOrigin(envSample.dir), envSample.dir), HitInfo(0), FLT_MAX);
                le = envSample.Le;

                directPdfW = envSample.pdf;
                emissionPdfW = directPdfW * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && !(gUseEnvLight && useEnvMap)) {
                float selectionPdf;
                uint triangleIndex = mEmissiveSampler.sampleLight(vertex.GetPosition(), vertexNormal, upperOnly, s.sg, selectionPdf);
                if (triangleIndex == LightCollection::kInvalidIndex)
                    return result;

                if (gUseEnvLight) selectionPdf *= 1 - mParams.mEnvMapProb;

                let tri = gScene.lightCollection.getTriangle(triangleIndex);
                const float3 barycentrics = sample_triangle(uv);
                const float directPdfA = selectionPdf / tri.area;

                const MeshLightData meshData = gScene.lightCollection.getMeshData(triangleIndex);

                TriangleHit triHit = {};
                triHit.instanceID     = { meshData.instanceID };
                triHit.primitiveIndex = triangleIndex - meshData.triangleOffset;
                triHit.barycentrics = barycentrics.yz;

                const float3 triPos = computeRayOrigin(tri.getPosition(barycentrics), tri.normal);
                const float3 rayOrigin = vertex.ComputeRayOrigin(triPos - vertex.GetPosition());
                float3 toLight = triPos - rayOrigin;
                const float dist = length(toLight);
                toLight /= dist;

                if (dot(tri.normal, toLight) > 0)
                    return result;

                l = PackedPathVertex(Ray(rayOrigin, toLight), HitInfo(triHit), dist);
                le = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(barycentrics));

                const float cosLight = abs(dot(tri.normal, l.mRayDirection));
                directPdfW = directPdfA * sqr(dist) / cosLight;
                emissionPdfW = directPdfA * CosHemispherePdfW(cosLight);
            } else {
                return result;
            }
        }

        if (!any(le > 0))
            return result;

        let f = vertex.EvaluateReflectance(s.sg, l.mRayDirection, false);

        const float3 contrib = s.throughput * f.mReflectance * le;
        if (!any(contrib > 0))
            return {};

        if (Occluded(Ray(l.mRayOrigin, l.mRayDirection, 0, l.mRayDistance)))
            return {};

        // compute mis weight
        if (gDebugDisableMIS) {
            weight = 1;
        } else {
            if (gBidirectional) {
                const float cosOut = vertex.GetCosN(l.mRayDirection);
                const float wLight = Mis(f.mFwdPdfW / directPdfW);
                const float wCamera = Mis(emissionPdfW * abs(cosOut) / (directPdfW * cosLight)) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
                weight = 1 / (wLight + 1 + wCamera);
            } else {
                weight = Mis(directPdfW, f.mFwdPdfW);
            }
        }

        // divide by pdf
        weight /= s.pdfW * directPdfW;

        const bool hasRcv = !bShift && s.rcvFound;
        ReconnectionVertex rcv;
        if (hasRcv) {
            // finalize the existing rcv
            rcv.mVertex = s.rcv;
            rcv.mCos  = s.rcvCos;
            rcv.mDist = s.rcvDist;
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
            s.seed, s.id,
            rcv,
            PathShiftType::eReconnection,
            1);

        return result;
    }
    PathSample ConnectToLightSubpath<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, out float weight) {
        weight = 0;

        PathSample result = {};

        const uint bounces = s.bounces + v.mBounces + 2;
        if (bounces > mParams.mMaxBounces)
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

            const float3 rayOrigin = vertex.ComputeRayOrigin(toLight);
            visibilityRay = Ray(rayOrigin, lightVertex.ComputeRayOrigin(-toLight) - rayOrigin, 0, 0);
            visibilityRay.tMax = length(visibilityRay.dir);
            visibilityRay.dir /= visibilityRay.tMax;

            cosLight = lightVertex.GetCosN(toLight);
            r = lightVertex.EvaluateReflectance(s.sg, -toLight, true);
            r.mReflectance /= cosLight;
        }
        if (all(r.mReflectance <= 0))
            return result;

        // evaluate the bsdf at the camera subpath vertex
        float cosCam = vertex.GetCosN(toLight);
        let f = vertex.EvaluateReflectance(s.sg, toLight, false);
        if (all(f.mReflectance <= 0))
            return result;

        if (Occluded(visibilityRay))
            return result;

        const float3 contrib = (s.throughput * v.mThroughput) * (f.mReflectance * r.mReflectance);

        // Convert pdfs to area measure
        const float invDist2 = 1 / sqr(dist);
        const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
        const float lightBsdfDirPdfA  = r.mFwdPdfW * cosCam   * invDist2;

        // mis weight
        if (gDebugDisableMIS) {
            weight = 1;
        } else {
            const float wLight = Mis(cameraBsdfDirPdfA) * (GetMisVmWeightFactor() + v.dVCM + v.dVC * Mis(r.mRevPdfW));
            const float wCamera = Mis(lightBsdfDirPdfA) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
            weight = 1 / (wLight + 1.f + wCamera);
        }

        weight *= cosLight * invDist2 / s.pdfW;

        const bool hasRcv = !bShift && s.rcvFound;
        ReconnectionVertex rcv;
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
            rcv.mDirOut = -v.GetDirIn();
            rcv.mCos = cosLight;
            rcv.mDist = dist;
        }

        result = PathSample(
            contrib,
            hasRcv ? s.pdfAtRcv : s.pdfW,
            bounces,
            hasRcv ? s.rcvBounces : s.bounces,
            s.seed, s.id,
            rcv,
            PathShiftType::eReconnection,
            v.mBounces + 2);

        return result;
    }
    PathSample MergeWithLightVertex<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, const LightSubpathVertex v, out float weight) {
        weight = 0;

        const uint bounces = s.bounces + v.mBounces + 1;
        if (bounces > mParams.mMaxBounces)
            return {};

        float3 d = PathVertex(v.mVertex).GetPosition() - vertex.GetPosition();
        if (dot(d, d) >= sqr(mParams.mMergeRadius))
            return {};

        ReflectanceEvalData f = vertex.EvaluateReflectance(s.sg, -v.GetDirIn(), false);
        f.mReflectance /= vertex.GetCosN(v.GetDirIn()); // divide out dot(N, wo)

        const float wLight  = v.dVCM * GetMisVcWeightFactor() + v.dVM * Mis(f.mFwdPdfW);
        const float wCamera = s.dVCM * GetMisVcWeightFactor() + s.dVM * Mis(f.mRevPdfW);
        const float misWeight = (gDebugDisableMIS || gProgressivePhotonMapping) ? 1.f : 1.f / (wLight + 1.f + wCamera);

        weight = misWeight / (s.pdfW * GetEtaVCM());

        assert(LightSubpathVertexSize == ReconnectionVertexSize);

        return PathSample(
            s.throughput * v.mThroughput * f.mReflectance,
            s.pdfW,
            bounces, s.bounces,
            s.seed, s.id,
            reinterpret<ReconnectionVertex>(v),
            PathShiftType::eMerge,
            v.mBounces + 2);
    }
    PathSample ConnectToCamera<let bShift : bool>(inout PathState<bShift> s, const PathVertex vertex, out int2 pixelCoord, out float weight) {
        weight = 0;
        pixelCoord = -1;

        PathSample result = {};

        float3 toCamera = gScene.camera.getPosition() - vertex.GetPosition();
        const float dist = length(toCamera);
        toCamera /= dist;

        float cosAtCamera;
        const float imageToSolidAngleFactor = GetImageToSolidAngleFactor(-toCamera, cosAtCamera);

        float4 clip = mul(gScene.camera.data.viewProjMat, float4(vertex.GetPosition(), 1));
        clip.y = -clip.y;
        pixelCoord = int2(((clip.xy / clip.w) * .5 + .5) * mParams.mOutputDim);
        if (any(pixelCoord < 0) || any(pixelCoord >= mParams.mOutputDim) || cosAtCamera <= 0)
            return result;

        if (gDebugBPT && (mParams.mDebugTotalVertices != s.bounces + 3 || mParams.mDebugLightVertices != s.bounces + 2))
            return result;

        const float cosOut = vertex.GetCosN(toCamera);

        let f = bShift ?
            vertex.Reversed(toCamera).EvaluateReflectance(s.sg, -vertex.GetDirIn(), false) :
            vertex.EvaluateReflectance(s.sg, toCamera, true);

        // Compute pdf conversion factor from image plane area to surface area
        const float cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);

        const float3 contrib = s.throughput * (bShift ? f.mReflectance : (f.mReflectance / cosOut));
        if (!any(contrib > 0))
            return result;

        const float samplePdf = s.pdfW * (1 / cameraPdfA) * mParams.mLightSubpathCount;
        if (!(samplePdf > 0))
            return result;

        if (Occluded(vertex, toCamera, dist))
            return result;

        const float wLight = Mis(cameraPdfA / mParams.mLightSubpathCount) * (GetMisVmWeightFactor() + s.dVCM + s.dVC * Mis(f.mRevPdfW));
        const float misWeight = (gLightTraceOnly || gDebugDisableMIS) ? 1.f : 1.f / (wLight + 1.f);

        weight = misWeight / samplePdf;

        result = PathSample(
            contrib,
            1,
            s.bounces + 1, max(s.bounces,2)-2,
            s.seed, s.id,
            {},
            PathShiftType::eLightPath,
            s.bounces + 2);

        if (!bShift) {
            result.mReconnectionVertex.mVertex = s.rcv;
            result.mReconnectionVertex.mRadiance = s.throughputAtRcv;
            result.mReconnectionVertex.mDirOut = s.rcvDirOut;
            result.mReconnectionVertex.mDist = 1;
            result.mReconnectionVertex.mCos = 1;
            result.mShiftType = PathShiftType::eReconnection;
        }

        return result;
    }
    PathSample ConnectToBasePath(inout PathState<true> s, const PathVertex vertex, const PathSample basePath) {
        if (vertex.IsDelta() || vertex.GetRoughness() < mParams.mReconnectionRoughness)
            return {};

        const PathVertex baseRcv = PathVertex(basePath.mReconnectionVertex.mVertex);

        float3 toRcv;
        float dist;
        if (baseRcv.IsFinite()) {
            toRcv = baseRcv.GetPosition() - vertex.ComputeRayOrigin(baseRcv.GetPosition() - vertex.GetPosition());
            dist = length(toRcv);
            toRcv /= dist;
        } else {
            toRcv = baseRcv.GetDirIn();
            dist = FLT_MAX;
        }

        if (dist < mParams.mReconnectionDistance)
            return {};

        float3 le = s.throughput * basePath.mReconnectionVertex.mRadiance;
        if (!any(le > 0)) {
            print("Reconnection failed: zero le", le);
            return {};
        }

        le *= vertex.EvaluateReflectance(s.sg, toRcv, false).mReflectance;
        if (!any(le > 0)) {
            print("Reconnection failed: zero vertex reflectance", dot(toRcv, vertex.GetFaceNormal()));
            return {};
        }

        const float cosLight = baseRcv.HasFaceNormal() ? dot(baseRcv.GetFaceNormal(), -toRcv) : 1;

        // Evaluate reflectance if reconnection vertex is not the last vertex
        if (basePath.mReplayBounces + 1 < basePath.mBounces) {
            le *= baseRcv.EvaluateReflectance(s.sg, basePath.mReconnectionVertex.mDirOut, false).mReflectance;
        } else {
            // triangle lights are one-sided
            if (cosLight <= 0) {
                print("Reconnection failed: backfacing light", cosLight);
                return {};
            }
        }

        if (!any(le > 0)) {
            print("Reconnection failed: zero rcv reflectance", le);
            return {};
        }

        Ray visibilityRay = Ray(vertex.ComputeRayOrigin(toRcv), toRcv, 0, dist);
        if (baseRcv.IsFinite()) {
            visibilityRay.dir = baseRcv.ComputeRayOrigin(-toRcv) - visibilityRay.origin;
            visibilityRay.tMax = length(visibilityRay.dir);
            visibilityRay.dir /= visibilityRay.tMax;
        }
        if (Occluded(visibilityRay)) {
            print("Reconnection failed: occluded", 0);
            return {};
        }
        PathSample ps = basePath;
        ps.mRadiance = le;
        ps.mReplayPdfW = s.pdfW;
        ps.mReconnectionVertex.mCos = abs(cosLight);
        ps.mReconnectionVertex.mDist = dist;
        return ps;
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
        s.sgBsdf = SampleGenerator(s.id, s.sg.next());

        s.bounces = 0;
        s.rcvBounces = 0;
        s.throughput = 1;
        s.pdfW = 1;
        if (gBidirectional) {
            float cosAtCamera;
            const float cameraPdfW = GetImageToSolidAngleFactor(vertex.GetDirIn(), cosAtCamera);
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

        s.sg     = SampleGenerator(s.id, s.seed);
        s.sgBsdf = SampleGenerator(s.id, s.sg.next());

        s.throughput = 0;

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
                branchlessONB(ray.dir, t, b);
                const float2 uv = sample_disk_concentric(sampleU.zw);

                ray = Ray(mParams.mSceneSphere.xyz + mParams.mSceneSphere.w * (t * uv.x + b * uv.y - envSample.dir), -envSample.dir);
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
        s.dVCM = Mis(directPdfA / emissionPdfW);
        s.dVC = finiteEmitter ? Mis(cosLight / emissionPdfW) : 0;
        s.dVM = s.dVC * GetMisVcWeightFactor();

        // set initial vertex as rcv
        s.rcv = l;
        s.throughputAtRcv = s.throughput;
        s.rcvBounces = 0;
        s.rcvCos = cosLight;

        vertex = PathVertex(TraceRay(ray));
        if (vertex.IsFinite()) {
            const float cosIn = vertex.GetCosN(ray.dir);
            if (finiteEmitter)
                s.dVCM *= Mis(sqr(vertex.mVertex.mRayDistance));
            s.dVCM /= Mis(cosIn);
            s.dVC  /= Mis(cosIn);
            s.dVM  /= Mis(cosIn);
            s.rcvDist = vertex.mVertex.mRayDistance;
        } else
            s.throughput = 0;

        return s;
    }
    // Merges a canonical sample into the pixel's reservoir
    void MergeSample(inout PathReservoir r, const float mergeRnd, const PathSample ps, const float weight) {
        if (gDebugBPT &&
                (mParams.mDebugTotalVertices != -1 && mParams.mDebugTotalVertices != ps.mBounces+2) ||
                (mParams.mDebugLightVertices != -1 && mParams.mDebugLightVertices != ps.mLightVertices))
            return;

        r.Merge(mergeRnd, ps, weight * ps.GetTargetPdf(), 1);
        r.mM = 1;
    }

    // ---------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- Top-level -----------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    PathReservoir SampleCameraPath(const uint seed, const uint2 id) {
        PathVertex vertex = PathVertex(LoadPrimaryVertex(id));

        // directly output primary radiance
        mOutputRadiance[id] = float4(vertex.GetEmission(), 1);

        PathReservoir r = {};
        if (!vertex.IsFinite())
            return r;

        PathState<false> s = InitializeCameraPath<false>(seed, id, vertex);

        while (s.bounces + 1 <= mParams.mMaxBounces && any(s.throughput > 0)) {
            if (!vertex.IsDelta()) {
                bool doDI = (gBidirectional || gSampleDirectLight) && !gProgressivePhotonMapping;
                bool doVC = gBidirectional && !gProgressivePhotonMapping;
                bool doVM = gBidirectional && gVertexMerging;
                float techniqueW = 1;
                /*
                if (gBidirectional && gWavefrontTechniqueSelection && !gProgressivePhotonMapping) {
                    float vcRnd = sampleNext1D(s.sg);
                    vcRnd = WaveReadLaneFirst(vcRnd);

                    bool doDI = false;
                    bool doVC = false;
                    bool doVM = false;

                    // pick between direct light, vertex connection, and vertex merging.
                    const float vcProb = gVertexMerging ? (1 - mParams.mDirectLightProb) * mParams.mMergeProb : (1 - mParams.mDirectLightProb);
                    if (vcRnd < mParams.mDirectLightProb) {
                        doDI = true;
                        techniqueW = 1 / mParams.mDirectLightProb;
                    } else if (vcRnd < mParams.mDirectLightProb + vcProb) {
                        doVC = true;
                        techniqueW = 1 / vcProb;
                    } else if (gVertexMerging) {
                        doVM = true;
                        techniqueW = 1 / mParams.mMergeProb;
                    }
                    // compute weight for the selecting the technique
                    if (doVC)
                        vcRnd /= vcProb;
                }
                */

                if (doDI) {
                    float weight = 0;
                    const PathSample ps = ConnectToLight(s, vertex, weight);
                    const float u = sampleNext1D(s.sg);
                    MergeSample(r, u, ps, weight * techniqueW);
                }

                if (gBidirectional) {
                    if (doVC) {
                        // connect to a random light subpath vertex in the LVC
                        const uint count = mLightVertexCache.size();
                        if (count > 0) {
                            const LightSubpathVertex v = mLightVertexCache[min(uint(sampleNext1D(s.sg) * count), count - 1)];

                            float weight;
                            PathSample ps = ConnectToLightSubpath(s, vertex, v, weight);
                            weight *= v.mIntegrationWeight * (count / (float)mParams.mLightSubpathCount);
                            MergeSample(r, sampleNext1D(s.sg), ps, weight * techniqueW);
                        }
                    }

                    if (gVertexMerging && doVM) {
                        // merge with a random vertex from the 8 nearest grid cells
                        uint cell = -1;
                        float cellW = 0.f;

                        // Iterate the 8 neighbor grid cells and merge with the vertices in them.
                        // pick one random grid cell to merge with, with probability proportional to its count
                        const int3 o = int3(lerp(
                            float3(0, 0, 0),
                            float3(-1, -1, -1),
                            float3(frac(vertex.GetPosition() / (2.f * mParams.mMergeRadius)) < 0.5f) ));

                        for (int x = 0; x < 2; x++) {
                            for (int y = 0; y < 2; y++) {
                                for (int z = 0; z < 2; z++) {
                                    const uint cellIndex = mPhotonMap.getCellIndex(vertex.GetPosition(), mParams.mMergeRadius, mParams.mPhotonCellCount, int3(x, y, z) + o);
                                    if (cellIndex == -1)
                                        continue;

                                    const uint count = mPhotonMap.getCellPhotonCount(cellIndex);
                                    if (count == 0)
                                        continue;

                                    cellW += count;
                                    if (sampleNext1D(s.sg) * cellW < count)
                                        cell = cellIndex;
                                }
                            }
                        }

                        // merge with the photon in the selected cell
                        if (cell != -1) {
                            float lightW;
                            PathSample ps = MergeWithLightVertex(s, vertex, mLightVertexCache[mPhotonMap.getCellOffset(cell)], lightW);
                            MergeSample(r, sampleNext1D(s.sg), ps, techniqueW * cellW * lightW);
                        }

                        if (gProgressivePhotonMapping)
                            break;
                    }
                }
            }

            {
                float weight;
                const PathSample ps = SampleNextVertex(s, vertex, weight);
                const float u = sampleNext1D(s.sg);
                MergeSample(r, u, ps, weight);
            }
        }

        r.FinalizeMerge();
        return r;
    }

    void SampleLightPath(const uint seed, const uint2 id) {
        PathVertex vertex;
        PathState<false> s = InitializeLightPath<false>(seed, id, vertex);

        while (any(s.throughput > 0)) {
            if (!vertex.IsDelta()) {
                // form a full path by connecting the vertex to the camera
                if (!gProgressivePhotonMapping) {
                    float weight;
                    int2 pixelCoord;
                    const PathSample ps = ConnectToCamera(s, vertex, pixelCoord, weight);
                    if (weight > 0 && all(pixelCoord >= 0) && all(pixelCoord < mParams.mOutputDim)) {
                        const uint flatId = pixelCoord.y * mParams.mOutputDim.x + pixelCoord.x;
                        if (gUseLightTraceReservoirs) {
                            mLightTraceReservoirs.Insert(flatId, PathReservoir(ps, weight, 1));
                        } else {
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3)                  , ps.mRadiance.r * weight);
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)  , ps.mRadiance.g * weight);
                            mAtomicRadiance.InterlockedAddF32(flatId * sizeof(float3) + sizeof(float)*2, ps.mRadiance.b * weight);
                        }
                    }
                }

                if (!gLightTraceOnly) {
                    // store vertex
                    LightSubpathVertex v;
                    v.mVertex = vertex.Pack();
                    v.mThroughput = s.throughput;
                    v.dVC  = s.dVC;
                    v.dVM  = s.dVM;
                    v.dVCM = s.dVCM;
                    v.mIntegrationWeight = 1/s.pdfW;
                    v.mRngIdx  = s.id;
                    v.mBounces = s.bounces;

                    const uint i = mLightVertexCache.addVertex(v);
                    if (gVertexMerging)
                        mPhotonMap.addPhoton(i, vertex.GetPosition(), mParams.mMergeRadius, mParams.mPhotonCellCount, sampleNext1D(s.sg));
                }
            }

            if (s.bounces + 1 == mParams.mMaxBounces)
                break;

            s.rcv = vertex.Pack();
            s.throughputAtRcv = s.throughput;
            s.rcvBounces = 0;
            s.rcvDirOut = -vertex.GetDirIn();

            float weight;
            SampleNextVertex(s, vertex, weight);
        }
    }

    PathSample ShiftPath(const PathSample basePath, const PackedPathVertex dstVertex, out float jacobian) {
        jacobian = 0;

        if (gBidirectional && gLightTraceOnly && !gUseLightTraceReservoirs)
            return {}; // only atomic radiance is used in this case

        if (!dstVertex.IsFinite())
            return {};

        // initialize path state

        PathVertex vertex;
        PathState<true> s;
        if (gBidirectional && gUseLightTraceReservoirs && basePath.mShiftType == PathShiftType::eLightPath) {
            // pure light path
            s = InitializeLightPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex);
            if (!any(s.throughput > 0))
                return {};
        } else {
            // camera path
            vertex = PathVertex(dstVertex);
            s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex);
        }

        PathSample result = {};

        // trace the new prefix
        while (s.bounces < basePath.mReplayBounces && any(s.throughput > 0)) {
            float weight;
            result = SampleNextVertex(s, vertex, weight);
        }

        // reconnection strategies
        if (basePath.mShiftType != PathShiftType::eReplay) {
            if (!vertex.IsFinite() || !any(s.throughput > 0)) {
                print("Shift failed: Prefix mismatch. Replay bounces:", basePath.mReplayBounces);
                return {};
            }

            switch (basePath.mShiftType) {
            default:
                return {};

            case PathShiftType::eReconnection:
                result = ConnectToBasePath(s, vertex, basePath);
                break;

            case PathShiftType::eMerge: {
                if (!gBidirectional || !gVertexMerging || gLightTraceOnly)
                    break;

                assert(LightSubpathVertexSize == ReconnectionVertexSize);

                float weight;
                result = MergeWithLightVertex(s, vertex, reinterpret<LightSubpathVertex>(basePath.mReconnectionVertex), weight);
                break;
            }

            case PathShiftType::eLightPath: {
                if (!gBidirectional || !gUseLightTraceReservoirs)
                    break;

                // connect vertex to dstVertex

                PathVertex dstVertexUnpacked = PathVertex(dstVertex);

                float3 toCameraVertex = dstVertexUnpacked.GetPosition() - vertex.GetPosition();
                const float dist = length(toCameraVertex);
                toCameraVertex /= dist;

                if (basePath.mBounces == 1) {
                    s.throughput *= vertex.GetCosN(toCameraVertex);
                } else {
                    s.throughput *= vertex.EvaluateReflectance(s.sg, toCameraVertex, true).mReflectance;
                    if (!any(s.throughput > 0)) {
                        print("Zero vertex reflectance\n", 0);
                        return {};
                    }
                    s.bounces++;
                }

                if (Occluded(vertex, toCameraVertex, dist)) {
                    print("Camera vertex occluded\n", 0);
                    return {};
                }

                // connect dstVertex to camera

                float weight;
                int2 pixelCoord;
                result = ConnectToCamera(s, dstVertexUnpacked, pixelCoord, weight);
                break;
            }
            }
        }

        if (!any(result.mRadiance > 0))
            return {};

        jacobian = basePath.mReplayPdfW / result.mReplayPdfW;

        if (basePath.mShiftType == PathShiftType::eReconnection) {
            if (basePath.mReconnectionVertex.mVertex.IsFinite())
                jacobian *= (result.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / result.mReconnectionVertex.mDist);
        }

        return result;
    }
};

ParameterBlock<PathGenerator> gPathGenerator;
