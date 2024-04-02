#include "Utils/Math/MathConstants.slangh"
import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
__exported import PathGenerator;
import PhotonMap;

extension PathGenerator {
    // shift a 1-bounce light subpath to a new primary hit. Copied from InitializeLightPath
    PathState<true> ShiftLightSubpath(const LightSubpathData basePath, const PackedPathVertex newPrimaryHit, out PathVertex vertex) {
        PathState<true> s = {};
        s.id = basePath.mId;
        s.isLightPath = true;

        s.throughput = 0;
        s.pdfW = 0;
        s.throughputAtRcv = 0;
        s.rcvDirOut = 0;

        s.sg     = MakeSampleGenerator(s.id.mId, s.id.mSeed);
        s.sgBsdf = MakeSampleGenerator(s.id.mId, -s.id.mSeed);

        PathVertex newVertex = PathVertex(newPrimaryHit);

        vertex.mVertex = {};

        float cosLight = 0;
        float directPdfA = 0;
        float emissionPdfW = 0;
        Ray ray = {};
        PackedPathVertex l = {};
        {
            const bool useEnvMap = sampleNext1D(s.sg) < mParams.mEnvMapProb;
            const float4 sampleU = sampleNext4D(s.sg);

            if (gUseEnvLight && !HitInfo(basePath.mFirstVertex).isValid()) {
                EnvMapSample envSample;
                if (!mEnvMapSampler.sample(sampleU.xy, envSample))
                    return s;

                if (gUseEmissiveLights) envSample.pdf *= mParams.mEnvMapProb;

                ray = Ray(newVertex.GetPosition() + mParams.mSceneSphere.w * envSample.dir, -envSample.dir);
                l = PackedPathVertex(ray, HitInfo(0), FLT_MAX);
                s.throughput = envSample.Le;
                cosLight = 1;

                directPdfA = envSample.pdf;
                emissionPdfW = directPdfA * gConcentricDiscPdfA / sqr(mParams.mSceneSphere.w);
            } else if (gUseEmissiveLights && HitInfo(basePath.mFirstVertex).isValid()) {
                const TriangleHit triHit = HitInfo(basePath.mFirstVertex).getTriangleHit();
                const uint triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
                if (triangleIndex == LightCollection::kInvalidIndex) {
                    print("triangleIndex == LightCollection::kInvalidIndex", triangleIndex);
                    return s;
                }
                float selectionPdf = mEmissiveSampler.evalTriangleSelectionPdf(triangleIndex);

                if (gUseEnvLight) selectionPdf *= 1 - mParams.mEnvMapProb;

                let tri = gScene.lightCollection.getTriangle(triangleIndex);

                directPdfA = selectionPdf / tri.area;

                const float3 barycentrics = float3(1 - triHit.barycentrics.x - triHit.barycentrics.y, triHit.barycentrics);
                const float3 rayOrigin = computeRayOrigin(tri.getPosition(barycentrics), tri.normal);

                float3 rayDir = newVertex.ComputeRayOrigin(rayOrigin - newVertex.GetPosition()) - rayOrigin;
                const float dist = length(rayDir);
                rayDir /= dist;

                cosLight = dot(tri.normal, rayDir);
                if (cosLight < kMinCosTheta) {
                    print("cosLight < kMinCosTheta", cosLight);
                    return s;
                }

                float dirPdfW = CosHemispherePdfW(cosLight);

                ray = Ray(rayOrigin, rayDir, 0, dist);

                l = PackedPathVertex(ray, HitInfo(triHit), dist);

                s.throughput = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(barycentrics));

                emissionPdfW = directPdfA * dirPdfW;
            } else {
                return s;
            }
        }

        if (Occluded(ray)) {
            print("Occluded", 0);
            s.Terminate();
            return s;
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
        s.throughputAtRcv = s.throughput;
        s.pdfAtRcv = directPdfA;
        s.rcvCos = cosLight;

        vertex = PathVertex(PackedPathVertex(ray, HitInfo(newPrimaryHit.mPackedHit), ray.tMax));
        if (vertex.IsFinite()) {
            const float cosIn = vertex.GetCosN(ray.dir);
            if (cosIn < kMinCosTheta) {
                print("cosIn < kMinCosTheta", cosIn);
                s.Terminate();
                return s;
            }

            s.mis.ProcessRay(vertex.mVertex, cosIn, finiteEmitter);

            s.rcvDist = vertex.mVertex.mRayDistance;
        } else
            s.Terminate();

        return s;
    }

    PathState<true> ShiftLightSubpath(const LightSubpathData basePath, out PathVertex vertex) {
        PathState<true> s = InitializeLightPath<true>(basePath.mId, vertex, basePath.mFirstVertex);

        while (any(s.throughput > 0) && s.pdfW > 0 && s.bounces + 2 < basePath.mVertexCount && vertex.IsFinite()) {
            // for light paths: rcv is always the 2nd to last vertex
            s.rcv = vertex.Pack();
            s.rcvDirOut = vertex.GetDirIn();
            s.throughputAtRcv = s.throughput;
            s.pdfAtRcv = s.pdfW;
            s.prefixPdf = s.pdfW;
            s.prefixBounces = 0;
            s.rcvDiffuseBounces = 0;

            const bool connectable = vertex.IsConnectable();
            const float3 prevN = vertex.GetFaceNormal();

            const uint lobe = SampleNextVertex(s, vertex);

            s.rcvCos = abs(dot(prevN, vertex.mVertex.mRayDirection));
            s.rcvDist = vertex.mVertex.mRayDistance;
            s.deltaRcv = !connectable || ((lobe & uint(LobeType::Delta)) != 0);
        }

        return s;
    }

    // Returns the path state at the end of the prefix
    PathState<true> ShiftPrefix(const PathSample basePath, inout PathVertex vertex) {
        // initialize path state
        PathState<true> s = InitializeCameraPath<true>(basePath.mId, vertex);
        s.lightSubpathVertex = basePath.mSuffix.mLightSubpath.mFirstVertex;
        s.lightSubpathId     = basePath.mSuffix.mLightSubpath.mId;

        // trace the new prefix
        while (s.bounces < basePath.mPrefixBounces && any(s.throughput > 0)) {
            SampleNextVertex(s, vertex);
        }
        if (s.bounces != basePath.mPrefixBounces || s.diffuseBounces != basePath.mPrefixDiffuseBounces || !any(s.throughput > 0)) {
            print("Shift failed: Prefix length mismatch:", uint2(s.bounces, basePath.mPrefixBounces));
            s.Terminate();
            return s;
        }

        return s;
    }

    // Recomputes the suffix vertex's incoming ray based on vertex. Does not update the suffix's contribution.
    PathSuffix GetShiftedSuffixVertex(const PathVertex vertex, const PathSuffix baseSuffix, const bool lightPath = false) {
        const PathVertex suffixVertex = PathVertex(baseSuffix.mVertex);

        PathSuffix result = baseSuffix;

        if (suffixVertex.IsFinite()) {
            // update ray direction
            result.mVertex.mRayDirection = suffixVertex.GetPosition() - vertex.GetPosition();
            // recompute origin and distance
            result.mVertex.mRayOrigin = vertex.ComputeRayOrigin(result.mVertex.mRayDirection);
            result.mVertex.mRayDirection = suffixVertex.ComputeRayOrigin(-result.mVertex.mRayDirection) - result.mVertex.mRayOrigin;
            result.mVertex.mRayDistance = length(result.mVertex.mRayDirection);
            result.mVertex.mRayDirection /= result.mVertex.mRayDistance;

            result.mCos = suffixVertex.GetCosN(result.mVertex.mRayDirection);
            result.mDist = result.mVertex.mRayDistance;
        } else
            result.mVertex.mRayOrigin = vertex.ComputeRayOrigin(result.mVertex.mRayDirection);

        return result;
    }

    // connect to suffix without shifting it. Assumes the path state is already at the end of the prefix
    PathSample ConnectToSuffix(inout PathState<true> s, inout PathVertex vertex, const PathSample basePath) {
        const uint  cameraBounces = basePath.mBounces - (basePath.mSuffix.mLightSubpath.mVertexCount + (gVertexMerging && basePath.HasFlag(PathFlags::eVertexMerging) ? 1 : 0));
        const float prefixPdf = s.pdfW;

        PathSample result = {};

        PathSuffix newSuffix = GetShiftedSuffixVertex(vertex, basePath.mSuffix);

        if (gBidirectional && !gDisableCameraConnection && basePath.HasFlag(PathFlags::eCausticLightPath) && !gUseCausticReservoirs) {
            // just copy caustic paths if they are in the same pixel (or close)
            const float2 basePixel = GetPixelCoord(PathVertex(newSuffix.mVertex).GetPosition());
            const float2 d = GetPixelCoord(vertex.GetPosition()) - basePixel;
            const float d2 = dot(d, d);
            const float r = mParams.mCausticReuseRadius;
            if ((r == 0 && any(d != 0.f)) || (r > 0 && d2 > r * r)) {
                print("Shift failed: Caustic path not in pixel", basePixel);
                return result;
            }
            result = basePath;
            return result;
        }

        if (newSuffix.mVertex.IsFinite()) {
            if (newSuffix.mCos < kMinCosTheta)
                return result;
            if (s.bounces < cameraBounces && newSuffix.mDist < mParams.mReconnectionDistance)
                return result;
        }

        // advance to the suffix reconnection vertex

        RecursiveMisState mis = s.mis; // cache mis state at end of prefix
        float prefixFwdPdfW, prefixRevPdfW;
        {
            const PathVertex rcv = PathVertex(newSuffix.mVertex);
            if (Occluded(Ray(rcv.mVertex.mRayOrigin, rcv.mVertex.mRayDirection, 0, rcv.mVertex.mRayDistance))) {
                print("Shift failed: suffix occluded", 0);
                return result;
            }
            SetNextVertex(s, vertex, rcv);
            prefixFwdPdfW = s.lastVertex.scatterFwdPdfW;
            prefixRevPdfW = s.lastVertex.scatterRevPdfW;
            if (!any(s.throughput > 0)) {
                print("Shift failed: Zero reconnection reflectance", 0);
                return result;
            }
        }

        ReflectanceEvalData rcvF = {};
        if (s.bounces == basePath.mBounces) {
            newSuffix.mContribution = vertex.GetEmission();
        } else {
            rcvF = vertex.EvaluateReflectance(s.sg, newSuffix.mDirOut, false);
            s.throughput *= rcvF.mReflectance;
        }

        result = basePath;
        result.mContribution = s.throughput * newSuffix.mContribution;
        result.mPrefixPdf = prefixPdf;
        result.mSuffix = newSuffix;

        // compute new MIS weight for the shifted path
        if (PathSample::gIntegrandMIS && !DebugDisableMIS() && gUseReconnectionMIS) {
            if (!gDisableCameraConnection && gBidirectional && basePath.HasFlag(PathFlags::eNonCausticLightPath)) {
                // non-caustic pure light path. caustic light paths are handled above.

                float3 toCamera = gScene.camera.getPosition() - s.lastVertex.position;
                const float distToCamera = length(toCamera);
                toCamera /= distToCamera;

                float cosAtCamera;
                const float imageToSolidAngleFactor = GetImageToSolidAngleFactor(-toCamera, cosAtCamera);
                const float cosToCamera = abs(dot(s.lastVertex.normal, toCamera));

                // Compute pdf conversion factor from image plane area to surface area
                const float cameraPdfA = imageToSolidAngleFactor * cosToCamera / sqr(distToCamera);

                // advance the cached MIS state to 's.lastVertex', which is the primary hit
                mis = basePath.mSuffix.mLightSubpathMis;
                {
                    // swap forward and reverse PDFs since light tracing would have evaluated the reverse BSDF
                    DirectionSampleData r = {};
                    r.mFwdPdfW = prefixRevPdfW;
                    r.mRevPdfW = prefixFwdPdfW;
                    mis.ProcessDirection(r, vertex.GetCosN(vertex.GetDirIn()), true);
                    PackedPathVertex v = {};
                    v.mRayDistance = vertex.mVertex.mRayDistance;
                    mis.ProcessRay(v, abs(dot(s.lastVertex.normal, vertex.GetDirIn())), false);
                }

                result.mMisWeight = mis.EvalLightTraceMIS(prefixFwdPdfW, cameraPdfA);
            } else {
                #if 0
                ReconnectionMisData d = basePath.mSuffix.mReconnectionMis;

                // at this point, the path state has been advanced to the reconnection vertex

                ReflectanceEvalData cameraF = {};
                ReflectanceEvalData lightF = {};

                if (s.bounces == basePath.mBounces) {
                    // rcv is the last vertex in the path.
                    d.cosCam = s.lastVertex.HasNormal() ? abs(dot(s.lastVertex.normal, vertex.GetDirIn())) : 1;
                    d.cosLight = vertex.GetCosN(vertex.GetDirIn());
                    GetLightSamplePdf( vertex, s.lastVertex.position, s.lastVertex.normal, s.lastVertex.upperHemisphereOnly, d.emissionPdfW, d.directPdfW, d.directPdfA );
                    if (basePath.mSuffix.mLightSubpath.mVertexCount == 0)
                        mis = s.mis;
                    cameraF = rcvF;
                } else if (s.bounces < cameraBounces) {
                    // advance mis state to the end of the camera subpath
                    mis.dVC = coeff * (
                        mis.dVC * Mis(r.mRevPdfW) * fullProd +
                        sumProd_dVCM +
                        GetMisVmWeightFactor() * sumProd_eta);
                    r = ;
                    f = ;
                }  else if (s.bounces == cameraBounces) {
                    // rcv is last camera subpath vertex. don't need to advance the mis state.
                    mis = s.mis;
                    r = f; // use reflectance at rcv for camera connection
                    cameraF = { 0, prefixFwdPdfW, prefixRevPdfW }
                } else if (s.bounces > cameraBounces) {
                    // rcv is on the light subpath. don't need to advance the mis state.
                    if (basePath.mSuffix.mLightSubpath.mVertexCount > 1) {
                        // swap pdfs to get adjoint pdfs
                        float tmp = f.mRevPdfW;
                        f.mRevPdfW = f.mFwdPdfW;
                        f.mFwdPdfW = tmp;
                    }
                }

                if (basePath.mSuffix.mLightSubpath.mVertexCount == 0) {
                    result.mMisWeight = mis.EvalCameraTraceMIS(prefixFwdPdfW, d.directPdfA, d.directPdfW, d.emissionPdfW);
                } else if (basePath.mSuffix.mLightSubpath.mVertexCount == 1) {
                    result.mMisWeight = mis.EvalDirectLightMIS(d.directPdfW, d.emissionPdfW, cameraF, d.cosCam, d.cosLight);
                } else {
                    if (basePath.HasFlag(PathFlags::eVertexMerging)) {
                        result.mMisWeight = mis.EvalVertexMergingMIS(basePath.mSuffix.mLightSubpathMis, cameraF);
                    } else {
                        result.mMisWeight = mis.EvalVertexConnectionMIS(basePath.mSuffix.mLightSubpathMis, lightF, cameraF, d.cosCam, d.cosLight, d.invDist2);
                    }
                }
                #endif
            }
        }

        return result;
    }

    // Shift and connect to a suffix. Assumes the path state is already at the end of the prefix
    PathSample ShiftSuffix(inout PathState<true> s, inout PathVertex vertex, const PathSample basePath, const LightSubpathVertex shiftedLightVertex) {
        PathSample result = {};

        const uint cameraBounces = basePath.mBounces - (basePath.mSuffix.mLightSubpath.mVertexCount + (gVertexMerging && basePath.HasFlag(PathFlags::eVertexMerging) ? 1 : 0));

        // advance to the 1st suffix vertex (the reconnection vertex)
        if (s.bounces < cameraBounces) {
            const PathSuffix newSuffix = GetShiftedSuffixVertex(vertex, basePath.mSuffix);
            if (newSuffix.mVertex.IsFinite() && (newSuffix.mDist < mParams.mReconnectionDistance || newSuffix.mCos < kMinCosTheta))
                return result;
            const PathVertex rcv = PathVertex(newSuffix.mVertex);
            if (Occluded(Ray(rcv.mVertex.mRayOrigin, rcv.mVertex.mRayDirection, 0, rcv.mVertex.mRayDistance)))
                return result;
            SetNextVertex(s, vertex, rcv);
            if (!any(s.throughput > 0))
                return result;
        }

        // advance to the 2nd suffix vertex
        if (s.bounces < cameraBounces) {
            SetNextVertex(s, vertex, PathVertex(TraceRay(Ray(vertex.ComputeRayOrigin(basePath.mSuffix.mDirOut), basePath.mSuffix.mDirOut))));
            if (!any(s.throughput > 0))
                return result;
        }

        // advance to the last camera vertex
        while (s.bounces < cameraBounces && any(s.throughput > 0) && vertex.IsFinite()) {
            SampleNextVertex(s, vertex);
        }
        if (s.bounces != cameraBounces || !any(s.throughput > 0)) {
            print("Shift failed: Camera subpath length mismatch:", uint2(s.bounces, cameraBounces));
            s.Terminate();
            return result;
        }

        // perform the original sampling technique

        float w;
        if (basePath.mSuffix.mLightSubpath.mVertexCount == 0) {
            result = EvalLightContribution(s, vertex, w);
        } else if (vertex.IsConnectable()) {
            if ((gBidirectional || gSampleDirectLight) && basePath.mSuffix.mLightSubpath.mVertexCount == 1) {
                result = ConnectToLight(s, vertex, w);
            } else if (gBidirectional && basePath.mSuffix.mLightSubpath.mVertexCount > 1) {
                if (gVertexMerging && basePath.HasFlag(PathFlags::eVertexMerging))
                    result =  MergeWithLightVertex(s, vertex, shiftedLightVertex, w);
                else
                    result = ConnectToLightSubpath(s, vertex, shiftedLightVertex, w);
            }
        }

        return result;
    }
}

// Top-level

extension PathGenerator {
    // Shift a path to a new primary vertex
    PathSample ShiftPath(const PathSample basePath, const PackedPathVertex newPrimaryHit, out float jacobian) {
        jacobian = 0;
        PathSample result = {};

        // Shift the light subpath

        const bool hasLightSubpath = gBidirectional && (
                (basePath.HasSuffix() && basePath.mSuffix.mLightSubpath.mVertexCount > 1)
                || (!gDisableCameraConnection && basePath.HasFlag(PathFlags::eNonCausticLightPath))
                || (!gDisableCameraConnection && basePath.HasFlag(PathFlags::eCausticLightPath) && !gUseCausticReservoirs) );

        LightSubpathVertex shiftedLightVertex = {};
        if (gShiftSuffixes && hasLightSubpath) {
            // trace light subpath, so that vertex is the end of the light subpath
            PathVertex vertex;
            PathState<true> s;
            if (!gDisableCameraConnection && basePath.HasFlag(PathFlags::eNonCausticLightPath) && basePath.mBounces == 1)
                s = ShiftLightSubpath(basePath.mSuffix.mLightSubpath, newPrimaryHit, vertex);
            else {
                s = ShiftLightSubpath(basePath.mSuffix.mLightSubpath, vertex);
                if (s.bounces + 2 != basePath.mSuffix.mLightSubpath.mVertexCount || !vertex.IsFinite())
                    s.Terminate();
            }
            if (!any(s.throughput > 0) || !(s.pdfW > 0)) {
                print("Shift failed: Light subpath shift failed", s.throughput);
                return result;
            }

            if (!gDisableCameraConnection) {
                if (basePath.HasFlag(PathFlags::eCausticLightPath) && !gUseCausticReservoirs) {
                    // Connect caustic paths to the camera directly
                    int2 offsetPixel;
                    float w;
                    result = ConnectToCamera(s, vertex, offsetPixel, w);
                    if (!result.IsValid()) {
                        result = {};
                        return result;
                    }

                    // reject path if it's not in the same pixel (or close)
                    const float2 d = GetPixelCoord(PathVertex(newPrimaryHit).GetPosition()) - offsetPixel;
                    const float d2 = dot(d, d);
                    const float r = mParams.mCausticReuseRadius;
                    if ((r == 0 && any(d != 0.f)) || (r > 0 && d2 > r * r)) {
                        print("Shift failed: Caustic path not in pixel", offsetPixel);
                        result = {};
                        return result;
                    }

                    jacobian = basePath.mSuffix.mLightSubpath.mPdf / result.mSuffix.mLightSubpath.mPdf;
                    return result;
                } else if (basePath.HasFlag(PathFlags::eNonCausticLightPath)) {
                    // connect non-caustic pure light paths to the new primary hit

                    if (basePath.mBounces > 1) {
                        PathVertex v = PathVertex(newPrimaryHit);
                        float3 to_v = v.GetPosition() - vertex.GetPosition();
                        const float dist = length(to_v);
                        to_v /= dist;

                        if (!vertex.IsConnectable() || !v.IsConnectable()) {
                            print("Shift failed: Unconnectable light subpath", dist);
                            return result;
                        }

                        // advance to primary hit
                        {
                            const float3 rayOrigin = vertex.ComputeRayOrigin(to_v);
                            Ray visibilityRay = Ray(rayOrigin, v.ComputeRayOrigin(-to_v) - rayOrigin);
                            visibilityRay.tMax = length(visibilityRay.dir);
                            visibilityRay.dir /= visibilityRay.tMax;
                            if (Occluded(visibilityRay)) {
                                print("Shift failed: Light subpath occluded", 0);
                                return result;
                            }

                            // for light paths: rcv is always the 2nd to last vertex
                            s.rcv = vertex.Pack();
                            s.rcvDirOut = vertex.GetDirIn();
                            s.throughputAtRcv = s.throughput;
                            s.pdfAtRcv = s.pdfW;
                            s.prefixPdf = s.pdfW;
                            s.prefixBounces = 0;
                            s.rcvDiffuseBounces = 0;

                            SetNextVertex(s, vertex, PathVertex(PackedPathVertex(Ray(rayOrigin, to_v), HitInfo(newPrimaryHit.mPackedHit), dist)), true);

                            s.rcvCos = abs(dot(s.lastVertex.normal, vertex.mVertex.mRayDirection));
                            s.rcvDist = vertex.mVertex.mRayDistance;
                            s.deltaRcv = false;
                        }
                    }

                    int2 offsetPixel;
                    float w;
                    result = ConnectToCamera(s, vertex, offsetPixel, w, false);
                    if (!result.IsValid()) {
                        print("Shift failed: ConnectToCamera failed", 0);
                        result = {};
                        return result;
                    }

                    jacobian = basePath.mSuffix.mLightSubpath.mPdf / result.mSuffix.mLightSubpath.mPdf;
                    jacobian *= (result.mSuffix.mCos / basePath.mSuffix.mCos) * sqr(basePath.mSuffix.mDist / result.mSuffix.mDist);
                    return result;
                }
            }

            // store light subpath vertex for reconnection
            shiftedLightVertex = PackLightSubpathVertex(s, vertex);
        }

        // Shift the camera subpath
        {
            PathVertex vertex = PathVertex(newPrimaryHit);
            PathState<true> s = ShiftPrefix(basePath, vertex);
            if (!any(s.throughput > 0)) {
                print("Shift failed: Invalid prefix", 0);
                return result;
            }

            if (basePath.HasSuffix() || hasLightSubpath) {
                if (!vertex.IsConnectable()) {
                    print("Shift failed: Non reconnectable prefix", 0);
                    return result;
                }

                if (gShiftSuffixes) {
                    // shift the suffix and apply the original sampling technique
                    #if 0
                    result = ShiftSuffix(s, vertex, basePath, shiftedLightVertex);
                    #else
                    var tmp_s = s;
                    var tmp_vertex = vertex;
                    PathSample tmp_basePath = ShiftSuffix(s, vertex, basePath, shiftedLightVertex);
                    if (tmp_basePath.IsValid())
                        result = ConnectToSuffix(tmp_s, tmp_vertex, tmp_basePath);
                    #endif
                } else {
                    // reconnect to the suffix vertex
                    result = ConnectToSuffix(s, vertex, basePath);
                }
            } else {
                // pure random replay path, no suffix
                float w;
                result = EvalLightContribution(s, vertex, w);
            }
        }

        if (!result.IsValid()) {
            print("Shift failed: Invalid result", 0);
            return {};
        }

        // random replay jacobian
        if (!(basePath.HasFlag(PathFlags::eCausticLightPath) && gUseCausticReservoirs)) {
            jacobian = basePath.mPrefixPdf / result.mPrefixPdf;

            if (hasLightSubpath || (basePath.HasSuffix() && basePath.mSuffix.mVertex.IsFinite())) {
                // reconnection jacobian
                jacobian *= (result.mSuffix.mCos / basePath.mSuffix.mCos) * sqr(basePath.mSuffix.mDist / result.mSuffix.mDist);

                // suffix jacobian
                jacobian *= basePath.mSuffix.mCameraFwdPdf / result.mSuffix.mCameraFwdPdf;
                if (hasLightSubpath)
                    jacobian *= basePath.mSuffix.mLightSubpath.mPdf / result.mSuffix.mLightSubpath.mPdf;
            }
        }

        return result;
    }
}
