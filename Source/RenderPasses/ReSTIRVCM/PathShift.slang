import Scene.RaytracingInline;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
__exported import PathGenerator;
import PhotonMap;

extension PathGenerator {
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

    float ComputeReconnectionMis(
        PathState<true> s,
        LightSubpathVertex v ) {

        float misWeight = 1;
        if (DebugDisableMIS())
            return misWeight;

        #if 0
        float lightSubpathPdf = v.mLightSubpath.mPdf;

        float denom = prefixFwdPdf * (
            prefixScatterFwdPdf * (
                suffixFwdPdf * (cameraScatterFwdPdf * lightSubpathSumProd + lightSubpathPdf)
                + suffixSumProd * lightScatterRevPdf * lightSubpathPdf)
            + suffixRevPdf * lightScatterRevPdf * lightSubpathPdf );
        denom += prefixSumProd * suffixScatterRevPdf * (suffixRevPdf * lightScatterRevPdf * lightSubpathPdf);

        misWeight = (s.pdfW * v.mLightSubpath.mPdf) / denom;
        #endif

        return misWeight;
    }

    PathSample ConnectToSuffix(inout PathState<true> s, inout PathVertex vertex, const PathSample basePath, const LightSubpathVertex lightSubpathVertex) {
        PathSample result = {};

        const bool lightSubpathOnly = gBidirectional && basePath.mSuffix.mLightSubpath.mVertexCount > 1 && !basePath.HasFlag(PathFlags::eCausticLightPath) &&
            basePath.mBounces == (basePath.mPrefixBounces + basePath.mSuffix.mLightSubpath.mVertexCount) - (gVertexMerging && basePath.HasFlag(PathFlags::eVertexMerging) ? 1 : 0);
        const bool mergeSuffix = lightSubpathOnly && !gLightTraceOnly && basePath.HasFlag(PathFlags::eVertexMerging);

        PathSuffix newSuffix = basePath.mSuffix;

        // if the suffix is just the light subpath, we can connect to it directly to get the new suffix
        if (gValidateSuffixes && lightSubpathOnly) {
            float w;
            if (mergeSuffix)
                newSuffix =  MergeWithLightVertex(s, vertex, lightSubpathVertex, w).mSuffix;
            else
                newSuffix = ConnectToLightSubpath(s, vertex, lightSubpathVertex, w).mSuffix;
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
                    const float cosOut = abs(dot(guideNormal, visibilityRay.dir));
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

                    // update mis quantities
                    if (gBidirectional) {
                        const float cosOut = vertex.GetCosN(visibilityRay.dir);
                        const float invPdf = 1.f / r.mFwdPdfW;
                        s.dVC = Mis(cosOut * invPdf) * (s.dVC * Mis(r.mRevPdfW) + s.dVCM + GetMisVmWeightFactor());
                        s.dVM = Mis(cosOut * invPdf) * (s.dVM * Mis(r.mRevPdfW) + s.dVCM * GetMisVcWeightFactor() + 1);
                        s.dVCM = Mis(invPdf);
                    }
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

            // update mis quantities
            if (gBidirectional && suffixVertex.IsFinite()) {
                const float cosIn = suffixVertex.GetCosN(visibilityRay.dir);
                s.dVCM *= Mis(sqr(suffixVertex.mVertex.mRayDistance) / cosIn);
                s.dVC /= Mis(cosIn);
                s.dVM /= Mis(cosIn);
            }

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

                    // update mis quantities
                    if (gBidirectional) {
                        const float cosOut = suffixVertex.GetCosN(newSuffix.mDirOut);
                        const float invPdf = 1.f / r.mFwdPdfW;
                        s.dVC = Mis(cosOut * invPdf) * (s.dVC * Mis(r.mRevPdfW) + s.dVCM + GetMisVmWeightFactor());
                        s.dVM = Mis(cosOut * invPdf) * (s.dVM * Mis(r.mRevPdfW) + s.dVCM * GetMisVcWeightFactor() + 1);
                        s.dVCM = Mis(invPdf);
                    }

                    // Trace first ray in suffix
                    suffixVertex = PathVertex( TraceRay(Ray(suffixVertex.ComputeRayOrigin(newSuffix.mDirOut), newSuffix.mDirOut)) );
                    s.bounces++;
                    s.diffuseBounces++;

                    // update mis quantities
                    if (gBidirectional && suffixVertex.IsFinite()) {
                        const float cosIn = suffixVertex.GetCosN(suffixVertex.GetDirIn());
                        s.dVCM *= Mis(sqr(suffixVertex.mVertex.mRayDistance) / cosIn);
                        s.dVC /= Mis(cosIn);
                        s.dVM /= Mis(cosIn);
                    }

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
                        newSuffix.mCameraFwdPdf = s.pdfW / s.pdfAtRcv;
                    } else {
                        if (!suffixVertex.IsConnectable()) {
                            print("Suffix shift failed: Unconnectable suffix vertex", 0);
                            return result;
                        }

                        float w, m;
                        if (newSuffix.mLightSubpath.mVertexCount == 1) {
                            // NEE path, formed by connecting a camera subpath to a light directly
                            const bool upperOnly = suffixVertex.HasFaceNormal() && !suffixVertex.HasLobe(LobeType::NonDeltaTransmission);
                            result = ConnectToLight(s, suffixVertex, upperOnly, w);
                        } else {
                            // Bidirectional path, formed by connecting a camera subpath to a light subpath
                            if (basePath.HasFlag(PathFlags::eVertexMerging))
                                result = MergeWithLightVertex(s, vertex, lightSubpathVertex, w);
                            else
                                result = ConnectToLightSubpath(s, vertex, lightSubpathVertex, w);
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
            if (gValidateSuffixes) jacobian *= basePath.mSuffix.mCameraFwdPdf / result.mSuffix.mCameraFwdPdf;
        }

        return result;
    }
}
