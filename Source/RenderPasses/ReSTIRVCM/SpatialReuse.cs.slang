import Random;
import PathGenerator;
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;

Texture2D<uint4> gVertices;

static const RMISType gMISType = (RMISType)SPATIAL_RMIS_TYPE;

SampleGenerator GetNeighborOffsetSampleGenerator(const uint2 id) {
    return SampleGenerator(id, gRandomSeed + 1);
}

// Generalized Talbot RMIS weight
float ComputeSpatialMisWeight(const uint2 center, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex, float candidateTargetPdf) {
    float mi = 0;
    float msum = centerM * sample.mSample.GetTargetPdf() * jacobianToCenter;
    if (candidateIndex == -1)
        mi = msum;
    SampleGenerator spatialSg = GetNeighborOffsetSampleGenerator(center);
    for (uint i = 0; i < gPathGenerator.mParams.mSpatialReuseSamples; i++) {
        const int2 p = center + int2(sample_disk_concentric(sampleNext2D(spatialSg)) * gPathGenerator.mParams.mSpatialReuseRadius);
        if (any(p < 0) || any(p >= gPathGenerator.mParams.mOutputDim) || all(p == center))
            continue;

        if (candidateIndex == i) {
            mi = sample.mM * candidateTargetPdf / jacobianToCenter;
            msum += mi;
            continue;
		}

        const PackedPathVertex vp = gPathGenerator.LoadPrimaryVertex(p);
        if (!vp.IsFinite())
            continue;

        const PathReservoir candidate = gPathGenerator.LoadInputReservoir(p);
        if (!(candidate.mM > 0))
            continue;

        // Determine whether p could have generated sample by shifting sample to p's vertex

        print("Shifting sample to candidate", i);

        float jacobian;
        const PathSample shifted = gPathGenerator.ShiftPath(sample.mSample, vp, jacobian);
        print("jacobian:", jacobian);
        if (jacobian > 0) {
            const float m = candidate.mM * shifted.GetTargetPdf() * jacobian;
            msum += m;
            if (candidateIndex == i)
                mi = m;
        }
    }
    return msum > 0 ? mi / msum : 0;
}
float ComputeSpatialZ(const uint2 center, PathReservoir sample, float centerM, int selectedIndex) {
    float msum = centerM;
    SampleGenerator spatialSg = GetNeighborOffsetSampleGenerator(center);
    for (uint i = 0; i < gPathGenerator.mParams.mSpatialReuseSamples; i++) {
        const int2 p = center + int2(sample_disk_concentric(sampleNext2D(spatialSg)) * gPathGenerator.mParams.mSpatialReuseRadius);
        if (any(p < 0) || any(p >= gPathGenerator.mParams.mOutputDim) || all(p == center))
            continue;
        const PackedPathVertex vp = gPathGenerator.LoadPrimaryVertex(p);
        if (!vp.IsFinite())
            continue;

        const PathReservoir candidate = gPathGenerator.LoadInputReservoir(p);
        if (!(candidate.mM > 0))
            continue;

        if (i == selectedIndex) {
            msum += candidate.mM;
            continue;
        }

        // Determine whether p could have generated sample by shifting sample to p's vertex

        print("Shifting sample to candidate", i);

        float jacobian;
        gPathGenerator.ShiftPath(sample.mSample, vp, jacobian);
        print("jacobian:", jacobian);
        if (jacobian > 0) {
            msum += candidate.mM;
        }
    }
    return msum;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim))
        return;
    printSetPixel(id);

    PathReservoir r = gPathGenerator.LoadInputReservoir(id);

    const PackedPathVertex vertex = gPathGenerator.LoadPrimaryVertex(id);
    if (!vertex.IsFinite()) {
        gPathGenerator.StoreReservoir(id, r);
        return;
    }

    PathReservoir center = r;
    const float centerM = r.mM;
    int selectedIndex = -1;
    uint validNeighbors = 0;
    float mc = 1;

    print("\nCenter f:", center.mSample.GetTargetPdf());
    print("Center W:", center.mW);

    if (gMISType == RMISType::ePairwise) {
        r = {};
    } else if (gMISType == RMISType::eTalbot) {
        r.PrepareMerge(ComputeSpatialMisWeight(id, r, 1, centerM, -1, r.mSample.GetTargetPdf()));
    } else if (gMISType == RMISType::eConstant) {
        r.PrepareMerge(centerM);
    }

    SampleGenerator sg = SampleGenerator(id, gRandomSeed);
    SampleGenerator spatialSg = GetNeighborOffsetSampleGenerator(id);
    for (uint i = 0; i < gPathGenerator.mParams.mSpatialReuseSamples; i++) {
        const int2 p = id + int2(sample_disk_concentric(sampleNext2D(spatialSg)) * gPathGenerator.mParams.mSpatialReuseRadius);
        if (any(p < 0) || any(p >= gPathGenerator.mParams.mOutputDim) || all(p == id))
            continue;

        PathReservoir candidate = gPathGenerator.LoadInputReservoir(p);

        print("Candidate id:", candidate.mSample.mRngIdx);
        print("bounces, replay bounces:", uint2(candidate.mSample.mBounces, candidate.mSample.mReplayBounces));
        print("f:", candidate.mSample.GetTargetPdf());
        print("W:", candidate.mW);

        if (gMISType == RMISType::ePairwise) {
            const PackedPathVertex candidateVertex = gPathGenerator.LoadPrimaryVertex(p);
            if (!candidateVertex.IsFinite())
                continue;
            validNeighbors++;
            mc += 1;
            if (center.mW > 0 && candidate.mM > 0) {
                float jacobianToCandidate;
                const float shiftedTargetPdf = gPathGenerator.ShiftPath(center.mSample, candidateVertex, jacobianToCandidate).GetTargetPdf();
                print("jacobianToCandidate:", jacobianToCandidate);
                if (jacobianToCandidate > 0 && shiftedTargetPdf > 0) {
                    mc -= SafeDivide(
                        candidate.mM * shiftedTargetPdf * jacobianToCandidate,
                        candidate.mM * shiftedTargetPdf * jacobianToCandidate + (centerM / gPathGenerator.mParams.mSpatialReuseSamples) * center.mSample.GetTargetPdf());
                }
            }
        }

        if (!(candidate.mM > 0) || isinf(candidate.mM))
            continue;

        const float candidateTargetPdf = candidate.mSample.GetTargetPdf();

        if (!(candidate.mW > 0) || isinf(candidate.mW) || !(candidateTargetPdf > 0)) {
            r.mM += candidate.mM;
            continue;
        }

        // shift candidate's sample to here
        print("Shifting candidate to center", i);
        float jacobianToCenter;
        candidate.mSample = gPathGenerator.ShiftPath(candidate.mSample, vertex, jacobianToCenter);
        print("jacobianToCenter:", jacobianToCenter);

        if (!(jacobianToCenter > 0)) {
            r.mM += candidate.mM;
            continue;
        }

        float m;
        if (gMISType == RMISType::ePairwise) {
            m = SafeDivide(
                candidate.mM * candidateTargetPdf / jacobianToCenter,
                candidate.mM * candidateTargetPdf / jacobianToCenter + (centerM / gPathGenerator.mParams.mSpatialReuseSamples) * candidate.mSample.GetTargetPdf() );
        } else if (gMISType == RMISType::eTalbot) {
            m = ComputeSpatialMisWeight(id, candidate, jacobianToCenter, centerM, i, candidateTargetPdf);
        } else if (gMISType == RMISType::eConstant) {
            m = candidate.mM;
        }

        candidate.PrepareMerge(m, jacobianToCenter);

        if (r.Merge(sampleNext1D(sg), candidate)) {
            selectedIndex = i;
        }
    }

    if (gMISType == RMISType::ePairwise) {
        if (center.mW > 0 && mc > 0 && isfinite(mc)) {
            center.PrepareMerge(mc);
            r.Merge(sampleNext1D(sg), center);
        } else
            r.mM += center.mM;

        r.mW /= validNeighbors + 1;
    } else if (gMISType == RMISType::eConstant) {
        if (r.mW > 0) r.mW = SafeDivide(r.mW, ComputeSpatialZ(id, r, centerM, selectedIndex));
    }

    r.FinalizeMerge();

    print("Final center f:", r.mSample.mRadiance);
    print("Final center W:", r.mW);

    if (gPathGenerator.mParams.mMCap > 0) r.mM = min(r.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.StoreReservoir(id, r);
}
