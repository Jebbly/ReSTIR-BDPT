import PathGenerator;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

static const RMISType gMISType = (RMISType)TEMPORAL_RMIS_TYPE;

void DoTemporalReuse(const uint2 id, inout PathReservoir r) {
    const PackedPathVertex primaryHit = gPathGenerator.LoadPrimaryVertex(id);
    if (!primaryHit.IsFinite()) return;

    const int2 prevPixel = int2((id + 0.5) + gPathGenerator.mParams.mOutputDim * gPathGenerator.mMotionVectors[id]);
    if (any(prevPixel < 0) || any(prevPixel >= gPathGenerator.mParams.mOutputDim)) return;

    const PackedPathVertex prevVertex = gPathGenerator.LoadPrevPrimaryVertex(prevPixel);
    if (!prevVertex.IsFinite())
        return;

    PathReservoir candidate = gPathGenerator.LoadInputReservoir(prevPixel);
    if (!(candidate.mM > 0) || isinf(candidate.mM))
        return;

    const float candidateTargetPdf = candidate.mSample.GetTargetPdf();

    float jacobianToCenter = 0;
    float jacobianToPrev = 0;

    if (r.mSample.IsValid()) {
        print("\nCenter id, seed:", uint3(r.mSample.mRngIdx, r.mSample.mRngSeed));
        print("f:", r.mSample.GetTargetPdf());
        print("W:", r.mW);
    } else {
        print("\nInvalid center sample", 0);
    }

    if (candidate.mSample.IsValid()) {
        print("Candidate id, seed:", uint3(candidate.mSample.mRngIdx, candidate.mSample.mRngSeed));
        print("f:", candidate.mSample.GetTargetPdf());
        print("W:", candidate.mW);
        print("M:", candidate.mM);

        candidate.mSample = gPathGenerator.ShiftPath(candidate.mSample, primaryHit, jacobianToCenter);
        print("jacobianToCenter:", jacobianToCenter);
    } else {
        print("Invalid candidate sample", 0);
    }

    if (gMISType == RMISType::eTalbot) {
        // apply mis weight for r.mSample
        {
            float mi = r.mM * r.mSample.GetTargetPdf();
            float msum = mi;
            if (r.mSample.IsValid()) {
                const float shiftedTargetPdf = gPathGenerator.ShiftPath(r.mSample, prevVertex, jacobianToPrev).GetTargetPdf();
                print("jacobianToPrev:", jacobianToPrev);
                msum += candidate.mM * shiftedTargetPdf * jacobianToPrev;
            }

            const float m = SafeDivide(mi, msum);
            r.PrepareMerge(m);
        }
        // apply mis weight for shifted sample
        {
            float msum = r.mM * candidate.mSample.GetTargetPdf() * jacobianToCenter;
            float mi = SafeDivide(candidate.mM * candidateTargetPdf, jacobianToCenter);
            msum += mi;

            const float m = SafeDivide(mi, msum);
            candidate.PrepareMerge(m, jacobianToCenter);
        }
    } else {
        candidate.PrepareMerge(candidate.mM / (candidate.mM + r.mM), jacobianToCenter);
        if (r.mSample.IsValid() && r.mW > 0) {
            const float shiftedTargetPdf = gPathGenerator.ShiftPath(r.mSample, prevVertex, jacobianToPrev).GetTargetPdf();
            print("Shifted center to candidate. new f:", shiftedTargetPdf);
            print("jacobianToPrev:", jacobianToPrev);
            r.PrepareMerge(r.mM / (r.mM + (shiftedTargetPdf * jacobianToPrev > 0 ? candidate.mM : 0)));
        }
    }

    //print("r ris w: %f\tcandidate ris w: %f\n", r.mW, candidate.mW);

    SampleGenerator sg = SampleGenerator(id, gRandomSeed);
    r.Merge(sampleNext1D(sg), candidate);

    r.FinalizeMerge();

    print("Final center f:", r.mSample.mContribution);
    print("Final center W:", r.mW);
}

[shader("compute")]
[numthreads(8,4,1)]
void main(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    printSetPixel(id);

    PathReservoir r = gPathGenerator.LoadOutputReservoir(id);

    DoTemporalReuse(id, r);

    if (gPathGenerator.mParams.mMCap > 0) r.mM = min(r.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.StoreReservoir(id, r);
}
