import PathGenerator;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

void TemporalReuse(const uint2 id, inout SampleGenerator sg, inout PathReservoir r) {
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

        if (candidate.mSample.IsValid() && jacobianToCenter > 0) {
            gPathGenerator.IncrementCounter(DebugCounterType::eTemporalShiftSuccesses);
            if (!gPathGenerator.gUseCausticReservoirs && candidate.mSample.HasFlag(PathFlags::eCausticLightPath))
                gPathGenerator.IncrementCounter(DebugCounterType::eCausticShiftSuccesses);
        }
    } else {
        print("Invalid candidate sample", 0);
    }

    // apply mis weight for r.mSample
    {
        float mi = r.mM * r.mSample.GetTargetPdf();
        float msum = mi;
        if (r.mSample.IsValid()) {
            float jacobianToPrev = 0;
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

    if (r.Merge(sampleNext1D(sg), candidate)) {
        gPathGenerator.IncrementCounter(DebugCounterType::eTemporalReuseAcceptances);
        if (!gPathGenerator.gUseCausticReservoirs && r.mSample.HasFlag(PathFlags::eCausticLightPath))
            gPathGenerator.IncrementCounter(DebugCounterType::eCausticReuseAcceptances);
    }

    r.FinalizeMerge();

    print("Final center f:", r.mSample.mContribution);
    print("Final center W:", r.mW);
}

void TemporalCausticReuse(const uint2 id, inout SampleGenerator sg) {
    const uint flatId = id.y * gPathGenerator.mParams.mOutputDim.x + id.x;

    // load the canonical sample from the current frame
    PathReservoir center = gPathGenerator.mCausticReservoirs[flatId];
    const float centerM = max(1, center.mM);

    // shift current sample to the previous frame
    float jacobianToPrev = 0;
    float shiftedTargetPdf = 0;
    float prevM = 0;
    if (center.mSample.IsValid()) {
        int2 prevId = id;
        jacobianToPrev = 1;
        shiftedTargetPdf = center.mSample /*gPathGenerator.ShiftCausticPath(center.mSample, jacobianToPrev, prevId)*/.GetTargetPdf();
        if (all(prevId >= 0) && all(prevId < gPathGenerator.mParams.mOutputDim)) {
            prevM = 1;
            const PathReservoir prevReservoir = gPathGenerator.mLastCausticReservoirs[prevId.y * gPathGenerator.mParams.mOutputDim.x + prevId.x];
            if (prevReservoir.mSample.IsValid())
                prevM = prevReservoir.mM;
        }
    }

    center.PrepareMerge(centerM / (centerM + prevM));
    center.mM += prevM;

    const uint2 range = gPathGenerator.mCausticReservoirMap.GetCellDataRange(flatId);
    if (range.y > range.x) {
        float Msum = 1;
        for (uint i = range.x; i < range.y; i++) {
            Msum += gPathGenerator.mCausticReservoirMap[i].mM;
        }

        for (uint i = range.x; i < range.y; i++) {
            PathReservoir candidate = gPathGenerator.mCausticReservoirMap[i];

            // HACK: store jacobian and original target pdf in reservoir
            const float jacobianToCenter   = candidate.mSample.mSuffix.mVertex.pad1;
            const float candidateTargetPdf = candidate.mSample.mPrefixPdf;
            candidate.mSample.mPrefixPdf = 1.f; // prefixPdf is not used (and always 1) for caustic paths

            candidate.PrepareMerge(candidate.mM / (1.0 + candidate.mM), jacobianToCenter);

            if (!center.Merge(sampleNext1D(sg), candidate))
                gPathGenerator.IncrementCounter(DebugCounterType::eCausticReuseAcceptances);
        }
    }
    center.FinalizeMerge();

    if (gPathGenerator.mParams.mMCap > 0) center.mM = min(center.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.mCausticReservoirs[flatId] = center;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    SampleGenerator sg = SampleGenerator(id, gRandomSeed);

    if (gPathGenerator.gUseCausticReservoirs)
        TemporalCausticReuse(id, sg);

    PathReservoir r = gPathGenerator.LoadOutputReservoir(id);

    TemporalReuse(id, sg, r);

    if (gPathGenerator.mParams.mMCap > 0) r.mM = min(r.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void ShiftCaustics(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    PathReservoir r = gPathGenerator.mLastCausticReservoirs[id.y * gPathGenerator.mParams.mOutputDim.x + id.x];
    if (!(r.mW > 0) || !r.mSample.IsValid() || !r.mSample.HasFlag(PathFlags::eCausticLightPath))
        return;

    print("Shifting caustic path. f, W:", float4(r.mSample.mContribution, r.mW));

    float originalTargetPdf = r.mSample.GetTargetPdf();

    int2 newId = id;
    float jacobian = 1;
    r.mSample = gPathGenerator.ShiftCausticPath(r.mSample, jacobian, newId);
    if (!(jacobian > 0) || !r.mSample.IsValid()) {
        gPathGenerator.IncrementCounter(DebugCounterType::eCausticShiftFailures);
        return;
    }

    print("Shifted caustic path. new f, jacobian:", float4(r.mSample.mContribution, jacobian));

    // HACK: store jacobian and original target pdf in reservoir
    r.mSample.mSuffix.mVertex.pad1 = jacobian;
    r.mSample.mPrefixPdf = originalTargetPdf; // prefixPdf is not used (and always 1) for caustic paths

    gPathGenerator.mCausticReservoirMap.Insert(newId.y * gPathGenerator.mParams.mOutputDim.x + newId.x, r);

    gPathGenerator.DebugSetPixel(newId);
    gPathGenerator.IncrementCounter(DebugCounterType::eCausticShiftSuccesses);
}
