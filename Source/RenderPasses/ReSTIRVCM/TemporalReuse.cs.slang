import PathShift;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

static const bool gUnbiasedTemporalReuse = UNBIASED_TEMPORAL_REUSE;

void TemporalReuse(const uint2 id, inout SampleGenerator sg, inout PathReservoir r) {
    const PackedPathVertex primaryHit = gPathGenerator.LoadPrimaryVertex(id);
    if (!primaryHit.IsFinite()) return;

    const int2 prevPixel = int2((id + 0.5) + gPathGenerator.mParams.mOutputDim * gPathGenerator.mMotionVectors[id]);
    if (any(prevPixel < 0) || any(prevPixel >= gPathGenerator.mParams.mOutputDim)) return;

    PathReservoir candidate = gPathGenerator.LoadPrevReservoir(prevPixel);
    if (!(candidate.mM > 0) || isinf(candidate.mM))
        return;

    const PackedPathVertex prevVertex = gPathGenerator.LoadPrevPrimaryVertex(prevPixel);
    if (!prevVertex.IsFinite())
        return;

    float candidateTargetPdf = candidate.mSample.GetTargetPdf();
    float jacobianToCenter = 0;

    if (r.mSample.IsValid()) {
        print("\nCenter id, seed, bounces:", uint4(r.mSample.mId.mId, r.mSample.mId.mSeed, r.mSample.mBounces));
        print("  f, W:", float2(r.mSample.GetTargetPdf(), r.mW));
    } else {
        print("\nInvalid center sample", 0);
    }

    // shift candidate to current frame

    if (candidate.mSample.IsValid()) {
        print("Candidate id, seed, bounces:", uint4(candidate.mSample.mId.mId, candidate.mSample.mId.mSeed, candidate.mSample.mBounces));
        print("  f, W, M:", float3(candidate.mSample.GetTargetPdf(), candidate.mW, candidate.mM));
        print("  suffix:", luminance(candidate.mSample.mSuffix.mContribution));

        if (gPathGenerator.gBidirectional && !gPathGenerator.gDisableCameraConnection && gPathGenerator.gUseCausticMotionVectors && candidate.mSample.HasFlag(PathFlags::eCausticLightPath)) {
            // sample was already shifted in ShiftCaustics
            // HACK: store jacobian and original target pdf in reservoir
            jacobianToCenter   = candidate.mSample.mSuffix.mVertex.pad1;
            candidateTargetPdf = candidate.mSample.mPrefixPdf;
            candidate.mSample.mPrefixPdf = 1.f; // prefixPdf is not used (and always 1) for caustic paths
        } else {
            candidate.mSample = gPathGenerator.ShiftPath(candidate.mSample, primaryHit, jacobianToCenter);
        }
        print("jacobianToCenter:", jacobianToCenter);

        if (candidate.mSample.IsValid() && jacobianToCenter > 0) {
            gPathGenerator.IncrementCounter(DebugCounterType::eTemporalShiftSuccesses);
            if (!gPathGenerator.gUseCausticReservoirs && candidate.mSample.HasFlag(PathFlags::eCausticLightPath)) {
                gPathGenerator.IncrementCounter(DebugCounterType::eCausticReservoirM, uint(candidate.mM));
            }
        } else
            gPathGenerator.IncrementCounter(DebugCounterType::eTemporalShiftFailures);
    } else {
        print("Invalid candidate sample", 0);
    }

    // Talbot RMIS

    static const bool useCombinedMIS = false;
    const float N = gPathGenerator.mParams.mLightSubpathCount;
    const float alpha = 1;
    const float beta = 1;
    const float gamma = candidate.mM;

    // apply mis weight to r.mSample
    {
        float f = 0;
        if (r.mSample.IsValid()) {
            // shift current sample to previous frame
            if (gUnbiasedTemporalReuse) {
                // HACK: storing 'shiftedTargetPdf * jacobianToPrev' in reservoir
                f = r.mSample.mSuffix.mVertex.pad1;
            } else {
                float jacobianToPrev = 0;
                const float shiftedTargetPdf = gPathGenerator.ShiftPath(r.mSample, prevVertex, jacobianToPrev).GetTargetPdf();
                f = shiftedTargetPdf * jacobianToPrev;
            }
        }

        float m = 0;
        if (useCombinedMIS && r.mSample.HasFlag(PathFlags::eCausticLightPath)) {
            m = SafeDivide(alpha * N * r.mSample.GetTargetPdf(), alpha * N * r.mSample.GetTargetPdf() + candidate.mM * f);
        } else if (useCombinedMIS && r.mSample.HasFlag(PathFlags::eNonCausticLightPath)) {
            m = SafeDivide(beta * r.mSample.GetTargetPdf(), beta * r.mSample.GetTargetPdf() + candidate.mM * f);
        } else {
            const float mi = r.mM * r.mSample.GetTargetPdf();
            m = SafeDivide(mi, mi + candidate.mM * f);
        }
        r.PrepareMerge(m);
    }

    // apply mis weight to candidate sample
    {
        const float mi = SafeDivide(candidate.mM * candidateTargetPdf, jacobianToCenter);
        float m = 0;
        if (useCombinedMIS && candidate.mSample.HasFlag(PathFlags::eCausticLightPath)) {
            m = SafeDivide(mi, mi + alpha * N * candidate.mSample.GetTargetPdf());
        } else if (useCombinedMIS && r.mSample.HasFlag(PathFlags::eNonCausticLightPath)) {
            m = SafeDivide(mi, mi + beta * candidate.mSample.GetTargetPdf());
        } else {
            m = SafeDivide(mi, mi + r.mM * candidate.mSample.GetTargetPdf() * jacobianToCenter);
        }
        candidate.PrepareMerge(m, jacobianToCenter);
    }

    // perform the merge
    if (r.Merge(sampleNext1D(sg), candidate)) {
        gPathGenerator.IncrementCounter(DebugCounterType::eTemporalReuseAcceptances);
        if (!gPathGenerator.gUseCausticReservoirs && r.mSample.HasFlag(PathFlags::eCausticLightPath))
            gPathGenerator.IncrementCounter(DebugCounterType::eCausticReuseAcceptances);
    }

    r.FinalizeMerge();

    print("Final center f, W:", float2(r.mSample.GetTargetPdf(), r.mW));
    print("  suffix f, bounces, lv, prefix:", float4(luminance(r.mSample.mSuffix.mContribution), r.mSample.mBounces, r.mSample.mSuffix.mLightSubpath.mVertexCount, r.mSample.mPrefixBounces));
}

void TemporalCausticReuse(const uint2 id, inout SampleGenerator sg) {
    PathReservoir center = gPathGenerator.LoadCausticReservoir(id);

    const uint flatId = id.y * gPathGenerator.mParams.mOutputDim.x + id.x;
    const uint2 range = gPathGenerator.mCausticReservoirMap.GetCellDataRange(flatId);

    float Msum = center.mM;
    for (uint i = range.x; i < range.y; i++)
        Msum += gPathGenerator.mCausticReservoirMap[i].mM;

    // apply RMIS to canonical sample from the current frame
    if (gUnbiasedTemporalReuse && center.mSample.mSuffix.mVertex.pad1 <= 0)
        center.PrepareMerge(1);
    else
        center.PrepareMerge(1 / Msum);

    // Merge shifted samples from previous frame

    for (uint i = range.x; i < range.y; i++) {
        PathReservoir candidate = gPathGenerator.mCausticReservoirMap[i];

        // HACK: store jacobian and original target pdf in reservoir
        const float jacobianToCenter   = candidate.mSample.mSuffix.mVertex.pad1;
        const float candidateTargetPdf = candidate.mSample.mPrefixPdf;
        candidate.mSample.mPrefixPdf = 1.f; // prefixPdf is not used (and always 1) for caustic paths

        candidate.PrepareMerge(candidate.mM / Msum, jacobianToCenter);

        if (!center.Merge(sampleNext1D(sg), candidate))
            gPathGenerator.IncrementCounter(DebugCounterType::eCausticReuseAcceptances);

        gPathGenerator.IncrementCounter(DebugCounterType::eCausticReservoirM, uint(candidate.mM));
    }

    center.FinalizeMerge();

    if (gPathGenerator.mParams.mMCap > 0) center.mM = min(center.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.StoreCausticReservoir(id, center);
}

// shift current frame's reservoirs to the previous frame
// Store the shifted target pdf and jacobian in extra space in the current frame's reservoir
[shader("compute")]
[numthreads(8, 4, 1)]
void ShiftToPrevFrame(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    if (gPathGenerator.gUseCausticReservoirs) {
        PathReservoir center = gPathGenerator.LoadCausticReservoir(id);

        int2 prevId;
        float jacobianToPrev;
        const float shiftedTargetPdf = gPathGenerator.ShiftCausticPath(center.mSample, jacobianToPrev, prevId).GetTargetPdf();

        // HACK: storing 'shiftedTargetPdf * jacobianToPrev' in reservoir
        center.mSample.mSuffix.mVertex.pad1 = shiftedTargetPdf * jacobianToPrev;

        gPathGenerator.StoreCausticReservoir(id, center);
    }

    const int2 prevPixel = int2((id + 0.5) + gPathGenerator.mParams.mOutputDim * gPathGenerator.mMotionVectors[id]);
    if (any(prevPixel < 0) || any(prevPixel >= gPathGenerator.mParams.mOutputDim)) return;

    const PackedPathVertex prevVertex = gPathGenerator.LoadPrevPrimaryVertex(prevPixel);
    if (!prevVertex.IsFinite()) return;

    PathReservoir r = gPathGenerator.LoadOutputReservoir(id);

    float jacobianToPrev;
    float shiftedTargetPdf = gPathGenerator.ShiftPath(r.mSample, prevVertex, jacobianToPrev).GetTargetPdf();

    // HACK: storing 'shiftedTargetPdf * jacobianToPrev' in reservoir
    r.mSample.mSuffix.mVertex.pad1 = shiftedTargetPdf * jacobianToPrev;

    gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    SampleGenerator sg = MakeSampleGenerator(id, gRandomSeed);

    if (gPathGenerator.gUseCausticReservoirs)
        TemporalCausticReuse(id, sg);

    PathReservoir r = gPathGenerator.LoadOutputReservoir(id);

    TemporalReuse(id, sg, r);

    if (gPathGenerator.mParams.mMCap > 0) r.mM = min(r.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.StoreReservoir(id, r);
}

// Shift last frame's caustics to the current frame.
// Shifted reservoirs are added to the mCausticReservoirMap and merged in the main temporal reuse dispatch above
[shader("compute")]
[numthreads(8, 4, 1)]
void ShiftCaustics(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    PathReservoir r = gPathGenerator.gUseCausticReservoirs ?
        gPathGenerator.LoadPrevCausticReservoir(id) :
        gPathGenerator.LoadPrevReservoir(id);
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

    gPathGenerator.IncrementCounter(DebugCounterType::eCausticShiftSuccesses);

    print("Shifted caustic path. new f, jacobian:", float4(r.mSample.mContribution, jacobian));

    // HACK: store jacobian and original target pdf in reservoir
    r.mSample.mSuffix.mVertex.pad1 = jacobian;
    r.mSample.mPrefixPdf = originalTargetPdf; // prefixPdf is not used (and always 1) for caustic paths

    const uint flatId = newId.y * gPathGenerator.mParams.mOutputDim.x + newId.x;

    if (gPathGenerator.gUseCausticReservoirs)
        gPathGenerator.mCausticReservoirMap.Insert(flatId, r);

    if (gPathGenerator.gUseCausticMotionVectors) {
        gPathGenerator.mLastReservoirs[id.y * gPathGenerator.mParams.mOutputDim.x + id.x] = r;

        const uint byte = flatId / 8;
        const uint bit = 1 << (flatId % 8);
        uint m;
        gPathGenerator.mCausticMotionVectorMutex.InterlockedOr(byte, bit, m);
        if ((m & bit) != 0)
            gPathGenerator.mMotionVectors[newId] = (id - (newId + 0.5)) / gPathGenerator.mParams.mOutputDim;
    }
}
