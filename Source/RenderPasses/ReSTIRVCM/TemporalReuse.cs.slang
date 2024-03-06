import PathShift;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

extension PathGenerator {
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

        float integrationWeight;
        PathSample result = ConnectToCamera(s, vertex, pixelCoord, integrationWeight);

        if (!result.IsValid()) {
            print("Caustic shift failed: invalid result", 0);
            return {};
        }

        if (!result.HasFlag(PathFlags::eCausticLightPath)) {
            print("Caustic shift failed: Result not caustic", 0);
            return {};
        }

        // random replay jacobian
        jacobian = basePath.mSuffix.mLightSubpath.mPdf / result.mSuffix.mLightSubpath.mPdf;

        return result;
    }
};

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
        print("\nCenter id, seed:", uint3(r.mSample.mId.mId, r.mSample.mId.mSeed));
        print("f, W:", float2(r.mSample.GetTargetPdf(), r.mW));
    } else {
        print("\nInvalid center sample", 0);
    }

    if (candidate.mSample.IsValid()) {
        print("Candidate id, seed:", uint3(candidate.mSample.mId.mId, candidate.mSample.mId.mSeed));
        print("f, W, M:", float3(candidate.mSample.GetTargetPdf(), candidate.mW, candidate.mM));

        if (gPathGenerator.gUseCausticMotionVectors && candidate.mSample.HasFlag(PathFlags::eCausticLightPath)) {
            // sample was already shifted in ShiftCaustics
            // HACK: store jacobian and original target pdf in reservoir
            jacobianToCenter   = candidate.mSample.mSuffix.mVertex.pad1;
            candidateTargetPdf = candidate.mSample.mPrefixPdf;
            candidate.mSample.mPrefixPdf = 1.f; // prefixPdf is not used (and always 1) for caustic paths

            // reject path if it's not in the same pixel (or close)
            const float2 d = float2(int2(id) - gPathGenerator.GetPixelCoord(PathVertex(candidate.mSample.mSuffix.mVertex).GetPosition()));
            const float d2 = dot(d, d);
            const float r = gPathGenerator.mParams.mCausticReuseRadius;
            if ((r == 0 && any(d != 0.f)) ||
                (r > 0 && d2 > r*r)) {
                candidate.mSample = {};
                jacobianToCenter = 0.f;
            }
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

    print("Final center f, W:", float2(r.mSample.GetTargetPdf(), r.mW));
}

void TemporalCausticReuse(const uint2 id, inout SampleGenerator sg) {
    PathReservoir center = gPathGenerator.LoadCausticReservoir(id);

    const uint flatId = id.y * gPathGenerator.mParams.mOutputDim.x + id.x;
    const uint2 range = gPathGenerator.mCausticReservoirMap.GetCellDataRange(flatId);

    float Msum = 1;
    for (uint i = range.x; i < range.y; i++)
        Msum += gPathGenerator.mCausticReservoirMap[i].mM;

    // apply RMIS to canonical sample from the current frame
    {
        /*
        // shift center sample to the previous frame
        float jacobianToPrev = 0;
        float shiftedTargetPdf = 0;
        float previousM = 0;
        if (center.mSample.IsValid()) {
            #if 1
            const int2 prevId = gPathGenerator.GetPrevPixelCoord(PathVertex(center.mSample.mSuffix.mVertex).GetPosition());
            jacobianToPrev = 1;
            shiftedTargetPdf = center.mSample.GetTargetPdf();
            #else
            int2 prevId;
            shiftedTargetPdf = gPathGenerator.ShiftCausticPath(center.mSample, jacobianToPrev, prevId).GetTargetPdf();
            #endif

            if (shiftedTargetPdf > 0 && all(prevId >= 0) && all(prevId < gPathGenerator.mParams.mOutputDim)) {
                previousM = max(1, gPathGenerator.LoadPrevCausticReservoir(prevId).mM);
            }
        }

        center.PrepareMerge(1.0 / (1.0 + previousM));
        */
        center.PrepareMerge(1 / Msum);
    }

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
