import PathGenerator;
import Utils.Color.ColorMap;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gPathGenerator.mParams.mOutputDim.x + index.x >= gPathGenerator.mParams.mLightSubpathCount) return;
    gPathGenerator.DebugSetPixel(-1);

    gPathGenerator.SampleLightPath(gRandomSeed, index.xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index : SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    const PathReservoir r = gPathGenerator.SampleCameraPath(gRandomSeed, id);

    gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ResolveLightTraceReservoirs(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    if (!(gPathGenerator.gBidirectional && gPathGenerator.gUseResampling))
        return;

    SampleGenerator sg = SampleGenerator(id, gRandomSeed);

    PathReservoir r = {};
    if (!gPathGenerator.gLightTraceOnly) {
        r = gPathGenerator.LoadOutputReservoir(id);
        r.PrepareMerge();
    }

    PathReservoir cr = {};

    const uint2 range = gPathGenerator.mLightTraceReservoirs.GetCellDataRange(id.y * gPathGenerator.mParams.mOutputDim.x + id.x);
    for (uint i = range.x; i < range.y; i++) {
        PathReservoir c = gPathGenerator.mLightTraceReservoirs[i];
        c.PrepareMerge();
        const float u = sampleNext1D(sg);
        if (gPathGenerator.gUseCausticReservoirs && c.mSample.HasFlag(PathFlags::eCausticLightPath)) {
            cr.Merge(u, c);
        } else {
            if (r.Merge(u, c)) {
                print("Selected path. contrib:", c.mSample.mContribution);
            }
        }
    }

    r.FinalizeMerge();
    gPathGenerator.StoreReservoir(id, r);

    if (gPathGenerator.gUseCausticReservoirs) {
        cr.FinalizeMerge();
        gPathGenerator.mCausticReservoirs[id.y * gPathGenerator.mParams.mOutputDim.x + id.x] = cr;

        gPathGenerator.IncrementCounter(DebugCounterType::eCausticReservoirM, uint(cr.mM));
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    float4 radiance = gPathGenerator.mOutputRadiance[id];

    const uint flatId = id.y * gPathGenerator.mParams.mOutputDim.x + id.x;

    // Copy radiance from the pixel's reservoir to the output buffer. This is skipped if we only used the atomic radiance.
    if (gPathGenerator.gUseResampling) {
        const PathReservoir r = gPathGenerator.LoadInputReservoir(id);
        if (r.mW > 0 && r.mSample.IsValid())
            radiance.rgb += r.mSample.mContribution * r.mW;

        if (gPathGenerator.gUseCausticReservoirs) {
            const PathReservoir cr = gPathGenerator.mCausticReservoirs[flatId];
            if (cr.mW > 0 && cr.mSample.IsValid())
                radiance.rgb += cr.mSample.mContribution * cr.mW;
        }
    } else if (gPathGenerator.gBidirectional) {
        // Add light trace contribution from atomic buffer
        radiance.rgb += gPathGenerator.mAtomicRadiance.Load<float3>(flatId * sizeof(float3));
    }

    if (gPathGenerator.gDebugHeatmap) {
        const float m = float(gPathGenerator.mOutputCounterData.Load(0));
        const float v = gPathGenerator.mOutputCounterData.Load((1 + flatId) * sizeof(uint));
        radiance = float4( colormapViridis( SafeDivide(v, m) ), 1);
    }

    gPathGenerator.mOutputRadiance[id] = radiance;

}
