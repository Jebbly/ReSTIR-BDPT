import Random;
import PathGenerator;
import Utils.Debug.PixelDebug;

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gPathGenerator.mParams.mOutputDim.x + index.x >= gPathGenerator.mParams.mLightSubpathCount) return;
    printSetPixel(-1);

    gPathGenerator.SampleLightPath(gRandomSeed, index.xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index : SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    printSetPixel(id);

    const PathReservoir r = gPathGenerator.SampleCameraPath(gRandomSeed, id);

	gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ComputeLightReservoirOffsets(uint3 index: SV_DispatchThreadID) {
    gPathGenerator.mLightTraceReservoirs.ComputeCellOffset(index.y * gPathGenerator.mParams.mOutputDim.x + index.x);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void SortLightTraceReservoirs(uint3 index: SV_DispatchThreadID) {
    gPathGenerator.mLightTraceReservoirs.Sort(index.y * gPathGenerator.mParams.mOutputDim.x + index.x);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ResolveLightTraceReservoirs(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    printSetPixel(id);

    if (!(gPathGenerator.gBidirectional && gPathGenerator.gUseLightTraceReservoirs))
        return;

    SampleGenerator sg = SampleGenerator(id, gRandomSeed);

    PathReservoir r = {};
    if (!gPathGenerator.gLightTraceOnly) {
        r = gPathGenerator.LoadOutputReservoir(id);
        r.PrepareMerge();
    }

    const uint2 range = gPathGenerator.mLightTraceReservoirs.GetCellDataRange(id.y * gPathGenerator.mParams.mOutputDim.x + id.x);
    for (uint i = range.x; i < range.y; i++) {
        PathReservoir c = gPathGenerator.mLightTraceReservoirs[i];
        c.PrepareMerge();
        r.Merge(sampleNext1D(sg), c);
        r.mM = 1;
    }

    r.FinalizeMerge();

	gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    printSetPixel(id);

    float4 radiance = gPathGenerator.mOutputRadiance[id];

    // Copy radiance from the pixel's reservoir to the output buffer. This is skipped if we only used the atomic radiance.
    if (gPathGenerator.gUseResampling && (!gPathGenerator.gLightTraceOnly || gPathGenerator.gUseLightTraceReservoirs)) {
        const PathReservoir r = gPathGenerator.LoadInputReservoir(id);
        if (r.mW > 0) radiance.rgb += r.mSample.mRadiance * r.mW;
    }

    // Add light trace contribution
    if (gPathGenerator.gBidirectional && !gPathGenerator.gUseLightTraceReservoirs) {
        const uint flatId = id.y * gPathGenerator.mParams.mOutputDim.x + id.x;
        radiance.rgb += gPathGenerator.mAtomicRadiance.Load<float3>(flatId * sizeof(float3));
    }

    gPathGenerator.mOutputRadiance[id] = radiance;
}
