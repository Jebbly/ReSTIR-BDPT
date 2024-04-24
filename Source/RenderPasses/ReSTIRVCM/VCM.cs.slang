import Utils.Color.ColorMap;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import PathShift;

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gPathGenerator.mParams.mOutputDim.x + index.x >= gPathGenerator.mParams.mLightSubpathCount) return;
    gPathGenerator.DebugSetPixel(-1);

    gPathGenerator.SampleLightPath(PathId(gRandomSeed, index.xy));
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index : SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    const PathReservoir r = gPathGenerator.SampleCanonicalPaths(PathId(gRandomSeed, id));

    gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void ResolveLightTraceReservoirs(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);
    SampleGenerator sg = MakeSampleGenerator(id, gRandomSeed);

    const PackedPathVertex primaryVertex = gPathGenerator.LoadPrimaryVertex(id);

    // merge light trace reservoirs in pixel

    PathReservoir r = {};
    r.mM = 1;

    const uint2 range = gPathGenerator.mLightTraceReservoirs.GetCellDataRange(id.y * gPathGenerator.mParams.mOutputDim.x + id.x);
    for (uint i = range.x; i < range.y; i++) {
        const LightTraceReservoir lr = gPathGenerator.mLightTraceReservoirs[i];
        PathReservoir c = PathReservoir(lr.path, lr.W, 0);
        c.PrepareMerge(1/float(gPathGenerator.mParams.mLightSubpathCount));

        const float u = sampleNext1D(sg);

        if (gPathGenerator.gShiftLightPathsToPixelCenters && c.mSample.HasFlag(PathFlags::eNonCausticLightPath)) {
            // Reject path if it is not samplable from the primary vertex
            float jacobian;
            const bool valid = gPathGenerator.ShiftPath(c.mSample, primaryVertex, jacobian).IsValid();
            if (!valid || !(jacobian > 0))
                continue;
        }

        r.Merge(u, c);
    }

    // Merge with camera tracing sample

    if (!gPathGenerator.gLightTraceOnly) {
        PathReservoir c = gPathGenerator.LoadOutputReservoir(id);
        c.PrepareMerge();
        r.Merge(sampleNext1D(sg), c);
    }

    r.FinalizeMerge();
    gPathGenerator.StoreReservoir(id, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gPathGenerator.mParams.mOutputDim)) return;
    gPathGenerator.DebugSetPixel(id);

    float4 radiance = gPathGenerator.mOutputRadiance[id];

    const uint flatId = id.y * gPathGenerator.mParams.mOutputDim.x + id.x;

    // Copy radiance from the pixel's reservoir to the output buffer.

    if (gPathGenerator.gUseResampling) {
        const PathReservoir r = gPathGenerator.LoadInputReservoir(id);
        if (r.mW > 0 && r.mSample.IsValid()) {
            radiance.rgb += r.GetRadiance();
        }
    } else if (gPathGenerator.gBidirectional && ! gPathGenerator.gDisableCameraConnection) {
        // Add light trace contribution from atomic buffer
        radiance.rgb += gPathGenerator.mAtomicRadiance.Load<float3>(flatId * sizeof(float3));
    }

    if (gPathGenerator.gDebugHeatmap && (uint)gPathGenerator.mParams.mDebugCounter != 0 ) {
        const float m = float(gPathGenerator.mOutputCounterData.Load(0));
        const float v = gPathGenerator.mOutputCounterData.Load((1 + flatId) * sizeof(uint));
        radiance = float4( colormapViridis( SafeDivide(v, m) ), 1);

        print("Heatmap value/max:", float2(v, m));
    }

    gPathGenerator.mOutputRadiance[id] = radiance;

}
