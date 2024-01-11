import Random;
import PathGenerator;
import Utils.Math.MathHelpers;
import Utils.Debug.PixelDebug;

Texture2D<uint4> gVertices;

static const RMISType gMISType = (RMISType)SPATIAL_RMIS_TYPE;

SampleGenerator GetNeighborOffsetSampleGenerator(const uint2 id) {
    return SampleGenerator(id, -gPathGenerator.mParams.mRandomSeed);
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

        //print("Shifting sample (%u %u) to candidate %u\n", sample.mSample.mRngSeed, sample.mSample.mPackedRngIdx, i);

        float jacobian;
        const PathSample shifted = gPathGenerator.ShiftPath(sample.mSample, vp, jacobian);
        print("jacobian: %f\n", jacobian);
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

        //print("Shifting sample (%u %u) to candidate %u\n", sample.mSample.mRngSeed, sample.mSample.mPackedRngIdx, i);

        float jacobian;
        gPathGenerator.ShiftPath(sample.mSample, vp, jacobian);
        print("jacobian: %f\n", jacobian);
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

    //print("\n\n---------- Performing spatial reuse (%u) ----------\n\n", gPathGenerator.mParams.gRandomSeed);

    SampleGenerator sg = SampleGenerator(id, gPathGenerator.mParams.mRandomSeed);

    PathReservoir center = r;
    const float centerM = r.mM;
    int selectedIndex = -1;
    uint validNeighbors = 0;
    float mc = 1;

    if (gMISType == RMISType::ePairwise) {
        r = {};
    } else if (gMISType == RMISType::eTalbot) {
        r.PrepareMerge(ComputeSpatialMisWeight(id, r, 1, centerM, -1, r.mSample.GetTargetPdf()));
    } else if (gMISType == RMISType::eConstant) {
        r.PrepareMerge(centerM);
    }

    SampleGenerator spatialSg = GetNeighborOffsetSampleGenerator(id);
    for (uint i = 0; i < gPathGenerator.mParams.mSpatialReuseSamples; i++) {
        const int2 p = id + int2(sample_disk_concentric(sampleNext2D(spatialSg)) * gPathGenerator.mParams.mSpatialReuseRadius);
        if (any(p < 0) || any(p >= gPathGenerator.mParams.mOutputDim) || all(p == id))
            continue;

        PathReservoir candidate = gPathGenerator.LoadInputReservoir(p);

        if (gMISType == RMISType::ePairwise) {
            const PackedPathVertex candidateVertex = gPathGenerator.LoadPrimaryVertex(p);
            if (!candidateVertex.IsFinite())
                continue;
            validNeighbors++;
            mc += 1;
            if (center.mW > 0 && candidate.mM > 0) {
                //print("Shifting center (%u %u) to candidate %u\n", center.mSample.mRngSeed, center.mSample.mPackedRngIdx, i);
                float jacobianToCandidate;
                const PathSample shifted = gPathGenerator.ShiftPath(center.mSample, candidateVertex, jacobianToCandidate);
                //print("center.f: %f -> %f, center.W %f, jacobianToCandidate: %f\n", center.mSample.GetTargetPdf(), shifted.GetTargetPdf(), center.mW, jacobianToCandidate);
                if (jacobianToCandidate > 0) {
                    const float f = shifted.GetTargetPdf() * jacobianToCandidate;
                    mc -= SafeDivide(candidate.mM * f, candidate.mM * f + (centerM / gPathGenerator.mParams.mSpatialReuseSamples) * center.mSample.GetTargetPdf());
                }
            }
        }

        if (!(candidate.mM > 0) || !isfinite(candidate.mM))
            continue;

        if (!(candidate.mW > 0) || !isfinite(candidate.mW)) {
            r.mM += candidate.mM;
            continue;
        }

        const float candidateTargetPdf = candidate.mSample.GetTargetPdf();

        // shift candidate's sample to here
        //print("Shifting candidate %u (%d %d) with sample (%u %u) to center\n", i, p.x, p.y, candidate.mSample.mRngSeed, candidate.mSample.mPackedRngIdx);
        float jacobianToCenter;
        candidate.mSample = gPathGenerator.ShiftPath(candidate.mSample, vertex, jacobianToCenter);

		//print("candidate.f: %f -> %f, candidate.W %f, jacobianToCenter: %f\n", candidateTargetPdf, candidate.mSample.GetTargetPdf(), candidate.mW, jacobianToCenter);

        if (jacobianToCenter <= 0 || isnan(jacobianToCenter)) {
            r.mM += candidate.mM;
            continue;
        }

        float f, m;
        if (gMISType == RMISType::ePairwise) {
            f = candidateTargetPdf / jacobianToCenter;
            m = SafeDivide( candidate.mM * f, candidate.mM * f + (centerM / gPathGenerator.mParams.mSpatialReuseSamples) * candidate.mSample.GetTargetPdf() );
        } else if (gMISType == RMISType::eTalbot) {
            m = ComputeSpatialMisWeight(id, candidate, jacobianToCenter, centerM, i, candidateTargetPdf);
        } else if (gMISType == RMISType::eConstant) {
            m = candidate.mM;
        }

        candidate.PrepareMerge(m, jacobianToCenter);

        //print("r ris w: %f\tcandidate ris w: %f\n", r.mW, candidate.mW);

        if (r.Merge(sampleNext1D(sg), candidate)) {
            selectedIndex = i;
        }
    }

    if (gMISType == RMISType::ePairwise) {
        //print("%u valid, canonicalWeight: %f\n", validNeighbors, mc);
        if (center.mW > 0) {
            center.PrepareMerge(mc);
            //print("r ris w: %f\tcenter ris w: %f\n", r.mW, center.mW);
            r.Merge(sampleNext1D(sg), center);
        } else
            r.mM += center.mM;

        r.mW /= validNeighbors + 1;
    } else if (gMISType == RMISType::eConstant) {
        if (r.mW > 0) r.mW = SafeDivide(r.mW, ComputeSpatialZ(id, r, centerM, selectedIndex));
    }

    r.FinalizeMerge();

    if (gPathGenerator.mParams.mMCap > 0) r.mM = min(r.mM, gPathGenerator.mParams.mMCap);
    gPathGenerator.StoreReservoir(id, r);
}
/*
#define ReadReservoirAtLane(r, lane) \
	r.mW = WaveReadLaneAt(r.mW, lane); \
	r.mM = WaveReadLaneAt(r.mM, lane); \
    r.mSample.mReconnectionVertex.mVertex.mPosition               = WaveReadLaneAt(r.mSample.mReconnectionVertex.mVertex.mPosition              , lane); \
    r.mSample.mReconnectionVertex.mVertex.mInstancePrimitiveIndex = WaveReadLaneAt(r.mSample.mReconnectionVertex.mVertex.mInstancePrimitiveIndex, lane); \
	r.mSample.mReconnectionVertex.mRadiance                       = WaveReadLaneAt(r.mSample.mReconnectionVertex.mRadiance                      , lane); \
	r.mSample.mReconnectionVertex.mPackedDirOut                   = WaveReadLaneAt(r.mSample.mReconnectionVertex.mPackedDirOut                  , lane); \
	r.mSample.mReconnectionVertex.mDist                           = WaveReadLaneAt(r.mSample.mReconnectionVertex.mDist                          , lane); \
	r.mSample.mReconnectionVertex.mCos                            = WaveReadLaneAt(r.mSample.mReconnectionVertex.mCos                           , lane); \
	r.mSample.mRadiance      = WaveReadLaneAt(r.mSample.mRadiance,      lane); \
	r.mSample.mReplayPdfW    = WaveReadLaneAt(r.mSample.mReplayPdfW,    lane); \
	r.mSample.mRngSeed       = WaveReadLaneAt(r.mSample.mRngSeed,       lane); \
	r.mSample.mPackedRngIdx  = WaveReadLaneAt(r.mSample.mPackedRngIdx,  lane); \
	r.mSample.mPackedBounces = WaveReadLaneAt(r.mSample.mPackedBounces, lane); \
	r.mSample.mPacked        = WaveReadLaneAt(r.mSample.mPacked,        lane);

[shader("compute")]
[numthreads(32, 1, 1)]
void WavefrontSpatialReuse(uint3 groupId: SV_GroupID) {
	const uint index = groupId.y * 2048 + groupId.x * 32 + WaveGetLaneIndex();
    const uint pixelIndex     = index / gSpatialReuseSamples;
    const uint candidateIndex = index % gSpatialReuseSamples;
    const uint2 id = uint2(pixelIndex % mOutputDim.x, pixelIndex / mOutputDim.x);
    if (any(id >= mOutputDim))
        return;
    InitDebugPixel(id, mOutputDim);

    PathReservoir center = LoadReservoir(mPathReservoirsIn, id);

    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE) {
		if (candidateIndex == 0)
        	StoreReservoir(mPathReservoirsOut, id, center);
        return;
    }

    if (candidateIndex == 0) {
        DebugPrintf("\n\n---------- Performing spatial reuse (%u) ----------\n\n", gRandomSeed);
    }

    PathReservoir candidate = PathReservoir();
    bool valid = false;
    float mc = 0;

    const int2 p = GetSampleLocation(id, candidateIndex);
    if (!(any(p < 0) || any(p >= mOutputDim) || all(p == id))) {
        const PackedVertex candidateVertex = reinterpret<PackedVertex>(gVertices[p]);
        if (candidateVertex.mIsSurface) {
            candidate = LoadReservoir(mPathReservoirsIn, p);
			valid = true;

			// shift center sample to candidate
			if (center.mW > 0) {
				float jacobianToCandidate;
				const float shifted = ShiftPath(center.mSample, candidateVertex, gCameraPosition, jacobianToCandidate).GetTargetPdf();
				if (jacobianToCandidate > 0) {
					const float f = shifted * jacobianToCandidate;
					mc = SafeDivide(candidate.mM * f, candidate.mM * f + (center.mM / gSpatialReuseSamples) * center.mSample.GetTargetPdf());
				}
			}

			if (candidate.mW > 0) {
				IncrementCounter(DebugCounterType::eSpatialSamples);

                DebugPrintf("Shifting candidate %u (%d %d) with sample (%u %u) to center\n", candidateIndex, p.x, p.y, candidate.mSample.mRngSeed, candidate.mSample.mPackedRngIdx);

				// shift candidate to center
				const float candidateTargetPdf = candidate.mSample.GetTargetPdf();
				float jacobianToCenter;
				candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobianToCenter);
				DebugPrintf("candidate.f: %f -> %f, candidate.W %f, jacobianToCenter: %f\n", candidateTargetPdf, candidate.mSample.GetTargetPdf(), candidate.mW, jacobianToCenter)
				if (jacobianToCenter > 0) {
					const float f = candidateTargetPdf / jacobianToCenter;
					const float m = SafeDivide(candidate.mM * f, candidate.mM * f + (center.mM / gSpatialReuseSamples) * candidate.mSample.GetTargetPdf());
					candidate.PrepareMerge(m, jacobianToCenter);
				} else
				candidate.mW = 0;
			}
        }
    }

	RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip((2 + gSpatialReusePass) * 10000);
	rng.Skip(gSpatialReuseSamples);

    uint numValid = 0;
    float canonicalWeight = 0;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        if (valid) {
			numValid++;
            canonicalWeight += 1 - mc;
        }
        mc    = WaveReadLaneAt(mc   , WaveGetLaneIndex() + 1);
        valid = WaveReadLaneAt(valid, WaveGetLaneIndex() + 1);
    }

    center.PrepareMerge(1 + canonicalWeight);
    if (candidateIndex == 0) {
        DebugPrintf("%u valid, canonicalWeight: %f\n", numValid, 1+canonicalWeight)
    }

	for (uint i = 0; i < gSpatialReuseSamples; i++) {
		if (candidateIndex == 0) {
			DebugPrintf("merging with candidate %u: %f\n", i, candidate.mW)
			if (center.Merge(rng.NextFloat().x, candidate))
                IncrementCounter(DebugCounterType::eSpatialAcceptances);
        }
		ReadReservoirAtLane(candidate, WaveGetLaneIndex() + 1);
	}

	if (candidateIndex == 0) {
		center.FinalizeMerge();
		center.mW /= 1 + numValid;

        if (gMCap > 0) center.mM = min(center.mM, gMCap);
        StoreReservoir(mPathReservoirsOut, id, center);
    }
}
*/
