#include "BitfieldMacros.slangh"
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.RaytracingInline;
__exported import PathVertex;

struct ReconnectionVertex {
    PackedPathVertex mVertex = {};
    float3 mRadiance = 0;
    float mDist = 0;
    float3 mDirOut = 0;
    float mCos = 0;
};
static const uint ReconnectionVertexSize = sizeof(ReconnectionVertex);

struct LightSubpathVertex {
    PackedPathVertex mVertex;
    float3 mThroughput;
    uint mPacked;
    float mIntegrationWeight;
    float dVCM;
    float dVC;
    float dVM;

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPacked, 0, 12), BF_GET(mPacked, 12, 12)); }
        set { BF_SET(mPacked, newValue.x, 0, 12); BF_SET(mPacked, newValue.y, 12, 12); }
    }
    property uint mBounces {
        get { return BF_GET(mPacked, 24, 8); }
        set { BF_SET(mPacked, newValue, 24, 8); }
    }

    float3 GetDirIn() {
        return mVertex.mRayDirection;
    }
};
static const uint LightSubpathVertexSize = sizeof(LightSubpathVertex);

// static_assert( LightSubpathVertexSize == ReconnectionVertexSize )

enum PathShiftType {
	eReplay,
	eReconnection,
	eMerge,
	eLightPath,
};

struct PathSample {
    ReconnectionVertex mReconnectionVertex = {};
    float3 mRadiance = 0;
    float mReplayPdfW = 0;
    uint mRngSeed = 0;
    uint mPackedRngIdx = 0;
    uint mPackedBounces = 0;
    uint mPacked = 0;

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 16), BF_GET(mPackedRngIdx, 16, 16)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 16); BF_SET(mPackedRngIdx, newValue.y, 16, 16); }
    }
    property uint mBounces {
        get { return BF_GET(mPackedBounces, 0, 12); }
        set { BF_SET(mPackedBounces, newValue, 0, 12); }
    }
    property uint mReplayBounces {
        get { return BF_GET(mPackedBounces, 12, 12); }
        set { BF_SET(mPackedBounces, newValue, 12, 12); }
    }
    property uint mLightVertices {
        get { return BF_GET(mPackedBounces, 24, 8); }
        set { BF_SET(mPackedBounces, newValue, 24, 8); }
    }
    property PathShiftType mShiftType {
        get { return (PathShiftType)BF_GET(mPacked, 0, 4); }
        set { BF_SET(mPacked, (uint)newValue, 0, 4); }
    }

    __init(const float3 radiance, const float replayPdfW, const uint bounces, const uint replayBounces, const uint rngSeed, const uint2 rngIdx, const ReconnectionVertex rcv, const PathShiftType shiftType, const uint lightVertices) {
        mRadiance = radiance;
        mReplayPdfW = replayPdfW;
        mBounces = bounces;
        mReplayBounces = replayBounces;
        mRngSeed = rngSeed;
        mRngIdx = rngIdx;
        mReconnectionVertex = rcv;
        mShiftType = shiftType;
        mLightVertices = lightVertices;
	}

    float GetTargetPdf() { return luminance(mRadiance); }
};
static const uint PathSampleSize = sizeof(PathSample);

struct PathReservoir {
    PathSample mSample = {};
    float mW = 0;
    float mM = 0;

    __init(const PathSample s, const float W, const float M) {
        mSample = s;
        mW = W;
        mM = M;
    }

    [mutating]
    void PrepareMerge(float misWeight = 1, float jacobian = 1) {
        mW *= mSample.GetTargetPdf() * misWeight * jacobian;
    }

    // note: PrepareMerge must be called on both reservoirs prior to calling Merge.
    [mutating]
    bool Merge(const float rnd, const PathSample s, const float W, const float M) {
        mM += M;

        if (W <= 0 || isnan(W) || isinf(W))
            return false;

        mW += W;
        if (rnd * mW < W) {
            mSample = s;
            return true;
        }
        return false;
    }

    // note: PrepareMerge must be called on both reservoirs prior to calling Merge.
    [mutating]
    bool Merge(const float rnd, const PathReservoir r) {
        return Merge(rnd, r.mSample, r.mW, r.mM);
    }

    [mutating]
    void FinalizeMerge() {
        const float p = mSample.GetTargetPdf();
        if (p > 0)
            mW /= p;
        else
            mW = 0;
    }
}
static const uint PathReservoirSize = sizeof(PathReservoir);
