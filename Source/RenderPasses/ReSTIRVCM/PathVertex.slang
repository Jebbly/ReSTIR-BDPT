#include "BitfieldMacros.slangh"
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Math.Ray;
import Scene.HitInfo;

float SafeDivide(const float a, const float b) {
    return abs(b) > 0 ? a/b : 0;
}

struct PackedPathVertex {
    PackedHitInfo mPackedHit = 0; ///< Packed hit information.
    float3 mRayOrigin;            ///< Origin of the incoming ray (or the outgoing ray for the first light subpath vertex)
    float mRayDistance;           ///< Hit distance of the incoming ray (or the outgoing ray for the first light subpath vertex)
    float3 mRayDirection;         ///< Direction of the incoming ray (or the outgoing ray for the first light subpath vertex)
    float pad1;

    __init(const Ray ray, const HitInfo hit, const float hitDist) {
        mRayOrigin = ray.origin;
        mRayDistance = hitDist;
        mRayDirection = ray.dir;
        mPackedHit = hit.getData();
    }

    bool IsFinite() {
        return HitInfo(mPackedHit).isValid();
    }
};

struct ReconnectionVertex {
    PackedPathVertex mVertex = {};
    float3 mRadiance = 0;
    float mDist = 0;
    float3 mDirOut = 0;
    float mCos = 0;
};

enum PathShiftType {
    eReplay,
    eReconnection,
	eMerge,
};

struct PathSample {
    float3 mRadiance = 0;
    float mReplayPdfW = 0;
    uint mRngSeed = 0;
    uint mPackedRngIdx = 0;
    uint mPackedBounces = 0;
    uint mPackedFlags = 0;
    ReconnectionVertex mReconnectionVertex = {};

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 16), BF_GET(mPackedRngIdx, 16, 16)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 16); BF_SET(mPackedRngIdx, newValue.y, 16, 16); }
    }
    property uint mBounces {
        get { return BF_GET(mPackedBounces, 0, 11); }
        set { BF_SET(mPackedBounces, newValue, 0, 11); }
    }
    property uint mReplayBounces {
        get { return BF_GET(mPackedBounces, 11, 11); }
        set { BF_SET(mPackedBounces, newValue, 11, 11); }
    }
    property uint mReplayDiffuseBounces {
        get { return BF_GET(mPackedBounces, 22, 10); }
        set { BF_SET(mPackedBounces, newValue, 22, 10); }
    }
    property PathShiftType mShiftType {
        get { return (PathShiftType)BF_GET(mPackedFlags, 0, 4); }
        set { BF_SET(mPackedFlags, (uint)newValue, 0, 4); }
    }
    property bool mIsCausticPath {
        get { return (bool)BF_GET(mPackedFlags, 4, 1); }
        set { BF_SET(mPackedFlags, (uint)newValue, 4, 1); }
    }

    __init(const float3 radiance, const float replayPdfW, const uint bounces, const uint replayBounces, const uint replayDiffuseBounces, const uint rngSeed, const uint2 rngIdx, const ReconnectionVertex rcv, const PathShiftType shiftType) {
        mRadiance = radiance;
        mReplayPdfW = replayPdfW;
        mBounces = bounces;
        mReplayBounces = replayBounces;
        mReplayDiffuseBounces = replayDiffuseBounces;
        mRngSeed = rngSeed;
        mRngIdx = rngIdx;
        mReconnectionVertex = rcv;
        mShiftType = shiftType;
	}

    float GetTargetPdf() { return luminance(mRadiance); }
    bool IsValid() { return any(mRadiance > 0) && !any(mRadiance < 0) && mReplayPdfW > 0; }
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
