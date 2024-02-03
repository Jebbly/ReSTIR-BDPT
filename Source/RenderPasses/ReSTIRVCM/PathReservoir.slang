#include "BitfieldMacros.slangh"
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Math.Ray;
import Scene.HitInfo;

float  SafeDivide(const float a, const float b) { return abs(b) > 0 ? a / b : 0; }
float3 SafeDivide(const float3 a, const float3 b) { return float3( SafeDivide(a.x, b.x), SafeDivide(a.y, b.y), SafeDivide(a.z, b.z) ); }

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

    bool IsFinite() { return HitInfo(mPackedHit).isValid(); }
};

struct PathSuffix {
    float3 mContribution = 0;      ///< Contribution from vertices after the reconnection vertex.
    float mPdf = 0;                ///< Product of solid-angle sampling PDFs after the reconnection vertex.
    PackedPathVertex mVertex = {}; ///< First vertex in the suffix.
    float3 mDirOut;                ///< Outgoing direction at the reconnection vertex.
    float mDist = 0;               ///< Geometry term between the prefix and the suffix.
    float mCos = 0;                ///< Geometry term between the prefix and the suffix.
    uint mNumVertices = 0;         ///< Includes the reconnection vertex.
};

enum PathFlags : uint {
    eNone             = 0,
    eMergeAtRcv       = 1,
    eCausticLightPath = 2
};

// Represents an unweighted path sample, which can be shifted between domains.
struct PathSample {
    float3 mContribution = 0; ///< Full path contribution. This is the product of BSDFs from each vertex, and emission from the last vertex.
    float  mPrefixPdf = 0;    ///< Product of solid-angle sampling PDFs before the reconnection vertex (excluding the bounce which ends on the reconnection vertex).

    uint2 mPackedBounces = 0;
    uint  mRngSeed;
    uint  mPackedRngIdx;

    PathSuffix mSuffix = {};
    PackedHitInfo mLightSubpathVertex = {}; ///< First vertex in the light subpath

    // Number of bounces before the reconnection vertex.
    property uint mPrefixBounces {
        get { return BF_GET(mPackedBounces[0], 0, 16); }
        set { BF_SET(mPackedBounces[0], newValue, 0, 16); }
    }

    // Number of diffuse bounces in the full path.
    property uint mPrefixDiffuseBounces {
        get { return BF_GET(mPackedBounces[0], 16, 16); }
        set { BF_SET(mPackedBounces[0], newValue, 16, 16); }
    }

    // Number of diffuse bounces in the full path.
    property uint mNumLightSubpathVertices {
        get { return BF_GET(mPackedBounces[1], 0, 16); }
        set { BF_SET(mPackedBounces[1], newValue, 0, 16); }
    }

    property uint mFlags {
        get { return BF_GET(mPackedBounces[1], 16, 16); }
        set { BF_SET(mPackedBounces[1], newValue, 16, 16); }
    }

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 16), BF_GET(mPackedRngIdx, 16, 16)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 16); BF_SET(mPackedRngIdx, newValue.y, 16, 16); }
    }

    bool IsValid() { return any(mContribution > 0) && mPrefixPdf > 0; }

    bool HasFlag(const PathFlags flag) { return (mFlags & uint(flag)) != 0; };

    float GetTargetPdf() { return luminance(mContribution); }

    static PathSample Create(
        const float3 contrib,
        const float prefixPdf,
        const uint prefixBounces,
        const uint prefixDiffuseBounces,
        const uint lightSubpathVertices,
        const uint rngSeed,
        const uint2 rngIdx,
        const PathSuffix suffix,
        const PathFlags flags ) {
        PathSample ps = {};
        ps.mContribution = contrib;
        ps.mPrefixPdf = prefixPdf;
        ps.mPrefixBounces = prefixBounces;
        ps.mPrefixDiffuseBounces = prefixDiffuseBounces;
        ps.mNumLightSubpathVertices = lightSubpathVertices;
        ps.mRngSeed = rngSeed;
        ps.mRngIdx = rngIdx;
        ps.mFlags = uint(flags);
        ps.mSuffix = suffix;
        return ps;
	}
};

// Represents a path sample and its weight for integration/resampling.
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

    // After calling this function, the reservoir can be used for integration.
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
