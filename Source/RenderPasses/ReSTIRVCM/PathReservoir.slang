#include "BitfieldMacros.slangh"
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Utils.Math.Ray;
import Scene.HitInfo;

float  SafeDivide(const float a, const float b) { return abs(b) > 0 ? a / b : 0; }
float3 SafeDivide(const float3 a, const float3 b) { return float3(SafeDivide(a.x, b.x), SafeDivide(a.y, b.y), SafeDivide(a.z, b.z)); }

struct PathId {
    uint mSeed;
    uint mPackedId;

    property uint2 mId {
        get {
            return uint2(
                BF_GET(mPackedId, 0, 16),
                BF_GET(mPackedId, 16, 16));
        }
        set {
            BF_SET(mPackedId, newValue.x, 0, 16);
            BF_SET(mPackedId, newValue.y, 16, 16);
        }
    }

    __init(uint seed, const uint2 id) {
        mSeed = seed;
        mId = id;
    }
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

    bool IsFinite() { return HitInfo(mPackedHit).isValid(); }
};

struct LightSubpathData {
    PackedHitInfo mFirstVertex = {};
    PathId mId;
    uint mPackedBounces = 0;
    uint mRefractMask = 0;

    property uint mVertexCount {
        get { return BF_GET(mPackedBounces, 0, 16); }
        set { BF_SET(mPackedBounces, newValue, 0, 16); }
    }

    property uint mDiffuseBounces {
        get { return BF_GET(mPackedBounces, 16, 16); }
        set { BF_SET(mPackedBounces, newValue, 16, 16); }
    }

    static LightSubpathData Create(
        const PackedHitInfo firstVertex,
        const PathId id,
        const uint vertexCount,
        const uint diffuseBounces,
        const uint refractMask ) {
        LightSubpathData d = {};
        d.mFirstVertex = firstVertex;
        d.mId = id;
        d.mVertexCount = vertexCount;
        d.mDiffuseBounces = diffuseBounces;
        d.mRefractMask = refractMask;
        return d;
    }
};

struct PathSuffix {
    PackedPathVertex mVertex = {};       ///< First vertex in the suffix.
    LightSubpathData mLightSubpath = {}; ///< Light subpath id
    float3 mContribution = 0;            ///< Contribution from reflectance at vertices after the reconnection vertex and emission.
    float mPdf = 0;                      ///< Product of solid-angle sampling PDFs after the reconnection vertex.
    float3 mDirOut;                      ///< Outgoing direction from the reconnection vertex.
    float mDist = 0;                     ///< Geometry term between the prefix and the suffix.
    float mCos = 0;                      ///< Geometry term between the prefix and the suffix.

    static PathSuffix Create(
        const float3 contribution,
        const float pdf,
        const PackedPathVertex reconnectionVertex,
        const float3 reconnectionDirOut,
        const float reconnectionDistance,
        const float reconnectionCos,
        const LightSubpathData lightSubpath ) {
        PathSuffix s = {};
        s.mContribution = contribution;
        s.mPdf = pdf;
        s.mDirOut = reconnectionDirOut;
        s.mDist   = reconnectionDistance;
        s.mCos    = reconnectionCos;
        s.mVertex = reconnectionVertex;
        s.mLightSubpath = lightSubpath;
        return s;
    }
};

enum PathFlags : uint {
    eNone             = 0,
    eCausticLightPath = 1,
    eVertexMerging    = 2,
};


// Represents an unweighted path sample, which can be shifted between domains.
struct PathSample {
    PathId mId;
    uint mPackedBounces = 0;
    uint mFlags = 0;

    float3 mContribution = 0; ///< Full path contribution. This is the product of BSDFs from each vertex, and emission from the last vertex.
    float  mPrefixPdf = 0;    ///< Product of solid-angle sampling PDFs before the reconnection vertex (excluding the bounce which ends on the reconnection vertex).

    PathSuffix mSuffix = {};

    // Number of bounces in the full path.
    property uint mBounces {
        get { return BF_GET(mPackedBounces, 0, 11); }
        set { BF_SET(mPackedBounces, newValue, 0, 11); }
    }

    // Number of bounces before the reconnection vertex.
    property uint mPrefixBounces {
        get { return BF_GET(mPackedBounces, 11, 11); }
        set { BF_SET(mPackedBounces, newValue, 11, 11); }
    }

    // Number of diffuse bounces before the reconnection vertex.
    property uint mPrefixDiffuseBounces {
        get { return BF_GET(mPackedBounces, 22, 10); }
        set { BF_SET(mPackedBounces, newValue, 22, 10); }
    }


    bool IsValid() { return any(mContribution > 0) && mPrefixPdf > 0; }
    bool HasFlag(const PathFlags flag) { return (mFlags & uint(flag)) != 0; };
    bool HasSuffix() { return mBounces > mPrefixBounces; }

    float GetTargetPdf() { return luminance(mContribution); }

    static PathSample Create(
        const float3 contrib,
        const uint bounces,
        const float prefixPdf,
        const uint prefixBounces,
        const uint prefixDiffuseBounces,
        const PathId id,
        const PathSuffix suffix,
        const PathFlags flags ) {
        PathSample ps = {};
        ps.mContribution = contrib;
        ps.mPrefixPdf = prefixPdf;
        ps.mBounces = bounces;
        ps.mPrefixBounces = prefixBounces;
        ps.mPrefixDiffuseBounces = prefixDiffuseBounces;
        ps.mFlags = uint(flags);
        ps.mId = id;
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
