#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class RMISType {
    eConstant,
    ePairwise,
    eTalbot
};
FALCOR_ENUM_INFO(
    RMISType, {
        { RMISType::eConstant, "Constant" },
        { RMISType::ePairwise, "Pairwise" },
        { RMISType::eTalbot  , "Talbot" },
    }
);
FALCOR_ENUM_REGISTER(RMISType);

enum class DebugCounterType : uint {
    eScatterRays              = (1 << 0),
    eOcclusionRays            = (1 << 1),

    eTemporalShiftSuccesses   = (1 << 2),
    eSpatialShiftSuccesses    = (1 << 3),
    eCausticShiftSuccesses    = (1 << 4),

    eCausticReservoirM        = (1 << 5),

    eTemporalReuseAcceptances = (1 << 6),
    eSpatialReuseAcceptances  = (1 << 7),
    eCausticReuseAcceptances  = (1 << 8),

    eCausticShiftFailures     = (1 << 9),

    eRays             = 3,
    eShiftSuccesses   = (1 << 2) | (1 << 3) | (1 << 4),
    eReuseAcceptances = (1 << 5) | (1 << 6) | (1 << 7)
};
FALCOR_ENUM_INFO(
    DebugCounterType, {
        { DebugCounterType::eScatterRays, "Scatter rays" },
        { DebugCounterType::eOcclusionRays, "Occlusion rays" },
        { DebugCounterType::eRays, "Rays" },

        { DebugCounterType::eTemporalShiftSuccesses, "Temporal shift successes" },
        { DebugCounterType::eSpatialShiftSuccesses, "Spatial shift successes" },
        { DebugCounterType::eCausticShiftSuccesses, "Caustic shift successes" },
        { DebugCounterType::eShiftSuccesses, "Shift successes" },

        { DebugCounterType::eCausticReservoirM, "Caustic reservoir M" },

        { DebugCounterType::eCausticShiftFailures, "Caustic shift failures" },

        { DebugCounterType::eTemporalReuseAcceptances, "Temporal reuse acceptances" },
        { DebugCounterType::eSpatialReuseAcceptances, "Spatial reuse acceptances" },
        { DebugCounterType::eCausticReuseAcceptances, "Caustic reuse acceptances" },
        { DebugCounterType::eReuseAcceptances, "Reuse acceptances" },
    }
);
FALCOR_ENUM_REGISTER(DebugCounterType);


struct PathGeneratorParams {
    static const uint kMaxBounces = 0xFFFF; ///< reservoirs use 16 bits to store the number of bounces

    float4 mSceneSphere;

    uint2 mOutputDim;

    uint  mMaxBounces = 20;
    uint  mMaxDiffuseBounces = 8;
    float mTerminationProbability = .2f;
    uint  mLightSubpathCount = 1000000;
    uint  mPhotonCellCount   = 10000000;
    float mMergeRadius       = 0.0025f;
    float mDynamicMergeRadius = 2.f;

    float mReconnectionDistance  = 0.01f;
    float mReconnectionRoughness = 0.08f; ///< sqrt(kMinGGXAlpha)
    float mCausticReuseRadius = 0.f; // in pixels

    uint  mSpatialReuseSamples = 4;
    float mSpatialReuseRadius  = 30.f;
    uint  mMCap                = 20;

    float mDirectLightProb = 0.8f; ///< Probability of performing direct light sampling at each path vertex.
    float mMergeProb       = 0.5f; ///< Probability of performing vertex merigng at each path vertex.
    float mEnvMapProb      = 0.5f; ///< Probability of sampling the environment map (instead of emissive triangles).

    int mDebugTotalVertices  = -1; ///< Only render paths with this many vertices.
    int mDebugLightVertices  = -1; ///< Only render paths with this many light vertices.
    int mDebugDisableMerging = 0;  ///< Don't render paths that use vertex merging.
    DebugCounterType mDebugCounter = DebugCounterType::eRays;
};

END_NAMESPACE_FALCOR
