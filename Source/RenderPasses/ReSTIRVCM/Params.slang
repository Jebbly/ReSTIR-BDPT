#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class RMISType {
    eConstant,
    ePairwise,
    eTalbot,
};

struct PathGeneratorParams {
    uint2 mOutputDim;
    uint mRandomSeed;
    uint mMaxBounces = 10;

    float4 mSceneSphere;

    uint  mLightSubpathCount = 1000000;
    uint  mPhotonCellCount   = 10000000;
    float mMergeRadius       = 0.0025f;

    float mReconnectionDistance  = 0.0001f;
    float mReconnectionRoughness = 0.2f;

    uint mPathReservoirIndex = 0;
    uint  mSpatialReuseSamples = 5;
    float mSpatialReuseRadius  = 30.f;
    uint  mMCap                = 20;

    float mDirectLightProb = 0.8f; ///< Probability of performing direct light sampling at each path vertex.
    float mMergeProb       = 0.5f; ///< Probability of performing vertex merigng at each path vertex.

    float mEnvMapProb      = 0.5f; ///< Probability of sampling the environment map (instead of emissive triangles).

    int mDebugTotalVertices = -1; ///< Only render paths with this many vertices.
    int mDebugLightVertices = -1; ///< Only render paths with this many light vertices.
};

END_NAMESPACE_FALCOR
