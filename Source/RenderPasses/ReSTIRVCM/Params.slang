#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class RMISType {
    eConstant,
    ePairwise,
    eTalbot,


};
FALCOR_ENUM_INFO(
    RMISType, {
        { RMISType::eConstant, "Constant" },
        { RMISType::ePairwise, "Pairwise" },
        { RMISType::eTalbot  , "Talbot" },
    }
);
FALCOR_ENUM_REGISTER(RMISType);

struct PathGeneratorParams {
    float4 mSceneSphere;

    uint2 mOutputDim;

    uint mMaxBounces = 10;
    uint  mLightSubpathCount = 1000000;
    uint  mPhotonCellCount   = 10000000;
    float mMergeRadius       = 0.0025f;

    float mReconnectionDistance  = 0.01f;
    float mReconnectionRoughness = 0.2f;
    float mCausticReuseRadius = 0.003f;

    uint  mSpatialReuseSamples = 4;
    float mSpatialReuseRadius  = 30.f;
    uint  mMCap                = 20;

    float mDirectLightProb = 0.8f; ///< Probability of performing direct light sampling at each path vertex.
    float mMergeProb       = 0.5f; ///< Probability of performing vertex merigng at each path vertex.
    float mEnvMapProb      = 0.5f; ///< Probability of sampling the environment map (instead of emissive triangles).

    int mDebugTotalVertices = -1; ///< Only render paths with this many vertices.
    int mDebugLightVertices = -1; ///< Only render paths with this many light vertices.
};

END_NAMESPACE_FALCOR
