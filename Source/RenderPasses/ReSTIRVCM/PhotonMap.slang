#include "BitfieldMacros.slangh"
import Scene.HitInfo;
import PathReservoir; // for PackedPathVertex

struct RecursiveMisState {
    float dVCM = 0;
    float dVC = 0;
    float dVM = 0;
    uint pad;
};

struct LightSubpathVertex {
    LightSubpathData mLightSubpath = {};
    PackedPathVertex mVertex = {};
    float3 mThroughput = 0;
    uint pad;
    RecursiveMisState mis = {};

    property uint mBounces {
        get { return mLightSubpath.mVertexCount - 2; }
    }
    property uint mDiffuseBounces {
        get { return mLightSubpath.mDiffuseBounces; }
        set { mLightSubpath.mDiffuseBounces = newValue; }
    }

    float3 GetDirIn() { return -mVertex.mRayDirection; }
};
static const uint LightSubpathVertexSize = sizeof(LightSubpathVertex);


// xxhash https://github.com/Cyan4973/xxHash
// implementation from https://www.shadertoy.com/view/Xt3cDn
uint xxhash32(const uint p) {
    const uint PRIME32_2 = 2246822519U, PRIME32_3 = 3266489917U;
    const uint PRIME32_4 = 668265263U, PRIME32_5 = 374761393U;
    uint h32 = p + PRIME32_5;
    h32 = PRIME32_4 * ((h32 << 17) | (h32 >> (32 - 17)));
    h32 = PRIME32_2 * (h32 ^ (h32 >> 15));
    h32 = PRIME32_3 * (h32 ^ (h32 >> 13));
    return h32 ^ (h32 >> 16);
}

// PCG (https://www.pcg-random.org/)
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct LightVertexCache {
    RWStructuredBuffer<LightSubpathVertex> lightVertices; ///< Diffuse light sub-path vertices.
    RWStructuredBuffer<uint> lightVertexCount;            ///< Number of items in lightVertices.

    uint size()
    {
        return lightVertexCount[0];
    }

    __subscript(uint i) -> LightSubpathVertex
    {
        get { return lightVertices[i]; }
        set { lightVertices[i] = newValue; }
    }

    uint Insert(const LightSubpathVertex v)
    {
        uint i;
        InterlockedAdd(lightVertexCount[0], 1, i);
        lightVertices[i] = v;
        return i;
    }
}

struct PhotonMap {
    RWStructuredBuffer<uint> cellSizes;   ///< Number of photons in each cell.
    RWStructuredBuffer<uint> cellOffsets; ///< Index in cellData where each cell starts. First element is the total number of elements.

    uint gHashOffset = 0;

    uint getCellOffset(const uint cellIndex)
    {
        return cellOffsets[cellIndex];
    }

    uint getCellPhotonCount(const uint cellIndex)
    {
        return cellSizes[cellIndex];
    }

    uint getCellIndex(const float3 posW, const float mergeRadius, const uint cellCount, const int3 offset = int3(0))
    {
        const int3 p = int3(floor(posW / (2.f * mergeRadius))) + offset;
        uint v = pcg(p.z + pcg(p.y + pcg(p.x)));
        v = pcg(gHashOffset + v);
        return v % cellCount;
    }

    void addPhoton(const uint lightVertexIndex, const float3 posW, const float mergeRadius, const uint cellCount, const float rnd)
    {
        const uint cellIndex = getCellIndex(posW, mergeRadius, cellCount);

        uint photonCount;
        InterlockedAdd(cellSizes[cellIndex], 1, photonCount);

        // stochastically write the photon
        if (rnd * (photonCount + 1) < 1)
        {
            cellOffsets[cellIndex] = lightVertexIndex;
        }
    }
};
