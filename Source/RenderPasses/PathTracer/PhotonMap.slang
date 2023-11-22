#pragma once

import Scene.HitInfo;

struct LightSubpathVertex
{
    PackedHitInfo packedHit;
    float3 L;               ///< Throughput at the vertex.
    uint vertexIndex;       ///< Index of the vertex in the light sub-path
    float3 rayOrigin;       ///< origin of the ray we traced to find the vertex.
    float dVCM;             ///< dVCM at the vertex.
    float3 rayDirection;    ///< direction of the ray we traced to find the vertex.
    float dVC;              ///< dVC at the vertex.
    float dVM;              ///< dVM at the vertex.
    float pathPdf;          ///< Sampling pdf of the subpath (product solid-angle measure)
    float hitDist;          ///< Hit distance along the ray.
    uint _pad;

    float3 getPosW()
    {
        return rayOrigin + rayDirection * hitDist;
    }
};

struct LightVertexCache
{
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

    uint addVertex(const LightSubpathVertex v)
    {
        uint i;
        InterlockedAdd(lightVertexCount[0], 1, i);
        lightVertices[i] = v;
        return i;
    }
}

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

struct PhotonMap
{
    RWStructuredBuffer<uint> cellSizes;   ///< Number of photons in each cell.
    RWStructuredBuffer<uint> cellOffsets; ///< Index in cellData where each cell starts. First element is the total number of elements.

    uint getCellDataOffset(const uint cellIndex)
    {
        return cellOffsets[cellIndex];
    }

    uint getCellDataSize(const uint cellIndex)
    {
        return cellSizes[cellIndex];
    }

    uint getCellIndex(const float3 posW, const float mergeRadius, const uint cellCount, const int3 offset = int3(0))
    {
        const int3 p = int3(floor(posW / (2.f * mergeRadius))) + offset;
        return pcg(pcg(p.z + pcg(p.y + pcg(p.x)))) % cellCount;
    }

    void addPhoton(const uint lightVertexIndex, const float3 posW, const float mergeRadius, const uint cellCount)
    {
        const uint cellIndex = getCellIndex(posW, mergeRadius, cellCount);
        cellOffsets[cellIndex] = lightVertexIndex;
        InterlockedAdd(cellSizes[cellIndex], 1);
    }
};
