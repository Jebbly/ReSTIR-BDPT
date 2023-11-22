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

struct PhotonMap
{
    RWStructuredBuffer<LightSubpathVertex> cellData;        ///< Sorted photons.
    RWStructuredBuffer<uint2>              dataIndices;     ///< Stores (cell index, indexInCell) for each photon.
    RWStructuredBuffer<uint>               cellChecksums;   ///< Cell checksums.
    RWStructuredBuffer<uint>               cellSizes;       ///< Number of photons in each cell.
    RWStructuredBuffer<uint>               cellDataOffsets; ///< Index in cellData where each cell starts. First element is the total number of elements.

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

    uint getCellIndex<let bAppending : bool>(const float3 posW, const float mergeRadius, const uint cellCount, const int3 offset = int3(0))
    {
        const int3 p = int3(floor(posW / (2.f * mergeRadius))) + offset;
        const uint checksum = max(1, xxhash32(xxhash32(p.z + xxhash32(p.y + xxhash32(p.x)))));
        const uint baseCellIndex = pcg(pcg(p.z + pcg(p.y + pcg(p.x)))) % cellCount;

        // resolve hash collisions with linear probing
        for (uint i = 0; i < 32; i++) {
            const uint cellIndex = (baseCellIndex + i) % cellCount;
            // find cell with matching checksum, or empty cell if inserting
            if (bAppending) {
                uint prevChecksum;
                InterlockedCompareExchange(cellChecksums[cellIndex], 0, checksum, prevChecksum);
                if (prevChecksum == 0 || prevChecksum == checksum)
                    return cellIndex;
            } else {
                if (cellChecksums[cellIndex] == checksum)
                    return cellIndex;
            }
        }

        // collision resolution failed
        return -1;
    }

    uint2 getCellDataRange(const uint cellIndex)
    {
        const uint offset = cellDataOffsets[cellIndex];
        return uint2(offset, offset + cellSizes[cellIndex]);
    }

    LightSubpathVertex getPhoton(const uint photonIndex)
    {
        return cellData[photonIndex];
    }

    void addPhoton(const uint lightVertexIndex, const float3 posW, const float mergeRadius, const uint cellCount)
    {
        const uint cellIndex = getCellIndex<true>(posW, mergeRadius, cellCount);

        uint indexInCell;
        InterlockedAdd(cellSizes[cellIndex], 1, indexInCell);

        dataIndices[lightVertexIndex] = uint2(cellIndex, indexInCell);
    }


    void computeCellOffsets(const uint cellIndex)
    {
        const uint size = cellSizes[cellIndex];
        uint offset;
        InterlockedAdd(cellDataOffsets[0], size, offset);
        cellDataOffsets[1 + cellIndex] = offset;
    }

    void sortPhotons(const LightVertexCache lvc, const uint lightVertexIndex)
    {
        const uint2 i = dataIndices[lightVertexIndex];
        const uint offset = cellDataOffsets[1 + i.x];
        cellData[offset + i.y] = lvc[lightVertexIndex];
    }
};
