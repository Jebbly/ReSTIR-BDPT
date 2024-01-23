#include "BitfieldMacros.slangh"
import Scene.HitInfo;
import Random;
import PathVertex;

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
        return pcg(gHashOffset + pcg(p.z + pcg(p.y + pcg(p.x)))) % cellCount;
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
