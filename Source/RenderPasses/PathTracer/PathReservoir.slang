import Scene.HitInfo;
import Utils.Color.ColorHelpers;

/** Stores the information required to shift a path.
*/
struct PathReservoir
{
    float  integrationWeight;    ///< Unbiased integration weight. This is W in the ReSTIR papers.
    uint   sampleCount;          ///< Number of samples streamed through this reservoir. This is M in the ReSTIR papers.
    uint   pathId;               ///< Path ID which encodes original pixel and sample index.
    uint   seed;                 ///< Path seed.

    float3 sampleF;              ///< Unweighted path contribution.
    uint   replayBounces;        ///< Number of bounces before the reconnection vertex.

    float3 suffixContribution;   ///< Unweighted contribution of the suffix, excluding the BSDF at the reconnection vertex. Only valid if reconnectionG > 0.
    float  replayPdf;            ///< Sampling pdf of the prefix. Used to compute the replay jacobian.

    float3 reconnectionDirOut;   ///< Direction towards the next vertex in the suffix. Only valid if reconnectionG > 0.
    float  reconnectionG;        ///< Reconnection geometry term. Used to compute the reconnection jacobian. If this is 0, only random replay is used.

    PackedHitInfo reconnectionHit; ///< Hit information for the reconnection vertex. Only valid if reconnectionG > 0.

    /** Returns the target weight for resampling.
    */
    float getTargetPdf() { return luminance(sampleF); }

    /** Returns the estimated radiance.
    */
    float3 getRadiance() { return sampleF * integrationWeight; }

    [mutating]
    void copySample(const PathReservoir r)
    {
        pathId = r.pathId;
        seed = r.seed;

        sampleF = r.sampleF;
        replayBounces = r.replayBounces;

        suffixContribution = r.suffixContribution;
        replayPdf = r.replayPdf;

        reconnectionDirOut = r.reconnectionDirOut;
        reconnectionG = r.reconnectionG;

        reconnectionHit = r.reconnectionHit;
    }

    [mutating]
    void prepareMerge(float misWeight = 1, float jacobian = 1) {
        integrationWeight *= getTargetPdf() * misWeight * jacobian;
    }

    // note: PrepareMerge must be called on both reservoirs prior to calling Merge.
    // FinalizeMerge must be called after merging.
    [mutating]
    bool merge(float rnd, const PathReservoir r) {
        sampleCount += r.sampleCount;

        if (r.integrationWeight <= 0 || isnan(r.integrationWeight) || isinf(r.integrationWeight))
            return false;

        integrationWeight += r.integrationWeight;
        if (rnd * integrationWeight < r.integrationWeight) {
            copySample(r);
            return true;
        }
        return false;
    }

    [mutating]
    void finalizeMerge() {
        const float p = getTargetPdf();
        integrationWeight = p > 0 ? integrationWeight / p : 0;
    }
};
