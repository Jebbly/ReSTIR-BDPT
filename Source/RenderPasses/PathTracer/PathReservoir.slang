import Scene.HitInfo;
import Utils.Color.ColorHelpers;

/** Stores the information required to shift a path.
*/
struct PathReservoir
{
    float  W;                    ///< Integration weight, or RIS weight between beginMerge/endMerge calls.
    uint   M;                    ///< Number of samples streamed through this reservoir.
    uint   pathId;               ///< Path ID which encodes original pixel and sample index.
    uint   seed;                 ///< Path seed.

    float3 sampleF;              ///< Unweighted path contribution.
    uint   replayBounces;        ///< Number of bounces before the reconnection vertex.

    float3 suffixContribution;   ///< Unweighted contribution of the suffix, excluding the BSDF at the reconnection vertex. Only valid if reconnectionG > 0.
    float  replayPdf;            ///< Sampling pdf of the prefix. Used to compute the replay jacobian.

    float3 reconnectionDirOut;   ///< Direction towards the next vertex in the suffix. Only valid if reconnectionG > 0.
    float  reconnectionG;        ///< Reconnection geometry term. Used to compute the reconnection jacobian. If this is 0, only random replay is used.

    float3 reconnectionPosW;     ///< World-space position of the reconection vertex. Only valid if reconnectionG > 0.
    uint   pad;

    PackedHitInfo reconnectionHit; ///< Hit information for the reconnection vertex. Only valid if reconnectionG > 0.

    /** Returns the target weight for resampling.
    */
    float getTargetPdf() { return luminance(sampleF); }

    /** Returns the estimated radiance.
    */
    float3 getRadiance() {
        return sampleF * W;
    }

    /** Copies a sample from another reservoir.
        \param[in] r The reservoir to copy the sample from.
    */
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
    void beginMerge(float resamplingMis = 1, float jacobian = 1)
    {
        W *= getTargetPdf() * resamplingMis * jacobian;
    }

    /** Merges two reservoirs that are in the same domain.
        \param[in] rnd Random number used for merging.
        \param[in] r Candidate to merge with.
    */
    [mutating]
    bool merge(float rnd, const PathReservoir candidate) {
        M += candidate.M;

        if (candidate.W <= 0 || isnan(candidate.W) || isinf(candidate.W))
            return false;

        W += candidate.W;
        if (rnd * W < candidate.W) {
            copySample(candidate);
            return true;
        }
        return false;
    }

    [mutating]
    void endMerge()
    {
        const float p = getTargetPdf();
        if (p > 0) W /= p;
    }
};
static const uint PathReservoirSize = sizeof(PathReservoir);
