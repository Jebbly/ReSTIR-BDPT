/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "Scheduler.slangh"

import Utils.Math.MathHelpers;

static const uint kMaxSpp = kSamplesPerPixel == 0 ? kMaxSamplesPerPixel : kSamplesPerPixel;

float safeDivide(float x, float y) { return y > 0 ? x / y : 0; }

uint getLightTraceSeed()
{
    const uint numSeeds =
        1 +             // light subpaths
        1 +             // canonical RIS merging
        kMaxSpp +       // camera subpaths
        gPathTracer.params.useTemporalReuse +  // temporal reuse merging
        gPathTracer.params.spatialReusePasses; // spatial reuse offsets + merging
    return gPathTracer.params.seed * numSeeds;
}
uint getCameraTraceSeed()
{
    return getLightTraceSeed() + 1;
}
uint getTemporalReuseSeed()
{
    return getCameraTraceSeed() + 1 + kMaxSpp;
}
uint getSpatialReuseSeed()
{
    return getTemporalReuseSeed() + 1 + gPathTracer.params.spatialReusePassIndex;
}


void sampleCanonicalPaths(uint2 pixel)
{
    if (kTraceFromLight)
    {
        if (pixel.y * gPathTracer.params.frameDim.x + pixel.x >= gPathTracer.params.lightSubpathCount) return;

        // Trace one light subpath per thread. For light tracing, samples are accumulated
        // into the lightImage directly, and reservoirs are ignored.
        gScheduler.tracePath(pixel.x | (pixel.y << 12) | ((-1) << 24), getLightTraceSeed());
    }
    else
    {
        if (any(pixel >= gPathTracer.params.frameDim)) return;

        // Determine number of samples to take.
        uint samplesRemaining = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            samplesRemaining = gPathTracer.sampleCount[pixel];
        }

        PathReservoir pixelReservoir = {};
        pixelReservoir.M = 0;
        pixelReservoir.W = 0;
        pixelReservoir.sampleF = 0;

        uint seed = getCameraTraceSeed();
        SampleGenerator sg = SampleGenerator(pixel, seed);
        seed++;

        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            const uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);

            const PathReservoir candidate = gScheduler.tracePath(pathID, seed + samplesRemaining);

            if (kDebugMIS)
            {
                pixelReservoir.sampleF += candidate.sampleF;
                pixelReservoir.M++;
            }
            else
            {
                pixelReservoir.merge(sampleNext1D(sg), candidate);
            }

            // Use SER to compact active threads.
            if (kSamplesPerPixel == 0)
            {
                gScheduler.compact(samplesRemaining == 0);
            }
        }

        pixelReservoir.endMerge();
        pixelReservoir.W /= pixelReservoir.M;

        gPathTracer.pathReservoirs[pixel.y * gPathTracer.params.frameDim.x + pixel.x] = pixelReservoir;
    }
}

void temporalReuse(uint2 pixel)
{
    if (any(pixel >= gPathTracer.params.frameDim)) return;

    const HitInfo primaryHit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!primaryHit.isValid()) return;

    const int2 lastPixel = int2(pixel + 0.5 + gPathTracer.params.frameDim * gPathTracer.motionVecs[pixel]);
    if (any(lastPixel < 0) || any(lastPixel >= gPathTracer.params.frameDim)) return;

    const HitInfo lastPrimaryHit = HitInfo(gPathTracer.lastVbuffer[lastPixel]);
    if (!lastPrimaryHit.isValid()) return;

    PathReservoir reservoir = gPathTracer.pathReservoirs[pixel.y * gPathTracer.params.frameDim.x + pixel.x];
    PathReservoir candidate = gPathTracer.lastPathReservoirs[lastPixel.y * gPathTracer.params.frameDim.x + lastPixel.x];

    // Apply resampling MIS to the current reservoir.
    if (reservoir.W > 0)
    {
        Ray lastCameraRay = gScene.camera.computeRayPinholePrevFrame(lastPixel, gPathTracer.params.frameDim);
        if (kUseViewDir) lastCameraRay.dir = -gPathTracer.lastViewDir[pixel];

        float jacobianToPrev;
        const PathReservoir shifted = gScheduler.shiftPath(reservoir, lastCameraRay, lastPrimaryHit, jacobianToPrev);

        const float mi = reservoir.M * reservoir.getTargetPdf();
        const float msum = mi + candidate.M * shifted.getTargetPdf() * jacobianToPrev;

        reservoir.beginMerge(safeDivide(mi, msum));
    }

    // Shift the candidate's sample to the current pixel and apply resampling MIS to the candidate.
    if (candidate.W > 0)
    {
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, gPathTracer.params.frameDim);
        if (kUseViewDir) cameraRay.dir = -gPathTracer.viewDir[pixel];

        const float prevTargetPdf = candidate.getTargetPdf();

        float jacobianToCurrent;
        candidate = gScheduler.shiftPath(candidate, cameraRay, primaryHit, jacobianToCurrent);

        const float mi = safeDivide(candidate.M * prevTargetPdf, jacobianToCurrent);
        const float msum = mi + reservoir.M * candidate.getTargetPdf() * jacobianToCurrent;

        candidate.beginMerge(safeDivide(mi, msum), jacobianToCurrent);
    }

    SampleGenerator sg = SampleGenerator(pixel, getTemporalReuseSeed());
    reservoir.merge(sampleNext1D(sg), candidate);

    reservoir.endMerge();

    reservoir.M = min(reservoir.M, gPathTracer.params.reservoirMCap);
    gPathTracer.pathReservoirs[pixel.y * gPathTracer.params.frameDim.x + pixel.x] = reservoir;

}

PathReservoir loadSpatialReservoir(const uint2 pixel)
{
    const uint idx = pixel.y * gPathTracer.params.frameDim.x + pixel.x;
    if ((gPathTracer.params.spatialReusePassIndex & 1) == 0)
    {
        return gPathTracer.pathReservoirs[idx];
    }
    else
    {
        return gPathTracer.lastPathReservoirs[idx];
    }
}
void storeSpatialReservoir(const uint2 pixel, const PathReservoir r)
{
    const uint idx = pixel.y * gPathTracer.params.frameDim.x + pixel.x;
    if ((gPathTracer.params.spatialReusePassIndex & 1) == 0)
    {
        gPathTracer.lastPathReservoirs[idx] = r;
    }
    else
    {
        gPathTracer.pathReservoirs[idx] = r;
    }
}
void spatialReuse(uint2 pixel)
{
    if (any(pixel >= gPathTracer.params.frameDim)) return;

    const HitInfo centerHit = HitInfo(gPathTracer.vbuffer[pixel]);
    if (!centerHit.isValid()) return;

    Ray centerRay = gScene.camera.computeRayPinhole(pixel, gPathTracer.params.frameDim);
    if (kUseViewDir) centerRay.dir = -gPathTracer.viewDir[pixel];

    PathReservoir centerReservoir = loadSpatialReservoir(pixel);

    PathReservoir reservoir = {};
    reservoir.W = 0;
    reservoir.M = 0;
    reservoir.sampleF = 0;

    uint validNeighbors = 0;
    float mc = 1;

    SampleGenerator sg = SampleGenerator(pixel, getSpatialReuseSeed());

    for (uint i = 0; i < gPathTracer.params.spatialReuseCandidates; i++) {
        const int2 p = int2(sample_disk_concentric(sampleNext2D(sg)) * gPathTracer.params.spatialReuseRadius);

        if (any(p < 0) || any(p >= gPathTracer.params.frameDim) || all(p == pixel))
            continue;

        const HitInfo candidateHit = HitInfo(gPathTracer.vbuffer[p]);
        if (!candidateHit.isValid()) continue;

        Ray candidateRay = gScene.camera.computeRayPinhole(p, gPathTracer.params.frameDim);
        if (kUseViewDir) candidateRay.dir = -gPathTracer.viewDir[p];

        PathReservoir candidate = loadSpatialReservoir(p);

        validNeighbors++;
        mc += 1;
        if (centerReservoir.W > 0 && candidate.M > 0) {
            float jacobianToCandidate;
            const PathReservoir shifted = gScheduler.shiftPath(centerReservoir, candidateRay, candidateHit, jacobianToCandidate);
            if (jacobianToCandidate > 0) {
                const float f = shifted.getTargetPdf() * jacobianToCandidate;
                mc -= safeDivide( candidate.M * f, candidate.M * f + (centerReservoir.M / float(gPathTracer.params.spatialReuseCandidates)) * centerReservoir.getTargetPdf());
            }
        }

        if (candidate.M == 0)
            continue;

        if (!(candidate.W > 0) || !isfinite(candidate.W)) {
            reservoir.M += candidate.M;
            continue;
        }

        const float candidateTargetPdf = candidate.getTargetPdf();

        // shift candidate's sample to here
        float jacobianToCenter;
        candidate = gScheduler.shiftPath(candidate, centerRay, centerHit, jacobianToCenter);

        if (jacobianToCenter <= 0 || isnan(jacobianToCenter)) {
            reservoir.M += candidate.M;
            continue;
        }

        const float f = candidateTargetPdf / jacobianToCenter;
        const float m = safeDivide(candidate.M * f, candidate.M * f + (centerReservoir.M / float(gPathTracer.params.spatialReuseCandidates)) * candidate.getTargetPdf());

        candidate.beginMerge(m, jacobianToCenter);

        reservoir.merge(sampleNext1D(sg), candidate);
    }

    {
        centerReservoir.beginMerge(mc);

        reservoir.merge(sampleNext1D(sg), centerReservoir);
    }

	reservoir.endMerge();
    reservoir.W /= validNeighbors + 1;

    reservoir.M = min(reservoir.M, gPathTracer.params.reservoirMCap);
    storeSpatialReservoir(pixel, reservoir);
}

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

#if defined(TEMPORAL_REUSE_PASS)
    temporalReuse(pixel);
#elif defined(SPATIAL_REUSE_PASS)
    spatialReuse(pixel);
#else
    sampleCanonicalPaths(pixel);
#endif
}
