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

void sampleCanonicalPaths(uint2 pixel)
{
    if (kTraceFromLight)
    {
        if (pixel.y * gPathTracer.params.frameDim.x + pixel.x >= gPathTracer.params.lightSubpathCount) return;

        // Trace one light subpath per thread.
        gScheduler.tracePath(pixel.x | (pixel.y << 12) | ((-1) << 24));
    }
    else
    {
        if (any(pixel >= gPathTracer.params.frameDim)) return;

        #if LIGHT_TRACE_ONLY == 0
        // Determine number of samples to take.
        uint samplesRemaining = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            samplesRemaining = gPathTracer.sampleCount[pixel];
        }

        gPathTracer.initializeReservoir(pixel, samplesRemaining);

        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);
            gScheduler.tracePath(pathID);

            // Use SER to compact active threads.
            if (kSamplesPerPixel == 0)
            {
                gScheduler.compact(samplesRemaining == 0);
            }
        }
        #endif

        gPathTracer.resolve(pixel);
    }
}

void temporalReuse(uint2 pixel)
{
    if (any(pixel >= gPathTracer.params.frameDim)) return;

}

void spatialReuse(uint2 pixel)
{
    if (any(pixel >= gPathTracer.params.frameDim)) return;

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
