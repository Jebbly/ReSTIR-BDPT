/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

/** This file contains helper functions for analytic light source sampling.
    The code supports Falcor's analytic point, directional, and area lights,
    which are all defined in the scene description.

    Mesh lights (emissive geometry) and light probes are handled separately.

    This is work in progress. The code is not very well-tested.
*/

#include "Utils/Math/MathConstants.slangh"

import Scene.Lights.LightData;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import PathState;
import Params;
import LoadShadingData;

static const float kMinLightDistSqr  = 1e-9f;
static const float kMaxLightDistance = FLT_MAX;
static const bool kUseEnvLight       = USE_ENV_LIGHT;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;

float3 generateSceneSphereSample(const float4 sceneSphere, const float3 dir, inout SampleGenerator sg)
{
    float2 uv = sample_disk_concentric(sampleNext2D(sg));
    float3 t, b;
    branchlessONB(dir, t, b);
    return sceneSphere.xyz + sceneSphere.w * (t * uv.x + b * uv.y + dir);
}
float getSceneSpherePdf(const float4 sceneSphere)
{
    return M_1_PI / (sceneSphere.w * sceneSphere.w);
}

/** Describes a light sample.
 */
struct LightSample
{
    float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float pdf;      ///< Pdf with respect to solid angle at the shading point.
    float emitPdf;  ///< Emission pdf with respect to solid angle at the light surface. This is already multiplied by the area-measure light sampling pdf.
    float cosLight; ///< Cosine term at the light surface for area lights, or 1 for non-area lights.
    float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;     ///< Ray direction for visibility evaluation (normalized).
    uint flags;

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }

    bool getIsFinite()   { return (flags & 1u) != 0; }
    bool getIsDelta()    { return (flags & 2u) != 0; }
    bool getIsAnalytic() { return (flags & 4u) != 0; }

    [mutating] void setIsFinite  (bool newValue) { flags = ((flags & ~1u) | (newValue ? 1u : 0u)); }
    [mutating] void setIsDelta   (bool newValue) { flags = ((flags & ~2u) | (newValue ? 2u : 0u)); }
    [mutating] void setIsAnalytic(bool newValue) { flags = ((flags & ~4u) | (newValue ? 4u : 0u)); }

};

/**
 * Computes new ray origin based on hit position to avoid self-intersections.
 */
float3 computeRayOrigin(const ShadingData sd, const float3 dir)
{
    return computeRayOrigin(sd.posW, dot(dir, sd.faceN) >= 0.f ? sd.faceN : -sd.faceN);
}


/** Internal helper function to finalize the shared computations for area light samples.
    The input sample must already have posW and normalW computed.
    \return True if the sample is valid, false otherwise.
*/
bool finalizeAreaLightSample(const Optional<ShadingData> sd, const float3 posW, const float3 normalW, const float surfaceArea, inout LightSample ls)
{
    ls.setIsDelta(false);
    ls.setIsFinite(true);

    // Compute direction and distance to light.
    // The distance is clamped to a small epsilon to avoid div-by-zero below.
    float3 toLight = posW - sd.value.posW;
    float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
    ls.origin = computeRayOrigin(sd.value, toLight);
    ls.distance = sqrt(distSqr);
    ls.dir = toLight / ls.distance;

    // Compute incident radiance at shading point.
    // The area lights are single-sided by default, so radiance is zero when seen from the back-facing side.
    ls.cosLight = dot(normalW, -ls.dir);
    if (ls.cosLight <= 0.f) return false;
    ls.Li *= surfaceArea * ls.cosLight / distSqr;

    ls.emitPdf = ls.cosLight * M_1_PI / surfaceArea;

    // Compute the PDF with respect to solid angle. Note this may be +inf.
    ls.pdf = distSqr / (ls.cosLight * surfaceArea);
    return true;
}

/** Samples a rectangular area light source.
    \param[in] shadingPosW Shading point in world space.
    \param[in] light Light data.
    \param[in] u Uniform random numbers.
    \param[out] ls Light sample struct.
    \return True if a sample was generated, false otherwise.
*/
bool sampleRectAreaLight(const Optional<ShadingData> sd, const LightData light, const float2 u, out LightSample ls)
{
    // Pick a random sample on the quad.
    // The quad is from (-1,-1,0) to (1,1,0) in object space, but may be scaled by its transform matrix.
    float3 pos = float3(u.x * 2.f - 1.f, u.y * 2.f - 1.f, 0.f);

    // Apply model to world transformation matrix.
    const float3 posW = mul(light.transMat, float4(pos, 1.f)).xyz;

    // Setup world space normal.
    // TODO: normalW is not correctly oriented for mesh instances that have flipped triangle winding.
    const float3 normalW = normalize(mul(light.transMatIT, float4(0.f, 0.f, 1.f, 0.f)).xyz);

    ls.Li = light.intensity;

    return finalizeAreaLightSample(sd, posW, normalW, light.surfaceArea, ls);
}

/** Samples a spherical area light source.
    \param[in] shadingPosW Shading point in world space.
    \param[in] light Light data.
    \param[in] u Uniform random numbers.
    \param[out] ls Light sample struct.
    \return True if a sample was generated, false otherwise.
*/
bool sampleSphereAreaLight(const Optional<ShadingData> sd, const LightData light, const float2 u, out LightSample ls)
{
    // Sample a random point on the sphere.
    // TODO: We should pick a random point on the hemisphere facing the shading point.
    float3 pos = sample_sphere(u);

    // Apply model to world transformation matrix.
    const float3 posW = mul(light.transMat, float4(pos, 1.f)).xyz;

    // Setup world space normal.
    const float3 normalW = normalize(mul(light.transMatIT, float4(pos, 0.f)).xyz);

    ls.Li = light.intensity;

    return finalizeAreaLightSample(sd, posW, normalW, light.surfaceArea, ls);
}

/** Samples disc area light source.
    \param[in] shadingPosW Shading point in world space.
    \param[in] light Light data.
    \param[in] Uniform random numbers.
    \param[out] ls Light sample struct.
    \return True if a sample was generated, false otherwise.
*/
bool sampleDiscAreaLight(const Optional<ShadingData> sd, const LightData light, const float2 u, out LightSample ls)
{
    // Sample a random point on the disk.
    // TODO: Fix spelling disagreement between disc vs disk.
    float3 pos = float3(sample_disk(u), 0.f);

    // Transform to world space.
    const float3 posW = mul(light.transMat, float4(pos, 1.f)).xyz;

    // Setup world space normal.
    const float3 normalW = normalize(mul(light.transMatIT, float4(0.f, 0.f, 1.f, 0.f)).xyz);

    ls.Li = light.intensity;

    return finalizeAreaLightSample(sd, posW, normalW, light.surfaceArea, ls);
}

/** Samples a distant light source.
    \param[in] shadingPosW Shading point in world space.
    \param[in] light Light data.
    \param[in] u Uniform random numbers.
    \param[out] ls Light sample struct.
    \return True if a sample was generated, false otherwise.
*/
bool sampleDistantLight(const Optional<ShadingData> sd, const LightData light, const float4 sceneSphere, const float2 u, out LightSample ls)
{
    ls = {};
    ls.setIsFinite(false);
    ls.setIsDelta(false);

    // Sample direction.
    float3 dir = sample_cone(u, light.cosSubtendedAngle);
    // Transform sampled direction to world space
    ls.dir = normalize(mul((float3x3)light.transMat, dir));

    ls.origin = computeRayOrigin(sd.value, ls.dir);
    ls.distance = kMaxLightDistance;
    ls.cosLight = 1.f;

    // Compute incident radiance at shading point.
    // A DistantLight's angle defines the solid angle it subtends.
    // But because the angle is intended to affect penumbra size, but not
    // perceived brigthness, we treat intensity as radiance.
    float solidAngle = (float)M_2PI * (1.f - light.cosSubtendedAngle);
    ls.Li = light.intensity;
    ls.pdf = 1.f / solidAngle;
    ls.emitPdf = getSceneSpherePdf(sceneSphere) * ls.pdf;
    return true;
}

/** Samples a directional light source.
    \param[in] shadingPosW Shading point in world space.
    \param[in] light Light data.
    \param[out] ls Light sample struct.
    \return True if a sample was generated, false otherwise.
*/
bool sampleDirectionalLight(const Optional<ShadingData> sd, const LightData light, const float4 sceneSphere, out LightSample ls)
{
    ls = {};
    ls.setIsFinite(false);
    ls.setIsDelta(true);

    // Setup ray.
    ls.origin = computeRayOrigin(sd.value, -light.dirW);
    ls.distance = kMaxLightDistance;
    ls.dir = -light.dirW;

    // Setup incident radiance. For directional lights there is no falloff or cosine term.
    ls.Li = light.intensity;

    // For a directional light, the PDF with respect to solid angle is a Dirac function. Set to zero.
    ls.pdf = 0.f;
    ls.emitPdf = getSceneSpherePdf(sceneSphere);
    ls.cosLight = 1.f;

    return true;
}

/** Samples a point (spot) light.
    \param[in] shadingPosW Shading point in world space.
    \param[in] light Light data.
    \param[out] ls Light sample struct.
    \return True if a sample was generated, false otherwise.
*/
bool samplePointLight(const Optional<ShadingData> sd, const LightData light, out LightSample ls)
{
    ls = {};
    ls.setIsFinite(true);
    ls.setIsDelta(true);

    // Compute direction and distance to light.
    // The distance is clamped to a small epsilon to avoid div-by-zero below.
    float3 toLight = light.posW - sd.value.posW;
    float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
    ls.origin = computeRayOrigin(sd.value, toLight);
    ls.distance = sqrt(distSqr);
    ls.dir = toLight / ls.distance;

    // Calculate the falloff for spot-lights.
    float cosTheta = -dot(ls.dir, light.dirW);
    float falloff = 1.f;
    if (cosTheta < light.cosOpeningAngle)
    {
        falloff = 0.f;
    }
    else if (light.penumbraAngle > 0.f)
    {
        float deltaAngle = light.openingAngle - acos(cosTheta);
        falloff = smoothstep(0.f, light.penumbraAngle, deltaAngle);
    }

    // Compute incident radiance at shading point.
    ls.Li = light.intensity * falloff / distSqr;

    // For a point light, the PDF with respect to solid angle is a Dirac function. Set to zero.
    ls.pdf = 0.f;
    ls.emitPdf = M_4_PI;
    ls.cosLight = 1.f;

    return true;
}

/** Samples an analytic light source.
    This function calls the correct sampling function depending on the type of light.
    \param[in] sd Shading point in world space.
    \param[in] light Light data.
    \param[in,out] sg Sample generator.
    \param[out] ls Sampled point on the light and associated sample data, only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleAnalyticLight(const Optional<ShadingData> sd, const LightData light, const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls)
{
    // Sample the light based on its type: point, directional, or area.
    switch (light.type)
    {
    case LightType::Point:
        return samplePointLight(sd, light, ls);
    case LightType::Directional:
        return sampleDirectionalLight(sd, light, sceneSphere, ls);
    case LightType::Rect:
        return sampleRectAreaLight(sd, light, sampleNext2D(sg), ls);
    case LightType::Sphere:
        return sampleSphereAreaLight(sd, light, sampleNext2D(sg), ls);
    case LightType::Disc:
        return sampleDiscAreaLight(sd, light, sampleNext2D(sg), ls);
    case LightType::Distant:
        return sampleDistantLight(sd, light, sceneSphere, sampleNext2D(sg), ls);
    default:
        ls = {};
        return false; // Should not happen
    }
}

struct LightSampler
{
    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    // ------------------------------------------------------------------------------------------
    // ----------------------------------- Environment lights -----------------------------------
    // ------------------------------------------------------------------------------------------

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sceneSphere Scene sphere.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const ShadingData vertex, const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        ls.setIsDelta(false);
        ls.setIsFinite(false);

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf;
        ls.origin = computeRayOrigin(vertex, lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;
        ls.emitPdf = lightSample.pdf * getSceneSpherePdf(sceneSphere);
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.cosLight = 1.f;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the environment map.
        \param[in,out] sceneSphere Scene sphere.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        ls.setIsDelta(false);
        ls.setIsFinite(false);

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf;
        ls.origin = generateSceneSphereSample(sceneSphere, lightSample.dir, sg);
        ls.distance = kRayTMax;
        ls.dir = -lightSample.dir;
        ls.emitPdf = ls.pdf * getSceneSpherePdf(sceneSphere);
        ls.Li = ls.emitPdf > 0.f ? lightSample.Le / ls.emitPdf : float3(0);
        ls.cosLight = 1.f;

        return any(ls.Li > 0.f);
    }

    // ------------------------------------------------------------------------------------------
    // ------------------------------------ Emissive lights -------------------------------------
    // ------------------------------------------------------------------------------------------

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const ShadingData vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(vertex.posW, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

        ls.setIsDelta(false);
        ls.setIsFinite(true);

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        ls.emitPdf = tls.emitPdf;
        ls.cosLight = abs(dot(tls.normalW, tls.dir));
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = computeRayOrigin(vertex, lightPos - vertex.posW);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(sg, tls)) return false;

        ls.setIsDelta(false);
        ls.setIsFinite(true);

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.emitPdf : float3(0);
        ls.pdf = tls.pdf;
        ls.emitPdf = tls.emitPdf;
        ls.cosLight = abs(dot(tls.normalW, tls.dir));
        // Offset shading and light position to avoid self-intersection.
        ls.origin = computeRayOrigin(tls.posW, tls.normalW);
        ls.distance = kMaxLightDistance;
        ls.dir = tls.dir;

        return any(ls.Li > 0.f);
    }

    // ------------------------------------------------------------------------------------------
    // ------------------------------------ Analytic lights -------------------------------------
    // ------------------------------------------------------------------------------------------

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sceneSphere Scene sphere.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const ShadingData vertex, const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        if (!sampleAnalyticLight(vertex, gScene.getLight(lightIndex), sceneSphere, sg, ls)) return false;

        ls.pdf     /= lightCount;
        ls.emitPdf /= lightCount;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in,out] sceneSphere Scene sphere.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        #if 1
        return false;
        #else
        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        if (!sampleAnalyticLight(none, gScene.getLight(lightIndex), sceneSphere, sg, ls)) return false;

        ls.pdf /= lightCount;
        ls.emitPdf /= lightCount;

        return any(ls.Li > 0.f);
        #endif
    }

    // ------------------------------------------------------------------------------------------
    // -------------------------------- Top-level light sampling --------------------------------
    // ------------------------------------------------------------------------------------------

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight       ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        lightType = {};
        pdf = {};

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample( const ShadingData vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls )
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == 0)
        {
            valid = generateEnvMapSample(vertex, sceneSphere, sg, ls);
        }
        if (kUseEmissiveLights && lightType == 1)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == 2)
        {
            valid = generateAnalyticLightSample(vertex, sceneSphere, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
        if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
            return false;

        // Account for light type selection.
        ls.pdf     *= selectionPdf;
        ls.emitPdf *= selectionPdf;
        ls.Li      /= selectionPdf;
        ls.setIsAnalytic(kUseAnalyticLights && lightType == 2);

        return true;
    }

    /** Samples a light ray from a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type. This
        overload samples a light ray emitting from the light instead of
        just a direction to the light.
        \param[in,out] sceneSphere Scene sphere.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == 0)
        {
            valid = generateEnvMapSample(sceneSphere, sg, ls);
        }
        if (kUseEmissiveLights && lightType == 1)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            valid = generateEmissiveSample(sg, ls);
        }
        if (kUseAnalyticLights && lightType == 2)
        {
            valid = generateAnalyticLightSample(sceneSphere, sg, ls);
        }
        if (!valid) return false;

        // Account for light type selection.
        ls.pdf     *= selectionPdf;
        ls.emitPdf *= selectionPdf;
        ls.Li      /= selectionPdf;
        ls.setIsAnalytic(kUseAnalyticLights && lightType == 2);

        return true;
    }
};
