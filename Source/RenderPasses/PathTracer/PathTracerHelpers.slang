#include "Utils/Math/MathConstants.slangh"

//import AnalyticLightHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import LoadShadingData;
import PathState;
import Params;

static const float kMinLightDistSqr = 1e-9f;
static const float kMaxLightDistance = FLT_MAX;

// Render settings that depend on the scene.
// TODO: Move into scene defines.
static const bool kUseEnvLight       = USE_ENV_LIGHT;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseCurves         = USE_CURVES;
static const bool kUseHairMaterial   = USE_HAIR_MATERIAL;

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] n Number of samples taken from the sampling strategy.
    \param[in] p Pdf for the sampling strategy.
    \return Partial MIS weight
*/
float evalMIS(float n, float p)
{
    switch (MISHeuristic(kMISHeuristic))
    {
    case MISHeuristic::Balance:
    {
        // Balance heuristic
        return n * p;
    }
    case MISHeuristic::PowerTwo:
    {
        // Power two heuristic
        return (n * p) * (n * p);
    }
    case MISHeuristic::PowerExp:
    {
        // Power exp heuristic
        return pow(n * p, kMISPowerExponent);
    }
    default:
        return 0.f;
    }
}

/** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
    \param[in] n0 Number of samples taken from the first sampling strategy.
    \param[in] p0 Pdf for the first sampling strategy.
    \param[in] n1 Number of samples taken from the second sampling strategy.
    \param[in] p1 Pdf for the second sampling strategy.
    \return Weight for the contribution from the first strategy (p0).
*/
float evalMIS(float n0, float p0, float n1, float p1)
{
    float q0 = evalMIS(n0, p0);
    float q1 = evalMIS(n1, p1);
    float qs = q0 + q1;
    return qs > 0 ? q0 / qs : 0.f;
}

/**
 * Returns the shading normal correction for light tracing.
 */
float getShadingNormalCorrection(const ShadingData sd, const float3 wo) {
    const float num = abs(dot(sd.frame.N, sd.V) * dot(sd.faceN, wo));
    const float denom = abs(dot(sd.frame.N, wo) * dot(sd.faceN, sd.V));
    return denom >= kMinCosTheta ? num / denom : 1;
}

/**
 * Computes new ray origin based on hit position to avoid self-intersections.
 */
float3 computeRayOrigin(const ShadingData sd, const float3 dir)
{
    return computeRayOrigin(sd.posW, dot(dir, sd.faceN) >= 0.f ? sd.faceN : -sd.faceN);
}

/** Types of samplable lights.
*/
enum class LightSampleType
{
    EnvMap,
    Emissive,
    Analytic
};

/** Describes a light sample.
 */
struct LightSample
{
    float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float pdf;      ///< Pdf with respect to solid angle at the shading point.
    float emitPdf;  ///< Emission pdf with respect to solid angle at the light surface. This is already multiplied by the area-measure light sampling pdf.
    float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;     ///< Ray direction for visibility evaluation (normalized).
    uint lightType; ///< Light type this sample comes from (LightSampleType casted to uint).

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

float3 generateSceneSphereSample(const float4 sceneSphere, const float3 dir, inout SampleGenerator sg)
{
    float2 uv = sample_disk_concentric(sampleNext2D(sg));
    float3 t, b;
    branchlessONB(dir, t, b);
    return sceneSphere.xyz + sceneSphere.w * (t * uv.x + b * uv.y - dir);
}
float getSceneSpherePdf(const float4 sceneSphere)
{
    return M_1_PI / (sceneSphere.w * sceneSphere.w);
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

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf;
        ls.origin = generateSceneSphereSample(sceneSphere, lightSample.dir, sg);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;
        ls.emitPdf = ls.pdf * getSceneSpherePdf(sceneSphere);
        ls.Li = ls.emitPdf > 0.f ? lightSample.Le / ls.emitPdf : float3(0);

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

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
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

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
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
#if 1
        return false;
#else
        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        if (!sampleLight(vertex.posW, gScene.getLight(lightIndex), sceneSphere, sg, ls)) return false;

        return any(ls.Li > 0.f);
#endif
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
        if (!sampleLight(gScene.getLight(lightIndex), sceneSphere, sg, ls)) return false;

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
        p[0] = kUseEnvLight ? 1.f : 0.f;
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

    float getEnvMapSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

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
        if (kUseEnvLight && lightType == (uint)LightSampleType::EnvMap)
        {
            valid = generateEnvMapSample(vertex, sceneSphere, sg, ls);
        }
        if (kUseEmissiveLights && lightType == (uint)LightSampleType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightSampleType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sceneSphere, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
        if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
            return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.emitPdf *= selectionPdf;
        ls.Li /= selectionPdf;

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
    bool generateLightSample(const float4 sceneSphere, inout SampleGenerator sg, out LightSample ls )
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightSampleType::EnvMap)
        {
            valid = generateEnvMapSample(sceneSphere, sg, ls);
        }
        if (kUseEmissiveLights && lightType == (uint)LightSampleType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            valid = generateEmissiveSample(sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightSampleType::Analytic)
        {
            valid = generateAnalyticLightSample(sceneSphere, sg, ls);
        }
        if (!valid) return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.emitPdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }
};
