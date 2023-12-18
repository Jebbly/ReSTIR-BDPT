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
#include "PathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

namespace
{
    const std::string kGeneratePathsFilename = "RenderPasses/PathTracer/GeneratePaths.cs.slang";
    const std::string kTracePassFilename = "RenderPasses/PathTracer/TracePass.rt.slang";
    const std::string kReflectTypesFile = "RenderPasses/PathTracer/ReflectTypes.cs.slang";

    // Render pass inputs and outputs.
    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "mvec";
    const std::string kInputViewDir = "viewW";
    const std::string kInputSampleCount = "sampleCount";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */ },
        { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
        { kInputSampleCount,    "gSampleCount",     "Sample count buffer (integer format)", true /* optional */, ResourceFormat::R8Uint },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputSpecularAlbedo = "specularAlbedo";
    const std::string kOutputIndirectAlbedo = "indirectAlbedo";
    const std::string kOutputReflectionPosW = "reflectionPosW";
    const std::string kOutputRayCount = "rayCount";
    const std::string kOutputPathLength = "pathLength";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,                                     "",     "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputAlbedo,                                    "",     "Output albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputSpecularAlbedo,                            "",     "Output specular albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputIndirectAlbedo,                            "",     "Output indirect albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputReflectionPosW,                            "",     "Output reflection pos (world space)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputRayCount,                                  "",     "Per-pixel ray count", true /* optional */, ResourceFormat::R32Uint },
        { kOutputPathLength,                                "",     "Per-pixel path length", true /* optional */, ResourceFormat::R32Uint },
    };

    // Scripting options.
    const std::string kSamplesPerPixel = "samplesPerPixel";
    const std::string kMaxSurfaceBounces = "maxSurfaceBounces";
    const std::string kMaxDiffuseBounces = "maxDiffuseBounces";
    const std::string kMaxSpecularBounces = "maxSpecularBounces";
    const std::string kMaxTransmissionBounces = "maxTransmissionBounces";

    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kUseBSDFSampling = "useBSDFSampling";
    const std::string kUseRussianRoulette = "useRussianRoulette";
    const std::string kUseNEE = "useNEE";
    const std::string kUseMIS = "useMIS";
    const std::string kUseBPT = "useBPT";
    const std::string kUseVM  = "useVM";
    const std::string kMISHeuristic = "misHeuristic";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";
    const std::string kUseRTXDI = "useRTXDI";
    const std::string kRTXDIOptions = "RTXDIOptions";

    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kAdjustShadingNormals = "adjustShadingNormals";
    const std::string kMaxNestedMaterials = "maxNestedMaterials";
    const std::string kUseLightsInDielectricVolumes = "useLightsInDielectricVolumes";
    const std::string kDisableCaustics = "disableCaustics";
    const std::string kSpecularRoughnessThreshold = "specularRoughnessThreshold";
    const std::string kPrimaryLodMode = "primaryLodMode";
    const std::string kLODBias = "lodBias";

    const std::string kUseSER = "useSER";

    const std::string kOutputSize = "outputSize";
    const std::string kFixedOutputSize = "fixedOutputSize";
    const std::string kColorFormat = "colorFormat";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PathTracer>();
    ScriptBindings::registerBinding(PathTracer::registerBindings);
}

void PathTracer::registerBindings(pybind11::module& m)
{
    pybind11::class_<PathTracer, RenderPass, ref<PathTracer>> pass(m, "PathTracer");
    pass.def("reset", &PathTracer::reset);
    pass.def_property_readonly("pixelStats", &PathTracer::getPixelStats);

    pass.def_property("useFixedSeed",
        [](const PathTracer* pt) { return pt->mParams.useFixedSeed ? true : false; },
        [](PathTracer* pt, bool value) { pt->mParams.useFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const PathTracer* pt) { return pt->mParams.fixedSeed; },
        [](PathTracer* pt, uint32_t value) { pt->mParams.fixedSeed = value; }
    );
}

PathTracer::PathTracer(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("PathTracer requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("PathTracer requires Raytracing Tier 1.1 support.");

    mSERSupported = mpDevice->isFeatureSupported(Device::SupportedFeatures::ShaderExecutionReorderingAPI);

    parseProperties(props);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);

    // Note: The other programs are lazily created in updatePrograms() because a scene needs to be present when creating them.

    mpPixelStats = std::make_unique<PixelStats>(mpDevice);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
}

void PathTracer::setProperties(const Properties& props)
{
    parseProperties(props);
    validateOptions();
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
        lightBVHSampler->setOptions(mLightBVHOptions);
    if (mpRTXDI)
        mpRTXDI->setOptions(mRTXDIOptions);
    mRecompile = true;
    mOptionsChanged = true;
}

void PathTracer::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        // Rendering parameters
        if (key == kSamplesPerPixel) mStaticParams.samplesPerPixel = value;
        else if (key == kMaxSurfaceBounces) mParams.maxSurfaceBounces = value;
        else if (key == kMaxDiffuseBounces) mParams.maxDiffuseBounces = value;
        else if (key == kMaxSpecularBounces) mParams.maxSpecularBounces = value;
        else if (key == kMaxTransmissionBounces) mParams.maxTransmissionBounces = value;

        // Sampling parameters
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mParams.fixedSeed = value; mParams.useFixedSeed = true; }
        else if (key == kUseBSDFSampling) mStaticParams.useBSDFSampling = value;
        else if (key == kUseRussianRoulette) mStaticParams.useRussianRoulette = value;
        else if (key == kUseNEE) mStaticParams.useNEE = value;
        else if (key == kUseMIS) mStaticParams.useMIS = value;
        else if (key == kUseBPT) mStaticParams.useBPT = value;
        else if (key == kUseVM) mStaticParams.useVM = value;
        else if (key == kMISHeuristic) mStaticParams.misHeuristic = value;
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;
        else if (key == kUseRTXDI) mStaticParams.useRTXDI = value;
        else if (key == kRTXDIOptions) mRTXDIOptions = value;

        // Material parameters
        else if (key == kUseAlphaTest) mStaticParams.useAlphaTest = value;
        else if (key == kAdjustShadingNormals) mStaticParams.adjustShadingNormals = value;
        else if (key == kMaxNestedMaterials) mStaticParams.maxNestedMaterials = value;
        else if (key == kUseLightsInDielectricVolumes) mStaticParams.useLightsInDielectricVolumes = value;
        else if (key == kDisableCaustics) mStaticParams.disableCaustics = value;
        else if (key == kSpecularRoughnessThreshold) mParams.specularRoughnessThreshold = value;
        else if (key == kPrimaryLodMode) mStaticParams.primaryLodMode = value;
        else if (key == kLODBias) mParams.lodBias = value;

        // Scheduling parameters
        else if (key == kUseSER) mStaticParams.useSER = value;

        // Output parameters
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kFixedOutputSize) mFixedOutputSize = value;
        else if (key == kColorFormat) mStaticParams.colorFormat = value;

        else logWarning("Unknown property '{}' in PathTracer properties.", key);
    }

    if (props.has(kMaxSurfaceBounces))
    {
        // Initialize bounce counts to 'maxSurfaceBounces' if they weren't explicitly set.
        if (!props.has(kMaxDiffuseBounces)) mParams.maxDiffuseBounces = mParams.maxSurfaceBounces;
        if (!props.has(kMaxSpecularBounces)) mParams.maxSpecularBounces = mParams.maxSurfaceBounces;
        if (!props.has(kMaxTransmissionBounces)) mParams.maxTransmissionBounces = mParams.maxSurfaceBounces;
    }
    else
    {
        // Initialize surface bounces.
        mParams.maxSurfaceBounces = std::max(mParams.maxDiffuseBounces, std::max(mParams.maxSpecularBounces, mParams.maxTransmissionBounces));
    }

    bool maxSurfaceBouncesNeedsAdjustment =
        mParams.maxSurfaceBounces < mParams.maxDiffuseBounces ||
        mParams.maxSurfaceBounces < mParams.maxSpecularBounces ||
        mParams.maxSurfaceBounces < mParams.maxTransmissionBounces;

    // Show a warning if maxSurfaceBounces will be adjusted in validateOptions().
    if (props.has(kMaxSurfaceBounces) && maxSurfaceBouncesNeedsAdjustment)
    {
        logWarning("'{}' is set lower than '{}', '{}' or '{}' and will be increased.", kMaxSurfaceBounces, kMaxDiffuseBounces, kMaxSpecularBounces, kMaxTransmissionBounces);
    }
}

void PathTracer::validateOptions()
{
    if (mParams.specularRoughnessThreshold < 0.f || mParams.specularRoughnessThreshold > 1.f)
    {
        logWarning("'specularRoughnessThreshold' has invalid value. Clamping to range [0,1].");
        mParams.specularRoughnessThreshold = std::clamp(mParams.specularRoughnessThreshold, 0.f, 1.f);
    }

    // Static parameters.
    if (mStaticParams.samplesPerPixel < 1 || mStaticParams.samplesPerPixel > kMaxSamplesPerPixel)
    {
        logWarning("'samplesPerPixel' must be in the range [1, {}]. Clamping to this range.", kMaxSamplesPerPixel);
        mStaticParams.samplesPerPixel = std::clamp(mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }

    auto clampBounces = [] (uint32_t& bounces, const std::string& name)
    {
        if (bounces > kMaxBounces)
        {
            logWarning("'{}' exceeds the maximum supported bounces. Clamping to {}.", name, kMaxBounces);
            bounces = kMaxBounces;
        }
    };

    clampBounces(mParams.maxSurfaceBounces, kMaxSurfaceBounces);
    clampBounces(mParams.maxDiffuseBounces, kMaxDiffuseBounces);
    clampBounces(mParams.maxSpecularBounces, kMaxSpecularBounces);
    clampBounces(mParams.maxTransmissionBounces, kMaxTransmissionBounces);

    // Make sure maxSurfaceBounces is at least as many as any of diffuse, specular or transmission.
    uint32_t minSurfaceBounces = std::max(mParams.maxDiffuseBounces, std::max(mParams.maxSpecularBounces, mParams.maxTransmissionBounces));
    mParams.maxSurfaceBounces = std::max(mParams.maxSurfaceBounces, minSurfaceBounces);

    if (mStaticParams.primaryLodMode == TexLODMode::RayCones)
    {
        logWarning("Unsupported tex lod mode. Defaulting to Mip0.");
        mStaticParams.primaryLodMode = TexLODMode::Mip0;
    }

    if (mStaticParams.useSER && !mSERSupported)
    {
        logWarning("Shader Execution Reordering (SER) is not supported on this device. Disabling SER.");
        mStaticParams.useSER = false;
    }
}

Properties PathTracer::getProperties() const
{
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Properties props;

    // Rendering parameters
    props[kSamplesPerPixel] = mStaticParams.samplesPerPixel;
    props[kMaxSurfaceBounces] = mParams.maxSurfaceBounces;
    props[kMaxDiffuseBounces] = mParams.maxDiffuseBounces;
    props[kMaxSpecularBounces] = mParams.maxSpecularBounces;
    props[kMaxTransmissionBounces] = mParams.maxTransmissionBounces;

    // Sampling parameters
    props[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed) props[kFixedSeed] = mParams.fixedSeed;
    props[kUseBSDFSampling] = mStaticParams.useBSDFSampling;
    props[kUseRussianRoulette] = mStaticParams.useRussianRoulette;
    props[kUseNEE] = mStaticParams.useNEE;
    props[kUseMIS] = mStaticParams.useMIS;
    props[kUseBPT] = mStaticParams.useBPT;
    props[kUseVM] = mStaticParams.useVM;
    props[kMISHeuristic] = mStaticParams.misHeuristic;
    props[kMISPowerExponent] = mStaticParams.misPowerExponent;
    props[kEmissiveSampler] = mStaticParams.emissiveSampler;
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) props[kLightBVHOptions] = mLightBVHOptions;
    props[kUseRTXDI] = mStaticParams.useRTXDI;
    props[kRTXDIOptions] = mRTXDIOptions;

    // Material parameters
    props[kUseAlphaTest] = mStaticParams.useAlphaTest;
    props[kAdjustShadingNormals] = mStaticParams.adjustShadingNormals;
    props[kMaxNestedMaterials] = mStaticParams.maxNestedMaterials;
    props[kUseLightsInDielectricVolumes] = mStaticParams.useLightsInDielectricVolumes;
    props[kDisableCaustics] = mStaticParams.disableCaustics;
    props[kSpecularRoughnessThreshold] = mParams.specularRoughnessThreshold;
    props[kPrimaryLodMode] = mStaticParams.primaryLodMode;
    props[kLODBias] = mParams.lodBias;

    // Scheduling parameters
    props[kUseSER] = mStaticParams.useSER;

    // Output parameters
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed) props[kFixedOutputSize] = mFixedOutputSize;
    props[kColorFormat] = mStaticParams.colorFormat;

    return props;
}

RenderPassReflection PathTracer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void PathTracer::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    auto prevScreenTiles = mParams.screenTiles;

    mParams.frameDim = frameDim;
    if (mParams.frameDim.x > kMaxFrameDimension || mParams.frameDim.y > kMaxFrameDimension)
    {
        FALCOR_THROW("Frame dimensions up to {} pixels width/height are supported.", kMaxFrameDimension);
    }

    // Tile dimensions have to be powers-of-two.
    FALCOR_ASSERT(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    FALCOR_ASSERT(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    mParams.screenTiles = div_round_up(mParams.frameDim, kScreenTileDim);

    if (any(mParams.frameDim != prevFrameDim) || any(mParams.screenTiles != prevScreenTiles))
    {
        mVarsChanged = true;
    }
}

void PathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mParams.frameCount = 0;
    mParams.frameDim = {};
    mParams.screenTiles = {};

    // Need to recreate the RTXDI module when the scene changes.
    mpRTXDI = nullptr;

    resetPrograms();
    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("PathTracer: This render pass does not support custom primitives.");
        }

        validateOptions();
    }
}

void PathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData)) return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Prepare the path tracer parameter block.
    // This should be called after all resources have been created.
    preparePathTracer(renderData);

    // Trace light sub-paths.
    if (mStaticParams.useBPT)
    {
        pRenderContext->clearUAV(mpLightImage->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(mpLightVertexCount->getUAV().get(), uint4(0.f));
        if (mStaticParams.useVM)
        {
            pRenderContext->clearUAV(mpPhotonCellSizes->getUAV().get(), uint4(0.f));
        }

        FALCOR_ASSERT(mpLightTracePass);
        tracePass(pRenderContext, renderData, *mpLightTracePass, uint2(mParams.frameDim.x, (mParams.lightSubpathCount + mParams.frameDim.x-1) / mParams.frameDim.x));
    }

    // Generate paths at primary hits.
    generatePaths(pRenderContext, renderData);

    // Update RTXDI.
    if (mpRTXDI)
    {
        const auto& pMotionVectors = renderData.getTexture(kInputMotionVectors);
        mpRTXDI->update(pRenderContext, pMotionVectors);
    }

    // Trace camera sub-paths.
    FALCOR_ASSERT(mpTracePass);
    tracePass(pRenderContext, renderData, *mpTracePass, mParams.frameDim);

    endFrame(pRenderContext, renderData);
}

void PathTracer::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Stats and debug options.
    renderStatsUI(widget);
    dirty |= renderDebugUI(widget);

    if (dirty)
    {
        validateOptions();
        mOptionsChanged = true;
    }
}

bool PathTracer::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    if (mFixedSampleCount)
    {
        dirty |= widget.var("Samples/pixel", mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }
    else widget.text("Samples/pixel: Variable");
    widget.tooltip("Number of samples per pixel. One path is traced for each sample.\n\n"
        "When the '" + kInputSampleCount + "' input is connected, the number of samples per pixel is loaded from the texture.");

    if (widget.var("Max surface bounces", mParams.maxSurfaceBounces, 0u, kMaxBounces))
    {
        // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
        mParams.maxDiffuseBounces = std::min(mParams.maxDiffuseBounces, mParams.maxSurfaceBounces);
        mParams.maxSpecularBounces = std::min(mParams.maxSpecularBounces, mParams.maxSurfaceBounces);
        mParams.maxTransmissionBounces = std::min(mParams.maxTransmissionBounces, mParams.maxSurfaceBounces);
        dirty = true;
    }
    widget.tooltip("Maximum number of surface bounces (diffuse + specular + transmission).\n"
        "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold are also classified as diffuse events.");

    dirty |= widget.var("Max diffuse bounces", mParams.maxDiffuseBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max specular bounces", mParams.maxSpecularBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max transmission bounces", mParams.maxTransmissionBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

    // Sampling options.

    if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        dirty = true;
    }

    dirty |= widget.checkbox("BSDF importance sampling", mStaticParams.useBSDFSampling);
    widget.tooltip("BSDF importance sampling should normally be enabled.\n\n"
        "If disabled, cosine-weighted hemisphere sampling is used for debugging purposes");

    dirty |= widget.checkbox("Russian roulette", mStaticParams.useRussianRoulette);
    widget.tooltip("Use russian roulette to terminate low throughput paths.");
    if (mStaticParams.useRussianRoulette)
    {
        dirty |= widget.var("Continuation probability", mParams.contProb, 0.f, 1.f);
        widget.tooltip("Probability of continuing the path at each vertex.");
    }

    dirty |= widget.checkbox("Bidirectional path tracing (BPT)", mStaticParams.useBPT);
    widget.tooltip("Use bidirectional path tracing.\nThis option automatically enables NEE and MIS.");

    if (mStaticParams.useBPT)
    {
        dirty |= widget.var("Light sub-path count", mParams.lightSubpathCount, 1u, 16000000u);
        widget.tooltip("Number of light sub-paths to trace when BPT is enabled.");

        dirty |= widget.checkbox("Light trace only", mStaticParams.lightTraceOnly);
        widget.tooltip("Only use light tracing.\nThis option disables tracing paths from the camera.");

        dirty |= widget.checkbox("Vertex merging (VM)", mStaticParams.useVM);
        widget.tooltip("Enable vertex merging.");

        if (mStaticParams.useVM)
        {
            dirty |= widget.checkbox("VM Only", mStaticParams.vmOnly);
            widget.tooltip("Only use vertex merging.\nThis is the same as progressive photon mapping.");

            dirty |= widget.checkbox("Fast vertex merging", mStaticParams.useFastVM);
            widget.tooltip("Only merge with a single grid cell, instead\nof all 8 neighbor grid cells.");

            dirty |= widget.var("Photon radius factor", mVMRadiusFactor, 1e-9f, 0.1f);
            widget.tooltip("Photon radius as a percentange of the scene radius.");

            dirty |= widget.slider("Photon radius alpha", mVMRadiusAlpha, 0.f, 1.f);
            widget.tooltip("Photon radius shrink factor.\nLower values cause the radius to shrink faster.");

            widget.text("Current radius: " + std::to_string(mParams.mergeRadius) + " at frame " + std::to_string(mParams.frameCount));

            dirty |= widget.var("Hash grid cells", mParams.cellCount, 1000u, 16000000u);
            widget.tooltip("Number of cells in the photon hash grid.");
        }
    }
    else
    {
        dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
        widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");

        if (mStaticParams.useNEE)
        {
            dirty |= widget.checkbox("Multiple importance sampling (MIS)", mStaticParams.useMIS);
            widget.tooltip("When enabled, BSDF sampling is combined with light sampling for the environment map and emissive lights.\n"
                "Note that MIS has currently no effect on analytic lights.");
        }
    }

    if ((mStaticParams.useNEE && mStaticParams.useMIS) || mStaticParams.useBPT)
    {
        dirty |= widget.dropdown("MIS heuristic", mStaticParams.misHeuristic);

        if (mStaticParams.misHeuristic == MISHeuristic::PowerExp)
        {
            dirty |= widget.var("MIS power exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
        }
    }

    dirty |= widget.checkbox("Use reconnection", mStaticParams.useReconnection);

    if ((mStaticParams.useNEE || mStaticParams.useBPT) && mpScene && mpScene->useEmissiveLights())
    {
        if (auto group = widget.group("Emissive sampler"))
        {
            if (widget.dropdown("Emissive sampler", mStaticParams.emissiveSampler))
            {
                resetLighting();
                dirty = true;
            }
            widget.tooltip("Selects which light sampler to use for importance sampling of emissive geometry.", true);

            if (mpEmissiveSampler)
            {
                if (mpEmissiveSampler->renderUI(group)) mOptionsChanged = true;
            }
        }
    }

    if (auto group = widget.group("RTXDI"))
    {
        dirty |= widget.checkbox("Enabled", mStaticParams.useRTXDI);
        widget.tooltip("Use RTXDI for direct illumination.");
        if (mpRTXDI) dirty |= mpRTXDI->renderUI(group);
    }

    if (auto group = widget.group("Material controls"))
    {
        dirty |= widget.checkbox("Alpha test", mStaticParams.useAlphaTest);
        widget.tooltip("Use alpha testing on non-opaque triangles.");

        dirty |= widget.checkbox("Adjust shading normals on secondary hits", mStaticParams.adjustShadingNormals);
        widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.\nDoes not apply to primary hits which is configured in GBuffer.", true);

        dirty |= widget.var("Max nested materials", mStaticParams.maxNestedMaterials, 2u, 4u);
        widget.tooltip("Maximum supported number of nested materials.");

        dirty |= widget.checkbox("Use lights in dielectric volumes", mStaticParams.useLightsInDielectricVolumes);
        widget.tooltip("Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.");

        dirty |= widget.checkbox("Disable caustics", mStaticParams.disableCaustics);
        widget.tooltip("Disable sampling of caustic light paths (i.e. specular events after diffuse events).");

        runtimeDirty |= widget.var("Specular roughness threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        widget.tooltip("Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.");

        dirty |= widget.dropdown("Primary LOD Mode", mStaticParams.primaryLodMode);
        widget.tooltip("Texture LOD mode at primary hit");

        runtimeDirty |= widget.var("TexLOD bias", mParams.lodBias, -16.f, 16.f, 0.01f);
    }

    if (auto group = widget.group("Scheduling options"))
    {
        dirty |= widget.checkbox("Use SER", mStaticParams.useSER);
        widget.tooltip("Use Shader Execution Reordering (SER) to improve GPU utilization.");
    }

    if (auto group = widget.group("Output options"))
    {
        // Switch to enable/disable path tracer output.
        dirty |= widget.checkbox("Enable output", mEnabled);

        // Controls for output size.
        // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
        if (widget.dropdown("Output size", mOutputSizeSelection)) requestRecompile();
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        {
            if (widget.var("Size in pixels", mFixedOutputSize, 32u, 16384u)) requestRecompile();
        }

        dirty |= widget.dropdown("Color format", mStaticParams.colorFormat);
        widget.tooltip("Selects the color format used for internal per-sample color and denoiser buffers");
    }

    if (dirty) mRecompile = true;
    return dirty || runtimeDirty;
}

bool PathTracer::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging"))
    {
        dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        group.tooltip("Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging.");
        if (mParams.useFixedSeed)
        {
            dirty |= group.var("Seed", mParams.fixedSeed);
        }

        bool recompile = false;
        recompile |= group.checkbox("Debug MIS", mStaticParams.debugMIS);

        recompile |= group.checkbox("Debug BPT", mStaticParams.debugBPT);
        if (mStaticParams.debugBPT)
        {
            recompile |= group.var("Debug path length", mParams.debugPathLength, -1);
            group.tooltip("Only render paths with this many segments.");
            recompile |= group.var("Debug light vertex count", mParams.debugLightVertexCount, -1);
            group.tooltip("Only render paths with this many light vertices.");
        }
        dirty |= recompile;
        mRecompile |= recompile;

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

void PathTracer::renderStatsUI(Gui::Widgets& widget)
{
    if (auto g = widget.group("Statistics"))
    {
        // Show ray stats
        mpPixelStats->renderUI(g);
    }
}

bool PathTracer::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

void PathTracer::reset()
{
    mParams.frameCount = 0;
}

PathTracer::TracePass::TracePass(ref<Device> pDevice, const std::string& name, const std::string& passDefine, const ref<Scene>& pScene, const DefineList& defines, const TypeConformanceList& globalTypeConformances)
    : name(name)
    , passDefine(passDefine)
{
    const uint32_t kRayTypeScatter = 0;
    const uint32_t kMissScatter = 0;

    bool useSER = defines.at("USE_SER") == "1";

    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(kTracePassFilename);
    if (pDevice->getType() == Device::Type::D3D12 && useSER)
        desc.addCompilerArguments({ "-Xdxc", "-enable-lifetime-markers" });
    desc.setMaxPayloadSize(208); // This is conservative but the required minimum is 200 bytes.
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(1);
    if (!pScene->hasProceduralGeometry()) desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    // Create ray tracing binding table.
    pBindingTable = RtBindingTable::create(1, 1, pScene->getGeometryCount());

    // Specify entry point for raygen and miss shaders.
    // The raygen shader needs type conformances for *all* materials in the scene.
    // The miss shader doesn't need need any type conformances because it does not use materials.
    pBindingTable->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    pBindingTable->setMiss(kMissScatter, desc.addMiss("scatterMiss"));

    // Specify hit group entry points for every combination of geometry and material type.
    // The code for each hit group gets specialized for the actual types it's operating on.
    // First query which material types the scene has.
    auto materialTypes = pScene->getMaterialSystem().getMaterialTypes();

    for (const auto materialType : materialTypes)
    {
        auto typeConformances = pScene->getMaterialSystem().getTypeConformances(materialType);

        // Add hit groups for triangles.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterTriangleClosestHit", "scatterTriangleAnyHit", "", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for displaced triangle meshes.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for curves.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::Curve, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for SDF grids.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::SDFGrid, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }
    }

    pProgram = Program::create(pDevice, desc, defines);
}

void PathTracer::TracePass::prepareProgram(ref<Device> pDevice, const DefineList& defines)
{
    FALCOR_ASSERT(pProgram != nullptr && pBindingTable != nullptr);
    pProgram->setDefines(defines);
    if (!passDefine.empty()) pProgram->addDefine(passDefine);
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);
}

void PathTracer::resetPrograms()
{
    mpTracePass = nullptr;
    mpLightTracePass = nullptr;
    mpGeneratePaths = nullptr;
    mpReflectTypes = nullptr;

    mRecompile = true;
}

void PathTracer::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    // If we get here, a change that require recompilation of shader programs has occurred.
    // This may be due to change of scene defines, type conformances, shader modules, or other changes that require recompilation.
    // When type conformances and/or shader modules change, the programs need to be recreated. We assume programs have been reset upon such changes.
    // When only defines have changed, it is sufficient to update the existing programs and recreate the program vars.

    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getTypeConformances();

    // Create trace pass.
    if (!mpTracePass)
        mpTracePass = std::make_unique<TracePass>(mpDevice, "tracePass", "", mpScene, defines, globalTypeConformances);

    mpTracePass->prepareProgram(mpDevice, defines);

    // Create light trace pass.
    if (mStaticParams.useBPT)
    {
        if (!mpLightTracePass)
            mpLightTracePass = std::make_unique<TracePass>(mpDevice, "tracePass", "LIGHT_TRACE_PASS", mpScene, defines, globalTypeConformances);

        mpLightTracePass->prepareProgram(mpDevice, defines);
    }

    // Create compute passes.
    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);

    if (!mpGeneratePaths)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kGeneratePathsFilename).csEntry("main");
        mpGeneratePaths = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpReflectTypes)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, false);
    }

    auto preparePass = [&](ref<ComputePass> pass)
    {
        // Note that we must use set instead of add defines to replace any stale state.
        pass->getProgram()->setDefines(defines);

        // Recreate program vars. This may trigger recompilation if needed.
        // Note that program versions are cached, so switching to a previously used specialization is faster.
        pass->setVars(nullptr);
    };
    preparePass(mpGeneratePaths);
    preparePass(mpReflectTypes);

    mVarsChanged = true;
    mRecompile = false;
}

void PathTracer::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.
    uint32_t spp = mFixedSampleCount ? mStaticParams.samplesPerPixel : kMaxSamplesPerPixel;
    uint32_t tileCount = mParams.screenTiles.x * mParams.screenTiles.y;
    const uint32_t sampleCount = tileCount * kScreenTileDim.x * kScreenTileDim.y * spp;
    const uint32_t screenPixelCount = mParams.frameDim.x * mParams.frameDim.y;
    const uint32_t pathCount = screenPixelCount * spp;
    const size_t lightVertexCount = mParams.lightSubpathCount * std::max(1u, mParams.maxSurfaceBounces);

    auto var = mpReflectTypes->getRootVar();

    if (!mpReservoirs || mpReservoirs->getElementCount() != mParams.frameDim.x * mParams.frameDim.y)
    {
        mpReservoirs = mpDevice->createStructuredBuffer(var["pathTracer"]["pathReservoirs"], mParams.frameDim.x * mParams.frameDim.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mVarsChanged = true;
    }

    if (mStaticParams.useBPT)
    {
        if (!mpLightImage || mpLightImage->getSize() != sizeof(float3) * mParams.frameDim.x * mParams.frameDim.y)
        {
            mpLightImage = mpDevice->createBuffer(sizeof(float3) * mParams.frameDim.x * mParams.frameDim.y, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mVarsChanged = true;
        }

        if (!mpLightVertices || mpLightVertices->getElementCount() != lightVertexCount || mVarsChanged)
        {
            mpLightVertices    = mpDevice->createStructuredBuffer(var["pathTracer"]["lightVertexCache"]["lightVertices"], lightVertexCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpLightVertexCount = mpDevice->createBuffer(sizeof(uint32_t)*2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mVarsChanged = true;
        }

        if (!mpPhotonCellSizes || mpPhotonCellSizes->getElementCount() != mParams.cellCount || mVarsChanged)
        {
            mpPhotonCellSizes     = mpDevice->createBuffer(sizeof(uint32_t)*mParams.cellCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mpPhotonCellOffsets   = mpDevice->createStructuredBuffer(var["pathTracer"]["photonMap"]["cellOffsets"], mParams.cellCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mVarsChanged = true;
        }
    }
}

void PathTracer::preparePathTracer(const RenderData& renderData)
{
    // Create path tracer parameter block if needed.
    if (!mpPathTracerBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpPathTracerBlock);
        mVarsChanged = true;
    }

    // Bind resources.
    auto var = mpPathTracerBlock->getRootVar();
    bindShaderData(var, renderData);
}

void PathTracer::resetLighting()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void PathTracer::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for scene changes that require shader recompilation.
    // Whenever materials or geometry is added/removed to the scene, we reset the shader programs to trigger
    // recompilation with the correct defines, type conformances, shader modules, and binding table.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        resetPrograms();
    }
}

bool PathTracer::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getLightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, mpScene, mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene);
                break;
            default:
                FALCOR_THROW("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext);
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->pProgram->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void PathTracer::prepareRTXDI(RenderContext* pRenderContext)
{
    if (mStaticParams.useRTXDI)
    {
        if (!mpRTXDI) mpRTXDI = std::make_unique<RTXDI>(mpScene, mRTXDIOptions);

        // Emit warning if enabled while using spp != 1.
        if (!mFixedSampleCount || mStaticParams.samplesPerPixel != 1)
        {
            logWarning("Using RTXDI with samples/pixel != 1 will only generate one RTXDI sample reused for all pixel samples.");
        }
    }
    else
    {
        mpRTXDI = nullptr;
    }
}

void PathTracer::bindShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (useLightSampling && mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["lightSampler"]["envMapSampler"]);

        var["lightImage"] = mpLightImage;
        var["lightVertexCache"]["lightVertices"] = mpLightVertices;
        var["lightVertexCache"]["lightVertexCount"] = mpLightVertexCount;

        var["photonMap"]["cellSizes"]   = mpPhotonCellSizes;
        var["photonMap"]["cellOffsets"] = mpPhotonCellOffsets;

        var["pathReservoirs"] = mpReservoirs;
    }

    ref<Texture> pViewDir;
    if (mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        pViewDir = renderData.getTexture(kInputViewDir);
        if (!pViewDir) logWarning("Depth-of-field requires the '{}' input. Expect incorrect rendering.", kInputViewDir);
    }

    ref<Texture> pSampleCount;
    if (!mFixedSampleCount)
    {
        pSampleCount = renderData.getTexture(kInputSampleCount);
        if (!pSampleCount) FALCOR_THROW("PathTracer: Missing sample count input texture");
    }

    var["params"].setBlob(mParams);
    var["vbuffer"] = renderData.getTexture(kInputVBuffer);
    var["viewDir"] = pViewDir; // Can be nullptr
    var["sampleCount"] = pSampleCount; // Can be nullptr
    var["outputColor"] = renderData.getTexture(kOutputColor);

    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->bindShaderData(var["lightSampler"]["emissiveSampler"]);
    }
}

bool PathTracer::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels) {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y)) resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("PathTracer I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Prepare RTXDI.
    prepareRTXDI(pRenderContext);
    if (mpRTXDI) mpRTXDI->beginFrame(pRenderContext, mParams.frameDim);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    bool gbufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);
    if (gbufferAdjustShadingNormals != mGBufferAdjustShadingNormals)
    {
        mGBufferAdjustShadingNormals = gbufferAdjustShadingNormals;
        mRecompile = true;
    }

    // Check if fixed sample count should be used. When the sample count input is connected we load the count from there instead.
    mFixedSampleCount = renderData[kInputSampleCount] == nullptr;

    // Enable pixel stats if rayCount or pathLength outputs are connected.
    if (renderData[kOutputRayCount] != nullptr || renderData[kOutputPathLength] != nullptr)
    {
        mpPixelStats->setEnabled(true);
    }

    mpPixelStats->beginFrame(pRenderContext, mParams.frameDim);
    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);


    if (mStaticParams.useVM)
    {
        bool resetFrameCount = false;

        auto refreshFlags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        if (refreshFlags != RenderPassRefreshFlags::None)
            resetFrameCount = true;

        auto sceneUpdates = mpScene->getUpdates();
        if ((sceneUpdates & ~Scene::UpdateFlags::CameraPropertiesChanged) != Scene::UpdateFlags::None)
            resetFrameCount = true;
        if (is_set(sceneUpdates, Scene::UpdateFlags::CameraPropertiesChanged))
        {
            auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
            auto cameraChanges = mpScene->getCamera()->getChanges();
            if ((cameraChanges & ~excluded) != Camera::Changes::None)
                resetFrameCount = true;
        }

        if (resetFrameCount)
            mParams.frameCount = 0;
    }

    // Update the random seed.
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mParams.frameCount;

    const auto& aabb = mpScene->getSceneBounds();
    mParams.sceneSphere = float4(aabb.maxPoint + aabb.minPoint, length(aabb.maxPoint - aabb.minPoint))*.5f;
    mParams.mergeRadius = mVMRadiusFactor * mParams.sceneSphere.w;
    if (mVMRadiusAlpha < 1.f)
    {
        mParams.mergeRadius /= std::pow(float(mParams.frameCount + 1), 0.5f * (1 - mVMRadiusAlpha));
        mParams.mergeRadius = std::max(mParams.mergeRadius, 1e-7f);
    }

    return true;
}

void PathTracer::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelStats->endFrame(pRenderContext);
    mpPixelDebug->endFrame(pRenderContext);

    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Copy pixel stats to outputs if available.
    copyTexture(renderData.getTexture(kOutputRayCount).get(), mpPixelStats->getRayCountTexture(pRenderContext).get());
    copyTexture(renderData.getTexture(kOutputPathLength).get(), mpPixelStats->getPathLengthTexture().get());

    if (mpRTXDI) mpRTXDI->endFrame(pRenderContext);

    mVarsChanged = false;
    mParams.frameCount++;
}

void PathTracer::generatePaths(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "generatePaths");

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.

    // Additional specialization. This shouldn't change resource declarations.
    mpGeneratePaths->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");

    // Bind resources.
    auto var = mpGeneratePaths->getRootVar()["CB"]["gPathGenerator"];
    bindShaderData(var, renderData, false);

    mpScene->bindShaderData(mpGeneratePaths->getRootVar()["gScene"]);

    if (mpRTXDI) mpRTXDI->bindShaderData(mpGeneratePaths->getRootVar());

    // Launch one thread per pixel.
    mpGeneratePaths->execute(pRenderContext, { mParams.frameDim.x, mParams.frameDim.y, 1u });
}

void PathTracer::tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass, const uint2 dispatchDims)
{
    FALCOR_PROFILE(pRenderContext, tracePass.name);

    FALCOR_ASSERT(tracePass.pProgram != nullptr && tracePass.pBindingTable != nullptr && tracePass.pVars != nullptr);

    // Additional specialization. This shouldn't change resource declarations.
    tracePass.pProgram->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");

    // Bind global resources.
    auto var = tracePass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    if (mVarsChanged) mpSampleGenerator->bindShaderData(var);
    if (mpRTXDI) mpRTXDI->bindShaderData(var);

    mpPixelStats->prepareProgram(tracePass.pProgram, var);
    mpPixelDebug->prepareProgram(tracePass.pProgram, var);

    // Bind the path tracer.
    var["gPathTracer"] = mpPathTracerBlock;

    // Full screen dispatch.
    mpScene->raytrace(pRenderContext, tracePass.pProgram.get(), tracePass.pVars, uint3(dispatchDims, 1));
}

DefineList PathTracer::StaticParams::getDefines(const PathTracer& owner) const
{
    DefineList defines;

    // Path tracer configuration.
    defines.add("SAMPLES_PER_PIXEL", (owner.mFixedSampleCount ? std::to_string(samplesPerPixel) : "0")); // 0 indicates a variable sample count
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", (useBPT || useNEE) ? "1" : "0");
    defines.add("USE_MIS", (useBPT || useMIS) ? "1" : "0");
    defines.add("USE_BPT", useBPT ? "1" : "0");
    defines.add("USE_VM", (useVM && useBPT) ? "1" : "0");
    defines.add("USE_FAST_VM", (useVM && useBPT && useFastVM) ? "1" : "0");
    defines.add("USE_VM_ONLY", (useVM && useBPT && vmOnly) ? "1" : "0");
    defines.add("USE_RECONNECTION", useReconnection ? "1" : "0");
    defines.add("DEBUG_MIS", debugMIS ? "1" : "0");
    defines.add("DEBUG_BPT", debugBPT ? "1" : "0");
    defines.add("LIGHT_TRACE_ONLY", (useBPT && lightTraceOnly) ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_RTXDI", useRTXDI ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("USE_SER", useSER ? "1" : "0");
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());
    if (owner.mpRTXDI) defines.add(owner.mpRTXDI->getDefines());

    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene) defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_CURVES", scene && (scene->hasGeometryType(Scene::GeometryType::Curve)) ? "1" : "0");
    defines.add("USE_SDF_GRIDS", scene && scene->hasGeometryType(Scene::GeometryType::SDFGrid) ? "1" : "0");
    defines.add("USE_HAIR_MATERIAL", scene && scene->getMaterialCountByType(MaterialType::Hair) > 0u ? "1" : "0");

    // Set default (off) values for additional features.
    defines.add("USE_VIEW_DIR", "0");
    defines.add("OUTPUT_GUIDE_DATA", "0");
    defines.add("OUTPUT_NRD_DATA", "0");
    defines.add("OUTPUT_NRD_ADDITIONAL_DATA", "0");

    return defines;
}
