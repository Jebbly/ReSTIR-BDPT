#include "ReSTIRVCM.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

namespace
{
    const std::string kVCMPassFilename           = "RenderPasses/ReSTIRVCM/VCM.cs.slang";
    const std::string kTemporalReusePassFilename = "RenderPasses/ReSTIRVCM/TemporalReuse.cs.slang";
    const std::string kSpatialReusePassFilename  = "RenderPasses/ReSTIRVCM/SpatialReuse.cs.slang";
    const std::string kReflectTypesFile          = "RenderPasses/ReSTIRVCM/ReflectTypes.cs.slang";

    // Render pass inputs and outputs.
    const std::string kInputVBuffer       = "vbuffer";
    const std::string kInputMotionVectors = "mvec";
    const std::string kInputViewDir       = "viewW";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,       "gVBuffer",       "Visibility buffer in packed format" },
        { kInputMotionVectors, "gMotionVectors", "Motion vector buffer (float format)", true /* optional */ },
        { kInputViewDir,       "gViewW",         "World-space view direction (xyz float format)", true /* optional */ },
    };

    const std::string kOutputColor = "color";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor, "", "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
    };

    // Scripting options.
    const std::string kMaxBounces = "maxBounces";
    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kUseNEE = "useNEE";
    const std::string kUseBPT  = "useVC";
    const std::string kUseVM  = "useVM";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";
    const std::string kOutputSize = "outputSize";
    const std::string kFixedOutputSize = "fixedOutputSize";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRVCM>();
    ScriptBindings::registerBinding(ReSTIRVCM::registerBindings);
}

void ReSTIRVCM::registerBindings(pybind11::module& m)
{
    pybind11::class_<ReSTIRVCM, RenderPass, ref<ReSTIRVCM>> pass(m, "ReSTIRVCM");
    pass.def("reset", &ReSTIRVCM::reset);

    pass.def_property("useFixedSeed",
        [](const ReSTIRVCM* pt) { return pt->mUseFixedSeed ? true : false; },
        [](ReSTIRVCM* pt, bool value) { pt->mUseFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const ReSTIRVCM* pt) { return pt->mFixedSeed; },
        [](ReSTIRVCM* pt, uint32_t value) { pt->mFixedSeed = value; }
    );
}

ReSTIRVCM::ReSTIRVCM(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("ReSTIRVCM requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("ReSTIRVCM requires Raytracing Tier 1.1 support.");

    parseProperties(props);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);

    // Note: The other programs are lazily created in updatePrograms() because a scene needs to be present when creating them.

    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
}

void ReSTIRVCM::setProperties(const Properties& props)
{
    parseProperties(props);
    validateOptions();
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
        lightBVHSampler->setOptions(mLightBVHOptions);
    mRecompile = true;
    mOptionsChanged = true;
}

void ReSTIRVCM::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        // Rendering parameters
        if (key == kMaxBounces) mParams.mMaxBounces = value;

        // Sampling parameters
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mFixedSeed = value; mUseFixedSeed = true; }
        else if (key == kUseNEE) mStaticParams.useNEE = value;
        else if (key == kUseBPT) mStaticParams.useBPT = value;
        else if (key == kUseVM) mStaticParams.useVM = value;
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;

        // Output parameters
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kFixedOutputSize) mFixedOutputSize = value;

        else logWarning("Unknown property '{}' in ReSTIRVCM properties.", key);
    }
}

void ReSTIRVCM::validateOptions()
{
    if (mParams.mReconnectionRoughness < 0.f || mParams.mReconnectionRoughness > 1.f)
    {
        logWarning("'mReconnectionRoughness' has invalid value. Clamping to range [0,1].");
        mParams.mReconnectionRoughness = std::clamp(mParams.mReconnectionRoughness, 0.f, 1.f);
    }
}

Properties ReSTIRVCM::getProperties() const
{
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Properties props;

    // Rendering parameters
    props[kMaxBounces] = mParams.mMaxBounces;

    // Sampling parameters
    props[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mUseFixedSeed) props[kFixedSeed] = mFixedSeed;
    props[kUseNEE] = mStaticParams.useNEE;
    props[kUseBPT] = mStaticParams.useBPT;
    props[kUseVM] = mStaticParams.useVM;
    props[kMISPowerExponent] = mStaticParams.misPowerExponent;
    props[kEmissiveSampler] = mStaticParams.emissiveSampler;
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) props[kLightBVHOptions] = mLightBVHOptions;

    // Output parameters
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed) props[kFixedOutputSize] = mFixedOutputSize;

    return props;
}

RenderPassReflection ReSTIRVCM::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void ReSTIRVCM::setFrameDim(const uint2 mOutputDim)
{
    auto prevFrameDim = mParams.mOutputDim;

    mParams.mOutputDim = mOutputDim;

    if (any(mParams.mOutputDim != prevFrameDim))
    {
        mVarsChanged = true;
    }
}

void ReSTIRVCM::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mFrameCount = 0;
    mParams.mOutputDim = {};

    resetPrograms();
    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ReSTIRVCM: This render pass does not support custom primitives.");
        }

        validateOptions();
    }
}

void ReSTIRVCM::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData)) return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Trace light sub-paths.
    if (mStaticParams.useBPT)
    {
        pRenderContext->clearUAV(mpLightVertexCount->getUAV().get(), uint4(0));
        if (mStaticParams.useVM)
        {
            pRenderContext->clearUAV(mpPhotonCellSizes->getUAV().get(), uint4(0));
        }

        if (mStaticParams.useLightTraceReservoirs)
        {
            pRenderContext->clearUAV(mpLightReservoirHashMapCellKeys->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(mpLightReservoirHashMapCellCounters->getUAV().get(), uint4(0));
            pRenderContext->clearUAV(mpLightReservoirHashMapCounters->getUAV().get(), uint4(0));
        }
        else
        {
            pRenderContext->clearUAV(mpLightImage->getUAV().get(), float4(0.f));
        }

        // one thread per light subpath
        FALCOR_ASSERT(mpSampleLightPathsPass);
        preparePass(pRenderContext, renderData, *mpSampleLightPathsPass);
        mpSampleLightPathsPass->execute(pRenderContext, mParams.mOutputDim.x, (mParams.mLightSubpathCount + mParams.mOutputDim.x-1) / mParams.mOutputDim.x);
        mParams.mRandomSeed++;
    }

    // Trace camera sub-paths.
    FALCOR_ASSERT(mpSampleCameraPathsPass);
    preparePass(pRenderContext, renderData, *mpSampleCameraPathsPass);
    mpSampleCameraPathsPass->execute(pRenderContext, mParams.mOutputDim.x, mParams.mOutputDim.y);
    mParams.mRandomSeed++;

    // Merge pure light tracing reservoirs with camera reservoirs
    if (mStaticParams.useBPT && mStaticParams.useLightTraceReservoirs)
    {
        FALCOR_ASSERT(mpComputeLightReservoirOffsetsPass);
        FALCOR_ASSERT(mpSortLightReservoirsPass);
        FALCOR_ASSERT(mpLightReservoirResolvePass);

        preparePass(pRenderContext, renderData, *mpComputeLightReservoirOffsetsPass);
        preparePass(pRenderContext, renderData, *mpSortLightReservoirsPass);
        preparePass(pRenderContext, renderData, *mpLightReservoirResolvePass);

        // one thread per pixel
        mpComputeLightReservoirOffsetsPass->execute(pRenderContext, mParams.mOutputDim.x, mParams.mOutputDim.y);
        // one thread per light subpath vertex
        mpSortLightReservoirsPass->execute(pRenderContext, mParams.mOutputDim.x, (mParams.mLightSubpathCount*mParams.mMaxBounces + mParams.mOutputDim.x-1) / mParams.mOutputDim.x);

        // one thread per light subpath vertex
        mpLightReservoirResolvePass->execute(pRenderContext, mParams.mOutputDim.x, (mParams.mLightSubpathCount*mParams.mMaxBounces + mParams.mOutputDim.x-1) / mParams.mOutputDim.x);
        mParams.mRandomSeed++;
    }

    if (mStaticParams.useTemporalReuse)
    {
        // Temporal reservoir reuse.
        FALCOR_ASSERT(mpTemporalReusePass);
        FALCOR_ASSERT(renderData.getTexture(kInputMotionVectors));
        preparePass(pRenderContext, renderData, *mpTemporalReusePass);
        if (!mVarsChanged)
            mpTemporalReusePass->execute(pRenderContext, mParams.mOutputDim.x, mParams.mOutputDim.y);
        mParams.mRandomSeed++;
    }
    mParams.mPathReservoirIndex ^= 1;

    if (mStaticParams.spatialReusePasses > 0)
    {
        // Spatial reservoir reuse.
        FALCOR_ASSERT(mpSpatialReusePass);
        for (uint i = 0; i < mStaticParams.spatialReusePasses; i++)
        {
            preparePass(pRenderContext, renderData, *mpSpatialReusePass);
            mpSpatialReusePass->execute(pRenderContext, mParams.mOutputDim.x, mParams.mOutputDim.y);
            mParams.mRandomSeed++;
            mParams.mPathReservoirIndex ^= 1;
        }
    }

    // Copy radiance from reservoirs to output.
    FALCOR_ASSERT(mpCopyRadiancePass);
    preparePass(pRenderContext, renderData, *mpCopyRadiancePass);
    mpCopyRadiancePass->execute(pRenderContext, mParams.mOutputDim.x, mParams.mOutputDim.y);

    endFrame(pRenderContext, renderData);
}

void ReSTIRVCM::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Stats and debug options.
    dirty |= renderDebugUI(widget);

    if (dirty)
    {
        validateOptions();
        mOptionsChanged = true;
    }
}

bool ReSTIRVCM::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    runtimeDirty |= widget.var("Max bounces", mParams.mMaxBounces, 0u);
    widget.tooltip("Maximum number of bounces.\n1 = direct only\n2 = one indirect bounce etc.");

    // Sampling options.

    if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        dirty = true;
    }

    dirty |= widget.checkbox("Bidirectional path tracing (BPT)", mStaticParams.useBPT);
    widget.tooltip("Use bidirectional path tracing.\nThis option automatically enables NEE and MIS.");

    if (mStaticParams.useBPT)
    {
        runtimeDirty |= widget.var("Light sub-path count", mParams.mLightSubpathCount, 1u, 16000000u);
        widget.tooltip("Number of light sub-paths to trace when BPT is enabled.");

        dirty |= widget.checkbox("Resample pure light paths", mStaticParams.useLightTraceReservoirs);
        widget.tooltip("Store pure light paths in reservoirs,\nallowing them to be resampled.");

        dirty |= widget.checkbox("Light trace only", mStaticParams.lightTraceOnly);
        widget.tooltip("Only use light tracing.\nThis option disables tracing paths from the camera.");

        dirty |= widget.checkbox("Vertex merging (VM)", mStaticParams.useVM);
        widget.tooltip("Enable vertex merging.");

        if (mStaticParams.useVM)
        {
            dirty |= widget.checkbox("Vertex merging only", mStaticParams.useVMOnly);
            widget.tooltip("Only use vertex merging.\nThis is the same as progressive photon mapping.");

            runtimeDirty |= widget.var("Photon radius factor", mVMRadiusFactor, 1e-9f, 0.1f);
            widget.tooltip("Photon radius as a percentange of the scene radius.");

            runtimeDirty |= widget.slider("Photon radius alpha", mVMRadiusAlpha, 0.f, 1.f);
            widget.tooltip("Photon radius shrink factor.\nLower values cause the radius to shrink faster.");

            widget.text("Current radius: " + std::to_string(mParams.mMergeRadius) + " at frame " + std::to_string(mFrameCount));

            runtimeDirty |= widget.var("Hash grid cells", mParams.mPhotonCellCount, 1000u, 16000000u);
            widget.tooltip("Number of cells in the photon hash grid.");
        }
    }
    else
    {
        dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
        widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");
    }

    if (mStaticParams.useNEE || mStaticParams.useBPT)
    {
        dirty |= widget.var("MIS power exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
    }

    dirty |= widget.checkbox("Temporal reuse", mStaticParams.useTemporalReuse);
    const uint prevSpatialPasses = mStaticParams.spatialReusePasses;
    if (widget.var("Spatial reuse passes", mStaticParams.spatialReusePasses))
    {
        if (prevSpatialPasses == 0 && mStaticParams.spatialReusePasses > 0)
            dirty = true;
        else
            runtimeDirty = true;
    }
    if (mStaticParams.spatialReusePasses > 0)
    {
        runtimeDirty |= widget.var("Spatial reuse candidates", mParams.mSpatialReuseSamples, 1u);
        runtimeDirty |= widget.var("Spatial reuse radius", mParams.mSpatialReuseRadius, 1.f);
    }
    if (mStaticParams.useTemporalReuse || mStaticParams.spatialReusePasses > 0)
    {
        runtimeDirty |= widget.var("M cap", mParams.mMCap, 1u);
        runtimeDirty |= widget.var("Reconnection distance", mParams.mReconnectionDistance, 0.f, 1.f);
        runtimeDirty |= widget.var("Reconnection roughness", mParams.mReconnectionRoughness, 0.f, 1.f);
    }

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
    }

    if (dirty) mRecompile = true;
    return dirty || runtimeDirty;
}

bool ReSTIRVCM::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging"))
    {
        dirty |= group.checkbox("Use fixed seed", mUseFixedSeed);
        group.tooltip("Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging.");
        if (mUseFixedSeed)
        {
            dirty |= group.var("Seed", mFixedSeed);
        }

        bool recompile = false;
        recompile |= group.checkbox("Debug BPT", mStaticParams.debugBPT);
        if (mStaticParams.debugBPT)
        {
            recompile |= group.var("Debug vertex count", mParams.mDebugTotalVertices, -1);
            group.tooltip("Only render paths with this many segments.");
            recompile |= group.var("Debug light vertex count", mParams.mDebugLightVertices, -1);
            group.tooltip("Only render paths with this many light vertices.");
        }
        dirty |= recompile;
        mRecompile |= recompile;

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

bool ReSTIRVCM::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

void ReSTIRVCM::reset()
{
    mFrameCount = 0;
}

void ReSTIRVCM::resetPrograms()
{
    mpReflectTypes = nullptr;
    mpSampleCameraPathsPass = nullptr;
    mpSampleLightPathsPass = nullptr;
    mpSortLightReservoirsPass = nullptr;
    mpLightReservoirResolvePass = nullptr;
    mpSpatialReusePass = nullptr;
    mpTemporalReusePass = nullptr;
    mpCopyRadiancePass = nullptr;

    mRecompile = true;
}

void ReSTIRVCM::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    // If we get here, a change that require recompilation of shader programs has occurred.
    // This may be due to change of scene defines, type conformances, shader modules, or other changes that require recompilation.
    // When type conformances and/or shader modules change, the programs need to be recreated. We assume programs have been reset upon such changes.
    // When only defines have changed, it is sufficient to update the existing programs and recreate the program vars.

    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getTypeConformances();

    auto preparePass = [&](ref<ComputePass> pass)
    {
        // Note that we must use set instead of add defines to replace any stale state.
        pass->getProgram()->setDefines(defines);

        // Recreate program vars. This may trigger recompilation if needed.
        // Note that program versions are cached, so switching to a previously used specialization is faster.
        pass->setVars(nullptr);
    };

    // Create compute passes.
    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);

    if (!mpSampleCameraPathsPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kVCMPassFilename).csEntry("SampleCameraPaths");
        mpSampleCameraPathsPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (mStaticParams.useBPT)
    {
        if (!mpSampleLightPathsPass)
        {
            ProgramDesc desc = baseDesc;
            desc.addShaderLibrary(kVCMPassFilename).csEntry("SampleLightPaths");
            mpSampleLightPathsPass = ComputePass::create(mpDevice, desc, defines, false);
        }

        if (mStaticParams.useLightTraceReservoirs)
        {
            if (!mpComputeLightReservoirOffsetsPass)
            {
                ProgramDesc desc = baseDesc;
                desc.addShaderLibrary(kVCMPassFilename).csEntry("ComputeLightReservoirOffsets");
                mpComputeLightReservoirOffsetsPass = ComputePass::create(mpDevice, desc, defines, false);
            }

            if (!mpSortLightReservoirsPass)
            {
                ProgramDesc desc = baseDesc;
                desc.addShaderLibrary(kVCMPassFilename).csEntry("SortLightTraceReservoirs");
                mpSortLightReservoirsPass = ComputePass::create(mpDevice, desc, defines, false);
            }

            if (!mpLightReservoirResolvePass)
            {
                ProgramDesc desc = baseDesc;
                desc.addShaderLibrary(kVCMPassFilename).csEntry("ResolveLightTraceReservoirs");
                mpLightReservoirResolvePass = ComputePass::create(mpDevice, desc, defines, false);
            }

            preparePass(mpComputeLightReservoirOffsetsPass);
            preparePass(mpSortLightReservoirsPass);
            preparePass(mpLightReservoirResolvePass);
        }

        preparePass(mpSampleLightPathsPass);
    }

    if (mStaticParams.useTemporalReuse)
    {
        if (!mpTemporalReusePass)
        {
            ProgramDesc desc = baseDesc;
            desc.addShaderLibrary(kTemporalReusePassFilename).csEntry("main");
            mpTemporalReusePass = ComputePass::create(mpDevice, desc, defines, false);
        }
        preparePass(mpTemporalReusePass);
    }

    if (mStaticParams.spatialReusePasses > 0)
    {
        if (!mpSpatialReusePass)
        {
            ProgramDesc desc = baseDesc;
            desc.addShaderLibrary(kSpatialReusePassFilename).csEntry("main");
            mpSpatialReusePass = ComputePass::create(mpDevice, desc, defines, false);
        }
        preparePass(mpSpatialReusePass);
    }

    if (!mpCopyRadiancePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kVCMPassFilename).csEntry("OutputRadiance");
        mpCopyRadiancePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpReflectTypes)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, false);
    }

    preparePass(mpSampleCameraPathsPass);
    preparePass(mpCopyRadiancePass);
    preparePass(mpReflectTypes);

    mVarsChanged = true;
    mRecompile = false;
}

void ReSTIRVCM::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.
    const uint32_t screenPixelCount = mParams.mOutputDim.x * mParams.mOutputDim.y;
    const size_t lightVertexCount = mParams.mLightSubpathCount * std::max(1u, mParams.mMaxBounces);

    auto var = mpReflectTypes->getRootVar();

    if (!mpReservoirs0 || mpReservoirs0->getElementCount() != screenPixelCount)
    {
        mpReservoirs0 = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mPathReservoirs0"], screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpReservoirs1 = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mPathReservoirs1"], screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mVarsChanged = true;
    }

    if (mStaticParams.useTemporalReuse)
    {
        if (!mpLastVbuffer || mpLastVbuffer->getWidth() != mParams.mOutputDim.x || mpLastVbuffer->getHeight() != mParams.mOutputDim.y)
        {
            mpLastVbuffer  = mpDevice->createTexture2D(mParams.mOutputDim.x, mParams.mOutputDim.y, mpScene->getHitInfo().getFormat(), 1, 1);
            mVarsChanged = true;
        }
        if (mpScene->getCamera()->getApertureRadius() > 0.f)
        {
            if (!mpLastViewDir || mpLastViewDir->getWidth() != mParams.mOutputDim.x || mpLastViewDir->getHeight() != mParams.mOutputDim.y)
            {
                const auto& pViewDir = renderData.getTexture(kInputViewDir);
                mpLastViewDir = mpDevice->createTexture2D(mParams.mOutputDim.x, mParams.mOutputDim.y, pViewDir->getFormat(), 1, 1);
                mVarsChanged = true;
            }
        }
    }

    if (mStaticParams.useBPT)
    {
        if (!mpLightVertices || mpLightVertices->getElementCount() != lightVertexCount || mVarsChanged)
        {
            mpLightVertices    = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mLightVertexCache"]["lightVertices"], lightVertexCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
            mpLightVertexCount = mpDevice->createBuffer(sizeof(uint32_t)*2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mVarsChanged = true;
        }

        if (mStaticParams.useVM) {
            if (!mpPhotonCellSizes || mpPhotonCellSizes->getElementCount() != mParams.mPhotonCellCount || mVarsChanged)
            {
                mpPhotonCellSizes     = mpDevice->createBuffer(sizeof(uint32_t)*mParams.mPhotonCellCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpPhotonCellOffsets   = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mPhotonMap"]["cellOffsets"], mParams.mPhotonCellCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
                mVarsChanged = true;
            }
        }

        if (mStaticParams.useLightTraceReservoirs)
        {
            if (!mpLightReservoirHashMapCellDataOffsets || mpLightReservoirHashMapCellDataOffsets->getElementCount() != screenPixelCount || mVarsChanged)
            {
                mpLightReservoirHashMapCellKeys        = mpDevice->createBuffer(sizeof(uint32_t)*screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpLightReservoirHashMapCellCounters    = mpDevice->createBuffer(sizeof(uint32_t)*screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mpLightReservoirHashMapCellDataOffsets = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mLightTraceReservoirs"]["mCellDataOffsets"], screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
                mVarsChanged = true;
            }
            if (!mpLightReservoirHashMapData || mpLightReservoirHashMapData->getElementCount() != mParams.mLightSubpathCount*mParams.mMaxBounces || mVarsChanged)
            {
                mpLightReservoirHashMapData            = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mLightTraceReservoirs"]["mData"]           , mParams.mLightSubpathCount*mParams.mMaxBounces, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
                mpLightReservoirHashMapSortedData      = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mLightTraceReservoirs"]["mSortedData"]     , mParams.mLightSubpathCount*mParams.mMaxBounces, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
                mpLightReservoirHashMapDataIndices     = mpDevice->createStructuredBuffer(var["gPathGenerator"]["mLightTraceReservoirs"]["mDataIndices"]    , mParams.mLightSubpathCount*mParams.mMaxBounces, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
                mVarsChanged = true;
            }
            if (!mpLightReservoirHashMapCounters || mVarsChanged)
            {
                mpLightReservoirHashMapCounters        = mpDevice->createBuffer(sizeof(uint32_t)*2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                mVarsChanged = true;
            }
        }
        else if (!mpLightImage || mpLightImage->getSize() != sizeof(float3) * screenPixelCount || mVarsChanged)
        {
            mpLightImage = mpDevice->createBuffer(sizeof(float3) * screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            mVarsChanged = true;
        }
    }
}

void ReSTIRVCM::resetLighting()
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

void ReSTIRVCM::prepareMaterials(RenderContext* pRenderContext)
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

bool ReSTIRVCM::prepareLighting(RenderContext* pRenderContext)
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
        if (mpSampleCameraPathsPass && mpSampleCameraPathsPass->getProgram()->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void ReSTIRVCM::bindShaderData(const ShaderVar& var, const RenderData& renderData) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        var["mAtomicRadiance"] = mpLightImage;

        var["mLightVertexCache"]["lightVertices"] = mpLightVertices;
        var["mLightVertexCache"]["lightVertexCount"] = mpLightVertexCount;

        var["mPhotonMap"]["cellSizes"]   = mpPhotonCellSizes;
        var["mPhotonMap"]["cellOffsets"] = mpPhotonCellOffsets;

        var["mPathReservoirs0"] = mpReservoirs0;
        var["mPathReservoirs1"] = mpReservoirs1;

        var["mLightTraceReservoirs"]["mCellKeys"]        = mpLightReservoirHashMapCellKeys;
        var["mLightTraceReservoirs"]["mCellCounters"]    = mpLightReservoirHashMapCellCounters;
        var["mLightTraceReservoirs"]["mCellDataOffsets"] = mpLightReservoirHashMapCellDataOffsets;
        var["mLightTraceReservoirs"]["mData"]            = mpLightReservoirHashMapData;
        var["mLightTraceReservoirs"]["mSortedData"]      = mpLightReservoirHashMapSortedData;
        var["mLightTraceReservoirs"]["mDataIndices"]     = mpLightReservoirHashMapDataIndices;
        var["mLightTraceReservoirs"]["mCounters"]        = mpLightReservoirHashMapCounters;

        var["mLightTraceReservoirs"]["mCellCount"] = mpLightReservoirHashMapCellDataOffsets ? mpLightReservoirHashMapCellDataOffsets->getElementCount() : 0u;
        var["mLightTraceReservoirs"]["mMaxSize"]   = mpLightReservoirHashMapData ? mpLightReservoirHashMapData->getElementCount() : 0u;

        mpSampleGenerator->bindShaderData(var);
    }

    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["mEnvMapSampler"]);
    if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(var["mEmissiveSampler"]);

    ref<Texture> pViewDir;
    if (mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        pViewDir = renderData.getTexture(kInputViewDir);
        if (!pViewDir) logWarning("Depth-of-field requires the '{}' input. Expect incorrect rendering.", kInputViewDir);
    }

    ref<Texture> pMotionVecs;
    if (mStaticParams.useTemporalReuse)
    {
        pMotionVecs = renderData.getTexture(kInputMotionVectors);
        if (!pMotionVecs) logWarning("Temporal reuse requires the '{}' input. Expect incorrect rendering.", kInputMotionVectors);
    }

    var["mParams"].setBlob(mParams);
    var["mVbuffer"] = renderData.getTexture(kInputVBuffer);
    var["mViewDir"] = pViewDir; // Can be nullptr
    var["mLastVbuffer"] = mpLastVbuffer;
    var["mLastViewDir"] = mpLastViewDir;
    var["mMotionVectors"] = pMotionVecs; // Required for temporal reuse
    var["mOutputRadiance"] = renderData.getTexture(kOutputColor);
}

bool ReSTIRVCM::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
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
            if (pTexture && (pTexture->getWidth() != mParams.mOutputDim.x || pTexture->getHeight() != mParams.mOutputDim.y)) resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("ReSTIRVCM I/O sizes don't match. The pass will be disabled.");
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

    mpPixelDebug->beginFrame(pRenderContext, mParams.mOutputDim);

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
            mFrameCount = 0;
    }

    // Update the random seed.
    uint seedsPerFrame = 1;
    if (mStaticParams.useBPT) seedsPerFrame++;
    if (mStaticParams.useBPT && mStaticParams.useLightTraceReservoirs) seedsPerFrame++;
    if (mStaticParams.useTemporalReuse) seedsPerFrame++;
    seedsPerFrame += mStaticParams.spatialReusePasses;

    mParams.mRandomSeed = mUseFixedSeed ? mFixedSeed : mFrameCount * seedsPerFrame;

    const auto& aabb = mpScene->getSceneBounds();
    mParams.mSceneSphere = float4(aabb.maxPoint + aabb.minPoint, length(aabb.maxPoint - aabb.minPoint))*.5f;
    mParams.mMergeRadius = mVMRadiusFactor * mParams.mSceneSphere.w;
    if (mVMRadiusAlpha < 1.f)
    {
        mParams.mMergeRadius /= std::pow(float(mFrameCount + 1), 0.5f * (1 - mVMRadiusAlpha));
        mParams.mMergeRadius = std::max(mParams.mMergeRadius, 1e-7f);
    }

    return true;
}

void ReSTIRVCM::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelDebug->endFrame(pRenderContext);

    // Copy pixel stats to outputs if available.
    if (mStaticParams.useTemporalReuse)
    {
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

        copyTexture( mpLastVbuffer.get(), renderData.getTexture(kInputVBuffer).get() );
        copyTexture( mpLastViewDir.get(), renderData.getTexture(kInputViewDir).get() );
    }

    mVarsChanged = false;
    mFrameCount++;
}

void ReSTIRVCM::preparePass(RenderContext* pRenderContext, const RenderData& renderData, ComputePass& pass) const
{
    ref<Program> program = pass.getProgram();

    FALCOR_ASSERT(program);

    auto var = pass.getRootVar();
    mpPixelDebug->prepareProgram(program, var);

    mpScene->setRaytracingShaderData(pRenderContext, var);

    bindShaderData(var["gPathGenerator"], renderData);

    pass.addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
}

DefineList ReSTIRVCM::StaticParams::getDefines(const ReSTIRVCM& owner) const
{
    DefineList defines;

    defines.add("USE_NEE", (useNEE || useBPT) ? "1" : "0");
    defines.add("USE_BIDIRECTIONAL", useBPT ? "1" : "0");
    defines.add("USE_VERTEX_MERGING", (useBPT && useVM && !lightTraceOnly) ? "1" : "0");
    defines.add("LIGHT_TRACE_ONLY", (useBPT && lightTraceOnly) ? "1" : "0");
    defines.add("USE_PPM_ONLY", (useBPT && useVM && useVMOnly && !lightTraceOnly) ? "1" : "0");
    defines.add("LIGHT_TRACE_RESERVOIRS", (useBPT && useLightTraceReservoirs) ? "1" : "0");
    defines.add("WAVEFRONT_TECHNIQUE_SELECTION", useWavefrontTechniqueSelection ? "1" : "0");
    defines.add("TEMPORAL_RMIS_TYPE", std::to_string(uint(temporalRMIS)));
    defines.add("SPATIAL_RMIS_TYPE", std::to_string(uint(spatialRMIS)));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("DEBUG_BPT", debugBPT ? "1" : "0");
    defines.add("USE_VIEW_DIR", debugBPT ? "1" : "0");

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene) defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT"      , scene && scene->useEnvLight()       ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");

    return defines;
}
