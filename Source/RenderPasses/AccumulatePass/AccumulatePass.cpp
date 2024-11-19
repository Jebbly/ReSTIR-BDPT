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
#include "AccumulatePass.h"
#include "RenderGraph/RenderPassStandardFlags.h"

static void regAccumulatePass(pybind11::module& m)
{
    pybind11::class_<AccumulatePass, RenderPass, ref<AccumulatePass>> pass(m, "AccumulatePass");
    pass.def_property("enabled", &AccumulatePass::isEnabled, &AccumulatePass::setEnabled);
    pass.def("reset", &AccumulatePass::reset);
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AccumulatePass>();
    ScriptBindings::registerBinding(regAccumulatePass);
}

namespace
{
const char kShaderFile[] = "RenderPasses/AccumulatePass/Accumulate.cs.slang";

const char kInputChannel[] = "input";
const char kInputMotionVecsChannel[] = "mvec";
const char kInputPosWChannel[] = "posW";
const char kInputNormalWChannel[] = "normalW";
const char kOutputChannel[] = "output";

// Serialized parameters
const char kEnabled[] = "enabled";
const char kOutputFormat[] = "outputFormat";
const char kOutputSize[] = "outputSize";
const char kFixedOutputSize[] = "fixedOutputSize";
const char kAutoReset[] = "autoReset";
const char kPrecisionMode[] = "precisionMode";
const char kMaxFrameCount[] = "maxFrameCount";
const char kOverflowMode[] = "overflowMode";
const char kUseMotionVectors[] = "useMotionVectors";
const char kUseGbuffer[] = "useGbuffer";
const char kMaxDistance[] = "maxDistance";
const char kMaxAngle[] = "maxAngle";
} // namespace

AccumulatePass::AccumulatePass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // Deserialize pass from dictionary.
    for (const auto& [key, value] : props)
    {
        if (key == kEnabled)
            mEnabled = value;
        else if (key == kOutputFormat)
            mOutputFormat = value;
        else if (key == kOutputSize)
            mOutputSizeSelection = value;
        else if (key == kFixedOutputSize)
            mFixedOutputSize = value;
        else if (key == kAutoReset)
            mAutoReset = value;
        else if (key == kPrecisionMode)
            mPrecisionMode = value;
        else if (key == kMaxFrameCount)
            mMaxFrameCount = value;
        else if (key == kOverflowMode)
            mOverflowMode = value;
        else if (key == kUseMotionVectors)
            mUseMotionVectors = value;
        else if (key == kUseGbuffer)
            mUseGbuffer = value;
        else if (key == kMaxDistance)
            mMaxDistance = value;
        else if (key == kMaxAngle)
            mMaxAngle = value;
        else
            logWarning("Unknown property '{}' in AccumulatePass properties.", key);
    }

    if (props.has("enableAccumulation"))
    {
        logWarning("'enableAccumulation' is deprecated. Use 'enabled' instead.");
        if (!props.has(kEnabled))
            mEnabled = props["enableAccumulation"];
    }

    mpState = ComputeState::create(mpDevice);
}

Properties AccumulatePass::getProperties() const
{
    Properties props;
    props[kEnabled] = mEnabled;
    if (mOutputFormat != ResourceFormat::Unknown)
        props[kOutputFormat] = mOutputFormat;
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        props[kFixedOutputSize] = mFixedOutputSize;
    props[kAutoReset] = mAutoReset;
    props[kPrecisionMode] = mPrecisionMode;
    props[kMaxFrameCount] = mMaxFrameCount;
    props[kOverflowMode] = mOverflowMode;
    props[kUseMotionVectors] = mUseMotionVectors;
    props[kUseGbuffer] = mUseGbuffer;
    props[kMaxDistance] = mMaxDistance;
    props[kMaxAngle] = mMaxAngle;
    return props;
}

RenderPassReflection AccumulatePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);
    const auto fmt = mOutputFormat != ResourceFormat::Unknown ? mOutputFormat : ResourceFormat::RGBA32Float;

    reflector.addInput(kInputChannel, "Input data to be temporally accumulated").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputMotionVecsChannel, "Input motion vectors").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kInputPosWChannel, "Input world space position").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kInputNormalWChannel, "Input world space normal").bindFlags(ResourceBindFlags::ShaderResource).flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(kOutputChannel, "Output data that is temporally accumulated")
        .bindFlags(ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(fmt)
        .texture2D(sz.x, sz.y);
    return reflector;
}

void AccumulatePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mAutoReset)
    {
        // Query refresh flags passed down from the application and other passes.
        auto& dict = renderData.getDictionary();
        auto refreshFlags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);

        // If any refresh flag is set, we reset frame accumulation.
        if (refreshFlags != RenderPassRefreshFlags::None)
            reset();

        // Reset accumulation upon all scene changes, except camera jitter and history changes.
        // TODO: Add UI options to select which changes should trigger reset
        if (mpScene)
        {
            auto sceneUpdates = mpScene->getUpdates();
            if ((sceneUpdates & ~Scene::UpdateFlags::CameraPropertiesChanged) != Scene::UpdateFlags::None)
            {
                reset();
            }
            if (is_set(sceneUpdates, Scene::UpdateFlags::CameraPropertiesChanged))
            {
                auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
                auto cameraChanges = mpScene->getCamera()->getChanges();
                if ((cameraChanges & ~excluded) != Camera::Changes::None)
                    reset();
            }
        }
    }

    // Check if we reached max number of frames to accumulate and handle overflow.
    if (mMaxFrameCount > 0 && mFrameCount == mMaxFrameCount)
    {
        switch (mOverflowMode)
        {
        case OverflowMode::Stop:
            return;
        case OverflowMode::Reset:
            reset();
            break;
        case OverflowMode::EMA:
            break;
        }
    }

    // Grab our input/output buffers.
    ref<Texture> pSrc = renderData.getTexture(kInputChannel);
    ref<Texture> pDst = renderData.getTexture(kOutputChannel);
    FALCOR_ASSERT(pSrc && pDst);

    const uint2 resolution = uint2(pSrc->getWidth(), pSrc->getHeight());
    const bool resolutionMatch = pDst->getWidth() == resolution.x && pDst->getHeight() == resolution.y;

    // Reset accumulation when resolution changes.
    if (any(resolution != mFrameDim))
    {
        mFrameDim = resolution;
        reset();
    }

    // Verify that output is non-integer format. It shouldn't be since reflect() requests a floating-point format.
    if (isIntegerFormat(pDst->getFormat()))
        FALCOR_THROW("AccumulatePass: Output to integer format is not supported");

    // Issue error and disable pass if unsupported I/O size. The user can hit continue and fix the config or abort.
    if (mEnabled && !resolutionMatch)
    {
        logError("AccumulatePass I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    // Decide action based on current configuration:
    // - The accumulation pass supports integer input but requires matching I/O size.
    // - Blit supports mismatching size but requires non-integer format.
    // - As a fallback, issue warning and clear the output.

    if (!mEnabled && !isIntegerFormat(pSrc->getFormat()))
    {
        // Only blit mip 0 and array slice 0, because that's what the accumulation uses otherwise.
        pRenderContext->blit(pSrc->getSRV(0, 1, 0, 1), pDst->getRTV(0, 0, 1));
    }
    else if (resolutionMatch)
    {
        accumulate(pRenderContext, pSrc, pDst, renderData);
    }
    else
    {
        logWarning("AccumulatePass unsupported I/O configuration. The output will be cleared.");
        pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0));
    }
}

void AccumulatePass::accumulate(RenderContext* pRenderContext, const ref<Texture>& pSrc, const ref<Texture>& pDst, const RenderData& renderData)
{
    FALCOR_ASSERT(pSrc && pDst);
    FALCOR_ASSERT(pSrc->getWidth() == mFrameDim.x && pSrc->getHeight() == mFrameDim.y);
    FALCOR_ASSERT(pDst->getWidth() == mFrameDim.x && pDst->getHeight() == mFrameDim.y);
    const FormatType srcType = getFormatType(pSrc->getFormat());

    // If for the first time, or if the input format type has changed, (re)compile the programs.
    if (mpProgram.empty() || srcType != mSrcType)
    {
        DefineList defines;
        switch (srcType)
        {
        case FormatType::Uint:
            defines.add("_INPUT_FORMAT", "INPUT_FORMAT_UINT");
            break;
        case FormatType::Sint:
            defines.add("_INPUT_FORMAT", "INPUT_FORMAT_SINT");
            break;
        default:
            defines.add("_INPUT_FORMAT", "INPUT_FORMAT_FLOAT");
            break;
        }
        // Create accumulation programs.
        // Note only compensated summation needs precise floating-point mode.
        mpProgram[Precision::Double] =
            Program::createCompute(mpDevice, kShaderFile, "accumulateDouble", defines, SlangCompilerFlags::TreatWarningsAsErrors);
        mpProgram[Precision::Single] =
            Program::createCompute(mpDevice, kShaderFile, "accumulateSingle", defines, SlangCompilerFlags::TreatWarningsAsErrors);
        mpProgram[Precision::SingleCompensated] = Program::createCompute(
            mpDevice,
            kShaderFile,
            "accumulateSingleCompensated",
            defines,
            SlangCompilerFlags::FloatingPointModePrecise | SlangCompilerFlags::TreatWarningsAsErrors
        );
        mpVars = ProgramVars::create(mpDevice, mpProgram[mPrecisionMode]->getReflector());

        mSrcType = srcType;
    }

    ref<Texture> pMVec = renderData.getTexture(kInputMotionVecsChannel);
    ref<Texture> pPosW = renderData.getTexture(kInputPosWChannel);
    ref<Texture> pNormalW = renderData.getTexture(kInputNormalWChannel);

    mpProgram[mPrecisionMode]->addDefine("_USE_MOTION_VECS", mUseMotionVectors && pMVec ? "1" : "0");
    mpProgram[mPrecisionMode]->addDefine("_USE_GBUFFER", mUseGbuffer && pPosW && pNormalW ? "1" : "0");

    // Setup accumulation.
    prepareAccumulation(pRenderContext, mFrameDim.x, mFrameDim.y, mUseMotionVectors && pMVec, mUseGbuffer && pPosW && pNormalW);

    // Set shader parameters.
    auto var = mpVars->getRootVar();
    var["PerFrameCB"]["gResolution"] = mFrameDim;
    var["PerFrameCB"]["gAccumCount"] = mFrameCount;
    var["PerFrameCB"]["gAccumulate"] = mEnabled;
    var["PerFrameCB"]["gMaxAccum"] = mMaxFrameCount;
    var["PerFrameCB"]["gMaxDistance"] = mMaxDistance * mpScene->getSceneBounds().extent().length();
    var["PerFrameCB"]["gCosMaxAngle"] = (float)std::cos(mMaxAngle * (M_PI / 180.f));
    var["gCurFrame"] = pSrc;
    var["gOutputFrame"] = pDst;
    var["gMotionVectors"] = pMVec;
    var["gCurPosW"] = pPosW;
    var["gCurNormalW"] = pNormalW;
    var["gLastPosW"] = mpLastPosW;
    var["gLastNormalW"] = mpLastNormalW;

    // Bind accumulation buffers. Some of these may be nullptr's.
    var["gSampleCount"]     = mpSampleCount[0];
    var["gFrameSum"]        = mpFrameSum[0];
    var["gFrameCorr"]       = mpFrameCorr[0];
    var["gFrameSumLo"]      = mpFrameSumLo[0];
    var["gFrameSumHi"]      = mpFrameSumHi[0];
    var["gLastSampleCount"] = mpSampleCount[1];
    var["gLastFrameSum"]    = mpFrameSum[1];
    var["gLastFrameCorr"]   = mpFrameCorr[1];
    var["gLastFrameSumLo"]  = mpFrameSumLo[1];
    var["gLastFrameSumHi"]  = mpFrameSumHi[1];

    // Update the frame count.
    // The accumulation limit (mMaxFrameCount) has a special value of 0 (no limit) and is not supported in the SingleCompensated mode.
    if (mMaxFrameCount == 0 || mPrecisionMode == Precision::SingleCompensated || mFrameCount < mMaxFrameCount)
    {
        mFrameCount++;
    }

    // Run the accumulation program.
    auto pProgram = mpProgram[mPrecisionMode];
    FALCOR_ASSERT(pProgram);
    uint3 numGroups = div_round_up(uint3(mFrameDim.x, mFrameDim.y, 1u), pProgram->getReflector()->getThreadGroupSize());
    mpState->setProgram(pProgram);
    pRenderContext->dispatch(mpState.get(), mpVars.get(), numGroups);

    if (mUseMotionVectors && pMVec) pRenderContext->copyResource(mpSampleCount[1].get(), mpSampleCount[0].get());
    if (mpFrameSum[0]) pRenderContext->copyResource(mpFrameSum[1].get(), mpFrameSum[0].get());
    if (mpFrameCorr[0]) pRenderContext->copyResource(mpFrameCorr[1].get(), mpFrameCorr[0].get());
    if (mpFrameSumLo[0]) pRenderContext->copyResource(mpFrameSumLo[1].get(), mpFrameSumLo[0].get());
    if (mpFrameSumHi[0]) pRenderContext->copyResource(mpFrameSumHi[1].get(), mpFrameSumHi[0].get());
    if (mUseGbuffer && pPosW && pNormalW)
    {
        pRenderContext->copyResource(mpLastPosW.get(), pPosW.get());
        pRenderContext->copyResource(mpLastNormalW.get(), pNormalW.get());
    }
}

void AccumulatePass::renderUI(Gui::Widgets& widget)
{
    // Controls for output size.
    // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
    if (widget.dropdown("Output size", mOutputSizeSelection))
        requestRecompile();
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
    {
        if (widget.var("Size in pixels", mFixedOutputSize, 32u, 16384u))
            requestRecompile();
    }

    if (bool enabled = isEnabled(); widget.checkbox("Enabled", enabled))
        setEnabled(enabled);

    if (mEnabled)
    {
        if (widget.button("Reset", true))
            reset();

        widget.checkbox("Auto Reset", mAutoReset);
        widget.tooltip("Reset accumulation automatically upon scene changes and refresh flags.");

        widget.checkbox("Use motion vectors", mUseMotionVectors);
        widget.tooltip("Accumulate using motion vectors.");

        widget.checkbox("Use gbuffer", mUseGbuffer);
        widget.tooltip("Accumulate using motion vectors.");

        if (mUseGbuffer)
        {
            widget.var("Max distance", mMaxDistance, 0.f, 1.f);
            widget.tooltip("Maximum distance to allow reprojection\nas a percentage of the scene radius.");

            widget.var("Max normal", mMaxAngle, 0.f, 180.f);
            widget.tooltip("Maximum angle (in degrees) to allow reprojection.");
        }

        if (widget.dropdown("Mode", mPrecisionMode))
        {
            // Reset accumulation when mode changes.
            reset();
        }

        if (mPrecisionMode != Precision::SingleCompensated)
        {
            // When mMaxFrameCount is nonzero, the accumulate pass will only compute the average of
            // up to that number of frames. Further frames will be accumulated in the exponential moving
            // average fashion, i.e. every next frame is blended with the history using the same weight.
            if (widget.var("Max Frames", mMaxFrameCount, 0u))
            {
                reset();
            }
            widget.tooltip("Maximum number of frames to accumulate before triggering overflow. 0 means infinite accumulation.");

            if (widget.dropdown("Overflow Mode", mOverflowMode))
            {
                reset();
            }
            widget.tooltip(
                "What to do after maximum number of frames are accumulated:\n"
                "  Stop: Stop accumulation and retain accumulated image.\n"
                "  Reset: Reset accumulation.\n"
                "  EMA: Switch to exponential moving average accumulation.\n"
            );
        }

        const std::string text = std::string("Frames accumulated ") + std::to_string(mFrameCount);
        widget.text(text);
    }
}

void AccumulatePass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    // Reset accumulation when the scene changes.
    reset();
}

void AccumulatePass::onHotReload(HotReloadFlags reloaded)
{
    // Reset accumulation if programs changed.
    if (is_set(reloaded, HotReloadFlags::Program))
        reset();
}

void AccumulatePass::setEnabled(bool enabled)
{
    if (enabled != mEnabled)
    {
        mEnabled = enabled;
        reset();
    }
}

void AccumulatePass::reset()
{
    mFrameCount = 0;
}

void AccumulatePass::prepareAccumulation(RenderContext* pRenderContext, uint32_t width, uint32_t height, bool useMotionVecs, bool useGbuffer)
{
    // Allocate/resize/clear buffers for intermedate data. These are different depending on accumulation mode.
    // Buffers that are not used in the current mode are released.
    auto prepareBuffer = [&](ref<Texture>& b, ResourceFormat format, bool bufUsed)
    {
        if (!bufUsed)
        {
            b = nullptr;
            return;
        }

        // (Re-)create buffer if needed.
        if (!b || b->getWidth() != width || b->getHeight() != height)
        {
            for (uint i = 0; i < 2; i++) {
                b = mpDevice->createTexture2D(
                    width, height, format, 1, 1, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
                FALCOR_ASSERT(b);
            }
            reset();
        }

        // Clear data if accumulation has been reset (either above or somewhere else).
        if (mFrameCount == 0)
        {
            if (getFormatType(format) == FormatType::Float)
                pRenderContext->clearUAV(b->getUAV().get(), float4(0.f));
            else
                pRenderContext->clearUAV(b->getUAV().get(), uint4(0));
        }
    };
    auto prepareBuffers = [&](auto& bufs, ResourceFormat format, bool bufUsed) {
        for (auto& b : bufs)
            prepareBuffer(b, format, bufUsed);
    };

    prepareBuffers(mpSampleCount, ResourceFormat::R32Uint, useMotionVecs);
    prepareBuffers(
        mpFrameSum, ResourceFormat::RGBA32Float, mPrecisionMode == Precision::Single || mPrecisionMode == Precision::SingleCompensated
    );
    prepareBuffers(mpFrameCorr, ResourceFormat::RGBA32Float, mPrecisionMode == Precision::SingleCompensated);
    prepareBuffers(mpFrameSumLo, ResourceFormat::RGBA32Uint, mPrecisionMode == Precision::Double);
    prepareBuffers(mpFrameSumHi, ResourceFormat::RGBA32Uint, mPrecisionMode == Precision::Double);
    prepareBuffer(mpLastPosW, ResourceFormat::RGBA32Float, useGbuffer);
    prepareBuffer(mpLastNormalW, ResourceFormat::RG32Float, useGbuffer);
}
