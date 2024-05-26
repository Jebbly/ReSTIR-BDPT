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
#include "ErrorMeasurePass.h"
#include "Core/AssetResolver.h"
#include <sstream>
#include <imgui.h>

namespace
{
const std::string kErrorComputationShaderFile = "RenderPasses/ErrorMeasurePass/ErrorMeasurer.cs.slang";
const std::string kConstantBufferName = "PerFrameCB";

// Input channels
const std::string kInputChannelWorldPosition = "WorldPosition";
const std::string kInputChannelSourceImage = "Source";
const std::string kInputChannelReferenceImage = "Reference";

// Output channel
const std::string kOutputChannelImage = "Output";

// Serialized parameters
const std::string kReferenceImagePath = "ReferenceImagePath";
const std::string kMeasurementsFilePath = "MeasurementsFilePath";
const std::string kIgnoreBackground = "IgnoreBackground";
const std::string kComputeSquaredDifference = "ComputeSquaredDifference";
const std::string kComputeAverage = "ComputeAverage";
const std::string kComputePercentage = "ComputePercentage";
const std::string kUseLoadedReference = "UseLoadedReference";
const std::string kReportRunningError = "ReportRunningError";
const std::string kRunningErrorSigma = "RunningErrorSigma";
const std::string kSelectedOutputId = "SelectedOutputId";
const std::string kOutputImageFilePath = "OutputImageFilePath";
const std::string kOutputFrameIndex = "OutputFrameIndex";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ErrorMeasurePass>();
}

const Gui::RadioButtonGroup ErrorMeasurePass::sOutputSelectionButtons = {
    {(uint32_t)OutputId::Source, "Source", true},
    {(uint32_t)OutputId::Reference, "Reference", true},
    {(uint32_t)OutputId::Difference, "Difference", true}};

const Gui::RadioButtonGroup ErrorMeasurePass::sOutputSelectionButtonsSourceOnly = {{(uint32_t)OutputId::Source, "Source", true}};

const Gui::RadioButtonGroup ErrorMeasurePass::sGraphModeSelectionButtons = {
    {(uint32_t)GraphAxisScale::Linear,    "Linear",     false},
    {(uint32_t)GraphAxisScale::LogLinear, "Log-Linear", true},
    {(uint32_t)GraphAxisScale::LogLog,    "Log-Log",    true}};

ErrorMeasurePass::ErrorMeasurePass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);

    mpParallelReduction = std::make_unique<ParallelReduction>(mpDevice);
    mpErrorMeasurerPass = ComputePass::create(mpDevice, kErrorComputationShaderFile);

    mMeasurementHistory.reserve(mMeasurementHistoryLength);
}

void ErrorMeasurePass::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kReferenceImagePath)
            mReferenceImagePath = value.operator std::filesystem::path();
        else if (key == kMeasurementsFilePath)
            mMeasurementsFilePath = value.operator std::filesystem::path();
        else if (key == kIgnoreBackground)
            mIgnoreBackground = value;
        else if (key == kComputeSquaredDifference)
            mComputeSquaredDifference = value;
        else if (key == kComputeAverage)
            mComputeAverage = value;
        else if (key == kComputePercentage)
            mComputePercentage = value;
        else if (key == kUseLoadedReference)
            mUseLoadedReference = value;
        else if (key == kReportRunningError)
            mReportRunningError = value;
        else if (key == kRunningErrorSigma)
            mRunningErrorSigma = value;
        else if (key == kSelectedOutputId)
            mSelectedOutputId = value;
        else if (key == kOutputImageFilePath)
            mOutputImageFilePath = value.operator std::filesystem::path();
        else if (key == kOutputFrameIndex)
            mOutputFrameIndex = value;
        else
        {
            logWarning("Unknown property '{}' in ErrorMeasurePass properties.", key);
        }
    }

    // Load/create files (if specified in config).
    loadReference();
    loadMeasurementsFile();
}


Properties ErrorMeasurePass::getProperties() const
{
    Properties props;
    props[kReferenceImagePath] = mReferenceImagePath;
    props[kMeasurementsFilePath] = mMeasurementsFilePath;
    props[kIgnoreBackground] = mIgnoreBackground;
    props[kComputeSquaredDifference] = mComputeSquaredDifference;
    props[kComputeAverage] = mComputeAverage;
    props[kComputePercentage] = mComputePercentage;
    props[kUseLoadedReference] = mUseLoadedReference;
    props[kReportRunningError] = mReportRunningError;
    props[kRunningErrorSigma] = mRunningErrorSigma;
    props[kSelectedOutputId] = mSelectedOutputId;
    props[kOutputImageFilePath] = mOutputImageFilePath;
    props[kOutputFrameIndex] = mOutputFrameIndex;
    return props;
}

RenderPassReflection ErrorMeasurePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputChannelSourceImage, "Source image");
    reflector.addInput(kInputChannelReferenceImage, "Reference image (optional)").flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addInput(kInputChannelWorldPosition, "World-space position").flags(RenderPassReflection::Field::Flags::Optional);
    // TODO: when compile() is available, match the format of the source image?
    reflector.addOutput(kOutputChannelImage, "Output image").format(ResourceFormat::RGBA32Float);
    return reflector;
}

void ErrorMeasurePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    ref<Texture> pSourceImageTexture = renderData.getTexture(kInputChannelSourceImage);

    if (mEnabled && mSetReference)
    {
        const uint32_t width = pSourceImageTexture->getWidth(), height = pSourceImageTexture->getHeight();
        if (!mpReferenceTexture || mpReferenceTexture->getWidth() != width || mpReferenceTexture->getHeight() != height)
        {
            mpReferenceTexture = mpDevice->createTexture2D(
                width,
                height,
                pSourceImageTexture->getFormat(),
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
            );
            FALCOR_ASSERT(mpReferenceTexture);
        }
        pRenderContext->copyResource(mpReferenceTexture.get(), pSourceImageTexture.get());
        mUseLoadedReference = true;
        mSetReference = false;
    }

    ref<Texture> pOutputImageTexture = renderData.getTexture(kOutputChannelImage);
    ref<Texture> pReference = getReference(renderData);

    if (!pReference)
    {
        // We don't have a reference image, so just copy the source image to the output.
        pRenderContext->blit(pSourceImageTexture->getSRV(), pOutputImageTexture->getRTV());
        return;
    }

    if (mEnabled) {
        // Create the texture for the difference image if this is our first
        // time through or if the source image resolution has changed.
        const uint32_t width = pSourceImageTexture->getWidth(), height = pSourceImageTexture->getHeight();
        if (!mpDifferenceTexture || mpDifferenceTexture->getWidth() != width || mpDifferenceTexture->getHeight() != height)
        {
            mpDifferenceTexture = mpDevice->createTexture2D(
                width,
                height,
                ResourceFormat::RGBA32Float,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
            );
            FALCOR_ASSERT(mpDifferenceTexture);
        }

        mMeasurements.valid = false;

        runDifferencePass(pRenderContext, renderData);
        runReductionPasses(pRenderContext, renderData);

        saveMeasurementsToFile();

        if (mMeasurementHistory.size() == mMeasurementHistoryLength) {
            for (size_t i = 0; i < mMeasurementHistory.size() - 1; i++)
                mMeasurementHistory[i] = mMeasurementHistory[i+1];
            mMeasurementHistory.back() = mMeasurements.avgError;
        } else
            mMeasurementHistory.emplace_back(mMeasurements.avgError);
        mMaxMeasurement = std::max(mMaxMeasurement, mMeasurements.avgError);
        mMinMeasurement = std::min(mMinMeasurement, mMeasurements.avgError);

        if (mCurrentFrameIndex == mOutputFrameIndex && !mOutputImageFilePath.empty())
        {
            pSourceImageTexture->captureToFile(0, 0, mOutputImageFilePath, Bitmap::FileFormat::ExrFile);
        }
        mCurrentFrameIndex++;
    }

    switch (mSelectedOutputId)
    {
    case OutputId::Source:
        pRenderContext->blit(pSourceImageTexture->getSRV(), pOutputImageTexture->getRTV());
        break;
    case OutputId::Reference:
        pRenderContext->blit(pReference->getSRV(), pOutputImageTexture->getRTV());
        break;
    case OutputId::Difference:
        pRenderContext->blit(mpDifferenceTexture->getSRV(), pOutputImageTexture->getRTV());
        break;
    default:
        FALCOR_THROW("ErrorMeasurePass: Unhandled OutputId case");
    }
}

void ErrorMeasurePass::runDifferencePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Bind textures.
    ref<Texture> pSourceTexture = renderData.getTexture(kInputChannelSourceImage);
    ref<Texture> pWorldPositionTexture = renderData.getTexture(kInputChannelWorldPosition);
    auto var = mpErrorMeasurerPass->getRootVar();
    var["gReference"] = getReference(renderData);
    var["gSource"] = pSourceTexture;
    var["gWorldPosition"] = pWorldPositionTexture;
    var["gResult"] = mpDifferenceTexture;

    // Set constant buffer parameters.
    const uint2 resolution = uint2(pSourceTexture->getWidth(), pSourceTexture->getHeight());
    var[kConstantBufferName]["gResolution"] = resolution;
    // If the world position texture is unbound, then don't do the background pixel check.
    var[kConstantBufferName]["gIgnoreBackground"] = (uint32_t)(mIgnoreBackground && pWorldPositionTexture);
    var[kConstantBufferName]["gComputeDiffSqr"] = (uint32_t)mComputeSquaredDifference;
    var[kConstantBufferName]["gComputeAverage"] = (uint32_t)mComputeAverage;
    var[kConstantBufferName]["gComputePercentage"] = (uint32_t)mComputePercentage;
    var[kConstantBufferName]["gDifferenceOffset"] = mDifferenceOffset;

    // Run the compute shader.
    mpErrorMeasurerPass->execute(pRenderContext, resolution.x, resolution.y);
}

void ErrorMeasurePass::runReductionPasses(RenderContext* pRenderContext, const RenderData& renderData)
{
    float4 error;
    mpParallelReduction->execute(pRenderContext, mpDifferenceTexture, ParallelReduction::Type::Sum, &error);

    const float pixelCountf = static_cast<float>(mpDifferenceTexture->getWidth() * mpDifferenceTexture->getHeight());
    mMeasurements.error = error.xyz() / pixelCountf - mDifferenceOffset;
    mMeasurements.avgError = (mMeasurements.error.x + mMeasurements.error.y + mMeasurements.error.z) / 3.f;
    mMeasurements.valid = true;

    if (mRunningAvgError < 0)
    {
        // The running error values are invalid. Start them off with the current frame's error.
        mRunningError = mMeasurements.error;
        mRunningAvgError = mMeasurements.avgError;
    }
    else
    {
        mRunningError = mRunningErrorSigma * mRunningError + (1 - mRunningErrorSigma) * mMeasurements.error;
        mRunningAvgError = mRunningErrorSigma * mRunningAvgError + (1 - mRunningErrorSigma) * mMeasurements.avgError;
    }

}

void ErrorMeasurePass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {
    mCurrentFrameIndex = 0;
}

void ErrorMeasurePass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled) return;

    const auto getFilename = [](const std::filesystem::path& path) { return path.empty() ? "N/A" : path.filename().string(); };

    // Create a button for loading the reference image.
    if (widget.button("Load reference"))
    {
        FileDialogFilterVec filters;
        filters.push_back({"exr", "High Dynamic Range"});
        filters.push_back({"pfm", "Portable Float Map"});
        std::filesystem::path path;
        if (openFileDialog(filters, path))
        {
            mReferenceImagePath = path;
            if (!loadReference())
                msgBox("Error", fmt::format("Failed to load reference image from '{}'.", path), MsgBoxType::Ok, MsgBoxIcon::Error);
        }
    }
    if (mReferenceImagePath.empty())
    {
        if (widget.button("Set reference", true))
            mSetReference = true;
    } else {
        widget.text(getFilename(mReferenceImagePath), true);
    }

    // Create a button for defining the measurements output file.
    if (widget.button("Set output data file"))
    {
        FileDialogFilterVec filters;
        filters.push_back({"csv", "CSV Files"});
        std::filesystem::path path;
        if (saveFileDialog(filters, path))
        {
            mMeasurementsFilePath = path;
            if (!loadMeasurementsFile())
                msgBox("Error", fmt::format("Failed to save measurements to '{}'.", path), MsgBoxType::Ok, MsgBoxIcon::Error);
        }
    }
    widget.text(getFilename(mMeasurementsFilePath), true);
    if (mMeasurementsFile && mMeasurementsFile.is_open())
    {
        if (widget.button("x", true))
        {
            mMeasurementsFile.close();
            mMeasurementsFilePath.clear();
        }
    }

    if (widget.button("Set output image"))
    {
        FileDialogFilterVec filters;
        filters.push_back({"exr", "High Dynamic Range"});
        std::filesystem::path path;
        if (saveFileDialog(filters, path))
        {
            mOutputImageFilePath = path;
        }
    }
    widget.text(getFilename(mOutputImageFilePath), true);
    if (!mOutputImageFilePath.empty())
    {
        if (widget.button("x", true))
        {
            mOutputImageFilePath.clear();
        }
    }
    widget.var<size_t>("Output frame", mOutputFrameIndex, 0, std::numeric_limits<size_t>::max(), 1.f);
    widget.text(std::to_string(mCurrentFrameIndex));
    if (widget.button("Reset counter", true))
    {
        mCurrentFrameIndex = 0;
    }

    // Radio buttons to select the output.
    widget.text("Show:");
    if (mMeasurements.valid)
    {
        widget.radioButtons(sOutputSelectionButtons, reinterpret_cast<uint32_t&>(mSelectedOutputId));
        widget.tooltip(
            "Press 'O' to change output mode; hold 'Shift' to reverse the cycling.\n\n"
            "Note: Difference is computed based on current - reference value.",
            true
        );
    }
    else
    {
        uint32_t dummyId = 0;
        widget.radioButtons(sOutputSelectionButtonsSourceOnly, dummyId);
    }

    widget.checkbox("Ignore background", mIgnoreBackground);
    widget.tooltip(
        "Do not include background pixels in the error measurements.\n"
        "This option requires the optional input '" +
            std::string(kInputChannelWorldPosition) + "' to be bound",
        true
    );
    widget.checkbox("Compute L2 error (rather than L1)", mComputeSquaredDifference);
    widget.checkbox("Compute RGB average", mComputeAverage);
    widget.tooltip(
        "When enabled, the average error over the RGB components is computed when creating the difference image.\n"
        "The average is computed after squaring the differences when L2 error is selected."
    );

    widget.checkbox("Compute percentage", mComputePercentage);
    widget.tooltip("When enabled, the error is divided by the reference value.");

    widget.var("Difference offset", mDifferenceOffset);
    widget.tooltip("Offset to apply to per-pixel differences.");

    widget.checkbox("Use loaded reference image", mUseLoadedReference);
    widget.tooltip(
        "Take the reference from the loaded image instead or the input channel.\n\n"
        "If the chosen reference doesn't exist, the error measurements are disabled.",
        true
    );
    // Display the filename of the reference file.
    const std::string referenceText = "Reference: " + getFilename(mReferenceImagePath);
    widget.text(referenceText);
    if (!mReferenceImagePath.empty())
    {
        widget.tooltip(mReferenceImagePath.string());
    }

    // Display the filename of the measurement file.
    const std::string outputText = "Output: " + getFilename(mMeasurementsFilePath);
    widget.text(outputText);
    if (!mMeasurementsFilePath.empty())
    {
        widget.tooltip(mMeasurementsFilePath.string());
    }

    // Print numerical error (scalar and RGB).
    if (widget.checkbox("Report running error", mReportRunningError) && mReportRunningError)
    {
        // The checkbox was enabled; mark the running error values invalid so that they start fresh.
        mRunningAvgError = -1.f;
    }
    widget.tooltip("Exponential moving average, sigma = " + std::to_string(mRunningErrorSigma));
    if (mMeasurements.valid)
    {
        // Use stream so we can control formatting.
        std::ostringstream oss;
        oss << std::scientific;

        const char* label = "MSE";
        if (mComputeSquaredDifference) {
            label = mComputePercentage ? "MSAPE" : "MSE";
        } else {
            label = mComputePercentage ? "MAPE" : "L1 error";
        }

        oss << label << " (avg): "
            << (mReportRunningError ? mRunningAvgError : mMeasurements.avgError) << std::endl;
        oss << label << " (rgb): "
            << (mReportRunningError ? mRunningError.r : mMeasurements.error.r) << ", "
            << (mReportRunningError ? mRunningError.g : mMeasurements.error.g) << ", "
            << (mReportRunningError ? mRunningError.b : mMeasurements.error.b);
        widget.text(oss.str());
    }
    else
    {
        widget.text("Error: N/A");
        widget.text("Error delta: N/A");
    }

    if (widget.var("History length", mMeasurementHistoryLength, size_t(100), size_t(100000))) {
        if (mMeasurementHistoryLength > mMeasurementHistory.size())
            mMeasurementHistory.reserve(mMeasurementHistoryLength);
        else
            mMeasurementHistory.resize(mMeasurementHistoryLength);
    }

    widget.radioButtons(sGraphModeSelectionButtons, reinterpret_cast<uint32_t&>(mGraphScaleMode));

    if (widget.button("Reset")) {
        mMeasurementHistory.clear();
        mMaxMeasurement = 0;
        mMinMeasurement = FLT_MAX;
    }
    if (mMeasurementHistory.size() > 1 && mMaxMeasurement > 0) {
        ImVec2 frame_size = { ImGui::GetWindowContentRegionWidth(), ImGui::GetTextLineHeight()*6 };

        ImGuiStyle& style = ImGui::GetStyle();

        ImVec2 windowCursor = ImGui::GetCursorScreenPos();
        ImVec2 graphMin { windowCursor.x + style.FramePadding.x, windowCursor.y + style.FramePadding.y};
        ImVec2 graphMax { windowCursor.x + frame_size.x - style.FramePadding.x, windowCursor.y + frame_size.y - style.FramePadding.y};

        ImGui::Dummy(frame_size);

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(graphMin, graphMax, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameRounding, 0);
        drawList->PathClear();

        ImVec2 mousePos = ImGui::GetIO().MousePos;

        const ImU32 lineColor = ImGui::GetColorU32(ImGuiCol_PlotLines);

        for (int i = 0; i < mMeasurementHistory.size(); i++)
        {
            ImVec2 p = { i+1.f, mMeasurementHistory[i] };

            // scale x axis
            if (mGraphScaleMode == GraphAxisScale::LogLog) {
                p.x = std::log10(p.x) / std::log10(mMeasurementHistory.size());
            } else {
                p.x = (p.x - 1) / mMeasurementHistory.size();
            }

            // scale y axis
            if (mGraphScaleMode == GraphAxisScale::LogLog || mGraphScaleMode == GraphAxisScale::LogLinear) {
                p.y = (std::log10(p.y) - std::log10(mMinMeasurement)) / (std::log10(mMaxMeasurement) - std::log10(mMinMeasurement));
            } else {
                p.y = (p.y - mMinMeasurement) / (mMaxMeasurement - mMinMeasurement);
            }

            drawList->PathLineTo({
                graphMin.x + (graphMax.x - graphMin.x) * p.x,
                graphMin.y + (graphMax.y - graphMin.y) * (1 - p.y) // invert y since screen coordinates are y-down
            });
        }

        drawList->PathStroke(lineColor, 0, 2.5f);
    }
}

bool ErrorMeasurePass::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.type == KeyboardEvent::Type::KeyPressed && keyEvent.key == Input::Key::O)
    {
        int32_t ofs = keyEvent.hasModifier(Input::Modifier::Shift) ? -1 : 1;
        int32_t index = (int32_t)mSelectedOutputId;
        index = (index + ofs + (int32_t)OutputId::Count) % (int32_t)OutputId::Count;
        mSelectedOutputId = (OutputId)index;
        return true;
    }

    return false;
}

bool ErrorMeasurePass::loadReference()
{
    if (mReferenceImagePath.empty())
        return false;

    // TODO: it would be nice to also be able to take the reference image as an input.
    std::filesystem::path resolvedPath = AssetResolver::getDefaultResolver().resolvePath(mReferenceImagePath);
    mpReferenceTexture = Texture::createFromFile(mpDevice, resolvedPath, false /* no MIPs */, false /* linear color */);
    if (!mpReferenceTexture)
    {
        logWarning("Failed to load texture from '{}'", mReferenceImagePath);
        mReferenceImagePath.clear();
        return false;
    }

    mUseLoadedReference = mpReferenceTexture != nullptr;
    mRunningAvgError = -1.f; // Mark running error values as invalid.
    return true;
}

ref<Texture> ErrorMeasurePass::getReference(const RenderData& renderData) const
{
    return mUseLoadedReference ? mpReferenceTexture : renderData.getTexture(kInputChannelReferenceImage);
}

bool ErrorMeasurePass::loadMeasurementsFile()
{
    if (mMeasurementsFilePath.empty())
        return false;

    mMeasurementsFile = std::ofstream(mMeasurementsFilePath, std::ios::trunc);
    if (!mMeasurementsFile)
    {
        logWarning(fmt::format("Failed to open file '{}'.", mMeasurementsFilePath));
        mMeasurementsFilePath.clear();
        return false;
    }
    else
    {
        if (mComputeSquaredDifference)
        {
            mMeasurementsFile << "avg_L2_error,red_L2_error,green_L2_error,blue_L2_error" << std::endl;
        }
        else
        {
            mMeasurementsFile << "avg_L1_error,red_L1_error,green_L1_error,blue_L1_error" << std::endl;
        }
        mMeasurementsFile << std::scientific;
    }

    mCurrentFrameIndex = 0;

    return true;
}

void ErrorMeasurePass::saveMeasurementsToFile()
{
    if (!mMeasurementsFile)
        return;

    FALCOR_ASSERT(mMeasurements.valid);
    mMeasurementsFile << mMeasurements.avgError << ",";
    mMeasurementsFile << mMeasurements.error.r << ',' << mMeasurements.error.g << ',' << mMeasurements.error.b;
    mMeasurementsFile << std::endl;
}
