#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Materials/TexLODTypes.slang"

#include "Params.slang"

using namespace Falcor;

/** Fast path tracer.
*/
class ReSTIRVCM : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRVCM, "ReSTIRVCM", "ReSTIR VCM.");

    static ref<ReSTIRVCM> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReSTIRVCM>(pDevice, props); }

    ReSTIRVCM(ref<Device> pDevice, const Properties& props);

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void reset();

    static void registerBindings(pybind11::module& m);

private:
    void parseProperties(const Properties& props);
    void validateOptions();
    void resetPrograms();
    void updatePrograms();
    void setFrameDim(const uint2 frameDim);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void resetLighting();
    void prepareMaterials(RenderContext* pRenderContext);
    bool prepareLighting(RenderContext* pRenderContext);
    void bindShaderData(const ShaderVar& var, const RenderData& renderData) const;
    void preparePass(RenderContext* pRenderContext, const RenderData& renderData, ComputePass& pass) const;
    bool renderRenderingUI(Gui::Widgets& widget);
    bool renderDebugUI(Gui::Widgets& widget);
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);

    /** Static configuration. Changing any of these options require shader recompilation.
    */
    struct StaticParams
    {
        // Sampling parameters
        bool        useNEE = true; ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        bool        useBPT = true; ///< Use bidirectional path tracing. Automatically enables NEE.
        bool        useVM  = false; ///< Use vertex merging when using BPT.
        bool        useBsdfImportanceSampling = true;
        bool        debugBPT = false;
        bool        useLightTraceReservoirs = true;
        bool        lightTraceOnly = false;
        bool        useVMOnly = false;
        bool        useResampling = true;
        bool        useTemporalReuse = true;
        uint        spatialReusePasses = 1;
        uint32_t    sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM;
        float       misPowerExponent = 2;

        bool useWavefrontTechniqueSelection = false;
        RMISType temporalRMIS = RMISType::eTalbot;
        RMISType spatialRMIS = RMISType::ePairwise;

        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Power;

        DefineList getDefines(const ReSTIRVCM& owner) const;
    };

    // Configuration
    PathGeneratorParams             mParams;                    ///< Runtime path tracer parameters.
    StaticParams                    mStaticParams;              ///< Static parameters. These are set as compile-time constants in the shaders.
    mutable LightBVHSampler::Options mLightBVHOptions;          ///< Current options for the light BVH sampler.

    bool                            mEnabled = true;            ///< Switch to enable/disable the path tracer. When disabled the pass outputs are cleared.
    RenderPassHelpers::IOSize       mOutputSizeSelection = RenderPassHelpers::IOSize::Default;  ///< Selected output size.
    uint2                           mFixedOutputSize = { 512, 512 };                            ///< Output size in pixels when 'Fixed' size is selected.

    float                           mVMRadiusFactor  = .001f;    ///< Initial merge radius as a percentage of the scene radius.
    float                           mVMRadiusAlpha = 0.75f;      ///< Merge radius shrink factor.
    uint                            mFrameCount = 0;
    uint                            mFixedSeed = 0;
    bool                            mUseFixedSeed = false;
    bool                            mSwapReservoirs = false;
    uint                            mCurrentSeed = 0;

    // Internal state
    ref<Scene>                      mpScene;                     ///< The current scene, or nullptr if no scene loaded.
    ref<SampleGenerator>            mpSampleGenerator;           ///< GPU pseudo-random sample generator.
    std::unique_ptr<EnvMapSampler>  mpEnvMapSampler;             ///< Environment map sampler or nullptr if not used.
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;     ///< Emissive light sampler or nullptr if not used.
    std::unique_ptr<PixelDebug>     mpPixelDebug;                ///< Utility class for pixel debugging (print in shaders).

    bool                            mRecompile = false;          ///< Set to true when program specialization has changed.
    bool                            mVarsChanged = true;         ///< This is set to true whenever the program vars have changed and resources need to be rebound.
    bool                            mOptionsChanged = false;     ///< True if the config has changed since last frame.

    ref<ComputePass>                mpReflectTypes;              ///< Helper for reflecting structured buffer types.
    ref<ComputePass>                mpSampleCameraPathsPass;     ///< Camera trace pass.
    ref<ComputePass>                mpSampleLightPathsPass;      ///< Light trace pass.
    ref<ComputePass>                mpTemporalReusePass;         ///< Temporal reservoir reuse pass.
    ref<ComputePass>                mpSpatialReusePass;          ///< Spatial reservoir reuse pass.
    ref<ComputePass>                mpComputeLightReservoirOffsetsPass;   ///<
    ref<ComputePass>                mpSortLightReservoirsPass;   ///<
    ref<ComputePass>                mpLightReservoirResolvePass; ///< Fullscreen compute pass merging light traced reservoirs within each pixel.
    ref<ComputePass>                mpCopyRadiancePass;          ///< Fullscreen compute pass writing reservoir samples to the output buffer.

    ref<Buffer>                     mpLightImage;                ///< Light trace image. Light subpath contributions are atomically added to this.
    ref<Buffer>                     mpLightVertices;             ///< Light sub-path vertices.
    ref<Buffer>                     mpLightVertexCount;          ///< Light vertex counter.
    ref<Buffer>                     mpPhotonCellSizes;           ///< Photon grid cell sizes.
    ref<Buffer>                     mpPhotonCellOffsets;         ///< Photon grid cell offsets.
    ref<Buffer>                     mpReservoirs0;               ///< Per-pixel reservoirs.
    ref<Buffer>                     mpReservoirs1;               ///< Per-pixel reservoirs.

    ref<Buffer>                     mpLightReservoirHashMapCellKeys;
    ref<Buffer>                     mpLightReservoirHashMapCellCounters;
    ref<Buffer>                     mpLightReservoirHashMapCellDataOffsets;
    ref<Buffer>                     mpLightReservoirHashMapData;
    ref<Buffer>                     mpLightReservoirHashMapSortedData;
    ref<Buffer>                     mpLightReservoirHashMapDataIndices;
    ref<Buffer>                     mpLightReservoirHashMapCounters;

    ref<Texture>                    mpLastVbuffer;               ///< Copy of the vbuffer from last frame.
    ref<Texture>                    mpLastViewDir;               ///< Copy of the view directions from last frame.
};
