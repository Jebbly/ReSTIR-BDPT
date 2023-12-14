add_plugin(PathTracer)

target_sources(PathTracer PRIVATE
    BidirLightHelpers.slang
    ColorType.slang
    GeneratePaths.cs.slang
    LoadShadingData.slang
    Params.slang
    PathReservoir.slang
    PathState.slang
    PathTracer.slang
    PathTracer.cpp
    PathTracer.h
    PhotonMap.slang
    ReflectTypes.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(PathTracer RenderPasses/PathTracer)

target_source_group(PathTracer "RenderPasses")
