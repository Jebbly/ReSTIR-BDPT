add_plugin(ReSTIRVCM)

target_sources(ReSTIRVCM PRIVATE
    BitfieldMacros.slangh
    HashMap.slang
    LoadShadingData.slang
    Params.slang
    PathGenerator.slang
    PathReservoir.slang
    PathVertex.slang
    PhotonMap.slang
    Random.slang
    ReflectTypes.cs.slang
    ReSTIRVCM.cpp
    ReSTIRVCM.h
    SpatialReuse.cs.slang
    TemporalReuse.cs.slang
    VCM.cs.slang
)

target_copy_shaders(ReSTIRVCM RenderPasses/ReSTIRVCM)

target_source_group(ReSTIRVCM "RenderPasses")
