from falcor import *

def render_graph_PathTracer():
    g = RenderGraph("ReSTIRVCM")

    ReSTIRVCM = createPass("ReSTIRVCM", {})
    g.addPass(ReSTIRVCM, "ReSTIRVCM")

    VBufferRT = createPass("VBufferRT", {
        'adjustShadingNormals': False,
        'samplePattern': 'Center',
        'sampleCount': 1,
        'useAlphaTest': True })
    g.addPass(VBufferRT, "VBufferRT")

    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    ErrorMeasurePass = createPass("ErrorMeasurePass")
    g.addPass(ErrorMeasurePass, "ErrorMeasurePass")

    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer",    "ReSTIRVCM.vbuffer")
    g.addEdge("VBufferRT.viewW",      "ReSTIRVCM.viewW")
    g.addEdge("VBufferRT.mvec",       "ReSTIRVCM.mvec")
    g.addEdge("ReSTIRVCM.color",      "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("ToneMapper.dst", "ErrorMeasurePass.Source")

    g.markOutput("ErrorMeasurePass.Output")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None
