from falcor import *
import os

def render_graph_PathTracer():
    g = RenderGraph("ReSTIRBDPT")

    ReSTIRBDPT = createPass("ReSTIRBDPT", {})
    g.addPass(ReSTIRBDPT, "ReSTIRBDPT")

    VBufferRT = createPass("VBufferRT", {
        'adjustShadingNormals': False,
        'samplePattern': 'Center',
        'sampleCount': 1,
        'useAlphaTest': True })
    g.addPass(VBufferRT, "VBufferRT")

    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    ErrorMeasurePass = createPass("ErrorMeasurePass")
    g.addPass(ErrorMeasurePass, "ErrorMeasurePass")

    g.addEdge("VBufferRT.vbuffer",       "ReSTIRBDPT.vbuffer")
    g.addEdge("VBufferRT.viewW",         "ReSTIRBDPT.viewW")
    g.addEdge("VBufferRT.mvec",          "ReSTIRBDPT.mvec")
    g.addEdge("ReSTIRBDPT.color",         "AccumulatePass.input")
    g.addEdge("VBufferRT.mvec",          "AccumulatePass.mvec")
    g.addEdge("VBufferRT.posW",          "AccumulatePass.posW")
    g.addEdge("VBufferRT.normalW",       "AccumulatePass.normalW")
    g.addEdge("AccumulatePass.output",   "ErrorMeasurePass.Source")
    g.addEdge("ErrorMeasurePass.Output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g

PathTracer = render_graph_PathTracer()
try: m.addGraph(PathTracer)
except NameError: None

m.loadScene(os.path.join(os.path.dirname(__file__), os.path.pardir, "scene", "veach-bidir.pyscene"))
