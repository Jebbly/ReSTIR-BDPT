from falcor import *
import os

def render_graph_ReSTIRVCM():
    g = RenderGraph("ReSTIRBDPT")

    ReSTIRBDPT = createPass("ReSTIRBDPT", {})
    g.addPass(ReSTIRBDPT, "ReSTIRBDPT")

    VBufferRT = createPass("VBufferRT", {
        'adjustShadingNormals': False,
        'samplePattern': 'Center',
        'sampleCount': 1,
        'useAlphaTest': True })
    g.addPass(VBufferRT, "VBufferRT")

    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")

    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    ErrorMeasurePass = createPass("ErrorMeasurePass")
    g.addPass(ErrorMeasurePass, "ErrorMeasurePass")

    g.addEdge("VBufferRT.vbuffer",     "ReSTIRBDPT.vbuffer")
    g.addEdge("VBufferRT.viewW",       "ReSTIRBDPT.viewW")
    g.addEdge("VBufferRT.mvec",        "ReSTIRBDPT.mvec")
    g.addEdge("ReSTIRBDPT.color",       "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ErrorMeasurePass.Source")
    g.addEdge("ErrorMeasurePass.Output", "ToneMapper.src")

    g.markOutput("ErrorMeasurePass.Output")
    return g

graph = render_graph_ReSTIRVCM()
try: m.addGraph(graph)
except NameError: None

m.resizeSwapChain(1920, 1080)
m.clock.stop()

folder = os.path.join(__file__, os.path.pardir, 'Experiments')

def Render(
    scene,
    dstFolder,
    dstImage,
    numAnimationFrames = 1,
    samplesPerFrame = 1,
    cameraPose = None):

    m.loadScene(scene)

    if cameraPose is not None:
        pos,targ,up = cameraPose
        m.scene.camera.position = pos
        m.scene.camera.target   = targ
        m.scene.camera.up       = up

    m.frameCapture.outputDir = dstFolder
    m.frameCapture.ui = False
    if not os.path.exists(dstFolder):
        os.makedirs(dstFolder)

    m.clock.framerate = 30

    for i in range(numAnimationFrames):
        for j in range(samplesPerFrame):
            m.renderFrame()
        m.frameCapture.baseFilename = '{}.{}'.format(dstImage, i)
        m.frameCapture.capture()
        m.clock.step()

# Full ReSTIR BDPT, 1spp
graph.updatePass("ReSTIRBDPT", {
    "useBPT":               True,
    "numLightSubpaths":     1920 * 1080,
    "enableResampling":     True,
    "spatialResamplingPasses": 5,
    "Mcap":                 20,
    "useCausticReservoirs": True,
    "useReconnectionMis":   True })
graph.updatePass("AccumulatePass", { "enabled": False })
Render(
    'D:/3d/pbrtv4/veach-bidir/VeachBidir.pyscene',
    os.path.join(folder, 'VeachBidir'),
    'VeachBidir')

exit()
