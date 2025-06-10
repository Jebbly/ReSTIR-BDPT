from falcor import *
from Experiments.ExperimentBase import *

dstName = "Staircase"
runs = 5
equalTimePtCandidates=7
equalTimeBdptCandidates=6

LoadScene(m, "D:/3d/pbrtv4/staircase/scene-v4.pbrt")

experiment = ExperimentBase(m)

#experiment.Run(Experiment.REFERENCE, dstName)

#experiment.Run(EXPERIMENT_ALL_OFFLINE, dstName)

#experiment.Run(EXPERIMENT_ALL_RESTIR_BDPT,      dstName, runs=runs)
#experiment.Run(Experiment.BDPT_EQUAL_TIME,      dstName, runs=runs, candidates=equalTimeBdptCandidates)
#experiment.Run(Experiment.RESTIR_PT_EQUAL_TIME, dstName, runs=runs, candidates=equalTimePtCandidates)

experiment.Run(Experiment.RESTIR_BDPT_OFFLINE_NO_RCV_MIS, dstName, runs=runs)

exit()
