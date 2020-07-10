
import numpy as np
from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer
from GaitAnaylsisToolkit.Session import ViconGaitingTrial
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner
from GaitCore.Core import Point
import matplotlib.pyplot as plt
from dtw import dtw
import numpy.polynomial.polynomial as poly

file = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig1_00.csv"

trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
markers = trial.vicon.get_markers()
markers.smart_sort()
markers.play()