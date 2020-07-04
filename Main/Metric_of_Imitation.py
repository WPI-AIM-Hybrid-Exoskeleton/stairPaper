import numpy as np

from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer
from GaitAnaylsisToolkit.Session import ViconGaitingTrial
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner
from GaitCore.Core import Point
import matplotlib.pyplot as plt
from dtw import dtw
import numpy.polynomial.polynomial as poly


frames = {}
frames["stairA"] = [Point.Point(0, 0, 0),
                    Point.Point(63, 0, 0),
                    Point.Point(0, 42, 0),
                    Point.Point(63, 49, 0)]

frames["stairB"] = [Point.Point(0, 0, 0),
                    Point.Point(49, 0, 0),
                    Point.Point(28, 56, 0),
                    Point.Point(70, 70, 0)]

file13 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_stair_config1_00.csv"
file12 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_01/subject_01 stairconfig1_02.csv"
file11 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig2_00.csv"
file10 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 stairclimbing_config1_01.csv"
file09 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_09/subject_09 stairclimbing_config1_00.csv"
file07 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 stairclimbing_config1_00.csv"
file06 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 stairclimbing_config1_02.csv"
file05 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_stair_config1_01.csv"
file03 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_stair_config0_02.csv"
file02 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_stair_config1_01.csv"
file01 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_01/subject_01 stairconfig1_03.csv"
file00 = "/home/nathanielgoldfarb/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig1_00.csv"


def calculate_imitation_metric_1(demos, imitation):
    M = len(demos)
    T = len(imitation)
    imitation = np.array(imitation)
    metric = 0.0
    paths = []
    t = []
    t.append(1.0)
    alpha = 1.0
    manhattan_distance = lambda x, y: abs(x - y)
    for i in range(1, T):
        t.append(t[i - 1] - alpha * t[i - 1] * 0.01)  # Update of decay term (ds/dt=-alpha s) )
    t = np.array(t)

    for m in range(M):
        d, cost_matrix, acc_cost_matrix, path = dtw(imitation, demos[m], dist=manhattan_distance)
        data_warp = [demos[m][path[1]][:imitation.shape[0]]]
        coefs = poly.polyfit(t, data_warp[0], 20)
        ffit = poly.Polynomial(coefs)
        y_fit = ffit(t)
        paths.append(y_fit)
        metric += np.sum(abs(y_fit - imitation.flatten()))

    return paths, metric/(M*T)


def make_toeY(files, hills):

    paths = []

    for hill, file in zip(hills, files):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        marker = trial.vicon.get_markers()
        markers = trial.vicon.get_markers()
        markers.smart_sort()
        markers.auto_make_transform(frames)
        toe = marker.get_marker("RTOE")
        stair = marker.get_frame("stairA")
        joint = []
        for t in toe:
            joint.append(abs(t.y ))
        arr = [joint[h[0]] for h in hill]
        paths.append(np.array(arr))

    return paths


def make_toeZ(files, hills):

    paths = []

    for hill, file in zip(hills, files):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        marker = trial.vicon.get_markers()
        toe = marker.get_marker("RTOE")
        joint = []
        for t in toe:
            joint.append(t.z)
        arr = [joint[h[0]] for h in hill]
        paths.append(np.array(arr))

    return paths


def get_index(files):

    paths = []
    for file in files:
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        markers = trial.vicon.get_markers()
        markers.smart_sort()
        markers.auto_make_transform(frames)
        hills = trial.get_stairs("RTOE", "stairA")
        paths.append(hills[0])

    return paths


nb_states = 10
filesZ = [ file00, file01, file02, file03, file05, file06, file07, file09]
filesY = [ file00, file01, file03, file11, file12, file13]

hillsZ = get_index(filesZ)
hillsY = get_index(filesY)

pathsZ = make_toeZ(filesZ, hillsZ)
pathsY = make_toeY(filesY, hillsY)

#runner_toeZ = make_toeZ(filesZ, hillsZ, nb_states, "toeZ_all2")
#runner_toeY = make_toeY(filesY, hillsY, nb_states, "toeY_all")

# runnerY = GMMRunner.GMMRunner("toeY_all" + ".pickle")
# runnerZ = GMMRunner.GMMRunner("toeZ_all1" + ".pickle")
#
# imitationY = runnerY.run()
# imitationZ = runnerZ.run()
metricsZ = []
metricsY = []
for i in range(2,25):
    runnerZ = GMMRunner.GMMRunner("toeZ_all" + str(i)  + ".pickle")
    imitationZ = runnerZ.run()
    pathZ, metric = calculate_imitation_metric_1(pathsZ, imitationZ)
    metricsZ.append( [i, metric])

for i in range(2,25):
    runnerY = GMMRunner.GMMRunner("toeY_all" + str(i)  + ".pickle")
    imitationY = runnerY.run()
    pathY, metric = calculate_imitation_metric_1(pathsY, imitationY)
    metricsY.append( [i, metric])

plt.rcParams.update({'font.size': 22})
fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 3))

metricsY = np.array(metricsY)
metricsZ = np.array(metricsZ)

ax1[0].plot(metricsY[:, 0], metricsY[:, 1], "-o")
ax1[1].plot(metricsZ[:, 0], metricsZ[:, 1], "-o" )
ax1[0].set_title("Y Imitation cost")
ax1[1].set_title("Z Imitation cost")
fig1.suptitle("Imitation cost")
ax1[1].set_xlabel("K")
ax1[0].set_ylabel("cost")
ax1[1].set_ylabel("cost")

for p in pathY:
    ax2[0].plot(p)

for p in pathZ:
    ax2[1].plot(p)

fig2.suptitle("Imitation Comparison")
ax2[0].plot(imitationY, linewidth=4)
ax2[1].plot(imitationZ, linewidth=4)

ax2[1].set_title("Z Trajectories")
ax2[0].set_title("Y Trajectories")

ax2[1].legend(["file00", "file01", "file02", "file03", "file05", "file06", "file07", "file09", "Learned Model"])
ax2[0].legend([ "file00", "file01", "file03", "file11", "file12", "Learned Model"])

ax2[0].set_xlabel("s")
ax2[0].set_ylabel("pos (mm)")
ax2[1].set_xlabel("s")
ax2[1].set_ylabel("pos (mm)")

plt.show()