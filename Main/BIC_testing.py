from lib.GaitAnalysisToolkit.lib.GaitCore.Core import Point

from lib.GaitAnalysisToolkit.LearningTools.Trainer import GMMTrainer
from lib.GaitAnalysisToolkit.Session import ViconGaitingTrial
import matplotlib.pyplot as plt
import numpy as np


frames = {}

frames["stairA"] = [Point.Point(0, 0, 0),
                    Point.Point(63, 0, 0),
                    Point.Point(0, 42, 0),
                    Point.Point(63, 49, 0)]

frames["stairB"] = [Point.Point(0, 0, 0),
                    Point.Point(49, 0, 0),
                    Point.Point(28, 56, 0),
                    Point.Point(70, 70, 0)]

#
# file1 = "/media/nathaniel/Data01/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_01.csv"
# file2 = "/media/nathaniel/Data01/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_00.csv"

#file1 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_stair_config1_00.csv"
# file2 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_stair_config1_01.csv"

# file1 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_stair_config2_00.csv"
# file2 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_stair_config2_01.csv"


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


#file_list = [ file1, file2]
file_list = [file13, file12, file11,file10,file09,file07,file06,file05,file03,file02,file01,file00]

def get_index(files):

    frames = {}

    frames["stairA"] = [Point.Point(0, 0, 0),
                        Point.Point(63, 0, 0),
                        Point.Point(0, 42, 0),
                        Point.Point(63, 49, 0)]

    frames["stairB"] = [Point.Point(0, 0, 0),
                        Point.Point(49, 0, 0),
                        Point.Point(28, 56, 0),
                        Point.Point(70, 70, 0)]

    paths = []
    for file in files:
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        markers = trial.vicon.get_markers()
        markers.smart_sort()
        markers.auto_make_transform(frames)
        hills = trial.get_stairs("LTOE", "stairA")
        paths.append(hills[0])

    return paths

def make_toe(hills, files, name):

    output = []
    paths = []

    for hill, file in zip(hills, files):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        marker = trial.vicon.get_markers()
        toe = marker.get_marker("RTOE")
        joint = []
        for t in toe:
            joint.append(t.y)
        arr = [joint[h[0]] for h in hill]
        paths.append(np.array(arr))

    for i in xrange(2,25):
        trainer = GMMTrainer.GMMTrainer(paths, name, i, 0.01, 12)
        bic = trainer.train(save=False)
        output.append([i, bic])
    return output



plt.rcParams.update({'font.size': 22})

nb_states = 10
hills = get_index(file_list)
min_values = []
min_scores = []
all_x = []
all_y = []
for i in xrange(1):
    output = make_toe(hills, file_list, "toe")
    x = []
    y = []
    for o in output:
        x.append(o[0])
        y.append(o[1])

    minY = min(y)
    min_values.append(x[y.index(minY)])
    all_x.append(x)
    all_y.append(y)
    plt.plot(x, y)


plt.ylabel("BIC_score")
plt.xlabel("K")
plt.title("BIC score (smaller is better)")
x = np.mean(all_x, axis=0).tolist()
y = np.mean(all_y, axis=0).tolist()

minY = min(y)
print "Score ", minY
print min_values
print 'K ', x[y.index(minY)]
plt.plot(x, y, c="k", linewidth=5)
plt.show()
