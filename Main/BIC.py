
import numpy as np
import matplotlib.pyplot as plt
from utilities import data, utilities
from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer

file13 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_stair_config1_00.csv"
file12 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_01/subject_01 stairconfig1_02.csv"
file11 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig2_00.csv"
file10 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 stairclimbing_config1_01.csv"
file09 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_09/subject_09 stairclimbing_config1_00.csv"
file07 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 stairclimbing_config1_00.csv"
file06 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 stairclimbing_config1_02.csv"
file05 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_stair_config1_01.csv"
file03 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_stair_config0_02.csv"
file02 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_stair_config1_01.csv"
file01 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_01/subject_01 stairconfig1_03.csv"
file00 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 stairconfig1_00.csv"

file_list = [file13, file12, file11,file10,file09,file07,file06,file05,file03,file02,file01,file00]

files = data.files
sides = data.sides
frames = data.frames
hills = utilities.get_index(frames, files, sides)
pathsZ, pathsY = utilities.make_toe(files, hills, sides)

output = []
min_values = []
min_scores = []
all_x = []
all_y = []


for i in range(2,25):
    trainer = GMMTrainer.GMMTrainer(pathsZ, "name", i, 0.01)
    bic = trainer.train(save=False)
    output.append([i, bic])

x = []
y = []
for o in output:
    x.append(o[0])
    y.append(o[1]["BIC"])

all_x.append(x)
all_y.append(y)
plt.plot(x, y)


plt.ylabel("BIC_score")
plt.xlabel("K")
plt.title("BIC score (smaller is better)")
x = np.mean(all_x, axis=0).tolist()
y = np.mean(all_y, axis=0).tolist()


plt.plot(x, y, c="k", linewidth=5)
plt.show()
