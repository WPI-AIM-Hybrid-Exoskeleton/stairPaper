import numpy as np
import matplotlib.pyplot as plt
from utilities import data, utilities
from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.signal import resample
import matplotlib


def plot_gmm(Mu, Sigma, ax=None):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    X = []
    nb_state = len(Mu[0])
    patches = []

    for i in range(nb_state):
        w, v = np.linalg.eig(Sigma[i])
        R = np.real(v.dot(np.lib.scimath.sqrt(np.diag(w))))
        x = R.dot(np.array([np.cos(t), np.sin(t)])) + np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, nbDrawingSeg)
        x = x.transpose().tolist()
        patches.append(Polygon(x, edgecolor='r'))
        ax.plot(Mu[0, i], Mu[1, i], 'r*')

    p = PatchCollection(patches, edgecolor='k', cmap=matplotlib.cm.jet, alpha=0.8)

    ax.add_collection(p)

    return p

nb_states = 15
files = data.files
sides = data.sides
frames = data.frames
hills = utilities.get_index(frames, files, sides)
pathsZ, pathsY = utilities.make_toe(files, hills, sides)

# trainerZ1 = GMMTrainer.GMMTrainer(pathsZ, "trainZ", 15, 0.01)
# trainerZ1.train()
runnerZ1 = GMMRunner.GMMRunner("trainZ.pickle")
#
# trainerY1 = GMMTrainer.GMMTrainer(pathsY, "trainY", 15, 0.01)
# trainerY1.train()
runnerY1 = GMMRunner.GMMRunner("trainY.pickle")

nb_states = 15
files = data.files[0:1]
sides = data.sides[0:1]
frames = data.frames
hills = utilities.get_index(frames, files, sides)
pathsZ, pathsY = utilities.make_toe(files, hills, sides)

# trainerZ2 = GMMTrainer.GMMTrainer(pathsZ, "trainZ_single", 15, 0.01)
# trainerZ2.train()
runnerZ2 = GMMRunner.GMMRunner("trainZ_single.pickle")

# trainerY2 = GMMTrainer.GMMTrainer(pathsY, "trainY_single", 15, 0.01)
# trainerY2.train()
runnerY2 = GMMRunner.GMMRunner("trainY_single.pickle")




sample_size = 10000
for p in pathsZ:
    if len(p) < sample_size:
        sample_size = len(p)

fig0, ax0 = plt.subplots(1)
fig1, ax1 = plt.subplots(1)
sIn = runnerZ1.get_sIn()

runnerY2.update_start(int(round(pathsY[0][0])))
runnerZ2.update_start(int(round(pathsZ[0][0])))
runnerY2.update_goal(int(round(pathsY[0][-1])))
runnerZ2.update_goal(int(round(pathsZ[0][-1])))

runnerY1.update_start(int(round(pathsY[0][0])))
runnerZ1.update_start(int(round(pathsZ[0][0])))
runnerY1.update_goal(int(round(pathsY[0][-1])))
runnerZ1.update_goal(int(round(pathsZ[0][-1])))


pathX2 = runnerY2.run()
pathZ2 = runnerZ2.run()

pathX1 = runnerY1.run()
pathZ1 = runnerZ1.run()

ax0.plot(pathX2)
ax1.plot(pathZ2)
ax0.plot(pathX1)
ax1.plot(pathZ1)


plt.show()



