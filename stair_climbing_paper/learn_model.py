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

trainerZ = GMMTrainer.GMMTrainer(pathsZ, "trainZ", 15, 0.01)
trainerZ.train()
runnerZ = GMMRunner.GMMRunner("trainZ.pickle")


trainerY = GMMTrainer.GMMTrainer(pathsY, "trainY", 15, 0.01)
trainerY.train()
runnerY = GMMRunner.GMMRunner("trainY.pickle")


sample_size = 10000
for p in pathsZ:
    if len(p) < sample_size:
        sample_size = len(p)

fig0, ax0 = plt.subplots(1)
fig1, ax1 = plt.subplots(1)
sIn = runnerZ.get_sIn()
for i in range(len(trainerY.data["demos"])):
    ax1.plot(sIn, np.flip(trainerY.data["demos"][i]))
    ax0.plot(sIn, np.flip(trainerZ.data["demos"][i]))


path = runnerZ.run()
ax0.plot(sIn, np.flip(path), "k", linewidth=5)
path = runnerY.run()
ax1.plot(sIn, np.flip(path), "k", linewidth=5)

plt.show()



