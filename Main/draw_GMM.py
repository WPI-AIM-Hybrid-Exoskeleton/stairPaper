import numpy as np
import matplotlib.pyplot as plt
from utilities import data, utilities
from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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


file1 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_stair_config1_00.csv"
file2 = "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_stair_config1_01.csv"
files = [file1, file2]
nb_states = 10
sides = ["R", "R"]
frames = data.frames
hills = utilities.get_index(frames, files, sides)
pathsZ, pathsY = utilities.make_toe(files, hills, sides)

trainer = GMMTrainer.GMMTrainer(pathsZ, "plotGMM", 10, 0.01)
trainer.train()
runner = GMMRunner.GMMRunner("plotGMM.pickle")
fig, ax = plt.subplots(2)

sIn = runner.get_sIn()
tau = runner.get_tau()
l = runner.get_length()
motion = runner.get_motion()
currF = runner.get_expData()[0].tolist()

# plot the forcing functions
p = plot_gmm(Mu=runner.get_mu()[:2, :], Sigma=runner.get_sigma()[:, :2, :2], ax=ax[0])
for i in range(2):
    ax[0].plot(sIn, tau[1, i * l: (i + 1) * l].tolist(), color="b")

ax[0].plot(sIn, currF, color="r", linewidth=2)
ax[0].set_xlabel('S')
ax[0].set_ylabel('F')
ax[0].set_title("Forcing Function")

ax[1].plot(sIn, np.flip(motion[0]), 'b')
ax[1].plot(sIn, np.flip(motion[1]), 'b')
#
# path = runner.run()
# ax[1].plot(sIn, np.flip(path),"r")

path = runner.run()
ax[1].plot(sIn, np.flip(path), "k")

ax[1].set_xlabel('s')
ax[1].set_ylabel('angle')
ax[1].set_title("Learned Trajectory")
ax[1].legend(["EX1", "EX2", "Replicated"])
plt.show()
