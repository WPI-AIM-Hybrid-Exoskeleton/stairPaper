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



nb_states = 10
files = data.files
sides = data.sides
frames = data.frames
hills = utilities.get_index(frames, files, sides)
pathsZ, pathsY = utilities.make_toe(files, hills, sides)

trainer = GMMTrainer.GMMTrainer(pathsZ, "plotGMM", 10, 0.01)
trainer.train()
runner = GMMRunner.GMMRunner("plotGMM.pickle")

fig0, ax0 = plt.subplots(1)
fig1, ax1 = plt.subplots(1)

sIn = runner.get_sIn()
tau = runner.get_tau()
l = runner.get_length()
motion = runner.get_motion()
currF = runner.get_expData()[0].tolist()

# plot the forcing functions
p = plot_gmm(Mu=runner.get_mu()[:2, :], Sigma=runner.get_sigma()[:, :2, :2], ax=ax0)
for i in range(len(files)):
    ax0.plot(sIn, tau[1, i * l: (i + 1) * l].tolist(), color="b")
    ax1.plot(sIn, np.flip(motion[i]), 'b')

ax0.plot(sIn, currF, color="r", linewidth=2)
ax0.set_xlabel('S')
ax0.set_ylabel('F')
ax0.set_title("Forcing Function")

path = runner.run()
# ax1.plot(sIn, np.flip(motion[0]), 'b')
# ax1.plot(sIn, np.flip(motion[1]), 'b')
ax1.plot(sIn, np.flip(path), "k")
ax1.set_xlabel('s')
ax1.set_ylabel('angle')
ax1.set_title("Learned Trajectory")
ax1.legend(["EX1", "EX2", "Replicated"])

plt.show()



