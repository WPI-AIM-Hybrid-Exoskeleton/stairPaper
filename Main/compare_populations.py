import numpy as np
import matplotlib.pyplot as plt
from utilities import data, utilities
from dtw import dtw
from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.signal import resample
import matplotlib
import numpy.polynomial.polynomial as poly

def calculate_imitation_metric_1(demos, imitation):
    M = len(demos)
    T = len(imitation)
    imitation = np.array(imitation)
    metric = 0.0
    paths = []
    t = []
    t.append(1.0)
    alpha = 1.0
    manhattan_distance = lambda x, y: np.abs(x - y)
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
        metric += np.sum(np.sqrt( np.power(y_fit - imitation.flatten(), 2)))


    return paths, metric/(M*T)


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

pathY2 = runnerY2.run()
pathZ2 = runnerZ2.run()

pathY1 = runnerY1.run()
pathZ1 = runnerZ1.run()

repoduction_pathY1, metricY1 = calculate_imitation_metric_1(np.array([pathsY[0]]), pathY1 )
repoduction_pathY2, metricY2 = calculate_imitation_metric_1(np.array([pathsY[0]]), pathY2 )

repoduction_pathZ1, metricZ1 = calculate_imitation_metric_1(np.array([pathsZ[0]]), pathZ1 )
repoduction_pathZ2, metricZ2 = calculate_imitation_metric_1(np.array([pathsZ[0]]), pathZ2 )

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

ax0.plot(repoduction_pathZ2[0] )
ax1.plot(repoduction_pathY2[0])

ax0.plot(pathZ1)
#ax0.plot(pathZ2)

ax1.plot(pathY1)
#ax1.plot(pathY2)


plt.show()



