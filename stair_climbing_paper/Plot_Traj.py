import numpy as np
import matplotlib.pyplot as plt
from utilities import data, utilities
import matplotlib


nb_states = 10
files = data.files
leg = data.leg
sides = data.sides
frames = data.frames
hills = utilities.get_index(frames, files, sides)

pathsZ, pathsY, pathsX = utilities.make_toe(files, hills, sides)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
for pz, py, px in zip(pathsZ, pathsY, pathsX):
    ax1.plot(pz)
    ax2.plot(py)
    ax3.plot(px)



f.suptitle("Toe marker trajectories during stair ascent")
ax1.set_title("Z Position")
ax2.set_title("Y Position")
ax3.set_title("X Position")
ax1.set_ylabel("Position (mm)")
ax2.set_ylabel("Position (mm)")
ax3.set_ylabel("Position (mm)")
ax3.set_xlabel("Frames")
#ax2.legend(leg)
plt.show()