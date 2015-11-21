import numpy as np
import sys
import matplotlib.pyplot as plt

def read_data(filename):
    x = []
    y = []
    w = []
    with open(filename) as f:
        for line in f:
            vals = [float(v) for v in line.split()]
            if vals[4] > 1e-3:
                x.append(vals[0])
                y.append(vals[3])
                w.append(vals[4])
    x = np.around(x, decimals=5)
    y = np.around(y, decimals=5)
    w = np.array(w)

    points = {}
    for e_x, e_y, e_z in zip(x, y, w):
        if (e_x, e_y) in points:
            points[(e_x, e_y)] += e_z
        else:
            points[(e_x, e_y)] = e_z

    x = []
    y = []
    w = []
    for key in points:
        x.append(key[0])
        y.append(key[1])
        w.append(points[key])

    return np.array(x), np.array(y), np.array(w)

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "unfolding.dat"

x, y, z = read_data(filename)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
sc = plt.scatter(x, y, c=z, s=30, vmin=0, edgecolor='', cmap='Greys')
plt.colorbar(sc)
plt.show()
