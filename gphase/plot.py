'''
Plot all samples into a folder
'''

import numpy as np
import matplotlib.pyplot as plt

xrd = np.genfromtxt('../data/AlCuMo_XRD_long300.csv', delimiter=',')
composition = np.genfromtxt('../data/AlCuMo_composition_long300.csv', delimiter=',')
two_theta = xrd[0, :]
xrd = xrd[1:, :]
label = composition[:, 3]
sample_number = len(xrd)
feature_number = len(xrd[0])

# plot all samples and save
for sample_index in range(sample_number):
    fig = plt.figure()
    plt.title('sample ' + str(sample_index + 1) + ' category ' + str(int(label[sample_index])))
    plt.plot(two_theta, xrd[sample_index])
    plt.ylim([500, 800])
    plt.savefig("/home/zheng/Desktop/alcumolong300/" + str(sample_index + 1) + '.png', format="png")

# plot std curve
std = np.std(xrd, axis=0)
plt.plot(two_theta[1:-2], std[1:-2])
plt.savefig("/home/zheng/Desktop/alcumolong300/std.jpg")
