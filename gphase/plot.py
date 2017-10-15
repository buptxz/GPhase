'''
Plot all samples into a folder
'''

import numpy as np
import matplotlib.pyplot as plt

xrd = np.genfromtxt('../data/FePdGa_XRD.csv', delimiter=',')
composition = np.genfromtxt('../data/FePdGa_composition.csv', delimiter=',')
two_theta = xrd[0, :]
xrd = xrd[1:, :]
label = composition[:, 3]
sample_number = len(xrd)
feature_number = len(xrd[0])

f = plt.figure()
plt.plot(range(10), range(10), "o")
plt.xlabel(r'$2\theta\,Angle (Deg.)$', fontsize=20)
plt.ylabel(r'$Intensity$', fontsize=20)
plt.plot(two_theta, xrd[37], color='b')
plt.xlim(two_theta[0], two_theta[-1])
# plt.ylim(500, 800)
f.show()
f.savefig("../figure/figure4(a).pdf")

# # plot all samples and save
# for sample_index in range(sample_number):
#     fig = plt.figure()
#     plt.title('sample ' + str(sample_index + 1) + ' category ' + str(int(label[sample_index])))
#     plt.plot(two_theta, xrd[sample_index])
#     # plt.ylim([500, 800])
#     plt.savefig("../figure/fegapd/" + str(sample_index + 1) + '.png', format="png")

# # plot std curve
# std = np.std(xrd, axis=0)
# plt.plot(two_theta[1:-2], std[1:-2])
# plt.savefig("../figure/alcumo/std.jpg")
