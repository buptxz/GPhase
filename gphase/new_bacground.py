import matplotlib.pyplot as plt
import matplotlib.colors
import phase_module
import argparse
# import matlab.engine
import os
from argparse import RawTextHelpFormatter
import numpy as np
import peakdetect
import warnings
import copy
import math

# def back_sub(data):

if __name__ == "__main__":
    xrd = np.genfromtxt('../data/xrd.csv', delimiter=',')
    composition = np.genfromtxt('../data/composition.csv', delimiter=',')
    two_theta = xrd[0, :]
    xrd = xrd[1:, :]
    label = composition[:, 3]
    sample_number = len(xrd)
    feature_number = len(xrd[0])

    # std = np.std(xrd, axis=0)
    # std = np.std(xrd.tolist(), axis=0)
    # std = xrd[10]
    # std = std.tolist()
    # plt.plot(two_theta[1:-2], std[1:-2])
    # plt.show()
    # std = phase_module.back_sub(std, neighbor=2, threshold=0.5, fitting_degree=50, if_plot=0, two_theta=two_theta)
    #
    # for i in range(feature_number):
    #     if std[i] < 0:
    #         std[i] = 0
    # # _max, _min = peakdetect.peakdetect(std, range(feature_number), lookahead=10, delta=0.35)
    # plt.plot(two_theta[1:-2], std[1:-2])
    # plt.show()

    # plt.plot(two_theta, xrd[164])
    # plt.show()
    xrd_peak = copy.deepcopy(xrd)
    prediction = []
    # manually labeled two points
    start = 428
    end = 464
    xrd_peak = xrd_peak[:, start:end]
    x = two_theta[start:end]
    for sample in xrd_peak:
        # for index in range(feature_number):
        #     # if (two_theta[index] < 3.09215 or two_theta[index] > 5.61117 or (two_theta[index] > 3.2226 and two_theta[index] < 5.52592)):
        #     if (index < start or index > end):
        #         sample[index] = 0
        y = sample

        curvature = []
        for i in range(1, len(y) - 1):
            x1 = x[i - 1]
            x2 = x[i]
            x3 = x[i + 1]
            y1 = y[i - 1]
            y2 = y[i]
            y3 = y[i + 1]
            K = 2 * ((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)) / math.sqrt(
                ((x2 - x1) ** 2 + (y2 - y1) ** 2) * ((x3 - x2) ** 2 + (y3 - y2) ** 2) * (
                (x1 - x3) ** 2 + (y1 - y3) ** 2))
            curvature.append(K)
        curvature = [abs(temp) for temp in curvature]
        max_value = max(curvature)
        if (max_value > 0.001):
            prediction.append(1)
        else:
            prediction.append(0)

        # base = 0
        # result = -1
        # sign = False
        # if sample[start] > sample[end]:
        #     base = sample[start]
        #     for index in range(start, end + 1):
        #         if (sample[index] < base):
        #             if index - start >= (end - start) * 0.5:
        #                 result = 0
        #             else:
        #                 result = 1
        #             break
        #
        # else:
        #     base = sample[end]
        #     for index in range(end, start - 1, -1):
        #         if (sample[index] < base):
        #             if end - index >= (end - start) * 0.5:
        #                 result = 0
        #             else:
        #                 result = 1
        #             break
        # if (result != -1):
        #     prediction.append(result)
        # else:
        #     prediction.append(1)

    print(prediction)
    phase_module.result_evaluation(label, prediction)

    # plot all samples
    # for sample in range(sample_number):
    #     # plt.figure(figsize = (9,6), dpi = 300)
    #     plt.subplot(211)
    #     plt.title('sample ' + str(sample + 1) + ' category ' + str(label[sample]))
    #     # this is the background samples
    #     if sample in [184, 207, 208, 231, 255, 278, 301, 324]:
    #         plt.axis([two_theta[0], two_theta[feature_number - 1], 0, 1000])
    #     else:
    #         plt.axis([two_theta[0], two_theta[feature_number - 1], 450, 2400])
    #     plt.plot(two_theta, xrd[sample])
    #     plt.subplot(212)
    #     # plt.xlim(xmax = two_theta[feature_number - 1], xmin = two_theta[0])
    #     # plt.plot(two_theta[440:464], xrd[sample][440:464])
    #     plt.plot(two_theta[start:end], xrd_peak[sample][start:end])
    #     plt.title('sample ' + str(sample + 1) + ' category ' + str(prediction[sample]))
    #     plt.savefig("/home/zheng/Desktop/figure0208/" + str(sample + 1) + '.png', format="png", dpi=200)
    #     plt.close()

    # std = back_sub(std, neighbor=2, threshold=0.5, fitting_degree=50, if_plot=0, two_theta=two_theta)