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

    # hand mark start and end point, inclusive
    start = 438
    end = 466

    std = np.std(xrd, axis=0)
    # plot standard deviation curve
    # plt.plot(two_theta[1:-2], std[1:-2])
    # plt.scatter(two_theta[start], std[start])
    # plt.scatter(two_theta[end], std[end])
    # plt.savefig("../figure/deviation.png", dpi=500)
    # plt.show()
    # std = phase_module.back_sub(std, neighbor=2, threshold=0.5, fitting_degree=50, if_plot=0, two_theta=two_theta)
    #
    # for i in range(feature_number):
    #     if std[i] < 0:
    #         std[i] = 0
    # # _max, _min = peakdetect.peakdetect(std, range(feature_number), lookahead=10, delta=0.35)
    # plt.plot(two_theta[1:-2], std[1:-2])
    # plt.show()

    xrd_peak = []
    prediction = []
    curvature = []
    window = []
    # manually labeled two points
    for sample in xrd:
        i = start
        j = end
        if sample[i] > sample[j]:
            while sample[i] > sample[j]:
                i -= 1
                j -= 1
            if abs(sample[i + 1] - sample[j + 1]) < abs(sample[i] - sample[j]):
                i += 1
                j += 1
            xrd_peak.append(sample[i:j+1])
        else:
            while sample[i] < sample[j]:
                i += 1
                j +=1
            if abs(sample[i - 1] - sample[j - 1]) < abs(sample[i] - sample[j]):
                i -= 1
                j -= 1
            xrd_peak.append(sample[i:j+1])
        window.append([i, j])

    x = two_theta[start:end+1]
    for sample_number in range(len(xrd_peak)):
        y = xrd_peak[sample_number]
        middle = (end - start) / 2
        half_offset = (end - start) / 4
        peak_location = np.argmax(y[half_offset : middle + half_offset]) + half_offset
        window[sample_number].append(window[sample_number][0] + peak_location - half_offset)
        window[sample_number].append(window[sample_number][0] + peak_location)
        window[sample_number].append(window[sample_number][0] + peak_location + half_offset)
        if (sample_number == 271 or sample_number == 366):
            peak_location = (end - start) / 2
        x1 = x[peak_location - half_offset]
        x2 = x[peak_location]
        x3 = x[peak_location + half_offset]
        y1 = y[peak_location - half_offset]
        y2 = y[peak_location]
        y3 = y[peak_location + half_offset]
        K = 2 * ((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)) / math.sqrt(
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) * ((x3 - x2) ** 2 + (y3 - y2) ** 2) * (
            (x1 - x3) ** 2 + (y1 - y3) ** 2))
        if (abs(K) < 0.0005):
            curvature.append(K)
            prediction.append(0)
        else:
            curvature.append(K)
            prediction.append(1)
    # print(prediction)
    # background samples
    for sample in [184, 207, 208, 231, 232, 255, 278, 301, 324]:
        prediction[sample] = 2
    phase_module.result_evaluation(label, prediction)

    # plot all samples
    # for sample in range(sample_number):
    #     plt.figure(figsize = (9,6), dpi = 300)
    #     plt.subplot(211)
    #     plt.title('sample ' + str(sample + 1) + ' category ' + str(label[sample]))
    #     # this is the background samples
    #     if sample in [184, 207, 208, 231, 232, 255, 278, 301, 324]:
    #         plt.axis([two_theta[0], two_theta[feature_number - 1], 0, 1000])
    #     else:
    #         plt.axis([two_theta[0], two_theta[feature_number - 1], 450, 2400])
    #     plt.plot(two_theta, xrd[sample])
    #     plt.scatter([two_theta[window[sample][0]], two_theta[window[sample][1]]], [xrd[sample][window[sample][0]], xrd[sample][window[sample][1]]])
    #     plt.subplot(212)
    #     # plt.xlim(xmax = two_theta[feature_number - 1], xmin = two_theta[0])
    #     plt.plot(two_theta[window[sample][0]:window[sample][1]+1], xrd_peak[sample])
    #     plt.scatter([two_theta[window[sample][2]], two_theta[window[sample][3]], two_theta[window[sample][4]]], [xrd[sample][window[sample][2]], xrd[sample][window[sample][3]], xrd[sample][window[sample][4]]])
    #     plt.title('sample ' + str(sample + 1) + ' category ' + str(prediction[sample]) + " k " + str(curvature[sample]))
    #     plt.savefig("/home/zheng/Desktop/figure0216/" + str(sample + 1) + '.png', format="png", dpi=200)
    #     # plt.savefig("D:/xiong/Desktop/figure0215/" + str(sample + 1) + '.png', format="png", dpi=200)
    #     plt.close()
    # std = back_sub(std, neighbor=2, threshold=0.5, fitting_degree=50, if_plot=0, two_theta=two_theta)

    # plot for deep learning input
    fig = plt.figure(figsize=(10, 10), dpi=100)
    for sample in range(sample):
        fig.set_xticks([])
        fig.set_yticks([])
        fig = plt.figure(figsize=(10,10),dpi=100)
        plt.xlim(xmax=two_theta[window[sample]][1], xmin=two_theta[window[sample]][0])
        plt.plot(two_theta[window[sample][0]:window[sample][1] + 1], xrd_peak[sample])
        plt.savefig("/home/zheng/Desktop/figure0220/" + str(sample + 1) + '.png', format="png", dpi=200)
        plt.close()