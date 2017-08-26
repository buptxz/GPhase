'''
Uses curvature to do classification.
'''

import matplotlib.pyplot as plt
from phase_module import back_sub, result_evaluation
import numpy as np
import math
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, precision_recall_curve
from scipy.misc import imread, imresize

# read data set
xrd = np.genfromtxt('../data/xrd_new.csv', delimiter=',')
composition = np.genfromtxt('../data/composition_new.csv', delimiter=',')
two_theta = xrd[0, :]
xrd = xrd[1:, :]
label = composition[:, 3]
sample_number = len(xrd)
feature_number = len(xrd[0])

# calculate standard deviation
std = np.std(xrd, axis=0)

# plot standard deviation curve
plt.plot(two_theta[50:-2], std[50:-2])
plt.title("standard deviation figure")
# plt.scatter(two_theta[start], std[start])
# plt.scatter(two_theta[end], std[end])
# plt.savefig("../figure/deviation.png", dpi=500)
plt.show()

# Background subtraction by curve fitting
std = back_sub(std, neighbor=2, threshold=0.9, fitting_degree=50, if_plot=0, two_theta=two_theta)

# Make value less than 0 as 0
for i in range(feature_number):
    if std[i] < 0:
        std[i] = 0

# Looking for the foot of max peak
max_value = max(std[50:-2])
max_index = list(std).index(max_value)
start = max_index
end = max_index
while std[start] != 0:
    start -= 1
while std[end] != 0:
    end += 1
print("left foot location: ", start)
print("right foot location: ", end)

# plot the curve after removing background
plt.title("curve after removing background")
plt.plot(two_theta[2:-2], std[2:-2])
plt.scatter(two_theta[start], std[start])
plt.scatter(two_theta[end], std[end])
plt.show()

# Divide the samples into a "binary set" and "others"
cnn_set = []
other_set = []
for sample_index in range(sample_number):
    if label[sample_index] == 0 or label[sample_index] == 1:
        cnn_set.append(sample_index)
    else:
        other_set.append(sample_index)

prediction = np.empty(sample_number)
curvature = np.empty(sample_number)
window = []
xrd_peak = []

# Adjust the window location to the center of the peak
for sample_index in range(sample_number):
    sample = xrd[sample_index]
    i = start
    j = end

    max_index = np.argmax(sample[i:j+1]) + i
    if max_index - i > 5 and j - max_index > 5:
        xrd_peak.append(sample[max_index - (j - i) // 2 : max_index + (j - i) // 2 + 1])
        window.append([max_index - (j - i) // 2, max_index + (j - i) // 2])
    else :
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
                j += 1
            if abs(sample[i - 1] - sample[j - 1]) < abs(sample[i] - sample[j]):
                i -= 1
                j -= 1
            xrd_peak.append(sample[i:j+1])
        if i >= 0 and j >= 0 and i < feature_number and j < feature_number:
            window.append([i, j])
        else:
            window.append([i, j])

threshold = 0.0005
x = two_theta[start:end+1]
for sample_index in range(sample_number):
    if sample_index in other_set:
        prediction[sample_index] = label[sample_index]
        curvature[sample_index] = 0
    else:
        y = xrd_peak[sample_index]
        middle = (int) ((end - start) / 2)
        half_offset = (int) ((end - start) / 4)
        peak_location = np.argmax(y[half_offset : (middle + half_offset)]) + half_offset
        window[sample_index].append(window[sample_index][0] + peak_location - half_offset)
        window[sample_index].append(window[sample_index][0] + peak_location)
        window[sample_index].append(window[sample_index][0] + peak_location + half_offset)
        # if (sample_index == 271 or sample_index == 366):
        #     peak_location = int((end - start) / 2)
        x1 = x[peak_location - half_offset]
        x2 = x[peak_location]
        x3 = x[peak_location + half_offset]
        y1 = y[peak_location - half_offset]
        y2 = y[peak_location]
        y3 = y[peak_location + half_offset]
        K = 2 * ((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)) / math.sqrt(
            ((x2 - x1) ** 2 + (y2 - y1) ** 2) * ((x3 - x2) ** 2 + (y3 - y2) ** 2) * (
            (x1 - x3) ** 2 + (y1 - y3) ** 2))
        curvature[sample_index] = K
        if (abs(K) < threshold):
            prediction[sample_index] = 0
        else:
            prediction[sample_index] = 1

result_evaluation(label, prediction)
# print("accuracy = ", accuracy_score(label, prediction))
# print("precision = ", precision_score(label, prediction, average='micro'))
# print("recall = ", recall_score(label, prediction, average='micro'))
# print("mcc = ", matthews_corrcoef(label, prediction))
# precision_recall_curve(label, prediction)

# save result into a csv file
with open('../result/result.csv', 'w', newline='') as csv_file:
    spamwriter = csv.writer(csv_file)
    for sample_index in range(sample_number):
        if label[sample_index] == prediction[sample_index]:
            spamwriter.writerow([label[sample_index], prediction[sample_index]])
        else:
            spamwriter.writerow([label[sample_index], prediction[sample_index], "error"])

# Prepare data for deep learning input
x_data = np.empty([len(cnn_set), 100, 100])
y_data = np.empty(len(cnn_set))
current_index = 0
for sample_index in range(sample_number):
    fig = plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.xlim(xmin=two_theta[window[sample_index]][0], xmax=two_theta[window[sample_index]][1])
    plt.plot(two_theta[window[sample_index][0]:window[sample_index][1] + 1], xrd_peak[sample_index], 'k')
    plt.savefig("./figure/" + str(sample_index + 1) + '.png', format="png")
    plt.close()
    if label[sample_index] == 0 or label[sample_index] == 1:
        y_data[current_index] = label[sample_index]
        img = imread("./figure/" + str(sample_index + 1) + '.png', mode='P')
        # img = imresize(img,(100,100))
        # img = np.reshape(img, 10000)
        x_data[current_index] = img
        current_index += 1
x_data = x_data.astype(np.uint8)
y_data = y_data.astype(np.uint8)
np.save("../data/x_data_new.npy", x_data)
np.save("../data/y_data_new.npy", y_data)

# # plot all samples
# for sample in range(0, sample_number):
#     plt.figure()
#     # plt.figure(figsize = (9,6), dpi = 300)
#     plt.subplot(211)
#     plt.title('sample ' + str(sample + 1) + ' category ' + str(label[sample]))
#     # this is the background samples
#     # if sample in [184, 207, 208, 231, 232, 255, 278, 301, 324]:
#     #     plt.axis([two_theta[0], two_theta[feature_number - 1], 0, 1000])
#     # else:
#     #     plt.axis([two_theta[0], two_theta[feature_number - 1], 450, 2400])
#     plt.plot(two_theta, xrd[sample])
#     plt.scatter([two_theta[window[sample][0]], two_theta[window[sample][1]]], [xrd[sample][window[sample][0]], xrd[sample][window[sample][1]]])
#     plt.subplot(212)
#     # plt.xlim(xmax = two_theta[feature_number - 1], xmin = two_theta[0])
#     plt.plot(two_theta[window[sample][0]:window[sample][1]+1], xrd_peak[sample])
#     plt.scatter([two_theta[window[sample][2]], two_theta[window[sample][3]], two_theta[window[sample][4]]], [xrd[sample][window[sample][2]], xrd[sample][window[sample][3]], xrd[sample][window[sample][4]]])
#     plt.title('sample ' + str(sample + 1) + ' category ' + str(prediction[sample]) + " k " + str(curvature[sample]))
#     # plt.savefig("/home/zheng/Desktop/figure0216/" + str(sample + 1) + '.png', format="png", dpi=200)
#     plt.savefig("D:/xiong/Desktop/figure0418/" + str(sample + 1) + '.png', format="png", dpi=200)
#     plt.close()
# # # std = back_sub(std, neighbor=2, threshold=0.5, fitting_degree=50, if_plot=0, two_theta=two_theta)