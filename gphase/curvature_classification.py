import matplotlib.pyplot as plt
from phase_module import back_sub, result_evaluation
import numpy as np
import math
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, precision_recall_curve
from scipy.misc import imread, imresize

xrd = np.genfromtxt('../data/xrd.csv', delimiter=',')
composition = np.genfromtxt('../data/composition.csv', delimiter=',')
two_theta = xrd[0, :]
xrd = xrd[1:, :]
label = composition[:, 3]
sample_number = len(xrd)
feature_number = len(xrd[0])

std = np.std(xrd, axis=0)

# plot standard deviation curve
plt.plot(two_theta[1:-2], std[1:-2])
# plt.scatter(two_theta[start], std[start])
# plt.scatter(two_theta[end], std[end])
# plt.savefig("../figure/deviation.png", dpi=500)
plt.show()

# Background subtraction by curve fitting
std = back_sub(std, neighbor=2, threshold=0.5, fitting_degree=50, if_plot=0, two_theta=two_theta)

for i in range(feature_number):
    if std[i] < 0:
        std[i] = 0

# Looking for the foot of max peak
max_value = max(std[2:-2])
max_index = list(std).index(max_value)
start = max_index
end = max_index
while std[start] != 0:
    start -= 1
while std[end] != 0:
    end += 1
print(start)
print(end)

# _max, _min = peakdetect.peakdetect(std, range(feature_number), lookahead=10, delta=0.35)
plt.plot(two_theta[2:-2], std[2:-2])
plt.scatter(two_theta[start], std[start])
plt.scatter(two_theta[end], std[end])
plt.show()

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
            j += 1
        if abs(sample[i - 1] - sample[j - 1]) < abs(sample[i] - sample[j]):
            i -= 1
            j -= 1
        xrd_peak.append(sample[i:j+1])
    window.append([i, j])

x = two_theta[start:end+1]
new_label = []
sample_numbering = []
for sample_index in range(len(xrd_peak)):
    if sample_index not in [162, 184, 207, 208, 231, 232, 255, 278, 301, 324]:
    # if sample_index not in []:
        sample_numbering.append(sample_index)
        new_label.append(label[sample_index])
        y = xrd_peak[sample_index]
        middle = (int) ((end - start) / 2)
        half_offset = (int) ((end - start) / 4)
        peak_location = np.argmax(y[half_offset : (middle + half_offset)]) + half_offset
        window[sample_index].append(window[sample_index][0] + peak_location - half_offset)
        window[sample_index].append(window[sample_index][0] + peak_location)
        window[sample_index].append(window[sample_index][0] + peak_location + half_offset)
        if (sample_index == 271 or sample_index == 366):
            peak_location = int((end - start) / 2)
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
# for sample in [184, 207, 208, 231, 232, 255, 278, 301, 324]:
#     prediction[sample] = 2

result_evaluation(new_label, prediction)
# save result into a csv file
with open('../result/result.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    for sample in range(len(prediction)):
        if label[sample] == prediction[sample]:
            spamwriter.writerow([sample_numbering[sample], label[sample], prediction[sample]])
        else:
            spamwriter.writerow([sample_numbering[sample], label[sample], prediction[sample], "error"])
# print("accuracy = ", accuracy_score(label, prediction))
# print("precision = ", precision_score(label, prediction))
# print("recall = ", recall_score(label, prediction))
# print("mcc = ", matthews_corrcoef(label, prediction))
# precision_recall_curve(label, prediction)

## plot all samples
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
x_data = np.empty([len(prediction), 100, 100])
y_data = np.empty(len(prediction))
current_index = 0
for sample in range(sample_number):
    fig = plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.xlim(xmax=two_theta[window[sample]][1], xmin=two_theta[window[sample]][0])
    plt.plot(two_theta[window[sample][0]:window[sample][1] + 1], xrd_peak[sample], 'k')
    plt.savefig("/home/zheng/Desktop/figure0320/" + str(sample + 1) + '.png', format="png")
    plt.close()
    if label[sample] == 0 or label[sample] == 1:
        y_data[current_index] = label[sample]
        img = imread("/home/zheng/Desktop/figure0320/" + str(sample + 1) + '.png', mode='P')
        # img = imresize(img,(100,100))
        # img = np.reshape(img, 10000)
        x_data[current_index] = img
        current_index += 1


    # if label[sample] == 0:
    #     plt.savefig("D:/Users/xiong/Desktop/data/train/0/" + str(sample + 1) + '.png', format="png")
    #     plt.savefig("D:/Users/xiong/Desktop/data/validation/0/" + str(sample + 1) + '.png', format="png")
    # elif label[sample] == 1:
    #     plt.savefig("D:/Users/xiong/Desktop/data/train/1/" + str(sample + 1) + '.png', format="png")
    #     plt.savefig("D:/Users/xiong/Desktop/data/validation/1/" + str(sample + 1) + '.png', format="png")
x_data = x_data.astype(np.uint8)
y_data = y_data.astype(np.uint8)
np.save("../data/x_data.npy", x_data)
np.save("../data/y_data.npy", y_data)