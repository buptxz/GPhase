import sys
import matplotlib.pyplot as plt
import matplotlib.colors
import phase_module
import argparse
import ternary
# import matlab.engine
import os
from argparse import RawTextHelpFormatter
from matplotlib import pyplot, gridspec
import numpy as np
# from peakdetect import *
import warnings
import copy

if __name__ == "__main__":

    # ./bin/GPhase.py -i data/xrd.csv -c data/composition.sv
    # ./bin/GPhase.py -i xrd.csv -c composition.sv
    # Call matlab function

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="GPhase--An XRD Phase Mapping Program.\n University of South Carolina",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-x', '--xrd', dest='xrd', help='x-ray diffraction csv file', default='../data/AlCuMo_XRD_600.csv')
    parser.add_argument('-c', '--comp', dest='comp', help='composition csv file',
                        default='../data/AlCuMo_composition_600.csv')
    parser.add_argument('-k', "--K", dest='K', help='threshold for merging phases (default:1.5)', type=float,
                        default=1.5)
    parser.add_argument('-b', '--background', dest='if_background_subtraction', help='background subtraction', type=int,
                        default=0)
    # parser.add_argument('-v', "--version")
    args = parser.parse_args()

    # # execute MATLAB peak finder code
    # if args.if_background_subtraction == 0:
    #     eng = matlab.engine.start_matlab()
    #     bin_path = os.path.dirname(os.path.realpath(__file__))
    #     eng.cd(bin_path, nargout=0)
    #     print("finding peaks... takes a few minutes...")
    #     xrd_peak_path = eng.peakfinder(args.xrd, nargout=1)
    # else:
    #     xrd_peak_path = args.xrd
    #     print("calculating...")

    # read data
    [xrd, two_theta, composition, label] = phase_module.read_data(args.xrd, args.comp)
    sample_number = len(xrd)
    feature_number = len(xrd[0])

    for row in composition:
        temp = row[0]
        row[0] = row[2]
        row[2] = temp

    [neighbor_list_original, neighbor_list, original_comp, coordinate, x_coordinate, y_coordinate] = \
        phase_module.construct_neighbor(composition)

    # Plot Delaunay triangulation
    position = None
    scale = 100.0
    font_size = 10
    text_content = ["(at.%)", "(at.%)", "(at.%)"]
    text_position = [(0, -7.5, 107.5), (102.5, -7.5, 5), (-5, 102, 3)]
    ternary_data = [scale, position, font_size, text_content]

    prediction_source = np.genfromtxt("../data/agilefd_AlCuMo_600.csv", delimiter=',')
    prediction = []
    for row in prediction_source:
        category = 0
        for i in range(len(row)):
            if row[i] > 0.85:
                category += 3 ** i
        prediction.append(category)
    # prediction = np.genfromtxt("../data/agilefd.csv", delimiter=',')
    print(prediction)

    # Evaluate results
    phase_module.result_evaluation(label, prediction)

    # Plot results
    label = label + 1
    len_truth = max(label)
    len_predicted = max(prediction)
    # [figure, tax] = phase_module.ternary_figure(ternary_data)
    # # for i in range(len(label)):
    # #     label[i] += (i+1)
    #
    # tax.scatter(original_comp, marker='o', c=label, s=50, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.jet,
    #             edgecolor="w")
    # # tax.show()


    # for i in range(len(label)):
    #     if int(label[i]) != 1:
    #         tax.annotate(int(label[i]), original_comp[i], fontsize=8)
    #
    # tax.savefig("../figure/demo.pdf",format='pdf', dpi=1000)
    # tax.show()

    # [figure, tax] = phase_module.ternary_figure(ternary_data)
    # tax.scatter(original_comp, marker='o', c=prediction, s=50, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.jet, edgecolor="w")
    # tax.savefig("../figure/" + "prediction.png", format="png")
    # # tax.show()
    # plt.show()
    # tax.close()
    # tax.savefig("../figure/figure9(b).pdf")

    # print(original_comp)




    ## Boundary and Gridlines
    scale = 30
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 10)
    figure.set_dpi(600)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=6)
    tax.gridlines(color="blue", multiple=2, linewidth=0.5)

    # Set Axis labels and Title
    fontsize = 20
    tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize)
    tax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize)
    tax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize)
    tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize)

    # Set ticks
    tax.ticks(axis='lbr', linewidth=1)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()

    ternary.plt.show()


    f = pyplot.figure()
    gs = gridspec.GridSpec(1, 2)

    ax1 = pyplot.subplot(gs[0, 0])
    figure, tax = ternary.figure(ax=ax1)


    # f = plt.figure()
    # plt.plot(range(10), range(10), "o")
    # temp, (ax1, ax2) = plt.subplots(1, 2)
    # # ax = f.add_subplot(121)

    # tax = ternary.TernaryAxesSubplot(ax=ax)
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=20, color="blue")
    # tax.left_axis_label("Al (at.%)", fontsize=font_size, offset=0.12)
    # tax.right_axis_label("Cu (at.%)", fontsize=font_size, offset=0.12)
    # tax.bottom_axis_label("Mo (at.%)", fontsize=font_size, offset=-0.02)
    # tax.ticks(axis='l', ticks=["0", "20", "40", "60", "80", "100"], offset=0.022)
    # tax.ticks(axis='b', ticks=["0", "20", "40", "60", "80", "100"], offset=0.022)
    # tax.ticks(axis='r', ticks=["0", "20", "40", "60", "80", "100"], offset=0.022)
    tax.scatter([(1,1,1,)])
    tax.scatter(original_comp, marker='o', edgecolor='w', s=40, color='red')
    tax.clear_matplotlib_ticks()

    ax2 = pyplot.subplot(gs[0, 1])
    figure, tax = ternary.figure(ax=ax2)
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=20, color="blue")
    tax.scatter(original_comp, marker='o', edgecolor='w', s=40, color='red')
    # tax.set_title("Simplex Boundary and Gridlines")
    # tax.left_axis_label("Al (at.%)", fontsize=font_size, offset=0.12)
    # tax.right_axis_label("Cu (at.%)", fontsize=font_size, offset=0.12)
    # tax.bottom_axis_label("Mo (at.%)", fontsize=font_size, offset=-0.02)
    # tax.ticks(axis='l', ticks=["0", "20", "40", "60", "80", "100"], offset=0.022)
    # tax.ticks(axis='b', ticks=["0", "20", "40", "60", "80", "100"], offset=0.022)
    # tax.ticks(axis='r', ticks=["0", "20", "40", "60", "80", "100"], offset=0.022)
    tax.clear_matplotlib_ticks()

    plt.show()
    f.savefig("../figure/foo.pdf", bbox_inches='tight')
    print("Calculation finished.")