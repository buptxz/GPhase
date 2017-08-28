import sys
import matplotlib.pyplot as plt
import matplotlib.colors
import phase_module
import argparse
# import matlab.engine
import os
from argparse import RawTextHelpFormatter
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
    font_size = 20
    text_content = ["(at.%)", "(at.%)", "(at.%)"]
    text_position = [(0, -7.5, 107.5), (102.5, -7.5, 5), (-5, 102, 3)]
    ternary_data = [scale, position, font_size, text_content]

    prediction_source = np.genfromtxt("../data/agilefd_AlCuMo_600.csv", delimiter=',')
    prediction = []
    for row in prediction_source:
        prediction.append(np.argmax(row))
    # prediction = np.genfromtxt("../data/agilefd.csv", delimiter=',')
    print(prediction)

    # Evaluate results
    phase_module.result_evaluation(label, prediction)

    # Plot results
    len_truth = max(label)
    len_predicted = max(prediction)
    [figure, tax] = phase_module.ternary_figure(ternary_data)
    tax.scatter(original_comp, marker='o', c=label, s=50, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.jet, edgecolor="w")
    tax.savefig("../figure/" + "labeled.png", format="png")
    # tax.show()

    [figure, tax] = phase_module.ternary_figure(ternary_data)
    tax.scatter(original_comp, marker='o', c=prediction, s=50, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.jet, edgecolor="w")
    tax.savefig("../figure/" + "prediction.png", format="png")
    # tax.show()

    print("Calculation finished.")