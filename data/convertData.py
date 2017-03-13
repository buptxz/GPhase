
import numpy as np

def read_data(xrd, comp):
    xrd_data = np.genfromtxt(xrd, delimiter=',')
    comp_data = np.genfromtxt(comp, delimiter=',')
    two_theta = xrd_data[0, :]
    xrd_data = xrd_data[1:, :]
    label = comp_data[:, 3]
    comp_data = comp_data[:, 0:3]
    return [xrd_data, two_theta, comp_data, label]

xrd_peak_path = 'FePdGa_XRD.csv'
composition_path = 'FePdGa_composition.csv'
[xrd, two_theta, composition, label] = read_data(xrd_peak_path, composition_path)
sample_number = len(xrd)
feature_number = len(xrd[0])

with open('FePdGa.txt', 'w') as f:
    f.write('// Metadata\n')
    f.write('M=3\n')
    f.write('Elements=Fe,Ga,Pd\n')
    f.write('Composition=Fe,Ga,Pd\n')
    f.write('N=278\n\n')

    f.write('// Composition data\n')
    f.write('Fe=')
    for i in range(sample_number):
        f.write(str(composition[i][0]))
        if (i < sample_number - 1):
            f.write(',')
    f.write('\n')

    f.write('Ga=')
    for i in range(sample_number):
        f.write(str(composition[i][2]))
        if (i < sample_number - 1):
            f.write(',')
    f.write('\n')


    f.write('Pd=')
    for i in range(sample_number):
        f.write(str(composition[i][1]))
        if (i < sample_number - 1):
            f.write(',')
    f.write('\n\n')

    f.write('//Integrated counts data\n')
    f.write('Q=')
    for i in range(feature_number):
        f.write(str(two_theta[i]))
        if (i < feature_number - 1):
            f.write(',')
    f.write('\n')
    
    for sample in range(sample_number):
        f.write('I' + str(sample + 1) + '=')
        for i in range(feature_number):
            f.write(str(xrd[sample][i]))
            if (i < feature_number - 1):
                f.write(',')
        f.write('\n')