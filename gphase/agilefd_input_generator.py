import phase_module

xrd_path = '../data/AlCuMo_XRD_long300.csv'
composition_path = '../data/AlCuMo_composition.csv'

# xrd_path = '../data/FePdGa_XRD.csv'
# composition_path = '../data/FePdGa_composition.csv'

[xrd, two_theta, composition, label] = phase_module.read_data(xrd_path, composition_path)
sample_number = len(xrd)
feature_number = len(xrd[0])

f = open('../data/AlCuMo_long300.txt', 'w')

f.write('// Metadata\n')
f.write('M=3\n')
f.write('Elements=Fe,Ga,Pd\n')
f.write('Composition=Fe,Ga,Pd\n')
f.write('N=')
f.write(str(sample_number))
f.write('\n\n')

f.write('// Composition data\n')

f.write('Fe=')
for i in range(sample_number):
    f.write(str(composition[i][0]))
    if i != sample_number - 1:
        f.write(",")

f.write('\n')

f.write('Ga=')
for i in range(sample_number):
    f.write(str(composition[i][1]))
    if i != sample_number - 1:
        f.write(",")
f.write('\n')

f.write('Pd=')
for i in range(sample_number):
    f.write(str(composition[i][2]))
    if i != sample_number - 1:
        f.write(",")
f.write('\n')

f.write('\n')

f.write('// Integrated counts data\n')

f.write('Q=')
for i in range(feature_number):
    f.write(str(two_theta[i]))
    if i != feature_number - 1:
        f.write(",")
f.write('\n')

for sample in range(sample_number):
    f.write('I')
    f.write(str(sample + 1))
    f.write('=')
    for i in range(feature_number):
        f.write(str(xrd[sample][i]))
        if i != feature_number - 1:
            f.write(",")
    f.write('\n')