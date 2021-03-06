## Graph-based Automated Phase Segmentation (GPhase)

Last modified 01/25/2017
Version 1.0

Copyright (C) 2017 University of South Carolina

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
ERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

### Prerequisites

Python, Python-tk, Scipy, Numpy, matplotlib

Also, GPhase depends on MATLAB engine api for Python:
```
    cd "MATLABROOT/extern/engines/python"
    python setup.py install
```
see <a href="https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html" target="_blank">https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html</a>

### Usage

gphase [OPTIONS]

Options:

    -x --xrd <string>
     xrd file path
    -c --comp <string>
     composition file path
    -b --background <int>
     whether or not do background subtraction
    -k --K <float>
     threshold for merging phases (default:1.5)

### Example
```
cd gphase
python gphase.py -x ../data/FePdGa_XRD.csv -c ../data/FePdGa_composition.csv -b 0
python gphase.py -x ../data/AlCuMo_XRD.csv -c ../data/AlCuMo_composition.csv -b 1
```
### Reference
If you want to learn more about how GPhase works please refer to the paper:

Zheng Xiong, Yinyan He, Jason Hattrick-Simpers and Jianjun Hu. Automated Phase Segmentation for Large-Scale X-ray Diffraction Data Using Graph-based Phase Segmentation (GPhase) Algorithm. ACS Comb. Sci., 2017. DOI: 10.1021/acscombsci.6b00121
http://pubs.acs.org/doi/abs/10.1021/acscombsci.6b00121

### Contact me

If you have any questions or suggestions, please feel free to contact me. This is my email address: zxiong@email.sc.edu.
