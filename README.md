# Python-PDV-Software
This is a translation of the Dlott Group PDV speed calculator into Python using the STFT method found in scipy 0.19.1 to create a spectrogram, then takes user defined boundaries to fit a long fft of the trajectory to a gaussian. 
It currently contains a single program with dependent functions placed within the code. If called from another code, you need to import all modules of the code for it to work:

from PDV_speed import *

