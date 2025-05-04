<img src="./imag/1_imag.png" style="zoom:20%;" />

# Seismic Imaging 25 Edition
Course materials for the Seismic Imaging course (2025 Edition) taught by Professor Nicola Bienati at the University of Pisa for the [MSc. In Exploration and Applied Geophysics](https://www.dst.unipi.it/home-wgf.html)

This is a computational exercise for a practical application of FWI highlighting the importance of the frequency infromation when performing FWI.
Developed by Felipe Rincón at University of Pisa, Italy. 

If you have any questions, please contact Felipe by email: felipe.rincon@phd.unipi.it

This repository contains:
- A jupyter notebook with all the scripts and user functions to analyze the results. 
- A synthetic velocity model in .npy format.

This repository contains three Jupyter notebooks and a utility functions file:
	•	Notebook 1: Demonstrates the sensitivity kernel in full waveform inversion using both low- and high-frequency seismic data.
	•	Notebook 2: Compares the behavior of the objective function when using low versus high frequencies.
	•	Notebook 3: Performs FWI using low frequencies, high frequencies, and a multiscale approach that combines both.
 

Felipe Rincón

Italy, 05.05.2025


## Installation 
### Step 1:  Install [Devito](https://www.devitoproject.org/)
Installation steps are taken from the Devito repository. Please check the official page for alternative ways to install Devito.

The easiest way to try Devito is through Docker using the following commands:
```
# get the code
git clone https://github.com/devitocodes/devito.git
cd devito

# start a jupyter notebook server on port 8888
docker-compose up devito
```
After running the last command above, the terminal will display a URL such as
`https://127.0.0.1:8888/?token=XXX`. Copy-paste this URL into a browser window
to start a [Jupyter](https://jupyter.org/) notebook.

[See here](http://devitocodes.github.io/devito/download.html) for detailed installation
instructions and other options.

### Step 2:  Download the scripts
```
# get the code
git clone https://github.com/lfrincond/seismic_imaging25.git
cd seismic_imaging25
```

## Use these exercises to understand some basic concepts behind FWI
<img src="./imag/2_imag.png" style="zoom:20%;" />

