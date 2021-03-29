# Machine-learning-based-detection-ofdeliberately-generated-artifacts-inERP-BCI-experiments
Project associated with master thesis of Angelo Groß in cooperation with Neural Information Processing Group (TU Berlin). 

## Required Software
The following software will be required in order to locally run the source code in this repository.
- [Python 3.7](https://www.python.org/downloads/)
- [MatLab](https://de.mathworks.com/products/get-matlab.html?s_tid=gn_getml)
### Required Python Packages
The following packages are required to run the complete project.
#### Data Evaluation
- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [covar](https://pythonhosted.org/covar/index.html)
- [wyrm](http://bbci.github.io/wyrm/index.html)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [sklearn](https://sklearn.org/)
### Generating Simulated Data
- [SEREEGA](https://github.com/lrkrol/SEREEGA) (added as submodule to this repository)
- [Fieldtrip](https://www.fieldtriptoolbox.org/) (added as submodule to this repository)
### Stimulus Presentation Program
- [PyQt5](https://pypi.org/project/PyQt5/)

## Program Execution
The evaluation scripts were executed using the [Spyder IDE](https://www.spyder-ide.org/). When directly executing the scripts from the console, it was observed that the program pauses when a *matplotlib* figure is rendered. To continue the script it is necessary to close the figure.

### Executable scripts (Python)
The following scripts are executable.
#### Evaluation scripts
They assess classification accuracies using different ML algorithms
- [evaluation_program/src/ldaclassification.py](evaluation_program/src/ldaclassification.py)
- [evaluation_program/src/cspclassification.py](evaluation_program/src/cspclassification.py)
- [evaluation_program/src/psdfeatureclassification.py](evaluation_program/src/psdfeatureclassification.py)
- [evaluation_program/src/evalSimulationData.py](evaluation_program/src/evalSimulationData.py), execution time: >10 minutes
#### Other scripts
- [evaluation_program/src/generateArtifactData.py](evaluation_program/src/generateArtifactData.py), generates the scalp data used as input for the simulation task.
- [evaluation_program/src/plot_data.py](evaluation_program/src/plot_data.py), plots scalpmaps and averaged curves
- [presentation_program/mainWindow.py](presentation_program/mainWindow.py), runs the stimulus presentation program

### Executable scripts (MatLab)
This script generates all the simulation data from chapter 4. The execution time can last longer than an hour.
- [simulation_program/src/sampledata.m](simulation_program/src/sampledata.m)

## Running all scripts
In order to make all scripts it is necessary to access the simulation data, unless you want to repeat the generation process which can last longer than an hour depending on the device you are using.
The zipped simulation data is found [here](https://tubcloud.tu-berlin.de/s/ZNdQ6jcfnJwKDK8) and must be extracted within the [data](data) folder of your local repository.
