# OCTA_Vascular_Analysis
This repository contains Python code developed as part of the thesis:

**"Assessing the stimulated dynamic response of cutaneous microvasculature using optical coherence tomography angiography."**

These scripts build upon previous work by **Mads Grønbeck** and **Magnus Borge**, which can be found here:  
[Hamdermadsik/Thesis_code_algorithms](https://github.com/Hamdermadsik/Thesis_code_algorithms/tree/main)


### Overview 
These codes generate Percentile Intensity Projections (PIPs) of the papillary plexus and reticular plexus and analysis code for the papillary plexus. 

- PIP Percentile Optimizer
- PIP Generator
- Segmentation Model Trainer 
- Papillary Plexus Analyzer

### Dependencies
Install required packages via pip:

pip install opencv-python pydicom scikit-image matplotlib numpy scipy tqdm pandas torch xlsxwriter

### How to run
PIP Percentile Optimizer, PIP Generator, and the Papillary Plexus Analyzer can be run in interactive mode or using the command line. A trained segmentation model is included. 

Segmentation Model Trainer can be run using the command line. The dataset is included. 
