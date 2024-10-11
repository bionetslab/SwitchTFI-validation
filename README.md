# SwitchTFI-validation

This repository contains all the necessary code and files to recreate the results presented in *Identifying transcription factors driving cell differentiation*.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Installation

Instructions on how to install and set up the project locally.

```bash
# Clone the repository
git clone git@github.com:bionetslab/SwitchTFI-validation.git

# Navigate to the project directory
cd SwitchTFI-validation

# Create and activate the Conda environment from the .yml file
conda env create -f switchtfi_val.yml
conda activate switchtfi_val
```


## Usage
All scripts are documented with inline comments explaining the individual steps.
Functions used in the workflow are documented with docstring comments providing further information.
To reproduce the results follow the workflow:

### Data Preprocessing
```bash
# Create conda environment for preprocessing
conda env create -f preprocessing/prepr.yml
conda activate preprocessing

# Run preprocessing scripts
python 00_data_preprocessing.py
```
### GRN Inference
The auxiliary files required by Scenic can be downloaded from [https://resources.aertslab.org/cistarget/](https://resources.aertslab.org/cistarget/) (10.10.24).
They must be stored in the correct subdirectory of *switchtfi_val/data/scenic_aux_data*. For some information on the auxiliary data see *switchtfi_val/data/scenic_aux_data/meta_data.txt*
```bash
# Create conda environment for GRN Inference with PyScenic
conda env create -f grn_inf/psc.yml
conda activate psc

# Run GRN inference scripts, this may take a while
python 01_grn_inference.py
```

### SwitchTFI Analyses
```bash
# Activate main environment
conda activate switchtfi_val

# Run SwitchTFI analyses with the preprocessed scRNA-seq data and the inferred GRN as an input
python 02_switchtfi_model_fitting.py
```

### Validation
```bash
# Activate main environment
conda activate switchtfi_val

# Run scripts with necessary computations for the validation of SwitchTFI
python 03_switchtfi_model_fitting.py
```

### Plotting
```bash
# Activate main environment
conda activate switchtfi_val

# Run scripts to produce the plots
python 03_switchtfi_model_fitting.py
```
**Note:**
- SwitchTFI provides the preprocessed data via the functions *switchtfi.data.preendocrine_alpha()/preendocrine_beta()/erythrocytes()*. To skip the preprocessing step comment in this option in the scripts for the later steps.
- The GRNs used to produce the results are provided in *switchtfi_val/results/01_grn_inf*. The GRN inference step can be omitted.
- The results of the SwitchTFI analyses are provided in *switchtfi_val/results/02_switchtfi*.
- The results of the validation procedures are provided in *switchtfi_val/results/02_validation*.
- The plots are provided in *switchtfi_val/results/03_plots*.


## License

This project is licensed under the MIT License - see the [GNU General Public License v3.0](LICENSE) file for details.

## Contact

Paul Martini - paul.martini@fau.de

Project Link: [https://github.com/bionetslab/SwitchTFI-validation](https://github.com/bionetslab/SwitchTFI-validation)
