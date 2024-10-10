# SwitchTFI-validation

This repository contains all the necessary code and files to recreate the results presented in *Identifying transcription factors driving cell differentiation*.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
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
```bash
# Create conda environment for GRN Inference with PyScenic, this may take a while
conda env create -f grn_inf/psc.yml
conda activate psc

# Run preprocessing scripts
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



## Features

- Bullet points of the key features
- Feature 1
- Feature 2
- Feature 3


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - [@your_twitter_handle](https://twitter.com/your_twitter_handle) - email@example.com

Project Link: [https://github.com/username/repository-name](https://github.com/username/repository-name)

