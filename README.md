# GraphSAGESurvival

# GraphSAGE for Predicting Patient Survival Length

## Overview

Predicting patient survival length is a critical aspect of personalized medicine, enabling healthcare providers to tailor treatment strategies and improve patient outcomes. This project leverages a Graph Neural Network (GNN)-based approach, specifically the GraphSAGE architecture, to predict patient survival lengths by integrating features extracted from 3D CT images and comprehensive clinical (tabular) data. By modeling patient similarities and interactions within a graph structure, the model captures both individual and contextual information.

## Features

- **Multimodal Data Integration**: Combines 3D CT image features with structured clinical data to form comprehensive patient representations.
- **GraphSAGE Architecture**: Utilizes the GraphSAGE model to aggregate and propagate information across patient nodes based on similarity.
- **Ablation Studies**: Conducts experiments to assess the impact of different feature modalities (image, tabular) and learning rates on model performance.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:


1. **Install Required Packages**

    Install the necessary Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Preparing the Data

You can either gain access from RADCURE Dataset or download the sample dataset from repo. 

If you gain the data from RADCURE dataset, you need to first use `data_extractor.py` to preprocess the data, and then follow the next step. 

For sample dataset, the data should be organized in `.pkl` files containing image data, tabular features, and survival lengths for each patient. You can directly run `ablation.py` without preprocess the data. 

### Running Ablation Studies

The core experiment script is `ablation.py`, which conducts ablation studies by varying learning rates and feature modalities to evaluate their impact on model performance.

To execute the ablation studies, run the following command:

```bash
python ablation.py
