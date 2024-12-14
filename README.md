# GraphSAGESurvival

# GraphSAGE for Predicting Patient Survival Length

## Overview

Predicting patient survival length is a critical aspect of personalized medicine, enabling healthcare providers to tailor treatment strategies and improve patient outcomes. This project leverages a Graph Neural Network (GNN)-based approach, specifically the GraphSAGE architecture, to predict patient survival lengths by integrating features extracted from 3D CT images and comprehensive clinical (tabular) data. By modeling patient similarities and interactions within a graph structure, the model captures both individual and contextual information, enhancing predictive accuracy and providing valuable insights for individualized treatment planning.

## Features

- **Multimodal Data Integration**: Combines 3D CT image features with structured clinical data to form comprehensive patient representations.
- **GraphSAGE Architecture**: Utilizes the GraphSAGE model to aggregate and propagate information across patient nodes based on similarity.
- **Ablation Studies**: Conducts experiments to assess the impact of different feature modalities (image, tabular) and learning rates on model performance.
- **Hyperparameter Sensitivity Analysis**: Explores the influence of various hyperparameters on the predictive capabilities of the model.
- **Reproducible Experiments**: Ensures consistency and reliability through the use of multiple random seeds and detailed experimental configurations.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/graphsage-survival-prediction.git
    cd graphsage-survival-prediction
    ```

2. **Create a Virtual Environment**

    It is recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Required Packages**

    Install the necessary Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    *If a `requirements.txt` file is not provided, install the following packages manually:*

    ```bash
    pip install torch torchvision torch-geometric sklearn pandas numpy matplotlib tqdm
    ```

    *Note: Ensure that you have the appropriate versions of PyTorch and Torch Geometric compatible with your CUDA version if using GPU acceleration.*

## Usage

### Preparing the Data

Ensure that your preprocessed patient data is stored in the specified directory (`/data37/xl693/RADCURE2024_processed`). The data should be organized in `.pkl` files containing image data, tabular features, and survival lengths for each patient.

### Running Ablation Studies

The core experiment script is `ablation.py`, which conducts ablation studies by varying learning rates and feature modalities to evaluate their impact on model performance.

To execute the ablation studies, run the following command:

```bash
python ablation.py
