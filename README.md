# Shallow Learning Classification for Disease Spread Prediction in a Population Graph

## Overview

This project focuses on predicting disease spread within a simulated population graph using various machine learning models. The study utilizes individual and graph-level features to classify infection statuses, aiming for a balance between accuracy and computational efficiency. The dataset used is sourced from the Kaggle challenge "Predict Simulated Disease Spread (Classification)."

## Abstract

Accurately predicting disease transmission is critical for public health interventions. This study evaluates shallow-learning techniques for predicting disease spread within a simulated population graph, focusing on computational efficiency and predictive performance. Through targeted feature engineering, shallow models achieved competitive accuracy compared to ensemble and deep-learning methods while offering significant reductions in computational time. The results demonstrate the effectiveness of shallow-learning approaches for epidemiological modeling, particularly in resource-constrained environments.

## Introduction

Modeling infectious disease spread is a complex challenge for public health. Traditional models often fall short in capturing the nuances of transmission dynamics. Recent studies, like those by Alali et al., have demonstrated the effectiveness of shallow learning methods, offering a computationally efficient alternative to deep learning, especially in resource-constrained environments.

This study explores the use of shallow-learning techniques for predicting disease spread within a population graph, incorporating features like age, constitution, and behavior. We compare the performance of shallow learning models with ensemble and deep-learning approaches to identify the most effective method for this task.


## Methods
### Dataset

The dataset for this study is sourced from the Kaggle challenge titled "Simulated Disease Spread EDA." This dataset comprises simulated data of disease spread within a population graph, including features such as age, constitution, and behavior of individuals. The dataset is pre-cleaned and preprocessed.

### Model

We implemented a variety of models to predict disease spread, focusing on the balance between computational efficiency and classification performance:

- **Shallow Learning Models**
  - **Quadratic Discriminant Analysis (QDA)**
  - **Logistic Regression (LR)**
  - **K-Nearest Neighbors (KNN)**

- **Ensemble Models**
  - **Extreme Gradient Boost (XGBoost)**
  - **Random Forest (RF)**

- **Deep Learning Models**
  - **Multi-Layer Perceptron (MLP)**
  - **Graph Neural Network (GNN)**
  - **Graph Attention Network (GAT)**

### Training and Evaluation

1. **Feature Engineering:** Engineered additional node-level and component-level features, including graph statistics, neighbor statistics, and distance to the index patient, to enhance model performance.
2. **Model Training:** Implemented and trained a range of models, including shallow-learning, ensemble, and deep-learning approaches, using the engineered features.
3. **Hyperparameter Tuning:** Employed Grid Search and Random Search with cross-validation to fine-tune the hyperparameters of each model.
4. **Evaluation Metric:** Evaluated model performance using standard classification metrics, with a primary focus on AUC-ROC to assess the model’s ability to discriminate between classes.

## Expected Outcomes

The study successfully demonstrated that shallow learning models, such as KNN, achieved competitive results compared to ensemble and deep learning models, particularly in terms of AUC-ROC scores. The inclusion of engineered features significantly improved performance across all models. Notably, shallow learning methods offered a clear advantage in computational efficiency while maintaining robust predictive capabilities for disease spread prediction. These findings highlight the effectiveness of shallow learning approaches in epidemiological modeling, providing a viable alternative to more computationally intensive deep learning models.

## Implementation

The implementation of this project is organized into several key components:

1. **Data Processing**:
   - The `utils_data` module contains scripts for loading, preprocessing, and transforming the raw data.

2. **Model Development**:
   - Model definitions and training scripts are located in the `utils_classification` and `utils_training` modules.
   - The `notebooks` directory contains Jupyter notebooks that document the development and fine-tuning of these models across different approaches: shallow learning, ensemble learning, and deep learning.

3. **Training and Evaluation**:
   - Models are trained using scripts in the `utils_training` module.
   - Evaluation scripts are located in the `utils_evaluation` module, where standard metrics such as accuracy, precision, recall, F1 score, and AUC-ROC are computed to assess model performance.

4. **Visualization**:
   - The `utils_plot` module contains scripts for generating visualizations, including confusion matrices and ROC curves.

5. **Running the Models**:
   - The `utils_running` module includes scripts (`Runner.py` and `GraphRunner.py`). These scripts are for getting model predictions and probabilities.

6. **Reproducibility**:
   - All experiments and results are documented in the `notebooks` directory, allowing for easy replication of the study. The provided utilities ensure that all models can be trained and evaluated consistently across different configurations.


## How to Run Code

The notebooks are set up so that they can just be run with the necessary libraries installed. Run the following command to install the required libraries:
```
pip install -r requirements.txt
```
Following this, all notebooks can be run without further steps.

## Repository Structure
```
.
├── data
│   └── raw                     # Raw data files
├── notebooks                   # Jupyter notebooks for data exploration and model 
│   ├── _exploration            # Data exploration notebooks
│   ├── deep_learning           # Notebooks for deep learning model development
│   ├── ensemble_learning       # Notebooks for ensemble learning model development
│   └── shallow_learning        # Notebooks for shallow learning model development
├── utils_classification        # Utility functions for classification tasks
│   ├── GaussianDiscriminantAnalysis.py
│   ├── GraphAttentionNetwork.py
│   ├── GraphConvolutionalNetwork.py
│   └── MultiLayerPerceptron.py
├── utils_data                  # Utility functions for data preprocessing and 
│   ├── PopulationData.py
│   └── PopulationDataset.py
├── utils_evaluation            # Utility functions for model evaluation
│   ├── evaluate_graph.py
│   ├── evaluate_numpy.py
│   └── evaluate_tensor.py
├── utils_plot                  # Utility functions for plotting and visualization
│   ├── plot_confusion_matrix.py
│   ├── plot_graph_nx.py
│   └── plot_roc.py
├── utils_running               # Utility functions for running the models
│   ├── GraphRunner.py
│   └── Runner.py
└── utils_training              # Utility functions for training the models
    ├── GraphTrainer.py
    └── Trainer.py
└── README.md
```

