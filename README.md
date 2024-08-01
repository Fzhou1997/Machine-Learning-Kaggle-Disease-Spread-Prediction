# Predicting Disease Spread Using Generalized Discriminant Analysis with Kernel Functions (GDA-KF)

## Overview

This project aims to predict the spread of infectious diseases within a simulated population graph using Generalized Discriminant Analysis (GDA) combined with Kernel Functions.
The proposed model leverages individual-level features such as age, constitution, and behavior to classify infection statuses efficiently and accurately.
This study is conducted using the dataset from the Kaggle challenge "Simulated Disease Spread."

## Abstract

The spread of infectious diseases poses a significant threat to public health.
Traditional epidemiological models often have limitations in capturing the complex dynamics of disease transmission.
This research proposes the use of Generalized Discriminant Analysis (GDA), a shallow learning technique that combines traditional Gaussian Discriminant Analysis and Kernel Functions, to predict simulated disease spread within a population graph.
The model incorporates individual features such as age, constitution, and behavior to classify infection status.
By leveraging GDA's ability to identify complex patterns, we aim to develop a more accurate and efficient predictive model for disease spread.
The results of this study could inform targeted interventions and control strategies, ultimately contributing to improved public health outcomes.

## Introduction

Recent studies, such as Alali et al. (2022), have demonstrated the effectiveness of shallow learning methods, including algorithms such as Gaussian Process Regression (GPR), in complex tasks like disease spread prediction. These findings challenge the prevailing trend towards deep learning, which often necessitates vast computational resources and large datasets. By contrast, shallow learning methods, like Generalized Discriminant Analysis with Kernel Functions (GDA-KF), offer a computationally efficient alternative, making them particularly well-suited for scenarios with limited data or computational power. However, to our knowledge, none have specifically applied GDA-KF to graph-based disease spread prediction, incorporating individual-level features like age, constitution, and behavior. Therefore, we propose to leverage the unique strengths of GDA-KF, namely its ability to identify complex patterns and incorporate individual-level features, to develop a novel and potentially more accurate predictive model for disease spread that is also computationally efficient.


## Methods
### Dataset

The dataset for this study is sourced from the Kaggle challenge titled "Simulated Disease Spread EDA." This dataset comprises simulated data of disease spread within a population graph, including features such as age, constitution, and behavior of individuals. The dataset is pre-cleaned and preprocessed.

### Model

To predict/classify infection status, we propose to use the Generalized Discriminant Analysis with Kernel Functions (GDA-KF). The choice of using GDA-KF is motivated by the underlying normal distribution of the simulated dataset. Traditional GDA is effective for normally distributed data, but the inclusion of kernel functions is necessary to capture any possible non-linear relationships between features and target, such as age.

### Training and Evaluation

1. **Kernel Functions:** Experiment with different kernel functions and their parameters to identify the best-performing one.
2. **Data Transformation:** Use the selected kernel to transform the data.
3. **Discriminant Functions:** Apply GDA to learn the discriminant functions.
4. **Hyperparameter Tuning:** Employ techniques such as Grid Search or Random Search combined with cross-validation to fine-tune the model’s hyperparameters.
5. **Evaluation Metric:** The performance of the GDA-KF model will be evaluated using AUC-ROC, which measures the model’s ability to discriminate between classes.

### Baseline Models

To demonstrate the efficacy and efficiency of shallow learning, specifically the GDA-KF algorithm, in more complex prediction/classification tasks such as graph spread prediction, we will evaluate the model performance against other baseline shallow learning models:
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)

Additionally, we will compare the GDA-KF model with a deep learning model:
- Graph Neural Network (GNN)

## Expected Outcomes

The expected outcome of this research is to develop a GDA-KF-based predictive model that is comparable with deep learning models in terms of accuracy for disease spread prediction. Additionally, the GDA-KF model aims to provide a robust tool for public health interventions by accurately predicting infection statuses based on individual features and the overall population graph, while also being computationally efficient.

## Implementation



## Repository Structure
```
.
├── data
│ ├── test.csv
│ └── train.csv
├── models                      # Trained models and model evaluation scripts
├── notebooks                   # Jupyter notebooks for data analysis and model development
├── utils_classification        # Utility functions for classification tasks
├── utils_data                  # Utility functions for data preprocessing and transformation
├── utils_graph                 # Utility functions for graph-based tasks
├── utils_plot                  # Utility functions for plotting and visualization
└── README.md                   # Project description and instructions
```

