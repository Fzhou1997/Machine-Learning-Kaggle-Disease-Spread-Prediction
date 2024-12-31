<H1 align="middle"> Machine Learning for Disease Spread Prediction</H1>

<p align="middle">
    <strong>
        Exploring machine learning architectures for predicting the spread of diseases in synthetic 
        population graphs
    </strong>
</p>

## Description

This repository contains the code, data, and models for our CS6140 Machine Learning course and
Kaggle [Predict Simulated Disease Spread (Classification)](https://www.kaggle.com/competitions/predict-simulated-disease-spread-classification/overview)
challenge project, which explores predicting disease spread within a simulated population graph using various machine
learning models. The study utilizes individual, adjacency, and graph-level features to classify infection statuses,
aiming for a balance between accuracy and computational efficiency.

This project was jointly developed by [Zhou Fang](https://github.com/Fzhou1997)
and [Ryan Quinn](https://github.com/ryqui) at Northeastern University. This is an ongoing project, and the repository
will be updated with additional code, data, and models as the research progresses. Additional information about the
project, including the methodology, results, and discussions, can be found in
the [research report](https://docs.google.com/document/d/1bdr3bJpXvj3zNDenKJ3U3PD26dp_gV2A9J4GW41zoJI/edit?usp=sharing).

## Motivation

Modeling infectious disease spread is a complex challenge for public health. Traditional models often fall short in
capturing the nuances of transmission dynamics. Recent studies, like those by Alali et al., have demonstrated the
effectiveness of shallow learning methods, offering a computationally efficient alternative to deep learning, especially
in resource-constrained environments.

This study explores the use of various shallow, ensemble, and deep-learning techniques for predicting disease spread
within a population graph, incorporating features like age, constitution, and behavior. We compare the performance of
shallow learning models with ensemble and deep-learning approaches to identify the most effective method for this task.

## Key Features

- Implements a range of machine learning models, including shallow learning, ensemble learning, and deep learning
  architectures, for predicting disease spread in a population graph.
- Utilizes a synthetic dataset from the Kaggle challenge "Simulated Disease Spread EDA," comprising individual and
  graph-level features, of over 650,000 nodes for training and inference, respectively.
- Incorporates engineered features, including graph statistics, neighbor statistics, and distance to the index patient,
  to enhance model performance.
- Offers insights into the tradeoffs between computational efficiency and classification performance across different
  machine learning architectures and paradigms.

## Implementation

This project is implemented in Python using Sci-kit Learn, XGBoost, and PyTorch machine learning libraries.

### Dataset

The dataset for this study is sourced from the Kaggle challenge
titled [Predict Simulated Disease Spread (Classification)](https://www.kaggle.com/competitions/predict-simulated-disease-spread-classification/overview).
This dataset comprises simulated data of disease spread within a population graph, including features such as age,
constitution, and behavior of individuals. The dataset is separated into training and inference sets, each with 650,000
nodes. The training set includes the infection status of each node, while the inference set requires predicting the
infection status based on the provided features.

### Model Selection

We implemented a variety of models to predict disease spread, focusing on the balance between computational efficiency
and classification performance:

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

### Results Summary

The following table summarizes the test-set metrics of the models on the Kaggle challenge dataset:

| Model                   | Accuracy | AUROC | F1-Score |
|-------------------------|----------|-------|----------|
| Quadratic Discriminant  | 0.65     | 0.71  | 0.62     |
| Logistic Regression     | 0.60     | 0.63  | 0.54     |
| K-Nearest Neighbors     | 0.75     | 0.83  | 0.76     |
| Extreme Gradient Boost  | 0.80     | 0.87  | 0.81     |
| Random Forest           | 0.67     | 0.76  | 0.58     |
| Multi-Layer Perceptron  | 0.80     | 0.87  | 0.81     |
| Graph Neural Network    | 0.58     | 0.60  | 0.47     |
| Graph Attention Network | 0.56     | 0.59  | 0.33     |

The results indicate that the Multi-Layer Perceptron and Extreme Gradient Boost models outperform other models in terms
of accuracy, AUROC, and F1-Score. The Graph Neural Network and Graph Attention Network models exhibit lower performance
due to the complexity of the graph structure and the limited number of features.

Highest performing models, namely MLP and XGBoost, outperform the Kaggle leaderboard models by over 20% in terms of
accuracy and AUROC.

### Future Directions

The study provides insights into the tradeoffs between computational efficiency and classification performance across
different machine learning architectures and paradigms. Future research will focus on optimizing the hyperparameters of
the models, exploring additional feature engineering techniques, and incorporating other features to improve
classification performance.

## Repository Structure

```
.
├── data
|   ├── out                     # Predicted probability output files
│   ├── processed               # Preprocessed data files
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
│   ├── GraphTrainer.py
│   └── Trainer.py
├── requirements.txt            # Python package dependencies
└── README.md
```

## Installation

To run the code in this repository, you will need to have Python 3.10 or 3.11 installed on your system. You will also
need to install the required Python packages listed in the `requirements.txt` file. You can install these packages using
the following command:

``` bash
pip install -r requirements.txt
```

## Usage

All driver code for training, evaluating, and testing the models is provided in the Jupyter notebooks located in the
`notebooks/` directory. You can run these notebooks using JupyterLab or Jupyter Notebook.