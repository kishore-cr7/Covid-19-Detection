COVID-19 Pneumonia Classification Using Chest CT Images
Project Overview
This project implements a Hybrid Deep Transfer Learning Model with Kernel Metric for the classification of COVID-19 pneumonia using Chest CT Images. The model utilizes a ResNet18 feature extractor combined with an Extreme Learning Machine (ELM) classifier to improve classification accuracy. The approach is based on the study paper titled "A Hybrid Deep Transfer Learning Model With Kernel Metric for COVID-19 Pneumonia Classification Using Chest CT Images". The goal is to effectively classify chest CT images into four categories: COVID-19, Viral Pneumonia, Lung Opacity, and Normal.

Key Features
ResNet18 Feature Extractor: The model uses ResNet18 for feature extraction, which is fine-tuned to improve its representation of CT image features.
Extreme Learning Machine (ELM) Classifier: ELM is used as the classifier to improve accuracy. The model utilizes two separate ELM classifiers, each with a hidden layer size of 1000.
Hybrid Learning Approach: The model incorporates domain loss, classification loss, and diversity loss to enhance the learning process.
Evaluation Metrics: The model evaluates performance using various metrics like Precision, Recall, F1-Score, and AUC.
Project Structure

project/
│
├── data/
│   └── COVID/
│   └── Normal/
│   └── Viral Pneumonia/
│   └── Lung Opacity/
├── main.py                # Main code to train and evaluate the model
├── model.py               # Contains the model architecture (ResNet18, ELM Classifier)
├── requirements.txt       # List of required Python packages
└── README.md              # Project description


Requirements
Python 3.x
PyTorch 1.x
scikit-learn
Matplotlib
torchvision
Pillow

git clone https://github.com/your-username/covid-pneumonia-classification.git
cd covid-pneumonia-classification
pip install -r requirements.txt
Dataset
This model uses chest CT images categorized into the following folders:

COVID: Contains images of chest CT scans showing COVID-19 pneumonia.
Normal: Contains images of healthy lungs.
Viral Pneumonia: Contains images of viral pneumonia.
Lung Opacity: Contains images showing other lung conditions.
The dataset path in the code is set as C:/Users/HP/Desktop/SAMPLE. You need to update this path to where your dataset is stored locally.

Usage
Run the main Python script to start training and evaluating the model:

python main.py
The model will train for 30 epochs (modifiable), outputting the training loss, classification loss, domain loss, and diversity loss for each epoch. After training, the model will validate on the test set and display the following metrics:

Accuracy
Precision
Recall
F1-Score
AUC (Area Under the Curve)
Training and Validation
The model trains using the Adam optimizer with a learning rate of 0.001. The learning rate is adjusted every 10 epochs using the StepLR scheduler. The validation phase computes performance metrics using scikit-learn’s functions for precision, recall, F1-score, and AUC.

Loss Functions
The model uses three types of losses:

Domain Loss: Measures the similarity between source and target features.
Classification Loss: Uses Cross-Entropy loss for class predictions.
Diversity Loss: Encourages diversity in the model’s predictions by minimizing the entropy of the output.
Results
The final output of the training and validation will include the following:

Losses: Training losses for each epoch (Classification, Domain, and Diversity Losses).
Metrics: Precision, Recall, F1-Score, and AUC for model evaluation.
ROC Curve: The Receiver Operating Characteristic curve for each class, plotting True Positive Rate against False Positive Rate.
