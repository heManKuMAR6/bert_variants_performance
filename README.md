**# bert_variants_performance**
README: Multi-Model Fact-Checking System
Overview
This project implements a multi-model fact-checking system using transformer-based language models to classify relationships between a given premise and hypothesis into one of three categories:

Entailment (Fact): The hypothesis logically follows from the premise.
Neutral: The hypothesis is neither supported nor contradicted by the premise.
Contradiction (Fake): The hypothesis directly conflicts with the premise.
The models used in this project are:

bert-base-multilingual-cased
distilbert-base-multilingual-cased
xlm-roberta-base
Steps to Run
Dataset Preparation

Place your dataset in the datasets folder and name it train.csv.
Ensure the dataset has the following columns:
premise: A statement or fact.
hypothesis: A related or conflicting statement.
label: One of 0 (Entailment), 1 (Neutral), or 2 (Contradiction).
Install Dependencies

Install required libraries:
bash
Copy code
pip install transformers datasets scikit-learn matplotlib seaborn
Run the Code

Follow the steps outlined in the project:
Tokenize the dataset: Preprocess the dataset for each model using the appropriate tokenizer.
Train the models: Train the three models (bert-base-multilingual-cased, distilbert-base-multilingual-cased, xlm-roberta-base) using the Hugging Face Trainer API.
Evaluate the models: Compute evaluation metrics (accuracy, precision, recall, F1-score) for each model.
Test external inputs: Use the best-performing model to predict relationships for custom inputs.
Visualize confusion matrices: Generate and analyze confusion matrices for all models.
Identify the Best Model

Evaluate all three models on the test dataset and identify the best-performing model based on accuracy.
Test External Inputs

Use the identified best model to classify a custom premise and hypothesis.
Key Features
Multi-Model Comparison:
Train and evaluate multiple models to identify the best performer.
Custom Predictions:
Use the trained system to classify relationships for new inputs.
Visualization:
Generate confusion matrices to analyze model performance.
Project Structure
bash
Copy code
.
├── datasets
│   └── train.csv                # Dataset file
├── results_<model_name>         # Directory to save model checkpoints and tokenizer files
├── logs_<model_name>            # Training logs
└── main.py                      # Main script for training and evaluation
Example Usage
Custom Input Prediction:

python
Copy code
premise = "The sky is blue."
hypothesis = "It is raining."
predict_with_best_model(premise, hypothesis)
Evaluate Model:

Run confusion matrices and classification reports for each model.
Evaluation Metrics
Accuracy: Overall correctness of predictions.
Precision: Correctness of positive predictions.
Recall: Coverage of actual positives.
F1-Score: Harmonic mean of precision and recall.
Dependencies
transformers
datasets
scikit-learn
matplotlib
seaborn
