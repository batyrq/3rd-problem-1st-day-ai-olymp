# Text Matching and Classification for AI Olympiad

-----

This repository contains the Jupyter Notebook `3rd-problem-1st-day-ai-olymp.ipynb`, which provides a solution to the 3rd problem of the Republican AI Olympiad's first day. The problem involves determining if a given "description" text correctly matches a "conclusion" text.

## Competition Details

  * **Kaggle Competition Link:** [Problem 3 - Republic Olymp TST Homework](https://www.kaggle.com/competitions/problem-3-republic-olymp-tst-homework)
  * **Evaluation Metric:** Accuracy
  * **Achieved Score:** 0.92632

## Project Overview

The core of this problem is a binary classification task: given a pair of texts (description, conclusion), predict whether they are a "match" (label 1) or "not a match" (label 0). The solution involves data augmentation to create negative samples, text preprocessing, TF-IDF vectorization, and a CatBoost Classifier.

The key steps in the solution are:

1.  **Data Loading and Negative Sampling:**
      * The `train.csv` dataset is loaded. Initially, all pairs are assumed to be "matches" (label 1.0).
      * **Negative Sampling:** To create negative samples (label 0), for each original `description`, a random `conclusion` from a *different* entry in the dataset is paired with it. This significantly increases the training data and allows the model to learn what constitutes a "non-match".
2.  **Text Preprocessing:**
      * A `clean_text` function is defined to:
          * Convert text to lowercase.
          * Remove extra whitespace and strip leading/trailing spaces.
      * This cleaning is applied to both `description` and `conclusion` columns in the training and later to test data.
3.  **Feature Engineering (TF-IDF Vectorization):**
      * All unique texts from both `description` and `conclusion` columns are collected to build the TF-IDF vocabulary.
      * `TfidfVectorizer` is used with `max_features=50000` to convert text into numerical TF-IDF representations.
      * For each pair of `description` and `conclusion`, the features are created by horizontally stacking:
          * The TF-IDF vector of the `description`.
          * The TF-IDF vector of the `conclusion`.
          * The absolute difference between the TF-IDF vectors of `description` and `conclusion`. This last component helps the model identify dissimilarities.
      * The target variable `y` is the `label` column.
4.  **Model Training (CatBoostClassifier):**
      * The prepared features `X` and labels `y` are split into training and validation sets using `train_test_split`.
      * A `CatBoostClassifier` is initialized and trained on the training data.
      * The model's performance is evaluated on the validation set using AUC (Area Under the Receiver Operating Characteristic Curve), providing an indication of its ability to distinguish between positive and negative classes.
5.  **Inference on Test Data:**
      * The test data consists of three files: `test_batches.csv`, `test_conclusions.csv`, and `test_descriptions.csv`.
      * `test_descriptions` and `test_conclusions` are loaded into dictionaries for quick lookup by ID, and their texts are preprocessed.
      * For each entry in `test_batches.csv`, which contains a `description_id` and a list of `conclusion_ids`:
          * The model iterates through each possible `description_id` and its associated `conclusion_ids`.
          * For each `(description, conclusion)` pair, TF-IDF features are generated in the same way as for training data.
          * The trained `CatBoostClassifier` predicts the probability of the pair being a "match".
          * The `conclusion_id` that yields the *highest prediction probability* for a given `description_id` is selected as the best match.
6.  **Submission File Generation:**
      * The selected `description_id` and their corresponding best-matching `conclusion_id` are assembled into a pandas DataFrame.
      * This DataFrame is saved to `submission.csv` in the format required for Kaggle submission.

## Setup and Running the Notebook

To run this notebook, you'll need a Kaggle environment or a local setup with the necessary libraries.

### Prerequisites

  * Python 3.x
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `catboost`
  * `re` (standard library)

### Installation

You can install the required Python packages using pip:

```bash
pip install pandas numpy scikit-learn catboost
```

### Running the Notebook

1.  **Download the data:** Download `train.csv`, `test_batches.csv`, `test_conclusions.csv`, and `test_descriptions.csv` from the Kaggle competition page and place them in your input directory (e.g., `/kaggle/input/problem-3-republic-olymp-tst-homework/` if on Kaggle).
2.  **Open the notebook:** Open `3rd-problem-1st-day-ai-olymp.ipynb` in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, or Kaggle Notebooks).
3.  **Run all cells:** Execute all cells in the notebook sequentially. The script will perform data loading, augmentation, model training, and generate the `submission.csv` file.

-----
