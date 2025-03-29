# Loan Approval Prediction Challenge

## Introduction

This challenge consists of a machine learning competition. The task is to apply the techniques learned in class to develop the best predictive model. A dataset of SME enterprises that have applied for a loan is provided, and the goal is to build a classifier that determines whether the loan should be granted or denied.

In this challenge, students must assume the role of a bank and answer the following question:  
**As a bank representative, should I grant a loan to a particular small business (Company X)? Why or why not?**  
The decision should be based on an assessment of the loan's risk.

## Evaluation

The evaluation metric for this competition is **Macro F1-Score**. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision `p` and recall `r`.

- **Precision** is the ratio of true positives (tp) to all predicted positives (tp + fp).
- **Recall** is the ratio of true positives to all actual positives (tp + fn).

The F1 score is given by the following formula:

\[ F1 = 2 * \frac{Precision * Recall}{Precision + Recall} \]

The F1 metric weights recall and precision equally, and a good retrieval algorithm should maximize both. Thus, moderately good performance on both will be favored over excellent performance on one and poor performance on the other.

## Submission Format

For every instance in the dataset, submission files should contain two columns: `id` and `Accept`.

The file should contain a header and have the following format:

```csv
id,Accept
f255a970557,1
a53cb1bb739,0
3f5acb12b9a,0
f4b380b1972,0
8f159c7c9c0,0
e7885e60901,1
c03a84e0050,0
461e6e6a201,1
6ffe6f2c7b8,1
```

## Submission Guidelines

### Before the Challenge Deadline:
- You can submit your test file with predictions **up to 100 times per day**.
- Before the deadline, select **two submissions** as your best ones for calculating the final leaderboard ranking.

### After the Kaggle Submission Deadline:
Use Moodle to submit the following:
- The notebooks containing your code.
- A PDF file with a table listing all group members, indicating the author of each notebook.
- Individual and group assignments.

## How to Submit (Code Example)

You should load the training dataset for training your algorithm:

```python
import pandas as pd

url = "data/train.csv"
df = pd.read_csv(url)
df.head()
```

You should also load the test dataset (without labels) to make predictions:

```python
url = "data/test_nolabel.csv"
df_test = pd.read_csv(url)
df_test.head()
```

If you apply preprocessing steps to the training dataset, remember to apply the same steps to the test dataset. For example, if you add a new column, your model will expect that column to be present in the test dataset as well.

### Generating the Submission File

Assuming `df_test_clean` is the transformed `df_test` after preprocessing:

```python
X = df_test_clean[features].values
df_test_clean['Accept'] = model.predict(X)

# Convert predictions to integers and save as CSV
import numpy as np
df_test_clean['Accept'] = df_test_clean['Accept'].astype(np.int)
df_test_clean.to_csv('my-model.csv', columns=['id','Accept'], index=False)
```

Ensure your final submission file is correctly formatted before submitting.

---

Good luck with the competition!

