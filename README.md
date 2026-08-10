**Overview**

This repository contains my completed submission for the WGU course **D682 – AI Model Optimization**.

The project is a controlled experimental framework for improving machine learning model performance. Rather than training a single model and reporting its score, the code establishes a documented baseline and then applies a sequence of distinct optimization techniques, measuring the effect of each one against that baseline under identical conditions.

Both classification and regression optimization tracks are implemented, and every run writes a comparison table and confusion matrices so that improvements can be verified rather than asserted.

**Scenario**

A model has been trained and deployed, but its performance is insufficient for the business requirement. The task is to determine which optimization strategies actually improve results on this data, quantify the improvement, and produce evidence that supports the recommendation.

Because optimization claims must be defensible, the experiment holds the random seed, train/test split, and evaluation metrics constant across every model variant.

**Project Objectives**

-Establish a reproducible baseline model for comparison

-Apply and evaluate multiple distinct optimization techniques

-Tune hyperparameters through systematic search rather than manual adjustment

-Optimize the classification decision threshold rather than defaulting to 0.50

-Compare ensemble strategies against single-model approaches

-Produce quantitative before-and-after evidence of improvement

**Optimization Techniques Evaluated**

**Baseline — L2 Regularized Logistic Regression**
A standardized logistic regression with L2 penalty, serving as the reference point for all subsequent comparisons.

**Hyperparameter Tuning**
RandomizedSearchCV over regularization penalty (L2 and elastic net), inverse regularization strength across a logarithmic range, L1 ratio, and class weighting, using stratified cross-validation and F1 as the optimization target.

**Decision Threshold Optimization**
The tuned model is refit on a training subset and its predicted probabilities are evaluated against a held-out validation split across a range of decision thresholds. The threshold maximizing validation F1 is selected and then applied to the test set.

**Regularization-Constrained Random Forest**
A random forest deliberately constrained through maximum depth and minimum samples per leaf, testing whether structural regularization outperforms linear regularization on this data.

**Bagging Ensemble**
Bootstrap aggregation applied to the tuned logistic regression with subsampling of both rows and features, testing variance reduction as an optimization lever.

**Stacked Ensemble**
A stacking classifier combining random forest and gradient boosting base learners with a logistic regression meta-learner, testing whether model diversity outperforms any single optimized model.

**Evaluation and Reporting**

Every model variant is scored on accuracy, precision, recall, F1, and ROC AUC against the same held-out test set. Results are sorted by F1 and written to a comparison table. A confusion matrix is exported for each variant, and a summary file records the best search parameters, the selected decision threshold, and the final model ranking.

A parallel regression experiment applies the equivalent optimization sequence to a continuous target.

**Skills Demonstrated**

-Experimental design for model optimization

-Baseline establishment and controlled comparison

-Randomized hyperparameter search with cross-validation

-Regularization strategy selection and tuning

-Decision threshold optimization for imbalanced outcomes

-Ensemble methods: bagging, boosting, and stacking

-Multi-metric classification evaluation

-Reproducible experiment tracking and reporting

**How to Run**

Classification experiments:

```
python experiments/run_experiments.py
```

Regression experiments:

```
python experiments/run_experiments_regression.py
```

The data source and target column can be overridden through the DATA_CSV and TARGET_COL environment variables. Results are written to a reports directory.

**Repository Contents**

-`experiments/run_experiments.py` – Classification optimization experiment suite

-`experiments/run_experiments_regression.py` – Regression optimization experiment suite

-`data/demo_classification.csv` – Classification dataset

-`data/demo_regression.csv` – Regression dataset

-`docs/Report_Template.docx` – Written report deliverable template
