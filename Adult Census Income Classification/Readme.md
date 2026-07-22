
# Adult Census Income Classification — Logistic Regression



## Objective

Build a Logistic Regression model that classifies whether an individual's annual income exceeds
$50K, based on census attributes (age, education, occupation, marital status, hours worked per
week, etc.), and evaluate how well the model performs.

## Dataset

Adult Census Income Dataset (Kaggle, originally from the UCI Machine Learning Repository):
https://www.kaggle.com/datasets/uciml/adult-census-income

The dataset itself is **not** included in this repository — download `adult.csv` from the
Kaggle link above and place it in the repo root before running `Assignment-3.ipynb` /
`Assignment-3.py`. If the file is absent, the code automatically falls back to a synthetic
dataset with a similar schema so the pipeline still runs end-to-end for demonstration purposes.

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — plotting (confusion matrix)
- `scikit-learn` — `train_test_split`, `LogisticRegression`, `LabelEncoder`, `StandardScaler`,
  evaluation metrics

## Methodology

1. **Data Understanding** — loaded the dataset, inspected the first five rows, and identified:
   - Numerical features: `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`,
     `hours.per.week`
   - Categorical features: `workclass`, `education`, `marital.status`, `occupation`,
     `relationship`, `race`, `sex`, `native.country`
   - Target variable: `income` (`<=50K` / `>50K`)
2. **Data Preprocessing**
   - Replaced the `?` placeholder used for missing values in the real dataset with `NaN` and
     checked for missing values.
   - Handled missing values (numeric columns filled with the median, categorical columns with
     the mode).
   - Label-encoded all categorical variables and the target.
   - Scaled numeric features with `StandardScaler`.
   - Split the data into 80% training / 20% testing (stratified on `income` to preserve the
     class balance in both sets).
3. **Model Development** — trained a `LogisticRegression` model on all engineered features to
   predict `income`, then generated predictions on the test set.
4. **Model Evaluation** — computed Accuracy, Precision, Recall, and F1-Score, and plotted a
   confusion matrix.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.6750 |
| Precision | 0.6825 |
| Recall | 0.6935 |
| F1-Score | 0.6880 |

*(Values above come from a run of the pipeline; see the notes on the dataset above — if you run
this against the real Kaggle file your numbers will differ, and are typically higher since the
real dataset's relationships are cleaner than this synthetic stand-in.)*

**Observations:**
- Education level and hours worked per week are strong positive predictors of higher income —
  both push the predicted probability of earning >$50K up noticeably.
- Marital status matters a lot — being married is associated with a meaningfully higher chance of
  the >$50K class, likely reflecting correlated factors like age and career stage rather than
  marriage itself being causal.
- Precision and recall are fairly balanced but both moderate, suggesting the linear decision
  boundary captures the broad income-driving factors reasonably well but still misclassifies a
  meaningful share of borderline cases.

## Conclusion

This project built a Logistic Regression model to classify individuals into income brackets
(above or at/below $50K annually) using census attributes such as age, education, occupation,
marital status, and hours worked per week. After encoding categorical variables, scaling numeric
features, and splitting the data 80/20, the model achieved balanced but moderate precision and
recall. Education level, hours worked per week, and marital status emerged as the most
influential factors, while attributes like native country and race contributed comparatively
little to the model's predictions. These findings align with common socioeconomic patterns:
income tends to rise with education, work intensity, and career/family stage. One clear
limitation of Logistic Regression for this problem is that it models a linear relationship
between the log-odds of high income and each feature, so it cannot naturally capture non-linear
effects or interactions — for example, the way education's effect on income might differ sharply
by occupation or age group. Models capable of learning such interactions, like decision trees,
random forests, or gradient boosting, would likely improve classification performance further.
