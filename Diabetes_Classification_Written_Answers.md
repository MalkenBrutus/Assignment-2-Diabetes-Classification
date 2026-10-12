# Diabetes classification written answers

IE500618 Machine Learning | Assignment 2

Jørund Sæten Birkeland, Matteo Castelli, Haakon Torvik Gullbekk, and Marius Alexander Langaas Klæbo

We used the supplied survey data to fit Logistic Regression and compare decision thresholds of 0.50 and 0.40. The model detects some of the positive cases, but accuracy alone conceals many missed cases. Lowering the threshold increases detection and also increases false positives. The evidence below comes from the executed notebook Assignment_2_Diabetes_Classification.ipynb.


## Q1 Understanding the data

The verified CSV contains 253,680 respondents, 21 predictors, and the target Diabetes_binary. Class 0 means no diabetes. Class 1 combines prediabetes or diabetes, following the supplied dataset guide. The analysis cannot distinguish those two conditions.

Notebook Section 1 reports 218,334 class-0 respondents (86.07%) and 35,346 class-1 respondents (13.93%). The target is imbalanced. A rule predicting class 0 for everyone would achieve about 86% accuracy while detecting none of class 1. We therefore need precision, recall, F1, and the confusion matrix to see what the model actually achieves. There are no missing cells. We retain the 24,206 rows matching earlier rows, as instructed: without respondent identifiers, identical recorded values do not establish duplicate people.


## Q2 Understanding the features

| Feature example | Meaning | Variable type |
| --- | --- | --- |
| BMI = 30 | A measured Body Mass Index value | Numerical measurement |
| HighBP = 1 | The respondent has high blood pressure | Binary category |
| Age = 9 | Age group 60-64, rather than age nine | Ordered category |

All columns are stored as numbers, but their meanings differ. HighBP is already encoded as 0/1. Age has an order, but its codes are not measured ages or equal distances in years. BMI is a numerical quantity. MentHlth and PhysHlth, unlike category codes, are genuine counts of days. The feature guide and examples in notebook Section 1 explain these distinctions before preprocessing.


## Q3 Findings from exploratory analysis

Notebook Section 3 explores every input feature using the 202,944 TRAIN observations. Numerical features receive overall and class-wise distributions, boxplots, and summary statistics. Binary and ordered features receive category counts and within-category class-1 percentages. Using within-class densities for numerical comparisons avoids confusing the larger class size with a different distribution shape. The following five findings are supported by Sections 3.1-3.4 and the corresponding saved tables.

BMI. Median BMI is 27 in class 0 and 31 in class 1. The distributions overlap, so the difference supports an association without giving a reliable single-value classification rule.

High blood pressure. Class 1 accounts for 6.03% of respondents without high blood pressure and 24.46% of those with it. The difference makes this a useful candidate predictor, but it may also reflect age or other shared characteristics.

General health. The class-1 proportion is 2.52% for excellent health and 38.06% for poor health. These self-reported categories relate strongly to the target. The data do not establish whether poor health precedes the condition or reflects its consequences.

Age. The class-1 proportion is 1.42% at ages 18-24. It peaks at 22.03% in the 70-74 group, compared with 18.38% at 80+. Read the actual category pattern rather than assuming that each older group must have a higher proportion. Age is a grouped predictor, not an exact age.

Heavy alcohol indicator. Class-1 proportions are 14.41% for code 0 and 5.81% for code 1. This observed association cannot establish a protective effect: group composition, reporting differences, or behaviour following diagnosis could contribute.

These comparisons describe TRAIN associations. They do not establish causal effects, future disease onset, or which features the fitted model relies on most. We retain all 21 predictors instead of deleting features solely because of an individual association. The notebook also shows the remaining features, including category sample sizes, so the detailed discussion does not replace full EDA coverage.


## Q4 Why TEST remains separate

Section 2 creates one 80/20 split with random_state=42 and stratify=y: 202,944 TRAIN rows and 50,736 TEST rows. TEST contains 7,069 positive cases. Stratification keeps class proportions approximately similar; it does not make the classes 50/50.

Detailed EDA uses TRAIN. In Sections 4-5, the pipeline learns scaling means and standard deviations from TRAIN only, then fits Logistic Regression on the prepared TRAIN features. Numerical measurements and day counts are scaled, ordered categories use explicit OrdinalEncoder categories, and binary inputs remain 0/1. Defining these operations does not learn parameters; fit() does. TEST receives the already fitted transformations. If TEST influenced preprocessing statistics or repeated model revisions, its scores would no longer provide an independent final check. We neither balance nor resample TEST.


## Q5 Errors and evaluation metrics

The positive class is prediabetes or diabetes (1). Notebook Section 6 gives the following TEST confusion matrix at the default 0.50 threshold. Rows are actual classes and columns are predicted classes.

| Actual class | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| 0: no diabetes | 42,616 TN | 1,051 FP |
| 1: prediabetes or diabetes | 5,947 FN | 1,122 TP |

The 1,051 false positives are respondents labelled class 0 who are predicted as class 1. The 5,947 false negatives are respondents labelled class 1 who are predicted as class 0. In a follow-up setting, false positives could generate unnecessary assessment, while false negatives could leave positive cases unflagged. This dataset does not measure those consequences or establish the model as a diagnostic tool.

| Metric | TEST result | Meaning |
| --- | --- | --- |
| Accuracy | 86.21% | Correct predictions / all respondents |
| Precision | 51.63% | True positives / all predicted positives |
| Recall | 15.87% | True positives / all actual positives |
| F1 | 0.2428 | Harmonic mean of precision and recall |

The model finds 1,122 of 7,069 actual positive cases. Its 86.21% accuracy should be compared with the always-class-0 rule's 86.07% TEST accuracy and zero recall. The fitted model detects positives that the reference rule misses, but its recall shows that many positives still go undetected. F1 summarises precision and recall at this threshold; it does not replace the separate metrics or error counts.


## Q6 Probability and threshold

Notebook Section 7 compares the same probabilities at both prescribed thresholds, predicting class 1 when its probability is at least the threshold. The following scores are proportions, rather than percentages.

| Threshold | Accuracy | Precision | Recall | F1 | False positives | False negatives |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.8621 | 0.5163 | 0.1587 | 0.2428 | 1,051 | 5,947 |
| 0.40 | 0.8564 | 0.4731 | 0.2699 | 0.3437 | 2,125 | 5,161 |

At 0.40, 1,860 respondents previously predicted as class 0 become class 1. Of them, 786 are actual positive cases and 1,074 are actual negative cases. False negatives decrease from 5,947 to 5,161, while false positives increase from 1,051 to 2,125. Recall rises from 15.87% to 26.99%; precision changes from 51.63% to 47.31%. Accuracy changes from 86.21% to 85.64%, and F1 from 0.2428 to 0.3437.

The fitted preprocessing, coefficients, input observations, and predicted probabilities do not change. The threshold only changes how a probability becomes a class. No retraining occurs. The lower threshold finds more positives while creating more false alarms; it is not universally better. Selecting a threshold for actual use would require evidence about error costs and external validation. We have not used TEST to search for one.


## Q7 Conclusion and limitations

The EDA shows differences associated with the binary outcome, especially in BMI, high blood pressure, and self-rated general health, while the class distributions still overlap. Logistic Regression combines the predictors and detects some positive respondents. The threshold comparison shows why a high overall accuracy is insufficient: finding additional positives changes the burden of false positives. The results support a reasoned discussion of this workflow, not a causal or clinical claim.

- Survey data and timing: reporting differences and sample composition can affect associations. We do not have a documented sequence showing that the features preceded the outcome. This is not evidence of causes or prospective prediction of new disease.
- Combined target: class 1 merges prediabetes and diabetes, so detection of those conditions cannot be assessed separately.
- Model representation: Logistic Regression assumes an additive linear relationship in log-odds. Ordinal coding gives equal steps to categories whose actual age or income intervals differ. Some nonlinearities and interactions may therefore be missed.
- Evaluation scope: one random split is not external or temporal validation. The file has no respondent identifiers to verify independence. We have not assessed calibration or subgroup performance, and the analysis is unweighted. The scores describe this file and split rather than a population guarantee. Notebook Section 8 discusses these limits.

## G1 Division of work

The following contribution account follows the supplied group scenario.

Jørund worked across the workflow and checked that the notebook and written answers were consistent. He also helped connect the interpretation of the data with the modelling decisions. Matteo developed exploratory code and visualisations and investigated patterns across the features. The group then consolidated that work into a clear sequence of notebook sections.

Marius focused on the preprocessing pipeline, model fitting, evaluation, and checking that the notebook could run from a fresh kernel. He shared programming and review tasks with Jørund and Matteo. Haakon focused on the meanings of the variables, interpretation of plots and classification errors, and review of the written explanations. He worked through the programming with the others. The responsibilities overlapped, so each section received review beyond its main contributor.


## G2 Understanding the complete solution

We reviewed one another’s sections and walked through the notebook together rather than only combining separate pieces. Each member explained a part they had not mainly implemented: why age codes need an ordered representation, what the scaler learns, which confusion-matrix cells represent missed cases, and how lowering the threshold changes decisions. When an explanation was unclear, we returned to the relevant table or code cell and checked it together. Running the notebook in order helped connect the split, preprocessing, predictions, and metrics as one workflow.


## Sources and evidence

Assignment 2: Diabetes Classification, supplied instructions and Canvas page; Diabetes Classification: Understanding the Dataset, supplied course guide; Heart Disease Classification - Lecture Code and Lecture Classification Heart Disease v4, IE500618 teaching materials. Dataset meanings and codes follow the supplied guide. All numerical findings come from the executed notebook and its results/ tables.

Repository: https://github.com/MalkenBrutus/Assignment-2-Diabetes-Classification
