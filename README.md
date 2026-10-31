# Datheon_Machine_Learning_-_Data_Science_Internship
Three ML projects from my Datheon internship: Iris classification, California housing price regression, and telecom churn prediction using Python, Scikit-learn, and XGBoost.
# Iris Flower Classification (TASK 1)
A supervised machine learning project that classifies iris flowers into three species (setosa, versicolor, virginica) using Logistic Regression and Decision Tree models.
# Problem Statement
Build a classification model on the Iris dataset that handles feature scaling and an 80/20 train/test split. Train Logistic Regression and Decision Tree classifiers, then benchmark them on accuracy, precision, recall, and F1-score.
# Approach
-Load data with Pandas (150 samples, 4 numeric features, 3 balanced classes, no missing values).
-Explore the data with Seaborn pairplots, colored by species.
-Preprocess by scaling features with StandardScaler and encoding labels with LabelEncoder.
-Split the data 80/20 into train and test sets (random_state=42).
-Train LogisticRegression and DecisionTreeClassifier.
-Evaluate with accuracy and classification reports (precision, recall, F1-score).
-Visualize the results with confusion matrix heatmaps and decision boundary plots.
# Results
-Model	Accuracy	Precision	Recall	F1-score
-Logistic Regression	1.00	1.00	1.00	1.00
-Decision Tree	1.00	1.00	1.00	1.00
-Both models classified all 30 test samples correctly. The Iris dataset is small and its classes are well separated, so perfect scores are expected here and may not carry over to harder datasets.
# Tech Stack
Python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
# How to Run
-Clone the repository.
-Install the dependencies: pip install pandas numpy scikit-learn seaborn matplotlib
-Place iris.csv in the project folder and update the file path in the notebook (it currently points to /content/iris.csv for Google Colab).
-Open and run week1___1_task_1.ipynb.


# California Housing Price Prediction (TASK 2)
A regression project that predicts median house values in California using Linear (OLS), Ridge (L2), and Lasso (L1) regression.
# Problem Statement
Predict median house values using the California Housing dataset. Impute missing values, handle multicollinearity, encode and scale features, and compare OLS, Ridge, and Lasso regression models using error and fit metrics.
# Dataset
-20,640 records, 10 columns
-Features include location (longitude, latitude), housing age, rooms, bedrooms, population, households, median income, and ocean proximity
# Target: median_house_value
# Approach
-Load and inspect the data with Pandas (info, describe, null checks).
-Impute the 207 missing total_bedrooms values with the column median.
-Encode the categorical ocean_proximity column with LabelEncoder.
-Analyze correlations with a Seaborn heatmap to spot multicollinearity.
-Scale the continuous features with StandardScaler.
-Split the data 80/20 into train and test sets (random_state=42).
-Tune Ridge and Lasso with GridSearchCV (5-fold CV) over alpha values [0.01, 0.1, 1, 10, 100].
-Evaluate Linear Regression with RMSE and R², plus predicted-vs-actual and residual distribution plots.
# Results
-Model	Best alpha	CV MSE
-Ridge	10	~4.84 × 10⁹
-Lasso	10	~4.84 × 10⁹
Linear Regression (test set): RMSE ≈ 71,148 | R² ≈ 0.614
Ridge and Lasso performed almost identically to each other, which suggests regularization adds little here because the feature set is small.
# Tech Stack
python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
# How to Run
-Clone the repository.
-Install the dependencies: pip install pandas numpy scikit-learn seaborn matplotlib
-Place housing.csv in the project folder and update the path in the notebook (it currently points to /content/housing.csv for Google Colab).
-Open and run Task_2_1.ipynb.


# Telecom Customer Churn Prediction (TASK 3)
An ensemble machine learning project that predicts which telecom customers are likely to churn, using Random Forest and XGBoost classifiers.
# Problem Statement
Train ensemble models to predict customer churn from telecom subscription records. Handle categorical features and class imbalance, train Random Forest and XGBoost with cross-validation, and evaluate with ROC-AUC curves (target: >80%).
# Dataset
Telco Customer Churn dataset: 7,043 customers, 21 columns
# Target: Churn (No: 5,174 | Yes: 1,869), so the classes are imbalanced
# Approach
-Load and inspect the data with Pandas.
-Clean by converting TotalCharges to numeric and filling the 11 resulting nulls with the median.
-Encode categorical features with LabelEncoder.
-Split the data 80/20 into train and test sets (random_state=42).
-Balance classes with SMOTE (training set goes from 4,138 / 1,496 to 4,138 / 4,138) and with scale_pos_weight in XGBoost.
-Train RandomForestClassifier and XGBClassifier.
-Cross-validate with StratifiedKFold (5 folds).
-Evaluate with ROC-AUC curves for both models.
-Interpret the results by extracting the top 10 feature importances from each model.
-Results
-Model	5-Fold CV Accuracy
-Random Forest	79.5%
-XGBoost	76.0%
Key drivers of churn: Contract type, tenure, MonthlyCharges, TotalCharges, OnlineSecurity, TechSupport, and InternetService. XGBoost ranked Contract highest by a wide margin (~45% importance).
# Tech Stack
Python, Pandas, NumPy, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), Matplotlib
# How to Run
-Clone the repository.
-Install the dependencies: pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib
-Place Telecom.csv in the project folder and update the path in the notebook (it currently points to /content/Telecom.csv for Google Colab).
-Open and run Task_3_1.ipynb.
