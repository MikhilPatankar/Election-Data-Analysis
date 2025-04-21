# Import numpy, pandas for data manipulation
import numpy as np
import pandas as pd
from collections import Counter
import time

# Import plotly, matplotlib and seaborn as visualization tools
import matplotlib.pyplot as plt
# NOTE: Plotly imports might cause issues if the environment doesn't have them.
# The original notebook had plotly, but execution failed.
# Keeping them commented out for potential use if the environment supports it.
# from plotly import tools
# import chart_studio.plotly as py
# import plotly.graph_objs as go
# import plotly.figure_factory as ff
# import plotly.express as px
# from plotly.subplots import make_subplots
# from plotly.offline import download_plotlyjs, init_notebook_mode
import seaborn as sns

# Import for resampling the data
from imblearn.over_sampling import SMOTE

# Import for scaling the data
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder

# Import machine learning models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# NOTE: CatBoost and LightGBM might require separate installation
# from catboost import CatBoostClassifier
# import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier

# Import model evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Plot should appear inside the jupyter notebook (if run in Jupyter)
# %matplotlib inline

# init_notebook_mode(connected=True) # Specific to Jupyter notebooks with Plotly offline mode

# --- 3. Gathering The Data ---
# Loading the dataset (Assuming 'data/LS_2.0.csv' is accessible)
# Replace 'data/LS_2.0.csv' with the actual path if different
try:
    df = pd.read_csv("data/LS_2.0.csv")
except FileNotFoundError:
    print("Error: Dataset file 'data/LS_2.0.csv' not found.")
    print("Please make sure the dataset is in the correct directory or update the path.")
    # Exit or handle the error appropriately
    exit()


# --- 5. Data Preprocessing and EDA ---
df.replace({'Not Available': np.nan}, inplace=True)
df.columns = df.columns.str.replace('\\r','')
df.columns = df.columns.str.replace('\\n','')

# Checking to see if the dataset contains any null values. We need to exclude NOTA votes while checking it.
df = df[df['PARTY']!= 'NOTA']
df = df.dropna()

# Function to clean currency values
def value_cleaner(x):
    try:
        # Ensure x is a string before attempting string operations
        if not isinstance(x, str):
            x = str(x) # Convert non-strings (like potential floats/ints) to string

        # Check if 'Rs' is in the string before splitting
        if 'Rs' in x:
            str_temp = (x.split('Rs')[1].split('\\n')[0].strip())
            str_temp_2 = ''
            for i in str_temp.split(","):
                str_temp_2 = str_temp_2+i
            # Handle potential non-numeric results after cleaning
            if str_temp_2.replace('.', '', 1).isdigit(): # Check if it's a number (int or float)
                 return float(str_temp_2)
            else:
                 return 0.0 # Return 0 if cleaning results in non-numeric string
        else:
             # If 'Rs' is not found, try to directly convert, otherwise return 0
             if x.replace('.', '', 1).isdigit():
                 return float(x)
             else:
                 return 0.0
    except Exception as e:
        # print(f"Error cleaning value '{x}': {e}") # Optional: print errors for debugging
        return 0.0 # Return 0.0 for float consistency on error

df['ASSETS'] = df['ASSETS'].apply(value_cleaner)
df['LIABILITIES'] = df['LIABILITIES'].apply(value_cleaner)


# Clean EDUCATION column
df['EDUCATION'].replace(to_replace='Post Graduate\\r\\n', value='Post Graduate', inplace=True)
df['EDUCATION'].replace(to_replace='Graduate Professional', value='Graduate', inplace=True)
df['EDUCATION'].replace(to_replace='Literate', value='8th Pass', inplace=True)
df['EDUCATION'].replace(to_replace='5th Pass', value='Illiterate', inplace=True)

# Convert columns to appropriate types
df['CRIMINALCASES'] = df['CRIMINALCASES'].astype(int, errors='raise')
df['ASSETS'] = df['ASSETS'].astype(float, errors='raise')
df['LIABILITIES'] = df['LIABILITIES'].astype(float, errors='raise')

# --- Feature Engineering (from section 6.3 for Party New) ---
# Sort out the top five parties according to the total number of votes
vote_share_top5 = df.groupby('PARTY')['TOTALVOTES'].sum().nlargest(5).index.tolist()

# Create a method to label parties as 'Other' if they are not top five in total number of votes.
def sort_party(data):
    '''
    Method to label parties as 'Other' if they are not top five in total number of votes.
    data: input rows
    '''
    if data['PARTY'] not in vote_share_top5:
        return 'Other'
    else:
        return data['PARTY']

# Apply the function to create 'Party New'
df['Party New'] = df.apply(sort_party, axis=1)


# Labeling parties with less than 10 candidates as others in original PARTY col
less_candidates = []
for i, j in df['PARTY'].value_counts().items():
    if j <= 10:
        less_candidates.append(i)

def small_party(data):
    if data in less_candidates:
        return 'Other'
    return data
df['PARTY'] = df['PARTY'].apply(small_party)


# --- 7. Preparing the data ---
# Define features (X) and target (y)
X = df.drop(['NAME', 'SYMBOL', 'WINNER', 'GENERALVOTES', 'POSTALVOTES', 'TOTALVOTES', 'OVER TOTAL ELECTORS IN CONSTITUENCY',
             'OVER TOTAL VOTES POLLED IN CONSTITUENCY', 'Party New', 'CONSTITUENCY'], axis=1)
y = df['WINNER']

# Identify categorical and numerical features in X
categorical = []
numerical = []
for label, content in X.items():
    if pd.api.types.is_string_dtype(content) or pd.api.types.is_categorical_dtype(content):
        X[label] = content.astype("category") # Ensure it's category type
        categorical.append(label)
    elif pd.api.types.is_numeric_dtype(content):
        numerical.append(label)

# Creating training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# --- 7.1 Scaling and Encoding ---
# Initializing the StandardScaler() and OrdinalEncoder()
scaler = MinMaxScaler()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) # Handle potential new categories in test set

# Encoding the categorical columns
for cat in categorical:
    # Fit encoder ONLY on training data
    encoder.fit(np.array(X_train[cat]).reshape(-1, 1))
    # Transform both training and test data
    X_train[cat] = encoder.transform(np.array(X_train[cat]).reshape(-1, 1))
    X_test[cat] = encoder.transform(np.array(X_test[cat]).reshape(-1, 1))

# Scaling the numerical dataset
# Fit scaler ONLY on training data
scaler.fit(X_train[numerical])
# Transform both training and test data
X_train[numerical] = scaler.transform(X_train[numerical])
X_test[numerical] = scaler.transform(X_test[numerical])

# Encoding the target
target_enc = LabelEncoder().fit(y_train)
y_train = target_enc.transform(y_train)
y_test = target_enc.transform(y_test)

# --- 7.2 Upsampling the dataset ---
print("Class distribution before SMOTE:", Counter(y_train))
oversample = SMOTE(random_state=42) # Add random_state for reproducibility
X_train, y_train = oversample.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train))


# --- 8. Machine learning model experimentation ---
np.random.seed(42)

# Create a dictionary of the models to experiment
models = {
          "Random Forest" : RandomForestClassifier(n_jobs=-1, random_state=42),
          "Support Vector Machines" : SVC(random_state=42),
          "K-Nearest Neighbors" : KNeighborsClassifier(n_jobs=-1),
          "Logistic Regression"  : LogisticRegression(random_state=42),
          "XG Boosting" : xgb.XGBClassifier(n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric='logloss'), # Added params as per warning
         # "Cat Boosting": CatBoostClassifier(random_state=42, verbose=0), # Added verbose=0 to silence CatBoost output
          "Gradient Boosting"  : GradientBoostingClassifier(random_state=42),
          "ADA Boosting" : AdaBoostClassifier(random_state=42),
         # "LG Boosting" : lgb.LGBMClassifier(random_state=42),
          "Decision Trees" : DecisionTreeClassifier(random_state=42)
         }
# NOTE: Commented out CatBoost and LGBM as they might require separate installation.
#       Added random_state=42 to models for reproducibility.
#       Added use_label_encoder=False and eval_metric='logloss' to XGBClassifier to address warnings.


# Create a function to fit and score the models
def fit_and_score(models, X_train, y_train, X_test, y_test):
    '''
    Fits and evaluates the machine learning models using cross-validation on train set
    and direct evaluation on the test set.
    models : a dict of different Scikit-Learn machine learning models.
    X_train : training data (no labels)
    y_train : training labels
    X_test : testing data (no labels)
    y_test : testing labels
    '''
    train_model_scores_acc = {}
    train_model_scores_pre = {}
    train_model_scores_rec = {}
    train_model_scores_f1 = {}

    test_model_scores_acc = {}
    test_model_scores_pre = {}
    test_model_scores_rec = {}
    test_model_scores_f1 = {}

    train_model_timing = {}

    # Use a consistent cross-validation strategy
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Using 5 folds for faster execution

    print("Starting Model Training and Evaluation...")
    for name, model in models.items():
        print(f"--- Evaluating {name} ---")
        start = time.time()
        # Cross-validation scores on the training set
        try:
            train_model_scores_acc[name] = np.mean(cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='accuracy')) * 100
            train_model_scores_pre[name] = np.mean(cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='precision')) * 100
            train_model_scores_rec[name] = np.mean(cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='recall')) * 100
            train_model_scores_f1[name] = np.mean(cross_val_score(model, X_train, y_train, cv=cv, n_jobs=-1, scoring='f1')) * 100
            stop = time.time()
            timing = stop - start
            train_model_timing[name] = timing
            print(f"Cross-validation for {name} completed in {timing:.2f} seconds.")

            # Fit on the full training data and evaluate on the test set
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            test_model_scores_acc[name] = accuracy_score(y_test, pred) * 100
            test_model_scores_pre[name] = precision_score(y_test, pred) * 100
            test_model_scores_rec[name] = recall_score(y_test, pred) * 100
            test_model_scores_f1[name] = f1_score(y_test, pred) * 100
            print(f"Test set evaluation for {name} completed.")

        except Exception as e:
             print(f"Could not evaluate {name}. Error: {e}")
             # Assign NaN or skip the model scores if evaluation fails
             train_model_scores_acc[name], train_model_scores_pre[name], train_model_scores_rec[name], train_model_scores_f1[name] = np.nan, np.nan, np.nan, np.nan
             test_model_scores_acc[name], test_model_scores_pre[name], test_model_scores_rec[name], test_model_scores_f1[name] = np.nan, np.nan, np.nan, np.nan
             train_model_timing[name] = np.nan


    print("Model Evaluation Finished.")
    # Create dictionary for scores
    scores = {'Training Accuracy': train_model_scores_acc,
              'Training Precision': train_model_scores_pre,
              'Training Recall': train_model_scores_rec,
              'Training F1': train_model_scores_f1,
              'Test Accuracy':  test_model_scores_acc,
              'Test Precision': test_model_scores_pre,
              'Test Recall': test_model_scores_rec,
              'Test F1': test_model_scores_f1,
              'Cross-Validation Timing (s)': train_model_timing # Updated name
             }

    # Create dataframe of scores
    scores_df = pd.DataFrame(scores)

    scores_df.rename_axis('Model', inplace=True)
    scores_df.reset_index(inplace=True)
    scores_df.sort_values('Test Accuracy', ascending=False, inplace=True)
    return scores_df

# --- 9. Evaluation ---
# Call the function to get scores
score_df = fit_and_score(models, X_train, y_train, X_test, y_test)

# Display the scores dataframe
print("\n--- Model Performance Comparison ---")
print(score_df)

# --- Select the best performing model based on Test F1 or Accuracy (e.g., XG Boosting from notebook output) ---
# Re-fitting the chosen model on the entire training data is good practice,
# but the fit_and_score function already does this before evaluating on the test set.
# We'll use the XGBoost model instance from the dictionary if needed for further analysis.
best_model_name = score_df.iloc[0]['Model'] # Get name of best model from sorted df
print(f"\nBest performing model based on Test Accuracy: {best_model_name}")
final_model = models[best_model_name]
# final_model.fit(X_train, y_train) # Already fitted inside fit_and_score
pred = final_model.predict(X_test)

# --- 9.1 Classification report ---
print("\n--- Classification Report for Best Model ---")
print(classification_report(y_test, pred))

# --- 9.2 Confusion matrix ---
print("\n--- Confusion Matrix for Best Model ---")
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, cmap= "Blues", annot=True, fmt='d') # Added fmt='d' for integer display
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- 9.3 ROC curve ---
print("\n--- ROC Curve for Best Model ---")
RocCurveDisplay.from_estimator(final_model, X_test, y_test)
plt.title(f'ROC Curve - {best_model_name}')
plt.show()

print("\n--- Feature Importance for Best Model (if applicable) ---")
# Feature importance plotting is specific to tree-based models like XGBoost, RandomForest, etc.
if hasattr(final_model, 'feature_importances_'):
    try:
        if isinstance(final_model, xgb.XGBClassifier):
             xgb.plot_importance(final_model)
             plt.title(f'Feature Importance - {best_model_name}')
             plt.show()
        else:
             # Generic importance plot for other tree models
             importances = final_model.feature_importances_
             indices = np.argsort(importances)[::-1]
             feature_names = X.columns # Get feature names from original X before scaling/encoding

             plt.figure(figsize=(10, 6))
             plt.title(f"Feature importances - {best_model_name}")
             plt.bar(range(X_train.shape[1]), importances[indices], align="center")
             plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
             plt.xlim([-1, X_train.shape[1]])
             plt.tight_layout()
             plt.show()
    except Exception as e:
        print(f"Could not plot feature importance for {best_model_name}: {e}")
else:
     print(f"{best_model_name} does not support feature importance plotting directly.")
