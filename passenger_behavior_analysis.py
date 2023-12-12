import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Improved settings for data visualization
sns.set(style="whitegrid", color_codes=True)
plt.rc("font", size=14)

# Load and preprocess data
data_path = 'path/to/data.csv'  # Updated for confidentiality
data = pd.read_csv(data_path).dropna()
data.drop(['Unnamed: 0', 'Sensitive_Column_1'], axis=1, inplace=True)  # Anonymized column names
print(data.shape)
print(data.dtypes)

# Exploratory Data Analysis
def plot_count(data, column, fig_size=(8, 4), save_as=None):
    plt.figure(figsize=fig_size)
    sns.countplot(x=column, data=data, palette='hls')
    plt.show()
    if save_as:
        plt.savefig(f'{save_as}.png')

plot_count(data, 'PAX_LOYALTY_FLAG', save_as='loyalty_flag_count')
plot_count(data, 'PAX_GENDER_CODE', save_as='gender_count')
# ... Add more plots as needed ...

# Passenger Profile by Product
com_count = pd.DataFrame(data['COMPONENT_NAME_ADJ'].value_counts())
com_name = data.groupby('COMPONENT_NAME_ADJ').mean()
com_count.index.name = 'COMPONENT_NAME_ADJ'
com_count.rename(columns={'COMPONENT_NAME_ADJ': 'Count'}, inplace=True)
com_name_total = pd.merge(com_name, com_count, on="COMPONENT_NAME_ADJ")
print(com_name_total)
com_name_total.to_csv('path/to/modified_com_name.csv')  # Updated for confidentiality

# Feature Selection and Preprocessing
data_pass = pd.read_csv('path/to/another_data.csv').dropna()  # Updated for confidentiality
data_pass.drop(['Unnamed: 0', 'Sensitive_Column_2'], axis=1, inplace=True)  # Anonymized column names

y_col = 'Target_Column'  # Anonymized target column
X_cols = data_pass.columns.difference([y_col])

scaler = StandardScaler()
data_pass_scaled = pd.DataFrame(scaler.fit_transform(data_pass), columns=data_pass.columns)
data_pass_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
data_pass_scaled.fillna(data_pass.mean(), inplace=True)

y = data_pass_scaled[y_col]
X = data_pass_scaled[X_cols]

# Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print(f'Accuracy: {logreg.score(X_test, y_test):.2f}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC Curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (area = {logit_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC.png')
plt.show()
