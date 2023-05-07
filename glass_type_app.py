import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)   

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
def prediction(model, features):
  glass_type = model.predict([features])
  if glass_type[0] ==1:
    return 'building windows float processed'.upper()
  elif glass_type[0] ==2:
    return 'building windows non float processed'.upper()
  elif glass_type[0] ==3:
    return 'vehicle windows float processed'.upper()
  elif glass_type[0]==4:
    return 'vehicle windows non float processed'.upper()
  elif glass_type[0] == 5:
    return 'containers'.upper()
  elif glass_type[0] ==6:
    return 'tableware'.upper()
  else:
    return 'headlamp'.upper()

st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Analysis')

# S5.1: Using the 'if' statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox('show raw data'):

  st.subheader('Glass type Data Set')

  st.write(glass_df)

st.sidebar.subheader('scatter plot')
features1 = st.sidebar.multiselect('select the x-axis values', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features1:
  st.subheader(f'scatter plot between {i} and glasstype')
  plt.figure(figsize=(18,7))
  sns.scatterplot(x=i, y='GlassType', data = glass_df)
  st.pyplot()

st.sidebar.subheader('Visualisation Selector')
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
plotypes = st.sidebar.multiselect('Select the Charts/Plots', ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
if 'Histogram' in plotypes:
  st.sidebar.subheader('histogram')
  features2 = st.sidebar.selectbox('select feature for histogram', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe')) 
  plt.figure(figsize=(18,7))
  plt.hist(glass_df[features2], bins='sturges', edgecolor='g')
  plt.title(f'Histogram for {features2}')
  st.pyplot()


if 'Box Plot' in plotypes:
  st.sidebar.subheader('Box Plot')
  features2 = st.sidebar.selectbox('select feature for Box Plot', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe')) 
  plt.figure(figsize=(20,2))
  sns.boxplot(glass_df[features2])
  plt.title(f'Box Plot for {features2}')
  st.pyplot()

if 'Count Plot' in plotypes:
  st.sidebar.subheader('Count Plot')
  features2 = st.sidebar.selectbox('select feature for Count Plot', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe')) 
  plt.figure(figsize=(20,5))
  sns.countplot(x = glass_df[features2])
  plt.title(f'Count Plot for {features2}')
  st.pyplot()

if 'Pie Chart' in plotypes:
  st.sidebar.subheader('Pie Chart')
  features2 = st.sidebar.selectbox('select feature for Pie Chart', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe')) 
  plt.figure(figsize=(20,5))
  plt.pie(glass_df[features2].value_counts())
  plt.title(f'Pie Chart for {features2}')
  st.pyplot()

if 'Correlation Heatmap' in plotypes:
  st.sidebar.subheader('Correlation Heatmap')
  plt.figure(figsize=(20,5))
  sns.heatmap(glass_df.corr(), annot=True)
  plt.title(f'Correlation Heatmap')
  st.pyplot()

if 'Pair Plot' in plotypes:
  st.sidebar.subheader('Pair Plot')
  plt.figure(figsize=(15,15))
  sns.pairplot(glass_df)
  plt.title(f'Pair Plot')
  st.pyplot()
st.sidebar.subheader("Select your values:")
ri = st.sidebar.slider("Input Ri", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))
k =  st.sidebar.slider("Input K", float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))



st.sidebar.subheader("Choose Classifier")

# Add a selectbox in the sidebar with label 'Classifier'.
# and with 2 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
# Store the current value of this slider in a variable 'classifier'.
classifier = st.sidebar.selectbox("Classifier", 
                                 ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

# Implement SVM with hyperparameter tuning
# if classifier == 'Support Vector Machine', ask user to input the values of 'C','kernel' and 'gamma'.
if classifier == 'Support Vector Machine':
    st.sidebar.subheader("Model Hyperparameters")
    c_value = st. sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
    kernel_input = st.sidebar.radio("Kernel", ("linear", "rbf", "poly"))
    gamma_input = st. sidebar.number_input("Gamma", 1, 100, step = 1)

    if st.sidebar.button('Classify'):
        st.subheader("Support Vector Machine")
        svc_model=SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = svc_model.score(X_test, y_test)
        glass_type = prediction(svc_model, [ri, na, mg, al, si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(svc_model, X_test, y_test)
        st.pyplot()
if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)
        
    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, [ri, na, mg, al, si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        plot_confusion_matrix(rf_clf, X_test, y_test)
        st.pyplot()

# S1.1: Implement Logistic Regression with hyperparameter tuning
if classifier == 'Logistic Regression':
  st.sidebar.subheader('Model Hyperparameters:')
  C = st.sidebar.number_input('C', 1,100, step = 10)
  max_iter = st.sidebar.number_input('max_iteration', 10,1000, step = 10)
  if st.sidebar.button('classify'):
    lg = LogisticRegression(C = C, max_iter = max_iter)
    lg.fit(X_train, y_train)
    score = lg.score(X_train, y_train)
    pred = prediction(lg, [ri, na, mg, al, si, k, ca, ba, fe])
    st.write("The Type of glass predicted is:", pred)
    st.write("Accuracy", score.round(2))
    plot_confusion_matrix(lg, X_test, y_test)
    st.pyplot()