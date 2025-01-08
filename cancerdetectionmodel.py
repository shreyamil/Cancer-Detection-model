#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# In[16]:


# Load your dataset (replace 'your_dataset.csv' with your actual file path)
dataset_path =r'C:\Users\shrey\Downloads\Cancer_Data.csv'

df = pd.read_csv(dataset_path)

# Explore the loaded data (e.g., check the first few rows)
print(df.head())


# In[17]:


# Load pre-trained ResNet50 model (without top classification layers)

base_model=df
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[22]:


# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)



# In[24]:


get_ipython().system('pip install xgboost')


# In[26]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Step 1: Load the Data
file_path = r'C:\Users\shrey\Downloads\Cancer_Data.csv'
data = pd.read_csv(file_path)

# Step 2: Preprocess the Data
# Drop unused columns
data.drop(columns=['id', 'Unnamed: 32'], inplace=True)

# Encode the target variable (diagnosis: M=1, B=0)
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Build and Evaluate Models

# 1. CNN Model for Tabular Data
def build_cnn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model(X_train.shape[1])
cnn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2)

# Evaluate CNN
cnn_preds = (cnn_model.predict(X_test) > 0.5).astype("int32")
print("CNN Accuracy:", accuracy_score(y_test, cnn_preds))
print("CNN ROC AUC Score:", roc_auc_score(y_test, cnn_preds))



# In[27]:


# Step 3: Build and Evaluate Models

# 1. CNN Model for Tabular Data
def build_cnn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model(X_train.shape[1])
cnn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_split=0.2)
# Evaluate CNN
cnn_preds = (cnn_model.predict(X_test) > 0.5).astype("int32")
print("CNN Accuracy:", accuracy_score(y_test, cnn_preds))
print("CNN ROC AUC Score:", roc_auc_score(y_test, cnn_preds))


# In[28]:


# 2. Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Random Forest ROC AUC Score:", roc_auc_score(y_test, rf_preds))



# In[30]:


from xgboost import XGBClassifier  # Import XGBClassifier

# 3. XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)  # Train the model
xgb_preds = xgb_model.predict(X_test)  # Make predictions

# Evaluate the model
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("XGBoost ROC AUC Score:", roc_auc_score(y_test, xgb_preds))


# In[31]:


# 4. Support Vector Machine (SVM)
svc_model = SVC(probability=True, kernel='linear', random_state=42)
svc_model.fit(X_train, y_train)
svc_preds = svc_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svc_preds))
print("SVM ROC AUC Score:", roc_auc_score(y_test, svc_preds))



# In[32]:


# Step 4: Find the Best Model
print("\nClassification Reports:\n")
print("CNN:\n", classification_report(y_test, cnn_preds))
print("Random Forest:\n", classification_report(y_test, rf_preds))
print("XGBoost:\n", classification_report(y_test, xgb_preds))
print("SVM:\n", classification_report(y_test, svc_preds))


# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Step 1: Load the Data
file_path = r'C:\Users\shrey\Downloads\Cancer_Data.csv' # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 2: Preprocess the Data
# Drop unused columns
data.drop(columns=['id', 'Unnamed: 32'], inplace=True)

# Encode the target variable (diagnosis: M=1, B=0)
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Models list for comparison
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Neural Network': Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
}

# Dictionary to store classification reports
model_reports = {}

# Train and evaluate each model
for model_name, model in models.items():
    if model_name == 'Neural Network':  # Special handling for Neural Network (Keras)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        nn_preds = (model.predict(X_test) > 0.5).astype('int32')  # Convert to binary predictions
        model_reports[model_name] = classification_report(y_test, nn_preds, output_dict=True)  # Store report
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        model_reports[model_name] = classification_report(y_test, preds, output_dict=True)  # Store report

# Extract Precision, Recall, and F1-Score from the classification reports
metrics = ['precision', 'recall', 'f1-score']
model_metrics = {metric: [] for metric in metrics}

# Fill in the metrics for each model
for model_name, report in model_reports.items():
    for metric in metrics:
        model_metrics[metric].append(report['1'][metric])  # Get metrics for class '1' (malignant)

# Plotting the metrics for each model
x = np.arange(len(models))  # The label locations for models
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot Precision, Recall, and F1-score for each model
ax.bar(x - width, model_metrics['precision'], width, label='Precision', color='blue')
ax.bar(x, model_metrics['recall'], width, label='Recall', color='green')
ax.bar(x + width, model_metrics['f1-score'], width, label='F1-Score', color='red')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Model')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison - Precision, Recall, and F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(models.keys())
ax.legend()

plt.tight_layout()
plt.show()


# In[ ]:




