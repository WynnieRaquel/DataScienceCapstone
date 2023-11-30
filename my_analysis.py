# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting and visualization
import seaborn as sns  # For advanced visualization
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For data scaling and label encoding
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.decomposition import PCA  # For principal component analysis
from sklearn.ensemble import RandomForestClassifier  # For the random forest classification algorithm
from sklearn.pipeline import Pipeline  # To create a processing pipeline
from sklearn.metrics import classification_report, davies_bouldin_score, calinski_harabasz_score, silhouette_score  # For model evaluation
from sklearn.cluster import KMeans  # For KMeans clustering
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


# Define paths to the datasets to streamline the data loading process
# These paths point to different demographic datasets for the analysis
paths = {
    "US": r"C:\Users\Wynette Vickers\Documents\Classes\Grad Courses\Fall23\DSCI602\CFD\CFD Version 3.0\CFD U.S. Norming Data.csv",
    "India": r"C:\Users\Wynette Vickers\Documents\Classes\Grad Courses\Fall23\DSCI602\CFD\CFD Version 3.0\CFD-I INDIA Norming Data.csv",
    "MR": r"C:\Users\Wynette Vickers\Documents\Classes\Grad Courses\Fall23\DSCI602\CFD\CFD Version 3.0\CFD-MR U.S. Norming Data.csv",
    "I_US": r"C:\Users\Wynette Vickers\Documents\Classes\Grad Courses\Fall23\DSCI602\CFD\CFD Version 3.0\CFD-I U.S. Norming Data.csv"
}

# Function to load and preprocess the dataset from a given path
# This includes reading the CSV file and performing initial data cleaning and formatting
def load_and_preprocess_dataset(path):
    df = pd.read_csv(path, skiprows=7)  # Skipping initial rows that might contain metadata or headers
    return df

# Load and preprocess individual datasets
# This process involves reading the CSV files and doing preliminary data cleaning
df_cfd_us = load_and_preprocess_dataset(paths["US"])
df_cfd_india = load_and_preprocess_dataset(paths["India"])
df_cfd_mr = load_and_preprocess_dataset(paths["MR"])
df_cfd_i_us = load_and_preprocess_dataset(paths["I_US"])

# Concatenate individual datasets to create a comprehensive dataset
# This combined dataset will be used for subsequent analysis
combined_dataset = pd.concat([df_cfd_us, df_cfd_india, df_cfd_mr, df_cfd_i_us])

# Handling missing values specifically in numeric columns
# We compute the mean of each column and use these values to fill in missing data
# This step is crucial to ensure the integrity of our dataset for modeling
numeric_cols = combined_dataset.select_dtypes(include=np.number).columns
numeric_means = combined_dataset[numeric_cols].mean()
combined_dataset[numeric_cols] = combined_dataset[numeric_cols].fillna(numeric_means)


# Encoding categorical columns to transform non-numeric data into numeric
# This is essential for ML models as they require numerical input
def encode_and_print_key(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Printing the encoding key helps in interpreting model predictions
    print("Encoding Key:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    return y_encoded

y_ethnicity_encoded = encode_and_print_key(combined_dataset['EthnicitySelf'])
y_gender_encoded = encode_and_print_key(combined_dataset['GenderSelf'])

# Selecting specific features for the prediction model
# These features are chosen based on prior analysis indicating their significance
features_for_prediction = [
    'LuminanceMedian', 'NoseWidth', 'NoseLength', 'LipThickness', 'FaceLength',
    'EyeHeightR', 'EyeHeightL', 'EyeHeightAvg', 'EyeWidthR', 'EyeWidthL',
    'EyeWidthAvg', 'FaceWidthCheeks', 'FaceWidthMouth', 'FaceWidthBZ', 'Forehead',
    'UpperFaceLength2', 'PupilTopR', 'PupilTopL', 'PupilTopAsymmetry', 'PupilLipR',
    'PupilLipL', 'PupilLipAvg', 'PupilLipAsymmetry', 'BottomLipChin', 'MidcheekChinR',
    'MidcheekChinL', 'CheeksAvg', 'MidbrowHairlineR', 'MidbrowHairlineL',
    'MidbrowHairlineAvg', 'FaceShape', 'Heartshapeness', 'NoseShape', 'LipFullness',
    'EyeShape', 'EyeSize', 'UpperHeadLength', 'MidfaceLength', 'ChinLength',
    'ForeheadHeight', 'CheekboneHeight', 'CheekboneProminence', 'FaceRoundness'
]

# Extracting the selected features from the dataset for modeling purposes
X = combined_dataset[features_for_prediction]

# Splitting the dataset into training and validation sets for the ethnicity prediction model
X_train_ethnicity, X_val_ethnicity, y_train_ethnicity, y_val_ethnicity = train_test_split(X, y_ethnicity_encoded, test_size=0.3, random_state=42)

# Applying SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train_ethnicity, y_train_ethnicity)

# Defining a pipeline for preprocessing and modeling for ethnicity prediction
# Pipelines streamline the process of data transformation and model fitting
ethnicity_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Function to evaluate and print model performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

# Hyperparameter tuning using RandomizedSearchCV to find optimal model parameters
# RandomizedSearchCV samples a fixed number of parameter settings from the distributions
# This approach can significantly reduce computation time and is effective in high-dimensional spaces
param_distributions = {
    'model__n_estimators': randint(100, 200),  # Randomly sample values between 100 and 200
    'model__max_depth': [10, 15, 20, None],  # Sample from these options
    'model__min_samples_split': randint(2, 11),  # Randomly sample values between 2 and 10
    'model__min_samples_leaf': randint(1, 5)  # Randomly sample values between 1 and 4
}

# Random search for hyperparameter tuning with specified number of iterations for the ethnicity model
random_search_ethnicity = RandomizedSearchCV(ethnicity_pipeline, param_distributions, n_iter=50, cv=5, random_state=42)
random_search_ethnicity.fit(X_resampled, y_resampled)
best_ethnicity_model = random_search_ethnicity.best_estimator_

# Evaluate the best ethnicity model
evaluate_model(best_ethnicity_model, X_val_ethnicity, y_val_ethnicity)

# Repeating similar steps for the gender prediction model
X_train_gender, X_val_gender, y_train_gender, y_val_gender = train_test_split(X, y_gender_encoded, test_size=0.3, random_state=42)

gender_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Random search for the gender model using the same parameter distributions
random_search_gender = RandomizedSearchCV(gender_pipeline, param_distributions, n_iter=50, cv=5, random_state=42)
random_search_gender.fit(X_train_gender, y_train_gender)
best_gender_model = random_search_gender.best_estimator_

# Evaluate the best gender model
evaluate_model(best_gender_model, X_val_gender, y_val_gender)

# Scaling features using StandardScaler for clustering analysis
# Feature scaling is necessary before clustering to ensure all features contribute equally
scaled_features = StandardScaler().fit_transform(combined_dataset[features_for_prediction])

# Applying PCA (Principal Component Analysis) to reduce the dimensionality of the feature set
# PCA is used here to simplify the dataset into 2 principal components for effective clustering
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Performing KMeans clustering on the PCA-reduced features
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(reduced_features)

# Evaluating KMeans clustering using Davies-Bouldin, Calinski-Harabasz, and Silhouette scores
# These metrics provide insights into the cluster quality and separation
db_index = davies_bouldin_score(reduced_features, clusters)
ch_index = calinski_harabasz_score(reduced_features, clusters)
silhouette_avg = silhouette_score(reduced_features, clusters)
print(f"KMeans with 2 PCA components - Davies-Bouldin Index: {db_index}, Calinski-Harabasz Index: {ch_index}, Silhouette Score: {silhouette_avg}")

# Visualizing the clusters using a scatter plot
# This visualization helps in understanding how the data points are grouped in the reduced feature space
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=clusters, palette="viridis")
plt.title("KMeans Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Append cluster labels to the original dataset for in-depth analysis
# This allows me to explore how different demographic attributes are distributed across clusters
combined_dataset_clustered = combined_dataset.copy()
combined_dataset_clustered['Cluster'] = clusters

# Creating crosstabs to analyze the distribution of ethnicity and gender in the formed clusters
# Crosstabs provide a simple way to observe the relationship between categorical variables
ethnicity_crosstab = pd.crosstab(combined_dataset_clustered['Cluster'], combined_dataset_clustered['EthnicitySelf'])
gender_crosstab = pd.crosstab(combined_dataset_clustered['Cluster'], combined_dataset_clustered['GenderSelf'])

print("KMeans Clustering - Ethnicity Distribution:\n", ethnicity_crosstab)
print("KMeans Clustering - Gender Distribution:\n", gender_crosstab)

# Function to plot a heatmap of the cluster distribution
# Heatmaps are effective in visualizing the concentration of categories within clusters
def plot_cluster_distribution(crosstab, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(title)
    plt.ylabel('Cluster')
    plt.xlabel('Category')
    plt.show()

# Plotting the distribution of ethnicity and gender for the KMeans clusters
# These plots will help in identifying any biases or trends in the clustering process
plot_cluster_distribution(ethnicity_crosstab, "KMeans - Ethnicity Distribution")
plot_cluster_distribution(gender_crosstab, "KMeans - Gender Distribution")


# --- Confusion Matrix for Ethnicity Model ---
# Predicting the ethnicity categories using the best ethnicity model
y_pred_ethnicity = best_ethnicity_model.predict(X_val_ethnicity)

# Generating the confusion matrix for ethnicity predictions
# This matrix compares the actual and predicted categories
cm_ethnicity = confusion_matrix(y_val_ethnicity, y_pred_ethnicity)

# Mapping the numeric labels to actual ethnicity categories for better readability
ethnicity_labels = ['A', 'B', 'I', 'L', 'M', 'W']
cm_ethnicity_df = pd.DataFrame(cm_ethnicity, index=ethnicity_labels, columns=ethnicity_labels)

# Visualizing the confusion matrix using a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_ethnicity_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Ethnicity Model')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

# --- Confusion Matrix for Gender Model ---
# Predicting the gender categories using the best gender model
y_pred_gender = best_gender_model.predict(X_val_gender)

# Generating the confusion matrix for gender predictions
cm_gender = confusion_matrix(y_val_gender, y_pred_gender)

# Mapping the numeric labels to actual gender categories for better readability
gender_labels = ['F', 'M']  # Female and Male
cm_gender_df = pd.DataFrame(cm_gender, index=gender_labels, columns=gender_labels)

# Visualizing the confusion matrix using a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_gender_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Gender Model')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')
plt.show()

# --- Feature Importance Plot for Ethnicity Model ---
# Extracting feature importances from the best ethnicity model
importances_ethnicity = best_ethnicity_model.named_steps['model'].feature_importances_
indices_ethnicity = np.argsort(importances_ethnicity)[::-1]

# Visualizing the feature importances in a bar chart
# The most important features will be highlighted
plt.figure(figsize=(15, 5))
plt.title("Feature Importances in Ethnicity Model")
plt.bar(range(X_train_ethnicity.shape[1]), importances_ethnicity[indices_ethnicity], align="center")
plt.xticks(range(X_train_ethnicity.shape[1]), X_train_ethnicity.columns[indices_ethnicity], rotation=90)
plt.xlim([-1, X_train_ethnicity.shape[1]])
plt.show()

# --- Feature Importance Plot for Gender Model ---
# Extracting feature importances from the best gender model
importances_gender = best_gender_model.named_steps['model'].feature_importances_
indices_gender = np.argsort(importances_gender)[::-1]

# Visualizing the feature importances in a bar chart
# This helps in understanding which features most influence gender predictions
plt.figure(figsize=(15, 5))
plt.title("Feature Importances in Gender Model")
plt.bar(range(X_train_gender.shape[1]), importances_gender[indices_gender], align="center")
plt.xticks(range(X_train_gender.shape[1]), X_train_gender.columns[indices_gender], rotation=90)
plt.xlim([-1, X_train_gender.shape[1]])
plt.show()
