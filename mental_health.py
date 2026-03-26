import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('survey.csv')
print("Original data shape:", df.shape)

# 2. Column name easier handling
age_col = 'What is your age?'
gender_col = 'What is your gender?'
country_col = 'What country do you live in?'
remote_col = 'Do you work remotely?'
tech_company_col = 'Is your employer primarily a tech company/organization?'
no_employees_col = 'How many employees does your company or organization have?'
family_history_col = 'Do you have a family history of mental illness?'
treatment_col = 'Have you ever sought treatment for a mental health issue from a mental health professional?'
work_interfere_col = 'Do you believe your productivity is ever affected by a mental health issue?'
benefits_col = 'Does your employer provide mental health benefits as part of healthcare coverage?'
seek_help_col = 'Do you know local or online resources to seek help for a mental health disorder?'
anonymity_col = 'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?'
leave_col = 'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:'

# List of columns to use
used_cols = [age_col, gender_col, country_col, remote_col, tech_company_col,
             no_employees_col, family_history_col, treatment_col, work_interfere_col,
             benefits_col, seek_help_col, anonymity_col, leave_col]

# check list exist
existing_cols = [col for col in used_cols if col in df.columns]
print("Columns used:", existing_cols)
df = df[existing_cols].copy()

# 3. Data cleaning
# 3.1 Age outlier removal (keep 18-65)
df = df[df[age_col].between(18, 65)]

# 3.2 Gender standardisation
def clean_gender(g):
    g = str(g).lower().strip()
    if g in ['male', 'm']:
        return 'Male'
    elif g in ['female', 'f']:
        return 'Female'
    else:
        return 'Other'
df[gender_col] = df[gender_col].apply(clean_gender)

# 3.3 Handle missing values
binary_cols = [remote_col, tech_company_col, family_history_col, treatment_col,
               benefits_col, seek_help_col, anonymity_col]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'No')

if work_interfere_col in df.columns:
    df[work_interfere_col] = df[work_interfere_col].fillna(df[work_interfere_col].mode()[0] if not df[work_interfere_col].mode().empty else 'No')

if no_employees_col in df.columns:
    df[no_employees_col] = df[no_employees_col].fillna(df[no_employees_col].mode()[0] if not df[no_employees_col].mode().empty else '1-5')

if leave_col in df.columns:
    df[leave_col] = df[leave_col].fillna(df[leave_col].mode()[0] if not df[leave_col].mode().empty else "Don't know")

# 3.4 Map binary variables to 0/1
binary_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'Y': 1, 'N': 0}
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map).fillna(0)

# 3.5 Work interference mapping
if work_interfere_col in df.columns:
    work_map = {'Yes': 1, 'No': 0, 'Sometimes': 1, 'Often': 1, 'Rarely': 0}
    df[work_interfere_col] = df[work_interfere_col].map(work_map).fillna(0)

# 3.6 Leave support mapping
if leave_col in df.columns:
    leave_map = {'Very easy': 3, 'Somewhat easy': 2, 'Somewhat difficult': 1, 'Very difficult': 0, "Don't know": 1}
    df[leave_col] = df[leave_col].map(leave_map).fillna(1)

# 3.7 Company size mapping
emp_map = {
    '1-5': 1,
    '6-25': 2,
    '26-100': 3,
    '100-500': 4,
    '500-1000': 5,
    'More than 1000': 6
}
if no_employees_col in df.columns:
    df[no_employees_col] = df[no_employees_col].map(emp_map).fillna(3)

# 3.8 Country processing: keep top 10 countries, others as 'Other'
if country_col in df.columns:
    top_countries = df[country_col].value_counts().index[:10]
    df[country_col] = df[country_col].apply(lambda x: x if x in top_countries else 'Other')

# 4. Feature selection and encoding
feature_cols_numeric = [age_col, no_employees_col, leave_col, work_interfere_col] + binary_cols
feature_cols_numeric = [col for col in feature_cols_numeric if col in df.columns]
categorical_cols = [gender_col, country_col]

# Build feature matrix with one-hot encoding
df_encoded = pd.get_dummies(df[feature_cols_numeric + categorical_cols], columns=categorical_cols, drop_first=False)
df_encoded = df_encoded.fillna(0)

# 4.1 Remove constant columns 
constant_cols = [col for col in df_encoded.columns if df_encoded[col].std() == 0]
if constant_cols:
    print("Removing constant columns (no variance):", constant_cols)
    df_encoded = df_encoded.drop(columns=constant_cols)
else:
    print("No constant columns found.")

X = df_encoded.values
print("Final number of features:", X.shape[1])

# 5. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. PCA for visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", sum(pca.explained_variance_ratio_))

# 7. Determine optimal number of clusters
inertias = []
silhouettes = []
k_range = range(2, 9)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow and silhouette
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.subplot(1,2,2)
plt.plot(k_range, silhouettes, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150)
plt.show()

# Choose k 
best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 8. Visualise clusters in PCA space
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'KMeans Clustering (k={best_k}) on PCA-reduced Data')
plt.savefig('clusters_pca.png', dpi=150)
plt.show()

# 9. Cluster characterisation
df_scaled = pd.DataFrame(X_scaled, columns=df_encoded.columns)
df_scaled['cluster'] = labels
cluster_means_scaled = df_scaled.groupby('cluster').mean()
print("\nCluster means (standardised):")
print(cluster_means_scaled)

df_original_cluster = df_encoded.copy()
df_original_cluster['cluster'] = labels
cluster_means_original = df_original_cluster.groupby('cluster').mean()
print("\nCluster means (original scale):")
print(cluster_means_original)

# Save original scale cluster means to CSV
cluster_means_original.to_csv('cluster_means_original.csv', index=True)
print("\nCluster means saved to 'cluster_means_original.csv'")

# Cluster sizes
cluster_counts = df_original_cluster['cluster'].value_counts().sort_index()
print("\nCluster sizes:")
for i in range(best_k):
    print(f"Cluster {i}: {cluster_counts[i]} people ({cluster_counts[i]/len(df_original_cluster)*100:.1f}%)")

print("\nAnalysis complete. Figures saved as 'elbow_silhouette.png' and 'clusters_pca.png'.")
