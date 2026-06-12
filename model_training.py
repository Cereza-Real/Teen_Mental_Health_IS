import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# 1. Load Data
df = pd.read_csv('Teen_Mental_Health_Dataset.csv')

target_col = 'depression_label'

categorical_cols = [
    'gender',
    'platform_usage',
    'social_interaction_level'
]

numerical_cols = [
    'age',
    'daily_social_media_hours',
    'sleep_hours',
    'screen_time_before_sleep',
    'academic_performance',
    'physical_activity',
    'stress_level',
    'anxiety_level',
    'addiction_level'
]

X = df.drop(columns=[target_col])
y = df[target_col]


# 2. Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Preprocessing
#numerical
num_transformer = SklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # handle missing/null values with median
    ('scaler', StandardScaler()) # normalization, all data weighs the same
])

#categorical
cat_transformer = SklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # handle missing/null values with most freq category
    ('encoder', OneHotEncoder(handle_unknown='ignore')) # changing categorical values into numerical
])

#merges both numerical and categorical data into one table
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])


print("SMOTE")
print("__________________________")
original_counts = y_train.value_counts()
print("Original Training Data Distribution:")
for cls, count in original_counts.items():
    print(f"  Class {cls}: {count} samples")

X_train_preprocessed = preprocessor.fit_transform(X_train)
smote_test = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote_test.fit_resample(X_train_preprocessed, y_train)

smote_counts = pd.Series(y_train_smote).value_counts()
print("\nPost-SMOTE Training Data Distribution:")
for cls, count in smote_counts.items():
    print(f"  Class {cls}: {count} samples")
print("__________________________\n")


# 4. Function to Train Models
def train_model(model_name, classifier, param_grid):
    print(f"\n--- OPTIMIZING {model_name} ---")

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)

    print(f"Best {model_name} Settings Found:")
    print(grid.best_params_)

    print(f"\n{model_name} Results:")
    print(classification_report(y_test, predictions))

    macro_f1 = f1_score(y_test, predictions, average='macro')

    return {
        'name': model_name,
        'grid': grid,
        'model': best_model,
        'predictions': predictions,
        'f1_macro': macro_f1
    }


# 5. Random Forest
rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

rf_result = train_model(
    'Random Forest',
    RandomForestClassifier(random_state=42),
    rf_param_grid
)


# 6. Gradient Boosting
gb_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

gb_result = train_model(
    'Gradient Boosting',
    GradientBoostingClassifier(random_state=42),
    gb_param_grid
)


# 7. Decision Tree
dt_param_grid = {
    'classifier__max_depth': [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__criterion': ['gini', 'entropy']
}

dt_result = train_model(
    'Decision Tree',
    DecisionTreeClassifier(random_state=42),
    dt_param_grid
)


# 8. Choose Best Model
results = [rf_result, gb_result, dt_result]

best_result = max(results, key=lambda x: x['f1_macro'])

best_model_name = best_result['name']
best_pipeline = best_result['model']
final_pred = best_result['predictions']
best_f1_score = best_result['f1_macro']

print("\n==============================================")
print(f"Winning Model: {best_model_name}")
print(f"Top Macro F1-Score: {best_f1_score:.4f}")
print("==============================================\n")


# 9. Save Best Full Pipeline
joblib.dump(best_pipeline, 'best_mental_health_model.pkl')
print("Model saved to best_mental_health_model.pkl")


# 10. Confusion Matrix
plt.figure(figsize=(8, 6))

safe_y_test = y_test.astype(int)
safe_pred = final_pred.astype(int)

cm = confusion_matrix(safe_y_test, safe_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Depressed', 'Depressed'],
    yticklabels=['Not Depressed', 'Depressed']
)

plt.title(f'Confusion Matrix: {best_model_name}')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("Saved confusion_matrix.png")


# 11. Feature Importance
classifier = best_pipeline.named_steps['classifier']

if hasattr(classifier, 'feature_importances_'):
    plt.figure(figsize=(10, 8))

    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = classifier.feature_importances_

    indices = np.argsort(importances)

    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f'Feature Importances: {best_model_name}')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    print("Saved feature_importance.png")
else:
    print("Feature importance is not available for this model.")


# --- NEW: 12. VISUALIZE SMOTE EFFECTS ---
# Map labels to readable format for the graph
label_mapping = {0: 'Not Depressed', 1: 'Depressed'}

df_before = pd.DataFrame({'Class': original_counts.index.map(label_mapping), 'Count': original_counts.values, 'Stage': 'Before SMOTE'})
df_after = pd.DataFrame({'Class': smote_counts.index.map(label_mapping), 'Count': smote_counts.values, 'Stage': 'After SMOTE'})
df_plot = pd.concat([df_before, df_after])

plt.figure(figsize=(8, 5))
sns.barplot(x='Class', y='Count', hue='Stage', data=df_plot, palette='Set2')
plt.title('Training Data Class Balance: Before vs After SMOTE')
plt.ylabel('Number of Samples')
plt.xlabel('Mental Health Label')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('smote_effects.png')
plt.close()

print("Saved smote_effects.png")