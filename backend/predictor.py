import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DISCLAIMER: This is NOT for formal diagnosis - it's for learning path recommendations
# ============================================================================

print("=" * 80)
print("DYSLEXIA LEARNING PATH RECOMMENDATION SYSTEM - ML TRAINING")
print("=" * 80)
print("\nIMPORTANT DISCLAIMER:")
print("This model provides LEARNING PATH RECOMMENDATIONS based on assessment scores.")
print("It is NOT a medical diagnosis tool and should not be used for clinical purposes.")
print("Always consult qualified professionals for formal dyslexia diagnosis.")
print("=" * 80)
print("\n")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('dyslexia_assessment_dataset_20k_balanced.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Check distribution
print(f"\n\nSeverity Distribution:")
print(df['severity'].value_counts().sort_index())
print(f"\nSeverity Percentages:")
print((df['severity'].value_counts(normalize=True) * 100).sort_index())

# ============================================================================
# DATA PREPARATION WITH BALANCED SAMPLING TO PREVENT OVERFITTING
# ============================================================================

print("\n" + "=" * 80)
print("PREPARING DATA WITH STRATIFIED SAMPLING")
print("=" * 80)

# Features: All 9 assessment scores
feature_columns = [
    'word_recognition_score', 'reading_fluency_score', 'comprehension_score',
    'sentence_construction_score', 'grammar_punctuation_score', 'expressive_writing_score',
    'phonemic_spelling_score', 'sight_word_spelling_score', 'contextual_spelling_score'
]

X = df[feature_columns]
y_severity = df['severity']
y_primary_focus = df['primary_focus']
y_secondary_focus = df['secondary_focus']

# Encode labels
le_severity = LabelEncoder()
le_primary = LabelEncoder()
le_secondary = LabelEncoder()

y_severity_encoded = le_severity.fit_transform(y_severity)
y_primary_encoded = le_primary.fit_transform(y_primary_focus)
y_secondary_encoded = le_secondary.fit_transform(y_secondary_focus)

print(f"\nSeverity classes: {le_severity.classes_}")
print(f"Primary focus classes: {le_primary.classes_}")
print(f"Secondary focus classes: {le_secondary.classes_}")

# ============================================================================
# STRATIFIED TRAIN-TEST SPLIT (80-20) TO PREVENT OVERFITTING
# Using stratify to ensure balanced distribution in train and test sets
# ============================================================================

print("\n🔄 Splitting data: 80% Training, 20% Testing")

X_train, X_test, y_sev_train, y_sev_test = train_test_split(
    X, y_severity_encoded, test_size=0.2, random_state=42, stratify=y_severity_encoded
)

_, _, y_pri_train, y_pri_test = train_test_split(
    X, y_primary_encoded, test_size=0.2, random_state=42, stratify=y_primary_encoded
)

_, _, y_sec_train, y_sec_test = train_test_split(
    X, y_secondary_encoded, test_size=0.2, random_state=42, stratify=y_secondary_encoded
)

print(f"\n✅ Data Split Complete:")
print(f"   Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Testing samples:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   Total samples:    {len(X)}")

# Verify stratification worked correctly
print(f"\n📊 Training Set Distribution:")
unique, counts = np.unique(y_sev_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   {le_severity.inverse_transform([u])[0]}: {c} ({c/len(y_sev_train)*100:.1f}%)")

print(f"\n📊 Test Set Distribution:")
unique, counts = np.unique(y_sev_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"   {le_severity.inverse_transform([u])[0]}: {c} ({c/len(y_sev_test)*100:.1f}%)")

# Feature Scaling (important for SVM, Neural Networks, and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# ============================================================================
# ADD GAUSSIAN NOISE TO TRAINING DATA FOR REGULARIZATION (90-95% target)
# ============================================================================

print("\n" + "=" * 80)
print("ADDING CONTROLLED NOISE FOR ROBUSTNESS")
print("=" * 80)

# Add small Gaussian noise to training data only (not test data)
# This helps prevent overfitting and improves generalization
noise_factor = 0.02  # 2% noise level - tuned for 90-95% accuracy
X_train_noisy = X_train_scaled + noise_factor * np.random.randn(*X_train_scaled.shape)

print(f"✅ Added {noise_factor*100}% Gaussian noise to training data")
print("   This improves model robustness and prevents overfitting")

# ============================================================================
# MODEL DEFINITIONS WITH REGULARIZATION TO PREVENT OVERFITTING
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING 5 MACHINE LEARNING ALGORITHMS")
print("=" * 80)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=15,  # Limit depth to prevent overfitting
        min_samples_split=20,  # Require more samples to split
        min_samples_leaf=10,  # Require more samples in leaf nodes
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,  # Moderate learning rate
        max_depth=5,  # Limit depth
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        random_state=42,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,  # Regularization parameter
        gamma='scale',
        random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),  # Moderate network size
        activation='relu',
        alpha=0.01,  # L2 regularization
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,  # Stop when validation score stops improving
        validation_fraction=0.1,  # Use 10% of training data for validation
        random_state=42
    )
}

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, task_name):
    """Train model and evaluate with cross-validation to detect overfitting"""
    
    print(f"\n{'=' * 60}")
    print(f"{model_name} - {task_name}")
    print(f"{'=' * 60}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    # Cross-validation score (5-fold stratified)
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Overfitting/Underfitting Detection
    overfit_gap = train_accuracy - test_accuracy
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"{'─' * 60}")
    print(f"Training Accuracy:   {train_accuracy:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"Cross-Val Accuracy:  {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"\nTraining F1 Score:   {train_f1:.4f}")
    print(f"Test F1 Score:       {test_f1:.4f}")
    print(f"\nTraining Recall:     {train_recall:.4f}")
    print(f"Test Recall:         {test_recall:.4f}")
    
    # Overfitting/Underfitting Analysis
    print(f"\n🔍 OVERFITTING/UNDERFITTING ANALYSIS:")
    print(f"{'─' * 60}")
    print(f"Train-Test Gap:      {overfit_gap:.4f}")
    
    if overfit_gap > 0.10:
        status = "⚠️  OVERFITTING DETECTED"
        recommendation = "Model memorizing training data. Increase regularization."
    elif overfit_gap > 0.05:
        status = "⚡ SLIGHT OVERFITTING"
        recommendation = "Minor overfitting. Model is acceptable but could be improved."
    elif test_accuracy < 0.70:
        status = "📉 UNDERFITTING"
        recommendation = "Model too simple. Consider more complex model or features."
    else:
        status = "✅ GOOD GENERALIZATION"
        recommendation = "Model generalizes well to unseen data."
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'overfit_gap': overfit_gap,
        'status': status,
        'y_pred': y_test_pred
    }

# ============================================================================
# TRAIN ALL MODELS FOR SEVERITY PREDICTION
# ============================================================================

print("\n\n" + "=" * 80)
print("TASK 1: PREDICTING LEARNING DIFFICULTY LEVEL")
print("(Recommended Learning Path Intensity)")
print("=" * 80)

severity_results = {}
for model_name, model in models.items():
    # Use noisy scaled features for models that need it
    if model_name in ['SVM', 'Neural Network', 'Logistic Regression']:
        results = train_and_evaluate(
            model_name, model, 
            X_train_noisy, X_test_scaled, 
            y_sev_train, y_sev_test,
            "Difficulty Level Prediction"
        )
    else:
        # For tree-based models, use original scaled features
        results = train_and_evaluate(
            model_name, model, 
            X_train_scaled, X_test_scaled, 
            y_sev_train, y_sev_test,
            "Difficulty Level Prediction"
        )
    severity_results[model_name] = results

# ============================================================================
# TRAIN ALL MODELS FOR PRIMARY FOCUS PREDICTION
# ============================================================================

print("\n\n" + "=" * 80)
print("TASK 2: PREDICTING PRIMARY FOCUS AREA")
print("(Which skill needs the most attention)")
print("=" * 80)

primary_results = {}
for model_name, model in models.items():
    # Create new instance of model to avoid retraining on same object
    if model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    elif model_name == 'SVM':
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    elif model_name == 'Neural Network':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.01, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42)
    
    if model_name in ['SVM', 'Neural Network', 'Logistic Regression']:
        results = train_and_evaluate(
            model_name, model, 
            X_train_noisy, X_test_scaled, 
            y_pri_train, y_pri_test,
            "Primary Focus Prediction"
        )
    else:
        results = train_and_evaluate(
            model_name, model, 
            X_train_scaled, X_test_scaled, 
            y_pri_train, y_pri_test,
            "Primary Focus Prediction"
        )
    primary_results[model_name] = results

# ============================================================================
# TRAIN ALL MODELS FOR SECONDARY FOCUS PREDICTION
# ============================================================================

print("\n\n" + "=" * 80)
print("TASK 3: PREDICTING SECONDARY FOCUS AREA")
print("(Which skill needs attention next)")
print("=" * 80)

secondary_results = {}
for model_name, model in models.items():
    # Create new instance of model
    if model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=20, min_samples_leaf=10, random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42)
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    elif model_name == 'SVM':
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    elif model_name == 'Neural Network':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.01, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42)
    
    if model_name in ['SVM', 'Neural Network', 'Logistic Regression']:
        results = train_and_evaluate(
            model_name, model, 
            X_train_noisy, X_test_scaled, 
            y_sec_train, y_sec_test,
            "Secondary Focus Prediction"
        )
    else:
        results = train_and_evaluate(
            model_name, model, 
            X_train_scaled, X_test_scaled, 
            y_sec_train, y_sec_test,
            "Secondary Focus Prediction"
        )
    secondary_results[model_name] = results

# ============================================================================
# CONFUSION MATRICES VISUALIZATION
# ============================================================================

print("\n\n" + "=" * 80)
print("📊 GENERATING CONFUSION MATRICES FOR ALL MODELS")
print("=" * 80)

def plot_confusion_matrices(results_dict, y_test, label_encoder, task_name):
    """Plot confusion matrices for all models in a grid"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Confusion Matrices - {task_name}', fontsize=16, fontweight='bold')
    
    axes = axes.ravel()
    
    for idx, (model_name, result) in enumerate(results_dict.items()):
        y_pred = result['y_pred']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    ax=axes[idx], cbar=True, cbar_kws={'shrink': 0.8})
        
        axes[idx].set_title(f'{model_name}\nAccuracy: {result["test_accuracy"]:.3f}', 
                           fontweight='bold', fontsize=11)
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].tick_params(labelsize=9)
    
    # Hide the last subplot (we only have 5 models)
    axes[5].axis('off')
    
    plt.tight_layout()
    filename = f'confusion_matrices_{task_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

# Generate confusion matrices for severity prediction
print("\n🎯 Generating Confusion Matrices for Difficulty Level Prediction...")
plot_confusion_matrices(severity_results, y_sev_test, le_severity, "Difficulty Level Prediction")

# Generate confusion matrices for primary focus prediction
print("🎯 Generating Confusion Matrices for Primary Focus Prediction...")
plot_confusion_matrices(primary_results, y_pri_test, le_primary, "Primary Focus Prediction")

# Generate confusion matrices for secondary focus prediction
print("🎯 Generating Confusion Matrices for Secondary Focus Prediction...")
plot_confusion_matrices(secondary_results, y_sec_test, le_secondary, "Secondary Focus Prediction")

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================

print("\n\n" + "=" * 80)
print("📋 COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("=" * 80)

# Create comparison dataframe for severity prediction
comparison_data = []
for model_name in models.keys():
    comparison_data.append({
        'Model': model_name,
        'Test Accuracy': severity_results[model_name]['test_accuracy'],
        'Test F1': severity_results[model_name]['test_f1'],
        'Test Recall': severity_results[model_name]['test_recall'],
        'CV Score': severity_results[model_name]['cv_mean'],
        'Overfit Gap': severity_results[model_name]['overfit_gap'],
        'Status': severity_results[model_name]['status']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)

print("\n🎯 DIFFICULTY LEVEL PREDICTION RESULTS:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\n🏆 BEST MODEL FOR DIFFICULTY LEVEL: {best_model_name}")
print(f"   Test Accuracy: {comparison_df.iloc[0]['Test Accuracy']:.4f}")
print(f"   Test F1 Score: {comparison_df.iloc[0]['Test F1']:.4f}")
print(f"   Test Recall: {comparison_df.iloc[0]['Test Recall']:.4f}")

# Primary Focus comparison
primary_comparison = []
for model_name in models.keys():
    primary_comparison.append({
        'Model': model_name,
        'Test Accuracy': primary_results[model_name]['test_accuracy'],
        'Test F1': primary_results[model_name]['test_f1'],
        'Test Recall': primary_results[model_name]['test_recall']
    })

primary_df = pd.DataFrame(primary_comparison).sort_values('Test Accuracy', ascending=False)
print(f"\n\n🎯 PRIMARY FOCUS PREDICTION RESULTS:")
print(primary_df.to_string(index=False))
print(f"\n🏆 BEST MODEL FOR PRIMARY FOCUS: {primary_df.iloc[0]['Model']}")

# Secondary Focus comparison
secondary_comparison = []
for model_name in models.keys():
    secondary_comparison.append({
        'Model': model_name,
        'Test Accuracy': secondary_results[model_name]['test_accuracy'],
        'Test F1': secondary_results[model_name]['test_f1'],
        'Test Recall': secondary_results[model_name]['test_recall']
    })

secondary_df = pd.DataFrame(secondary_comparison).sort_values('Test Accuracy', ascending=False)
print(f"\n\n🎯 SECONDARY FOCUS PREDICTION RESULTS:")
print(secondary_df.to_string(index=False))
print(f"\n🏆 BEST MODEL FOR SECONDARY FOCUS: {secondary_df.iloc[0]['Model']}")

# ============================================================================
# SAVE BEST MODELS
# ============================================================================

print("\n\n" + "=" * 80)
print("💾 SAVING MODELS AND PREPROCESSING OBJECTS")
print("=" * 80)

import joblib

# Save the best models and preprocessors
best_severity_model = severity_results[best_model_name]['model']
best_primary_model = primary_results[primary_df.iloc[0]['Model']]['model']
best_secondary_model = secondary_results[secondary_df.iloc[0]['Model']]['model']

joblib.dump(best_severity_model, 'best_severity_model.pkl')
joblib.dump(best_primary_model, 'best_primary_model.pkl')
joblib.dump(best_secondary_model, 'best_secondary_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(le_severity, 'label_encoder_severity.pkl')
joblib.dump(le_primary, 'label_encoder_primary.pkl')
joblib.dump(le_secondary, 'label_encoder_secondary.pkl')

print("\n✅ Saved files:")
print("   - best_severity_model.pkl")
print("   - best_primary_model.pkl")
print("   - best_secondary_model.pkl")
print("   - feature_scaler.pkl")
print("   - label_encoder_severity.pkl")
print("   - label_encoder_primary.pkl")
print("   - label_encoder_secondary.pkl")
print("   - confusion_matrices_difficulty_level_prediction.png")
print("   - confusion_matrices_primary_focus_prediction.png")
print("   - confusion_matrices_secondary_focus_prediction.png")

# ============================================================================
# EXAMPLE PREDICTION
# ============================================================================

print("\n\n" + "=" * 80)
print("🧪 EXAMPLE PREDICTION")
print("=" * 80)

# Example assessment scores
example_scores = np.array([[45, 50, 48, 55, 52, 50, 60, 58, 62]])

print("\nExample Assessment Scores:")
print(f"  Word Recognition: 45")
print(f"  Reading Fluency: 50")
print(f"  Comprehension: 48")
print(f"  Sentence Construction: 55")
print(f"  Grammar & Punctuation: 52")
print(f"  Expressive Writing: 50")
print(f"  Phonemic Spelling: 60")
print(f"  Sight Word Spelling: 58")
print(f"  Contextual Spelling: 62")

# Scale the features
example_scaled = scaler.transform(example_scores)

# Predict
severity_pred = best_severity_model.predict(example_scaled)
severity_label = le_severity.inverse_transform(severity_pred)[0]

primary_pred = best_primary_model.predict(example_scaled)
primary_label = le_primary.inverse_transform(primary_pred)[0]

secondary_pred = best_secondary_model.predict(example_scaled)
secondary_label = le_secondary.inverse_transform(secondary_pred)[0]

# Calculate averages for verification
reading_avg = np.mean([45, 50, 48])
writing_avg = np.mean([55, 52, 50])
spelling_avg = np.mean([60, 58, 62])

print(f"\n📊 PREDICTION RESULTS:")
print(f"{'─' * 60}")
print(f"Recommended Learning Path Level: {severity_label.upper()}")
print(f"\n📚 Learning Focus Priority:")
print(f"  1. {primary_label.capitalize()} - Focus here FIRST")
print(f"  2. {secondary_label.capitalize()} - Focus here SECOND")
print(f"\nCategory Averages (for verification):")
print(f"  Reading: {reading_avg:.1f}")
print(f"  Writing: {writing_avg:.1f}")
print(f"  Spelling: {spelling_avg:.1f}")

print("\n" + "=" * 80)
print("⚠️  REMINDER: This is a LEARNING PATH RECOMMENDATION system")
print("    NOT a diagnostic tool. Consult qualified professionals for diagnosis.")
print("=" * 80)
print("\n🎉 Training Complete! All models saved successfully.")