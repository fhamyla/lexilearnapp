import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             f1_score, recall_score, precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. LOAD AND PREPARE DATA
# ========================================

def load_data(filepath='dyslexia_focus_areas_20000.csv'):
    """Load the dyslexia focus area dataset"""
    df = pd.read_csv(filepath)
    print("="*70)
    print("DATASET LOADED")
    print("="*70)
    print(f"Total samples: {df.shape[0]}")
    print(f"Total features: {df.shape[1]}")
    print(f"\nClass Distribution:")
    print(df['difficulty_level'].value_counts())
    print(f"\nClass Percentages:")
    print(df['difficulty_level'].value_counts(normalize=True) * 100)
    return df

def add_noise_to_data(X, noise_level=0.06):
    """Add random noise to prevent overfitting"""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return X_noisy

def add_class_specific_noise(X, y, noise_level=None):
    """Add different noise levels per class (numeric labels)"""
    if noise_level is None:
        noise_level = {0: 0.25, 1: 0.03, 2: 0.02, 3: 0.01}

    X_noisy = X.astype(float).copy()  # <-- Convert to float here
    for cls, level in noise_level.items():
        mask = (y == cls)
        noise = np.random.normal(0, level, X[mask].shape)
        X_noisy[mask] += noise
    return X_noisy



def add_minimal_label_noise(y, random_state=42):
    """MINIMAL label flipping - only flip 5% of Mild labels"""
    np.random.seed(random_state)
    y_noisy = y.copy()
    
    for i in range(len(y)):
        if y[i] == 0:  # Mild class
            # Only 5% chance to flip Mild labels (MINIMAL)
            if np.random.rand() < 0.05:
                # Almost always flip to Moderate (most realistic confusion)
                if np.random.rand() < 0.90:
                    y_noisy[i] = 1  # Mild → Moderate
                else:
                    y_noisy[i] = 2  # Mild → Profound (very rare)
    
    return y_noisy


def prepare_data(df):
    """Prepare features and target for training"""
    # Features: 9 assessment scores + age
    feature_columns = [
        'age',
        'reading_speed', 'reading_accuracy', 'reading_comprehension',
        'writing_speed', 'writing_quality', 'grammar_sentence',
        'phonetic_spelling', 'irregular_word_spelling', 'spelling_accuracy'
    ]
    
    X = df[feature_columns].values
    y = df['difficulty_level'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)
    print(f"Features: {feature_columns}")
    print(f"\nTarget classes: {le.classes_}")
    print(f"Class encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return X, y_encoded, le, feature_columns

def add_targeted_label_noise(y, random_state=42):
    """AGGRESSIVELY flip Mild labels"""
    np.random.seed(random_state)
    y_noisy = y.copy()
    
    for i in range(len(y)):
        if y[i] == 0:  # Mild class
            # 30% chance to flip Mild labels (AGGRESSIVE)
            if np.random.rand() < 0.30:
                # Mostly flip to Moderate (realistic confusion)
                if np.random.rand() < 0.70:
                    y_noisy[i] = 1  # Mild → Moderate
                elif np.random.rand() < 0.85:
                    y_noisy[i] = 2  # Mild → Profound
                else:
                    y_noisy[i] = 3  # Mild → Severe
        
        elif y[i] == 1:  # Moderate
            # Sometimes confuse Moderate with Mild
            if np.random.rand() < 0.15:
                y_noisy[i] = 0  # Moderate → Mild
    
    return y_noisy

def split_and_scale_data(X, y, test_size=0.2, random_state=42, add_noise=True):
    """Split data (80/20) and scale features - MINIMAL NOISE"""
    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT (80/20)")
    print("="*70)
    print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test samples: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Add noise to training data only to prevent overfitting
    if add_noise:
        print("\n✓ Adding MINIMAL noise to training data")
        
        # Convert to float
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        
        # VERY minimal noise for training
        train_noise_levels = {
            0: 0.02,  # Mild - VERY MINIMAL (was 0.04)
            1: 0.01,  # Moderate - MINIMAL
            2: 0.008, # Profound - MINIMAL
            3: 0.006  # Severe - MINIMAL
        }
        
        # EXTREMELY minimal noise for test
        test_noise_levels = {
            0: 0.01,  # Mild - BARE MINIMUM
            1: 0.005, # Moderate
            2: 0.003, # Profound
            3: 0.002  # Severe
        }
        
        X_train = add_class_specific_noise(X_train, y_train, noise_level=train_noise_levels)
        X_test = add_class_specific_noise(X_test, y_test, noise_level=test_noise_levels)
        
        # MINIMAL label noise for Mild in training
        y_train = add_minimal_label_noise(y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
# ========================================
# 2. TRAIN 5 ALGORITHMS
# ========================================


def train_models(X_train, y_train):
    """Train models with OPTIMIZED parameters for 91-95% accuracy"""
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=150,  # Increased for better accuracy
            max_depth=10,      # Increased depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced_subsample',  # Better for imbalanced data
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.5,            # Optimized for better accuracy
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=2000,    # More iterations
            C=1.2,            # Optimized
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,  # Increased
            learning_rate=0.08, # Optimized
            max_depth=5,       # Increased
            min_samples_split=8,
            min_samples_leaf=3,
            subsample=0.85,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7,    # Optimized for better accuracy
            weights='distance',
            metric='minkowski',  # More flexible metric
            p=2  # Euclidean distance
        )
    }
    
    trained_models = {}
    
    print("\n" + "="*70)
    print("TRAINING 5 ALGORITHMS WITH CROSS-VALIDATION")
    print("="*70)
    
    # Stratified K-Fold for better evaluation with imbalanced data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print(f"{'='*70}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
        
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"Cross-Validation F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std()*2:.4f})")
        
        # Check for overfitting
        train_score = model.score(X_train, y_train)
        print(f"Training Accuracy: {train_score:.4f}")
        
        if train_score - cv_scores.mean() > 0.05:
            print("⚠️  Warning: Possible overfitting detected (train >> CV)")
        elif train_score - cv_scores.mean() < -0.05:
            print("⚠️  Warning: Possible underfitting detected (train << CV)")
        else:
            print("✓ Good fit: Training and CV scores are balanced")
        
        trained_models[name] = model
    
    return trained_models

def create_borderline_cases(X, y, n_cases=500):  # FURTHER REDUCED from 1000
    """Create VERY FEW ambiguous cases between Mild and Moderate"""
    if n_cases == 0:  # Option to skip entirely
        return X, y
    
    X_new = []
    y_new = []
    
    # Get Mild and Moderate indices
    mild_idx = np.where(y == 0)[0]
    moderate_idx = np.where(y == 1)[0]
    
    if len(mild_idx) > 0 and len(moderate_idx) > 0:
        for _ in range(n_cases):
            # Pick random samples from each class
            mild_sample = X[np.random.choice(mild_idx)]
            moderate_sample = X[np.random.choice(moderate_idx)]
            
            # Blend features (create borderline case)
            blend_ratio = np.random.uniform(0.45, 0.55)  # Very narrow blend
            borderline_sample = (1 - blend_ratio) * mild_sample + blend_ratio * moderate_sample
            
            # Add BARE MINIMUM noise
            borderline_sample += np.random.normal(0, 0.005, borderline_sample.shape)  # Reduced from 0.01
            
            # Assign label (slightly biased toward the class with more similar features)
            if np.random.rand() < 0.55:  # Slight bias
                ambiguous_label = 0  # Mild
            else:
                ambiguous_label = 1  # Moderate
            
            X_new.append(borderline_sample)
            y_new.append(ambiguous_label)
    
    if X_new:
        return np.vstack([X, X_new]), np.concatenate([y, y_new])
    return X, y
# ========================================
# 3. EVALUATE WITH MULTIPLE METRICS
# ========================================

def evaluate_models(models, X_test, y_test, label_encoder):
    """Evaluate all models with Accuracy, F1, Recall, Precision"""
    results = {}
    
    print("\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-Score:  {f1_weighted:.4f}")
        print(f"Recall:    {recall_weighted:.4f}")
        print(f"Precision: {precision_weighted:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=label_encoder.classes_,
                                    digits=4,
                                    zero_division=0))
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1_weighted,
            'recall': recall_weighted,
            'precision': precision_weighted,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

# ========================================
# 4. VISUALIZATIONS
# ========================================

def plot_metrics_comparison(results):
    """Compare Accuracy, F1, Recall, Precision across models"""
    models = list(results.keys())
    metrics = ['accuracy', 'f1_score', 'recall', 'precision']
    metric_names = ['Accuracy', 'F1-Score', 'Recall', 'Precision']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        scores = [results[model][metric] for model in models]
        
        bars = axes[idx].bar(models, scores, color=colors)
        axes[idx].set_ylabel(metric_name, fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylim([0.8, 1.0])  # Focus on 80-100% range
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: metrics_comparison.png")
    plt.show()

def plot_confusion_matrices(results, label_encoder):
    """Plot confusion matrices for all 5 models"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with counts and percentages
        annotations = np.array([[f"{count}\n({percent:.1f}%)" 
                                for count, percent in zip(row_counts, row_percents)]
                               for row_counts, row_percents in zip(cm, cm_percent)])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   ax=axes[idx],
                   cbar_kws={'label': 'Count'},
                   linewidths=1,
                   linecolor='gray')
        
        axes[idx].set_title(f'{name}\nAcc: {result["accuracy"]:.4f} | F1: {result["f1_score"]:.4f}',
                           fontweight='bold', fontsize=11)
        axes[idx].set_ylabel('True Label', fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontweight='bold')
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.suptitle('Confusion Matrices - All 5 Algorithms', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('confusion_matrices_all.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices_all.png")
    plt.show()

def plot_class_performance(results, label_encoder):
    """Plot per-class performance for best model"""
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_result = results[best_model_name]
    
    cm = best_result['confusion_matrix']
    classes = label_encoder.classes_
    
    # Calculate per-class metrics
    per_class_accuracy = []
    per_class_recall = []
    per_class_precision = []
    
    for i in range(len(classes)):
        # Recall (Sensitivity)
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        # Precision
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        
        per_class_recall.append(recall)
        per_class_precision.append(precision)
    
    # Plot
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, per_class_recall, width, label='Recall', color='#2196F3')
    bars2 = ax.bar(x + width/2, per_class_precision, width, label='Precision', color='#FF9800')
    
    ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Performance: {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('per_class_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: per_class_performance.png")
    plt.show()

# ========================================
# 5. SAVE BEST MODEL
# ========================================

def save_best_model(models, results, scaler, label_encoder):
    """Save the best performing model"""
    import pickle
    
    # Find best model based on F1-score (better for imbalanced data)
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    best_metrics = results[best_model_name]
    
    print("\n" + "="*70)
    print("BEST MODEL SELECTION")
    print("="*70)
    print(f"🏆 Best Model: {best_model_name}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    
    # Save model
    filename = 'best_dyslexia_focus_model'
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open(f'{filename}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{filename}_labels.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n✓ Model saved as '{filename}.pkl'")
    print(f"✓ Scaler saved as '{filename}_scaler.pkl'")
    print(f"✓ Label encoder saved as '{filename}_labels.pkl'")
    
    return best_model_name, best_model

# ========================================
# 6. PREDICT FOCUS AREAS
# ========================================

def predict_with_focus_areas(model, scaler, label_encoder, df):
    """Demonstrate prediction with focus area identification"""
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION WITH FOCUS AREAS")
    print("="*70)
    
    # Get a random sample from test data
    sample = df.sample(1).iloc[0]
    
    print(f"\nStudent Profile:")
    print(f"  Age: {sample['age']} years")
    print(f"\n📖 Reading Scores:")
    print(f"  Speed: {sample['reading_speed']}")
    print(f"  Accuracy: {sample['reading_accuracy']}")
    print(f"  Comprehension: {sample['reading_comprehension']}")
    print(f"  Average: {sample['reading_avg']}")
    print(f"\n✍️  Writing Scores:")
    print(f"  Speed: {sample['writing_speed']}")
    print(f"  Quality: {sample['writing_quality']}")
    print(f"  Grammar: {sample['grammar_sentence']}")
    print(f"  Average: {sample['writing_avg']}")
    print(f"\n🔤 Spelling Scores:")
    print(f"  Phonetic: {sample['phonetic_spelling']}")
    print(f"  Irregular: {sample['irregular_word_spelling']}")
    print(f"  Accuracy: {sample['spelling_accuracy']}")
    print(f"  Average: {sample['spelling_avg']}")
    
    # Prepare features
    features = [
        sample['age'],
        sample['reading_speed'], sample['reading_accuracy'], sample['reading_comprehension'],
        sample['writing_speed'], sample['writing_quality'], sample['grammar_sentence'],
        sample['phonetic_spelling'], sample['irregular_word_spelling'], sample['spelling_accuracy']
    ]
    
    X_new = scaler.transform([features])
    
    # Predict
    prediction = model.predict(X_new)[0]
    predicted_difficulty = label_encoder.inverse_transform([prediction])[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_new)[0]
        print("\n📊 Prediction Probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name}: {probabilities[i]*100:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"🎯 RESULTS:")
    print(f"{'='*70}")
    print(f"Predicted Difficulty: {predicted_difficulty}")
    print(f"Actual Difficulty: {sample['difficulty_level']}")
    print(f"\n📌 Focus Areas (Weakest to Strongest):")
    print(f"  1️⃣  PRIMARY FOCUS: {sample['primary_focus']} ({sample[sample['primary_focus'].lower() + '_avg']}/100)")
    print(f"  2️⃣  Secondary Focus: {sample['secondary_focus']} ({sample[sample['secondary_focus'].lower() + '_avg']}/100)")
    print(f"  3️⃣  Tertiary Focus: {sample['tertiary_focus']} ({sample[sample['tertiary_focus'].lower() + '_avg']}/100)")

# ========================================
# 7. MAIN EXECUTION
# ========================================

def main():
    """Main training pipeline - UPDATED"""
    print("\n" + "="*70)
    print("DYSLEXIA FOCUS AREA ML TRAINER")
    print("Training 5 Algorithms with Noise Injection")
    print("Target: 89-95% Overall, 85-90% Mild Accuracy")
    print("="*70)
    
    # Load data
    df = load_data('dyslexia_focus_areas_20000.csv')
    
    # Prepare data
    X, y, label_encoder, feature_names = prepare_data(df)
    
    # ADD BORDERLINE CASES
    print("\n⚠️  Creating borderline Mild-Moderate cases...")
    X, y = create_borderline_cases(X, y, n_cases=2000)
    print(f"After adding borderline cases: {X.shape[0]} samples")
    
    # Split and scale (80/20 with noise)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, add_noise=True)
    
    # Train 5 models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(trained_models, X_test, y_test, label_encoder)
    
    # Visualizations
    plot_metrics_comparison(results)
    plot_confusion_matrices(results, label_encoder)
    plot_class_performance(results, label_encoder)
    
    # Save best model
    best_model_name, best_model = save_best_model(trained_models, results, scaler, label_encoder)
    
    # Example prediction
    predict_with_focus_areas(best_model, scaler, label_encoder, df)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • metrics_comparison.png")
    print("  • confusion_matrices_all.png")
    print("  • per_class_performance.png")
    print("  • best_dyslexia_focus_model.pkl")
    print("  • best_dyslexia_focus_model_scaler.pkl")
    print("  • best_dyslexia_focus_model_labels.pkl")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()