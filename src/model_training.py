import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

def load_data_from_db():
    """
    Load data from the SQLite database
    """
    # Define database path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    db_path = os.path.join(data_dir, 'penguins.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Please run data_preparation.py first.")
        return None
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Load measurements and join with species
    query = """
    SELECT m.*, s.species
    FROM measurements m
    JOIN species s ON m.species_id = s.rowid
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Close connection
    conn.close()
    
    print(f"Loaded {len(df)} records from database")
    return df

def perform_feature_selection(df):
    """
    Perform feature selection and analysis
    """
    # Select only the features we're interested in
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    target = 'species'
    
    # Create a copy of the dataframe with only the features and target
    data = df[features + [target]].copy()
    
    # Display correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    
    # Save the correlation matrix plot
    images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'correlation_matrix.png'))
    
    # Display feature distributions by species
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x=target, y=feature, data=data)
        plt.title(f'{feature} by Species')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'feature_distributions.png'))
    
    # Apply SelectKBest to select the most important features
    X = data[features]
    y = data[target]
    
    # Create a SelectKBest object to select features with k best scores
    selector = SelectKBest(f_classif, k=3)  # Select top 3 features out of 4
    X_new = selector.fit_transform(X, y)
    
    # Get the scores and p-values for each feature
    scores = selector.scores_
    pvalues = selector.pvalues_
    
    # Create a DataFrame to display feature importance
    feature_scores = pd.DataFrame({
        'Feature': features,
        'Score': scores,
        'P-value': pvalues,
        'Selected': selector.get_support()
    })
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    print("\nFeature Selection Results (SelectKBest with f_classif):")
    print(feature_scores)
    
    # Visualize feature importance scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='Score', data=feature_scores)
    plt.title('Feature Importance Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'feature_importance.png'))
    
    # Get the selected features
    selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
    print(f"\nSelected features: {selected_features}")
    
    return data, selected_features, target

def train_model(data, features, target):
    """
    Train a machine learning model
    """
    # Split the data into features and target
    X = data[features]
    y = data[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', 'confusion_matrix.png'))
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Save the model and scaler
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'penguin_classifier.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    selected_features_path = os.path.join(models_dir, 'selected_features.txt')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save the selected features to a text file
    with open(selected_features_path, 'w') as f:
        f.write(','.join(features))
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Selected features saved to {selected_features_path}")
    
    return model, scaler

def main():
    # Load data from database
    df = load_data_from_db()
    
    if df is not None:
        # Perform feature selection
        data, features, target = perform_feature_selection(df)
        
        # Train model
        model, scaler = train_model(data, features, target)
        
        return model, scaler
    
    return None, None

if __name__ == "__main__":
    main()