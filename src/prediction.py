import requests
import pandas as pd
import joblib
import os
import json
from datetime import datetime

def fetch_new_penguin_data():
    """
    Fetch new penguin data from the API
    """
    api_url = "http://130.225.39.127:8000/new_penguin/"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        penguin_data = response.json()
        print(f"Fetched new penguin data: {penguin_data}")
        
        return penguin_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def predict_species(penguin_data):
    """
    Predict the species of a penguin using the trained model
    """
    # Load the model and scaler
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model_path = os.path.join(models_dir, 'penguin_classifier.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Please train the model first.")
        return None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load the model and get the selected features
    # We need to use the same features that were selected during training
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Default features (in case we can't determine the selected features)
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # Try to load selected features if available
    selected_features_path = os.path.join(models_dir, 'selected_features.txt')
    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = f.read().strip().split(',')
        if selected_features:
            features = selected_features
            print(f"Using selected features: {features}")
        else:
            print("Using all features as fallback")
    
    # Check if all required features are present
    for feature in features:
        if feature not in penguin_data:
            print(f"Missing feature: {feature}")
            return None
    
    # Create a DataFrame with the features
    X = pd.DataFrame([penguin_data])
    
    # Scale the features
    X_scaled = scaler.transform(X[features])
    
    # Make prediction
    species_prediction = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    # Get the probability for the predicted class
    species_index = list(model.classes_).index(species_prediction)
    confidence = prediction_proba[species_index]
    
    result = {
        'species': species_prediction,
        'confidence': float(confidence),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'features': penguin_data
    }
    
    print(f"Prediction: {species_prediction} with confidence {confidence:.4f}")
    
    return result

def update_prediction_history(prediction):
    """
    Update the prediction history file for GitHub Pages
    """
    if prediction is None:
        return
    
    # Define the path for the prediction history file
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    
    history_path = os.path.join(docs_dir, 'prediction_history.json')
    
    # Load existing history or create a new one
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = {'predictions': []}
    else:
        history = {'predictions': []}
    
    # Add the new prediction to the history
    history['predictions'].append(prediction)
    
    # Save the updated history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create or update the index.html file for GitHub Pages
    create_github_pages(history, docs_dir)
    
    print(f"Prediction history updated at {history_path}")

def create_github_pages(history, docs_dir):
    """
    Create or update the GitHub Pages files
    """
    # Create index.html with Jekyll front matter
    html_content = f"""
    ---
    layout: default
    title: Penguins of Madagascar - Species Classification
    ---
    <div class="container">
            <h1 class="my-4">Penguins of Madagascar - Species Classification</h1>
            <p class="lead">Looking for Skipper, Private, Rico, and Kowalski (Adelie penguins) in New York!</p>
            
            <h2 class="mt-4">Latest Prediction</h2>
    """
    
    if history['predictions']:
        latest = history['predictions'][-1]
        is_adelie = latest['species'] == 'Adelie'
        card_class = 'adelie' if is_adelie else 'other'
        
        html_content += f"""
            <div class="card prediction-card {card_class}">
                <div class="card-body">
                    <h5 class="card-title">Species: {latest['species']}</h5>
                    <h6 class="card-subtitle mb-2 text-muted">Confidence: {latest['confidence']:.2%}</h6>
                    <p class="card-text">Timestamp: {latest['timestamp']}</p>
                    <div class="card-text">
                        <strong>Features:</strong>
                        <ul>
                            <li>Bill Length: {latest['features']['bill_length_mm']} mm</li>
                            <li>Bill Depth: {latest['features']['bill_depth_mm']} mm</li>
                            <li>Flipper Length: {latest['features']['flipper_length_mm']} mm</li>
                            <li>Body Mass: {latest['features']['body_mass_g']} g</li>
                        </ul>
                    </div>
                    <p class="card-text">
                        <strong>Result: </strong>
                        {"This could be one of our penguins!" if is_adelie else "This is not one of our penguins."}
                    </p>
                </div>
            </div>
        """
    
    html_content += f"""
            <h2 class="mt-4">Prediction History</h2>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Species</th>
                            <th>Confidence</th>
                            <th>Bill Length (mm)</th>
                            <th>Bill Depth (mm)</th>
                            <th>Flipper Length (mm)</th>
                            <th>Body Mass (g)</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for pred in reversed(history['predictions']):
        row_class = 'table-success' if pred['species'] == 'Adelie' else ''
        html_content += f"""
                        <tr class="{row_class}">
                            <td>{pred['timestamp']}</td>
                            <td>{pred['species']}</td>
                            <td>{pred['confidence']:.2%}</td>
                            <td>{pred['features']['bill_length_mm']}</td>
                            <td>{pred['features']['bill_depth_mm']}</td>
                            <td>{pred['features']['flipper_length_mm']}</td>
                            <td>{pred['features']['body_mass_g']}</td>
                        </tr>
        """
    
    html_content += """
                    </tbody>
                </table>
            </div>
            
            <footer class="mt-5 text-center text-muted">
                <p>Penguin Classification System - MLOps Assignment</p>
            </footer>
        </div>
        
        {% include scripts.html %}
    </body>
    </html>
    """
    
    # Write the HTML content to the index.html file
    index_path = os.path.join(docs_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"GitHub Pages files updated at {docs_dir}")

def main():
    """
    Main function to run the prediction pipeline
    """
    # Fetch new penguin data from the API
    penguin_data = fetch_new_penguin_data()
    
    if penguin_data is not None:
        # Predict the species of the penguin
        prediction = predict_species(penguin_data)
        
        if prediction is not None:
            # Update the prediction history
            update_prediction_history(prediction)
            return prediction
    
    return None

if __name__ == "__main__":
    main()