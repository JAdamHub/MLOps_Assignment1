# Penguins of Madagascar - Species Classification

This repository contains a machine learning solution for classifying penguin species based on physical measurements. The project is specifically designed to identify Adelie penguins (like Skipper, Private, Rico, and Kowalski from the "Penguins of Madagascar") among other species.

## Project Overview

Every day at 7AM, new penguin data is available at the API endpoint. This project:
1. Fetches the data
2. Processes it
3. Makes a prediction about the penguin species
4. Updates the results on GitHub Pages

## Technical Implementation

### Data Preparation Process
The `data_preparation.py` script handles the initial data pipeline, transforming the raw penguins dataset into a normalized SQLite database. The process..:

1. **Data Download**
   - Uses seaborn's built-in dataset loader to fetch the penguins dataset
   - Removes any rows with missing values to ensure data quality
   - Provides initial dataset statistics including shape and species distribution

2. **Database Schema**
   The data is organized into a normalized database structure with the following tables:
   - `species`: Unique penguin species (Adelie, Chinstrap, Gentoo)
   - `island`: Different islands where penguins were observed
   - `sex`: Gender categories of the penguins
   - Measurements: `bill_length_mm`,	`bill_depth_mm`,	`flipper_length_mm`,`body_mass_g`: Contains the physical measurements with foreign keys to other tables
     - Links to species, islands, and sex tables
     - Stores bill length, bill depth, flipper length, and body mass

3. **Data Transformation**
   - Creates unique identifiers for species, islands, and sex categories
   - Establishes proper relationships between tables using foreign keys
   - Ensures data integrity and efficient querying capabilities

This structured database approach allows for:
- Efficient data storage and retrieval
- Maintaining data relationships
- Easy integration with the machine learning pipeline

### Model Training Process
The `model_training.py` script implements a comprehensive machine learning pipeline for penguin species classification. Here's a detailed breakdown of the process:

1. **Data Loading**
   - Connects to the SQLite database to fetch penguin measurements
   - Joins measurement data with species information
   - Performs initial data validation and preprocessing

2. **Feature Analysis & Selection**
   - Generates correlation matrix to understand feature relationships
   - Creates feature distribution plots by species
   - Utilizes SelectKBest with f_classif for feature importance ranking
   - Selects top 3 most discriminative features based on statistical significance
   - Visualizes feature importance scores for transparency

3. **Model Training Pipeline**
   - Implements data splitting (70% training, 30% testing)
   - Applies StandardScaler for feature normalization
   - Trains a RandomForest classifier with 100 estimators
   - Performs 5-fold cross-validation for robust performance estimation

4. **Model Evaluation**
   - Generates comprehensive classification metrics
   - Creates confusion matrix visualization
   - Calculates per-class precision, recall, and F1-scores
   - Reports cross-validation scores for model stability assessment

5. **Model Persistence**
   - Saves the trained model for production use
   - Stores the feature scaler for consistent preprocessing
   - Preserves selected feature list for prediction pipeline

The training process emphasizes both model performance and interpretability, ensuring reliable species classification while maintaining transparency in the feature selection process.

### Model Metrics

#### Feature Selection Results (SelectKBest with f_classif):

| Feature              | Score        | P-value          | Selected |
|----------------------|--------------|------------------|----------|
| flipper_length_mm    | 567.406992   | 1.587418e-107    | True     |
| bill_length_mm       | 397.299437   | 1.380984e-88     | True     |
| bill_depth_mm        | 344.825082   | 1.446616e-81     | True     |
| body_mass_g          | 341.894895   | 3.744505e-81     | False    |

**Selected features**: `['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']`  
**Model accuracy**: `0.9900`

#### Classification Report:

| Class        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| Adelie       | 0.98      | 1.00   | 0.99     | 48      |
| Chinstrap    | 1.00      | 0.94   | 0.97     | 18      |
| Gentoo       | 1.00      | 1.00   | 1.00     | 34      |

|              | Accuracy  | Macro Avg | Weighted Avg |
|--------------|-----------|-----------|---------------|
|              | 0.99      | 0.99      | 0.99          |

**Cross-validation scores**: `[0.98507463, 1.0, 0.97014925, 0.96969697, 0.93939394]`  
**Mean CV score**: `0.9729`

### Daily Prediction Pipeline (prediction.py)
The `prediction.py` script orchestrates the daily penguin species prediction process with several key components:

1. **Data Fetching (`fetch_new_penguin_data`)**
   - Connects to the API endpoint
   - Handles HTTP requests and error management
   - Returns structured penguin measurement data

2. **Species Prediction (`predict_species`)**
   - Loads the trained RandomForest model and feature scaler
   - Uses selected features from training phase
   - Processes new data through the same preprocessing pipeline
   - Generates species prediction with confidence score
   - Returns prediction results with timestamp

3. **History Management (`update_prediction_history`)**
   - Maintains a JSON file of all predictions
   - Stores features, predictions, and confidence scores
   - Enables tracking of prediction history

4. **GitHub Pages Update (`create_github_pages`)**
   - Generates dynamic HTML content
   - Creates visually appealing prediction cards
   - Highlights Adelie penguin predictions
   - Maintains a searchable prediction history table
   - Updates the website automatically

### GitHub Actions Workflow (7.30am-daily_prediction.yml)
The workflow file `.github/workflows/7.30am-daily_prediction.yml` automates the daily prediction process:

1. **Scheduling**
   - Runs automatically at 7:30 AM UTC daily
   - Supports manual triggering via workflow_dispatch

2. **Environment Setup**
   - Uses Ubuntu latest runner
   - Sets up Python 3.10
   - Installs required dependencies

3. **Execution Process**
   - Runs the prediction script
   - Handles potential errors gracefully

4. **Repository Management**
   - Configures Git user credentials
   - Commits new prediction results
   - Updates GitHub Pages content
   - Uses GitHub token for authentication

### Deployment and Automation
- GitHub Pages set up with custom Jekyll theme for result visualization
- Workflow steps:
  - Check out repository code
  - Set up Python 3.10 environment
  - Install required dependencies from requirements.txt
  - Execute prediction script
  - Commit and push changes to the docs directory
  - Authentication handled via GitHub token
- **Important Repository Settings**:
  - Workflow permissions must be set to 'Read and write permissions' in the repository settings
  - Navigate to: Repository → Settings → Actions → General → Workflow permissions
  - Select 'Read and write permissions' to allow the workflow to commit changes to the repository

## Repository Structure
- `data/`: Contains the SQLite database
- `models/`: Stores the trained machine learning model
- `src/`: Source code
  - `data_preparation.py`: Script to download data and create the database
  - `model_training.py`: Script for feature selection and model training
  - `prediction.py`: Script to fetch new data and make predictions
- `.github/workflows/`: Contains GitHub Actions workflow definitions

## Clone:

1. Clone the repository
```bash
git clone https://github.com/JAdamHub/MLOps_Assignment1.git
cd MLOps_Assignment1
