# Penguins of Madagascar - Species Classification

This repository contains a machine learning solution for classifying penguin species based on physical measurements. The project is specifically designed to identify Adelie penguins (like Skipper, Private, Rico, and Kowalski from the "Penguins of Madagascar") among other species.

## Project Overview

Every day at 7AM, new penguin data is available at the API endpoint. This project:
1. Fetches the data
2. Processes it
3. Makes a prediction about the penguin species
4. Updates the results on GitHub Pages

## Technical Implementation

### Data Pipeline
- Data is fetched from the original penguins dataset
- Transformed and stored in a SQLite database
- Features selected based on correlation analysis and domain knowledge

### Machine Learning Model
- Classification model trained on historical penguin data
- Features used: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
- Model can be trained using file 'model_training.py'

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

### Automation
- GitHub Actions workflow runs daily at 7:30 AM
- Fetches new penguin data from the API
- Makes predictions using the trained model
- Updates the GitHub Pages with the latest prediction

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
git clone https://github.com/yourusername/penguins-classification.git
cd penguins-classification
