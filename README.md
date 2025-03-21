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
- Model evaluation metrics included in the training notebook

### Automation
- GitHub Actions workflow runs daily at 7:30 AM
- Fetches new penguin data from the API
- Makes predictions using the trained model
- Updates the GitHub Pages with the latest prediction

### Deployment and Automation
- GitHub Pages set up with custom Jekyll theme for result visualization
- GitHub Actions workflow configured to run daily at 7:30 AM UTC
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

## How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/yourusername/penguins-classification.git
cd penguins-classification