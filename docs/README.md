# Penguins of Madagascar - GitHub Pages

This folder contains the files for the GitHub Pages website for the Penguins of Madagascar project.

## Structure

- `_config.yml`: Jekyll configuration file
- `_layouts/`: Contains HTML layouts
- `_includes/`: Contains reusable HTML components
- `index.html`: Main page (generated automatically by prediction.py)

## Functionality

The GitHub Pages website displays the daily predictions of penguin species, which are fetched from the API every morning at 7:30 AM via GitHub Actions. The website is updated automatically when new predictions are added.

## Technical Implementation

- Jekyll static site generator
- Bootstrap 4.6 for styling
- GitHub Actions for automation