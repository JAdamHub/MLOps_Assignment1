# Penguins of Madagascar - GitHub Pages

Denne mappe indeholder filerne til GitHub Pages-websitet for Penguins of Madagascar projektet.

## Struktur

- `_config.yml`: Jekyll konfigurationsfil
- `_layouts/`: Indeholder HTML layouts
- `_includes/`: Indeholder genbrugelige HTML komponenter
- `index.html`: Hovedsiden (genereres automatisk af prediction.py)

## Funktionalitet

GitHub Pages-websitet viser de daglige forudsigelser af pingvinarter, som hentes fra API'en hver morgen kl. 7:30 via GitHub Actions. Websitet opdateres automatisk, når nye forudsigelser tilføjes.

## Teknisk implementering

- Jekyll static site generator
- Bootstrap 4.6 for styling
- GitHub Actions for automatisering