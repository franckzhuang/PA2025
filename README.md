# PA2025

### Main Model task

Image classification task :
Take an image as an input and predict wether it is AI generated or not.

## Dataset Construction Stages

- Non-AI Generated Images:

  1. Bots & Web scrapers : Use web scrapers and bots to gather non-AI generated images on the internet
  2. Taking non-AI generated pictures manually (Phone, cameras, screenshots?, ...)
  3. Algorithmic validation of images : check valid formats, extensions (ex. no webp, no svg, etc...) and other details
  4. Human validation : Human review of each image to validate usefulness and or eliminate if could bias the models

- AI Generated images:
  1. Use **multiple** open source image generation models as our dataset (multiple to reduce biases). Run models overnight
  2. Manually search for AI-generated images from trusted sources (Midjourney, ...)
  3. Human validation : Human review of each image to validate usefulness and or eliminate if could bias the models

### AGID App (Anti Generated Image Detection)

- Interface : Faire une page pour générer un modèle entraîné à partir d'un modèle non entrainé et d'un jeu de donnée.
- Interface : Ajouter la possibilité de choisir différents modèles (ajouter quelque chose de ludique comme le plus utilisé, un modèle meilleur en dessin ou en photo etc ...)
