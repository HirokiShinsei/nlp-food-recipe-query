# ML Recipe Web App

This repository contains a machine learning model that has been trained on a dataset of recipes and a web application to help users choose food based on their preferences (e.g., vegan, meat, mixed, chicken, beef, fish, pork).

## Directory Structure

```
ml-recipe-web-app
├── src
│   ├── web
│   │   └── app.py         # Flask web app
│   ├── utils
│   │   └── data.py        # Utility functions for data/model processing
│   ├── train.py           # Training script for CRF model
│   └── train2.py          # Alternate training script with a data sampling approach
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.8 or higher
- pip

## Setup

1. **Clone the repository**

   ```batch
   git clone <repository-url>
   cd ml-recipe-web-app
   ```

2. **Create and activate a virtual environment**

   On Windows (using Command Prompt):

   ```batch
   python -m venv venv
   venv\Scripts\activate
   ```

   Or on PowerShell:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```batch
   pip install -r requirements.txt
   ```

4. **Download the Dataset**

   Ensure you have a local copy of `food_recipes.parquet`. If using the Hugging Face URL in `train2.py`, the script will download & sample the data. Otherwise, place your `food_recipes.parquet` in the repository root.

## Training the Model

Two scripts are provided for training:

- **train.py**: Trains a CRF model on the full dataset (or a fraction) and saves the model as `crf_ner_model.pkl`.
- **train2.py**: Loads data from a URL, samples 20% of the dataset, trains a CRF model, and saves the model as `crf_food_recipes_20pct.pkl`.

To train using either script, run:

```batch
python src/train.py
```

or

```batch
python src/train2.py
```

The trained models will be saved to the repository root.

## Running the Web App

Ensure you have a trained model (default expected filename is `crf_food_recipes_20pct.pkl` in `app.py`). Then run:

```batch
python src/web/app.py
```

Open a web browser and navigate to `http://127.0.0.1:5000` to use the Recipe Finder app.

## Notes

- The web app uses a pre-trained CRF model to predict recipe tags and match the recipes based on the user query.
- The dataset and the approach for predictions rely on ingredients and tags extraction.
- Edit the file paths in the code if necessary, depending on your environment and model location.
