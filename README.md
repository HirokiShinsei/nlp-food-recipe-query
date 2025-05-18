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
│   ├── train.py           # Training script using a local parquet file
│   └── train2.py          # Training script that loads data from a URL and samples 20% of it
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

4. **Download the spaCy Language Model**

   The app requires the `en_core_web_sm` model. Install it with:

   ```batch
   python -m spacy download en_core_web_sm
   ```

5. **Download the Dataset**

   - For **train.py**: Place your `food_recipes.parquet` close to the project root or update the path in `train.py`.
   - For **train2.py**: The script downloads the dataset from the Hugging Face URL.

## Training the Model

Two training scripts are provided:

- **train.py**: 
  - Loads the dataset from a local `food_recipes.parquet` file.
  - Samples the data if needed.
  - Trains a CRF model and saves it as `crf_ner_model.pkl`.

  To run:

  ```batch
  python src/train.py
  ```

- **train2.py**:
  - Downloads and samples 20% of the dataset from the Hugging Face URL.
  - Splits the data into training and development sets.
  - Trains a CRF model and saves it as `crf_food_recipes_20pct.pkl`.

  To run:

  ```batch
  python src/train2.py
  ```

## Running the Web App

Ensure you have a trained model. By default, the web app in `app.py` expects a locally available dataset and may use one of the saved models. Start the web server with:

```batch
python src/web/app.py
```

Open a web browser and navigate to `http://127.0.0.1:5000` to use the Recipe Finder app.

## Notes

- The web app uses a pre-trained CRF model to predict recipe tags and match recipes based on the user query.
- Depending on the training script you use, the model filename may differ (`crf_ner_model.pkl` from train.py versus `crf_food_recipes_20pct.pkl` from train2.py). Update `app.py` accordingly if needed.
- The dataset and the approach for predictions rely on ingredients and tags extraction.
