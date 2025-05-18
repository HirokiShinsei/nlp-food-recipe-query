from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import joblib
from sklearn_crfsuite import CRF

app = Flask(__name__)

# Load the model and dataset
model = joblib.load("crf_food_recipes_20pct.pkl")  # Pretrained CRF model
df = pd.read_parquet("food_recipes.parquet", engine="pyarrow")

# Function to process user input and predict using the model
def predict_from_input(user_input):
    input_ingredients = user_input.split(", ")
    feature_list = [{"word": word.lower()} for word in input_ingredients]
    predicted_tags = model.predict_single(feature_list)  # Predict tags for the input
    return predicted_tags

# Function to find recipes based on predicted tags or ingredients
def find_matching_recipes(predicted_tags, user_input):
    user_ingredients = user_input.split(", ")
    matched_recipes = []

    # Iterate over dataset and match recipes
    for _, row in df.iterrows():
        recipe_ingredients = row['ingredients'].lower()
        recipe_tags = row['tags'].lower()

        # Check for ingredient and tag matches
        ingredient_match = all(ingredient in recipe_ingredients for ingredient in user_ingredients)
        tag_match = any(tag in recipe_tags for tag in predicted_tags)

        if ingredient_match or tag_match:
            matched_recipes.append({
                "name": row['name'],
                "ingredients": row['ingredients'],
                "steps": row['steps']
            })

        # Limit to top 3 matches
        if len(matched_recipes) >= 3:
            break

    return matched_recipes

# Define the home route for the frontend
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("query")
        predicted_tags = predict_from_input(user_query)
        matching_recipes = find_matching_recipes(predicted_tags, user_query)
        return render_template_string(HTML_TEMPLATE, recipes=matching_recipes, query=user_query)
    return render_template_string(HTML_TEMPLATE, recipes=None, query="")

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recipe Finder</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: auto; }
        .recipe { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; }
        .recipe h2 { margin-top: 0; }
        .recipe p { margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Recipe Finder</h1>
        <form method="POST">
            <input type="text" name="query" placeholder="Enter ingredients (e.g., carrot, chicken, potatoes)" value="{{ query }}" required>
            <button type="submit">Search</button>
        </form>
        {% if recipes %}
            <h2>Results:</h2>
            {% for recipe in recipes %}
                <div class="recipe">
                    <h2>{{ recipe.name }}</h2>
                    <p><strong>Ingredients:</strong> {{ recipe.ingredients }}</p>
                    <p><strong>Steps:</strong> {{ recipe.steps }}</p>
                </div>
            {% endfor %}
        {% endif %}
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
