import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import numpy as np

# --- Data Loading & Preprocessing ---
def load_data(parquet_path, sample_frac=1.0):
    df = pd.read_parquet(parquet_path)
    df = df.sample(frac=sample_frac, random_state=42)
    texts, tag_lists = [], []
    for _, row in df.iterrows():
        text = f"{row['name']}. Description: {row['description']}. Ingredients: {row['ingredients']}. Steps: {row['steps']}"
        tags = row.get('tags', [])
        tag_list = [t.strip() for t in tags.split(',')] if isinstance(tags, str) else tags
        texts.append(text)
        tag_lists.append(tag_list)
    return texts, tag_lists

# --- Feature Extraction ---
def sent2features(doc, idx, tag_set):
    token = doc[idx]
    word = token.text
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': token.pos_,
        'postag[:2]': token.pos_[:2],
        'in_tags': word.lower() in {t.lower() for t in tag_set}
    }
    if idx > 0:
        prev = doc[idx-1].text
        features.update({
            '-1:word.lower()': prev.lower(),
            '-1:postag': doc[idx-1].pos_
        })
    else:
        features['BOS'] = True
    if idx < len(doc)-1:
        nxt = doc[idx+1].text
        features.update({
            '+1:word.lower()': nxt.lower(),
            '+1:postag': doc[idx+1].pos_
        })
    else:
        features['EOS'] = True
    return features

def docs2dataset(nlp, texts, tag_lists):
    X, y = [], []
    for text, tags in zip(texts, tag_lists):
        doc = nlp(text)
        tag_set = set(tags)
        feats = [sent2features(doc, i, tag_set) for i in range(len(doc))]
        labels = ['ING' if token.text.lower() in tag_set else 'O' for token in doc]
        X.append(feats)
        y.append(labels)
    return X, y

# --- Main Function ---
def main(parquet_path,
         sample_frac=1.0,
         test_size=0.3,
         c1=0.1,
         c2=0.1,
         max_iterations=100):
    # Load & split
    texts, tag_lists = load_data(parquet_path, sample_frac)
    X, y = docs2dataset(spacy.load('en_core_web_sm', disable=['ner', 'parser']), texts, tag_lists)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Predictions
    y_pred = crf.predict(X_test)

    # Evaluation
    labels = list(crf.classes_)
    labels.remove('O')
    metrics_report = metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=4
    )
    accuracy = metrics.flat_accuracy_score(y_test, y_pred)

    print("Classification Report:\n", metrics_report)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Save model
    import joblib
    joblib.dump(crf, 'crf_ner_model.pkl')
    print("Model saved as crf_ner_model.pkl")

if __name__ == '__main__':
    main('food_recipes.parquet', sample_frac=0.5, test_size=0.2, c1=0.2, c2=0.1, max_iterations=200)