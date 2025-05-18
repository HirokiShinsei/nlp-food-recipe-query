import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib

# 1) Load & Sample 20% of the dataset
def load_and_sample(url: str, sample_frac: float = 0.2, random_state: int = 42):
    print("Loading Parquet data from URL...")
    df_full = pd.read_parquet(url, engine="pyarrow")
    print(f"  Full dataset size: {len(df_full)} rows")
    df_sample = df_full.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    print(f"  Sampled {sample_frac*100:.0f}% → {len(df_sample)} rows")
    return df_sample

# 2) Split into train/dev
def split_data(df: pd.DataFrame, dev_frac: float = 0.2, random_state: int = 42):
    df_train, df_dev = train_test_split(df, test_size=dev_frac, random_state=random_state, shuffle=True)
    print(f"Train size: {len(df_train)}, Dev size: {len(df_dev)}")
    return df_train, df_dev

# 3) Feature extraction helpers
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
def token2features(tok, prev_tok, next_tok, tag_set):
    feats = {
        "bias": 1.0,
        "word.lower()": tok.text.lower(),
        "word[-3:]":    tok.text[-3:],
        "word.isupper()": tok.is_upper,
        "word.istitle()": tok.is_title,
        "postag":        tok.pos_,
        "in_tags":       tok.text.lower() in tag_set
    }
    if prev_tok:
        feats["-1:word.lower()"] = prev_tok.text.lower()
        feats["-1:postag"]       = prev_tok.pos_
    else:
        feats["BOS"] = True
    if next_tok:
        feats["+1:word.lower()"] = next_tok.text.lower()
        feats["+1:postag"]       = next_tok.pos_
    else:
        feats["EOS"] = True
    return feats

def doc2features_and_labels(row):
    text = (
        f"{row['name']}. Description: {row['description']}."
        f" Ingredients: {row['ingredients']}. Steps: {row['steps']}"
    )
    tag_set = set(t.lower() for t in row.get("tags", []))
    doc = nlp(text)
    feats, labs = [], []
    for i, tok in enumerate(doc):
        prev_tok = doc[i-1] if i>0 else None
        next_tok = doc[i+1] if i < len(doc)-1 else None
        feats.append(token2features(tok, prev_tok, next_tok, tag_set))
        labs.append("ING" if tok.text.lower() in tag_set else "O")
    return feats, labs

# 4) Build corpora
def build_corpus(df):
    X, y = zip(*df.apply(doc2features_and_labels, axis=1))
    return list(X), list(y)

# 5) Train & evaluate CRF
def train_and_evaluate(df_train, df_dev):
    print("Building feature corpora…")
    X_train, y_train = build_corpus(df_train)
    X_dev,   y_dev   = build_corpus(df_dev)

    print("Training CRF model…")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,       # L1 penalty
        c2=0.1,       # L2 penalty
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    print("Evaluating on dev set…")
    y_pred = crf.predict(X_dev)
    labels = [lbl for lbl in crf.classes_ if lbl != "O"]

    report = metrics.flat_classification_report(
        y_dev, y_pred, labels=labels, digits=4
    )
    accuracy = metrics.flat_accuracy_score(y_dev, y_pred)

    print("\n--- Dev Classification Report ---\n")
    print(report)
    print(f"Overall Dev Accuracy: {accuracy:.4f}\n")

    # Save model
    model_path = "crf_food_recipes_20pct.pkl"
    joblib.dump(crf, model_path)
    print(f"Model saved to: {model_path}")

def main():
    HF_PARQUET_URL = "https://huggingface.co/datasets/jojogo9/Food_Recipes/resolve/main/food_recipes.parquet"
    df = load_and_sample(HF_PARQUET_URL, sample_frac=0.2)
    df_train, df_dev = split_data(df, dev_frac=0.2)
    train_and_evaluate(df_train, df_dev)

if __name__ == "__main__":
    main()
