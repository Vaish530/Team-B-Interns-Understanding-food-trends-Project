

import re
import sys
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------- CONFIG ----------
INPUT_XLSX = "fb_reviews_translated(final).xlsx"
OUTPUT_XLSX = "fb_reviews_with_sentiment.xlsx"
POS_THRESH = 0.05
NEG_THRESH = -0.05

# Candidate normalized names to try (lowercase, spaces collapsed)
CANDIDATE_COLS = [
    "commenttextenglish", "comment_text_english", "commenttext", "comment", "comments",
    "feedbackdescription", "feedback", "text", "review", "comment english", "comment_english"
]
# ----------------------------

def download_vader_if_missing():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=False)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c_str = str(c)
        c_str = re.sub(r"[\t\r\n]+", " ", c_str)
        c_str = re.sub(r"\s+", " ", c_str)  # collapse whitespace
        c_str = c_str.strip()
        c_str = c_str.lower()
        c_str = c_str.replace("-", " ").replace("/", " ").replace("\\", " ")
        new_cols.append(c_str)
    df.columns = new_cols
    return df

def choose_text_column(columns):
    cols = list(columns)
    # 1) exact candidate match
    for c in CANDIDATE_COLS:
        if c in cols:
            return c
    # 2) prioritized substring checks
    priority = ["commenttextenglish", "commenttext", "comments", "comment", "feedback", "review", "text"]
    for pat in priority:
        for col in cols:
            if pat in col:
                return col
    # 3) any column containing keywords
    for col in cols:
        if any(k in col for k in ("comment", "feedback", "text", "review")):
            return col
    return None

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def label_from_compound(c: float) -> str:
    if c >= POS_THRESH:
        return "positive"
    if c <= NEG_THRESH:
        return "negative"
    return "neutral"

def compute_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    scores = []
    labels = []
    for txt in df[text_col].fillna("").astype(str):
        t = clean_text(txt)
        s = sid.polarity_scores(t)
        comp = s["compound"]
        scores.append(comp)
        labels.append(label_from_compound(comp))
    df["_sentiment_score"] = scores
    df["_sentiment_label"] = labels
    return df

def main():
    print("Loading:", INPUT_XLSX)
    try:
        df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    except Exception as e:
        print("Error reading Excel:", e)
        sys.exit(1)

    df = normalize_columns(df)
    print("Available (normalized) columns:", list(df.columns))

    chosen = choose_text_column(df.columns)
    if chosen is None:
        raise ValueError("Could not auto-detect a text column. Available columns: " + ", ".join(list(df.columns)))

    print(f"Using column for sentiment: '{chosen}'")

    download_vader_if_missing()

    print("Computing sentiment (VADER)...")
    out = compute_sentiment(df, chosen)

    out.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print("Done. Results saved to", OUTPUT_XLSX)
    print("\nLabel counts:")
    print(out["_sentiment_label"].value_counts())

if __name__ == "__main__":
    main()
