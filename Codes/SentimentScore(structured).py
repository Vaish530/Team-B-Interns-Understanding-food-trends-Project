# sentiment_survey_fixed.py
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon', quiet=True)

# ---------- CONFIG ----------
INPUT_XLSX = "master.xlsx"   # your file
RAW_TEXT_COL = "FeedbackDescription"  # the human name (optional) - script will normalize
TEXT_COL_CLEAN = "feedbackdescription"  # normalized name used in code
OUTPUT_XLSX = "survey_with_sentiment.xlsx"
POS_THRESH = 0.05
NEG_THRESH = -0.05
# ----------------------------

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

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # remove leading/trailing spaces, collapse multiple spaces, and lower-case
    new_cols = []
    for c in df.columns:
        c_str = str(c)
        c_str = re.sub(r"\s+", " ", c_str)  # collapse internal whitespace
        c_str = c_str.strip()               # trim ends
        c_str = c_str.lower()               # lowercase for robustness
        new_cols.append(c_str)
    df.columns = new_cols
    return df

def main():
    print("Loading", INPUT_XLSX)
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

    # normalize column names so you don't get bitten by spaces/caps
    df = normalize_column_names(df)
    print("Available (normalized) columns:", list(df.columns))

    if TEXT_COL_CLEAN not in df.columns:
        raise ValueError(
            f"Normalized column '{TEXT_COL_CLEAN}' not found. Available columns: {list(df.columns)}"
        )

    print("Computing sentiment (VADER)...")
    out = compute_sentiment(df, TEXT_COL_CLEAN)
    out.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print("Done. Results saved to", OUTPUT_XLSX)

if __name__ == "__main__":
    main()
