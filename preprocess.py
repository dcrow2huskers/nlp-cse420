# Load go_emotions and turn it into a single-label sentiment dataset
import os
import re
import pandas as pd
from datasets import load_dataset

OUT_DIR = "data"
DATASET_NAME = "SetFit/go_emotions"

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# GoEmotions sentiment grouping
SENTIMENT_MAP = {
    "admiration": "positive", "amusement": "positive", "approval": "positive",
    "caring": "positive", "desire": "positive", "excitement": "positive",
    "gratitude": "positive", "joy": "positive", "love": "positive",
    "optimism": "positive", "pride": "positive", "relief": "positive",
    "anger": "negative", "annoyance": "negative", "disappointment": "negative",
    "disapproval": "negative", "disgust": "negative", "embarrassment": "negative",
    "fear": "negative", "grief": "negative", "nervousness": "negative",
    "remorse": "negative", "sadness": "negative",
    "confusion": "ambiguous", "curiosity": "ambiguous",
    "realization": "ambiguous", "surprise": "ambiguous",
    "neutral": "neutral",
}


def clean_text(text):
    # collapse whitespace and strip
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def to_single_label(df):
    df = df.copy()
    df["num_labels"] = df[EMOTIONS].values.sum(axis=1)

    # keep rows with exactly one emotion
    single = df[df["num_labels"] == 1].reset_index(drop=True)

    # emotion is the column set to 1, then map it to a sentiment
    emo_idx = single[EMOTIONS].values.argmax(axis=1)
    single["emotion"] = [EMOTIONS[i] for i in emo_idx]
    single["sentiment"] = single["emotion"].map(SENTIMENT_MAP)

    return single[["text", "emotion", "sentiment"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)
    print(ds)

    # clean and process each split
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        df["text"] = df["text"].apply(clean_text)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)

        before = len(df)
        proc = to_single_label(df)

        out_path = os.path.join(OUT_DIR, f"{split_name}.csv")
        proc.to_csv(out_path, index=False)
        print(f"\n[{split_name}] kept {len(proc)}/{before} single-label rows -> {out_path}")
        print(proc["sentiment"].value_counts())

    print("\nDone. Data written to ./data/")


if __name__ == "__main__":
    main()
