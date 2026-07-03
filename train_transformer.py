# Model 2: fine-tuned DistilBERT
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

from utils import load_split, evaluate, LABELS

RESULTS_DIR = "results"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor([label2id[l] for l in labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS),
        id2label=id2label, label2id=label2id,
    ).to(device)

    train_loader = DataLoader(
        TextDataset(X_train, y_train, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TextDataset(X_test, y_test, tokenizer),
        batch_size=BATCH_SIZE,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Fine-tuning DistilBERT ...")
    model.train()
    for epoch in range(EPOCHS):
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            running += out.loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"loss {out.loss.item():.4f}")
        print(f"epoch {epoch+1} avg loss {running/len(train_loader):.4f}")

    print("Evaluating on test set ...")
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())

    y_pred = [id2label[int(p)] for p in preds]

    print("\n" + classification_report(y_test, y_pred, labels=LABELS, digits=3))

    metrics = evaluate(y_test, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.3f} "
          f"(95% CI {metrics['accuracy_ci'][0]:.3f}-{metrics['accuracy_ci'][1]:.3f})")
    print(f"Macro-F1: {metrics['macro_f1']:.3f} "
          f"(95% CI {metrics['macro_f1_ci'][0]:.3f}-{metrics['macro_f1_ci'][1]:.3f})")

    # save predictions for the comparison script (confusion matrix, etc.)
    import pandas as pd
    pd.DataFrame({"true": y_test, "pred": y_pred}).to_csv(
        f"{RESULTS_DIR}/transformer_preds.csv", index=False)

    metrics["model"] = "distilbert"
    with open(f"{RESULTS_DIR}/transformer.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR}/transformer.json")


if __name__ == "__main__":
    main()
