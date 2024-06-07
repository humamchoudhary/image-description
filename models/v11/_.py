import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
import torch
import pandas as pd
import numpy as np
import random
import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from v11 import model, criterion
from dataloader import dataloader, CaptionDataset, transform


# Set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed = 42
# set_seed(seed)


# Load the test dataset


# Generate predictions
def generate_predictions(model, test_loader, device):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            imgs, trg = batch
            imgs = imgs.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            output = model(imgs, trg)
            preds = torch.argmax(output, dim=-1)

            predictions.extend(preds.cpu().numpy())
            references.extend(trg.cpu().numpy())

    return predictions, references


# Calculate BLEU score
def calculate_bleu(predictions, references):
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        bleu_scores.append(sentence_bleu([ref], pred, smoothing_function=smoothie))
    return np.mean(bleu_scores)


# Calculate ROUGE score
def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(" ".join(map(str, ref)), " ".join(map(str, pred)))
        rouge_scores.append(scores)
    return rouge_scores


from dataloader import tokenizer

# Main function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = "./model-v11-2.pth"
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.to(device)

    # Load test data
    test_data_path = "test_dataset.csv"
    test_df = pd.read_csv("./dataset/csv/test_dataset.csv")

    cap_test_dl, _ = dataloader(
        test_df,
        CaptionDataset,
        16,
        transform,
        transform,
        train_size=1.0,
        tokenizer=tokenizer,
    )
    print(len(_))

    # Generate predictions
    predictions, references = generate_predictions(model, cap_test_dl, device)

    # Calculate BLEU score
    bleu_score = calculate_bleu(predictions, references)
    print(f"BLEU Score: {bleu_score:.4f}")

    # Calculate ROUGE scor
    rouge_scores = calculate_rouge(predictions, references)
    avg_rouge1 = np.mean([score["rouge1"].fmeasure for score in rouge_scores])
    avg_rouge2 = np.mean([score["rouge2"].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score["rougeL"].fmeasure for score in rouge_scores])
    print(f"ROUGE-1 Score: {avg_rouge1:.4f}")
    print(f"ROUGE-2 Score: {avg_rouge2:.4f}")
    print(f"ROUGE-L Score: {avg_rougeL:.4f}")
