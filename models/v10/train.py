import random
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
sys.path.insert(0, parent_dir)
import torch
import tqdm
import gc
from matplotlib import pyplot as plt
from rouge_score import rouge_scorer

from const import *
from hyperparm import *
from v10 import model, optimizer, criterion
from dataloader import cap_train_dl, cap_val_dl, tokenizer


# Set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# seed = 42
# set_seed(seed)


def decode_batch(sequences, tokenizer):
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]


def calculate_rouge(predictions, references, tokenizer):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    decoded_predictions = decode_batch(predictions, tokenizer)
    decoded_references = decode_batch(references, tokenizer)

    for pred_text, ref_text in zip(decoded_predictions, decoded_references):
        score = scorer.score(ref_text, pred_text)
        scores["rouge1"].append(score["rouge1"].fmeasure)
        scores["rouge2"].append(score["rouge2"].fmeasure)
        scores["rougeL"].append(score["rougeL"].fmeasure)

    avg_scores = {key: np.mean(value) for key, value in scores.items()}
    return avg_scores


def generate_predictions(model, data_loader, device):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
            imgs, trg = batch
            imgs = imgs.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

            output = model(imgs, trg)
            preds = torch.argmax(output, dim=-1)

            predictions.extend(preds.cpu().numpy())
            references.extend(trg.cpu().numpy())

    return predictions, references


# Training and validation function
def train_val_fn(
    model, train_loader, val_loader, optimizer, criterion, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    epoch_rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
        imgs, trg = batch
        imgs, trg = imgs.to(device, non_blocking=True), trg.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()
        output = model(imgs, trg)
        batch_size = trg.size(0)
        target_seq_len = trg.size(1)
        output_seq_len = output.size(1)

        if output_seq_len != target_seq_len:
            # Adjust the sequence length to match
            min_seq_len = min(output_seq_len, target_seq_len)
            output = output[:, :min_seq_len, :]
            trg = trg[:, :min_seq_len]

        # Reshape output and target tensors
        output = output.contiguous().view(-1, output.shape[-1])  # flatten
        trg = trg.contiguous().view(-1)  # flatten

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        preds = torch.argmax(output, dim=-1)
        batch_rouge_scores = calculate_rouge(
            preds.cpu().numpy(), trg.cpu().numpy(), tokenizer
        )
        for key in epoch_rouge_scores:
            epoch_rouge_scores[key] += batch_rouge_scores[key]

        del imgs, trg, output, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss = epoch_loss / len(train_loader)
    train_rouge_scores = {
        key: value / len(train_loader) for key, value in epoch_rouge_scores.items()
    }
    print(
        f"\tTrain Loss: {train_loss:.3f} | Train ROUGE-1: {train_rouge_scores['rouge1']:.4f} | Train ROUGE-2: {train_rouge_scores['rouge2']:.4f} | Train ROUGE-L: {train_rouge_scores['rougeL']:.4f}"
    )

    model.eval()
    val_loss = 0
    val_rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, total=len(val_loader)):
            imgs, trg = batch
            imgs, trg = imgs.to(device, non_blocking=True), trg.to(
                device, non_blocking=True
            )

            output = model(imgs, trg)

            target_seq_len = trg.size(1)
            output_seq_len = output.size(1)

            if output_seq_len != target_seq_len:
                # Adjust the sequence length to match
                min_seq_len = min(output_seq_len, target_seq_len)
                output = output[:, :min_seq_len, :]
                trg = trg[:, :min_seq_len]

            # Reshape output and target tensors
            output = output.contiguous().view(-1, output.shape[-1])  # flatten
            trg = trg.contiguous().view(-1)  # flatten

            loss = criterion(output, trg)
            val_loss += loss.item()

            preds = torch.argmax(output, dim=-1)
            batch_rouge_scores = calculate_rouge(
                preds.cpu().numpy(), trg.cpu().numpy(), tokenizer
            )
            for key in val_rouge_scores:
                val_rouge_scores[key] += batch_rouge_scores[key]

            del imgs, trg, output, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss = val_loss / len(val_loader)
    val_rouge_scores = {
        key: value / len(val_loader) for key, value in val_rouge_scores.items()
    }
    print(
        f"\tVal Loss: {val_loss:.3f} | Val ROUGE-1: {val_rouge_scores['rouge1']:.4f} | Val ROUGE-2: {val_rouge_scores['rouge2']:.4f} | Val ROUGE-L: {val_rouge_scores['rougeL']:.4f}"
    )
    return train_loss, val_loss, train_rouge_scores, val_rouge_scores


# Save model function
def save_model(model, optimizer, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# Training function
def train(
    model,
    train_dl,
    val_dl,
    optim,
    crit,
    model_path,
    checkpoint_path=None,
    load_model=False,
):
    best_val_loss = float("inf")
    best_val_acc = -float("inf")
    train_losses = []
    val_losses = []
    train_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    val_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    patience = 5
    epochs_no_improve = 0

    if load_model:

        checkpoint = torch.load(checkpoint_path)
        # print(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, val_loss, train_rouge, val_rouge = train_val_fn(
            model, train_dl, val_dl, optim, crit, teacher_forcing_ratio, device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for key in train_rouge_scores:
            train_rouge_scores[key].append(train_rouge[key])
            val_rouge_scores[key].append(val_rouge[key])

        print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f} | Train ROUGE-1: {train_rouge['rouge1']:.4f} | Train ROUGE-2: {train_rouge['rouge2']:.4f} | Train ROUGE-L: {train_rouge['rougeL']:.4f} | Val Loss: {val_loss:.3f} | Val ROUGE-1: {val_rouge['rouge1']:.4f} | Val ROUGE-2: {val_rouge['rouge2']:.4f} | Val ROUGE-L: {val_rouge['rougeL']:.4f}"
        )

        if val_rouge["rouge1"] > best_val_acc:
            best_val_acc = val_rouge["rouge1"]
            save_model(model, optim, "mode-v10-3-2-acc.pth")
            # epochs_no_improve = 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optim, model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
        plt.savefig("model-v10-3-2-t-v-loss.png")

        plt.figure(figsize=(10, 6))
        plt.plot(train_rouge_scores["rouge1"], label="Train ROUGE-1")
        plt.plot(val_rouge_scores["rouge1"], label="Validation ROUGE-1")
        plt.plot(train_rouge_scores["rouge2"], label="Train ROUGE-2")
        plt.plot(val_rouge_scores["rouge2"], label="Validation ROUGE-2")
        plt.plot(train_rouge_scores["rougeL"], label="Train ROUGE-L")
        plt.plot(val_rouge_scores["rougeL"], label="Validation ROUGE-L")
        plt.xlabel("Epoch")
        plt.ylabel("ROUGE Score")
        plt.title("Training and Validation ROUGE Scores")
        plt.legend()
        plt.show()
        plt.savefig("model-v10-3-2-t-v-rouge.png")

        if epochs_no_improve == patience:
            print("Early stopping")
            break

    save_model(model, optimizer, "./model-v10-3-2-final.pth")


# Main function
if __name__ == "__main__":
    train(
        model,
        cap_train_dl,
        cap_val_dl,
        optimizer,
        criterion,
        "./model-v10-3-2.pth",
        load_model=True,
        checkpoint_path="./model-v10-3.pth",
    )
