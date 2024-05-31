import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
import torch
import numpy as np
from const import *
from hyperparm import *
import tqdm
from matplotlib import pyplot as plt
from v8 import model, criterion, optimizer
from dataloader import cap_train_dl, cap_val_dl
import gc
import multiprocessing as mp


def calculate_accuracy(output, target):
    """Calculate accuracy based on predictions and target labels."""
    _, pred = torch.max(output, dim=1)
    correct = (pred == target).sum().item()
    total = target.size(0)
    return correct / total


def train_val_fn(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    teacher_forcing_ratio,
    device,
):
    model.train()
    epoch_loss = 0
    epoch_bleu = 0

    # for epoch in tqdm.tqdm(range(n_epochs)):
    for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
        # print(i)

        imgs, trg = batch
        imgs, trg = imgs.to(device, non_blocking=True), trg.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()
        output = model(imgs, trg, teacher_forcing_ratio)

        # Reshape output for loss calculation
        output_dim = output.shape[-1]
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_bleu += calculate_accuracy(output, trg)
        del imgs, trg, output, loss
        torch.cuda.empty_cache()
        gc.collect()
    train_bleu = epoch_bleu / len(train_loader)

    train_loss = epoch_loss / len(train_loader)
    # print(
    #     f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_bleu:.2f}"
    # )
    model.eval()
    val_loss = 0
    val_bleu = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, trg = batch
            imgs, trg = imgs.to(device, non_blocking=True), trg.to(
                device, non_blocking=True
            )

            # Forward pass with no teacher forcing
            output = model(imgs, trg, 0.0)

            # Reshape output for loss calculation
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            val_loss += loss.item()
            val_bleu += calculate_accuracy(output, trg)

            del imgs, trg, output, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss = val_loss / len(val_loader)
    val_bleu = val_bleu / len(val_loader)
    # print(
    #     f"\tVal Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f} | Val Acc: {val_bleu:.2f}"
    # )
    return train_loss, val_loss, train_bleu, val_bleu


mp.set_start_method("fork", force=True)


def save_model(model, optimizer, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train(model, train_dl, val_dl, optim, crit, model_path):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_bleus = []
    val_bleus = []

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, val_loss, train_bleu, val_bleu = train_val_fn(
            model,
            train_dl,
            val_dl,
            optim,
            crit,
            teacher_forcing_ratio,
            device,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_bleus.append(train_bleu)
        val_bleus.append(val_bleu)
        print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_bleu:.2f} "
            f"| Val Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f} | Val Acc: {val_bleu:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, crit, model_path)

        torch.cuda.empty_cache()
        gc.collect()

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
        plt.savefig("model-v8-4-t-v-loss.png")

        plt.figure(figsize=(10, 6))
        plt.plot(train_bleus, label="Train Acc")
        plt.plot(val_bleus, label="Validation BLEU")
        plt.xlabel("Epoch")
        plt.ylabel("BLEU Score")
        plt.title("Training and Validation BLEU Score")
        plt.legend()
        plt.show()

        plt.savefig("model-v8-4-t-v-accu.png")
    save_model(model, optimizer, "./model-v8-4-full.pth")


if __name__ == "__main__":
    train(model, cap_train_dl, cap_val_dl, optimizer, criterion, "./model-v8-4.pth")
