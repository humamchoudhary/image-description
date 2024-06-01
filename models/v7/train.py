import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
sys.path.insert(0, parent_dir)

import torch
import random
import numpy as np


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
set_seed(seed)

import torch
import numpy as np
from const import *
from hyperparm import *
import tqdm
from matplotlib import pyplot as plt
from v7 import model, criterion, optimizer
from dataloader import cap_train_dl, cap_val_dl
import gc
import multiprocessing as mp

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
    # print("train")
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
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_bleu += calculate_accuracy(output, trg)

        del imgs, trg, output, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss = epoch_loss / len(train_loader)
    train_bleu = epoch_bleu / len(train_loader)
    print(
        f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_bleu:.2f}"
    )

    model.eval()
    val_loss = 0
    val_bleu = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, trg = batch
            imgs, trg = imgs.to(device, non_blocking=True), trg.to(
                device, non_blocking=True
            )

            output = model(imgs, trg, 0.0)  # Turn off teacher forcing for validation
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            val_loss += loss.item()
            val_bleu += calculate_accuracy(output, trg)
            del imgs, trg, output, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss = val_loss / len(val_loader)
    val_bleu = val_bleu / len(val_loader)
    print(
        f"\tVal Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f} | Val Acc: {val_bleu:.2f}"
    )
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
    train_losses = []
    val_losses = []
    train_bleus = []
    val_bleus = []
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000, mode='triangular')

    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

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
        # scheduler.step(val_loss)
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
        plt.savefig("model-v7-4-t-v-loss.png")

        plt.figure(figsize=(10, 6))
        plt.plot(train_bleus, label="Train Acc")
        plt.plot(val_bleus, label="Validation BLEU")
        plt.xlabel("Epoch")
        plt.ylabel("BLEU Score")
        plt.title("Training and Validation BLEU Score")
        plt.legend()
        plt.show()

        plt.savefig("model-v7-5-t-v-accu.png")
    save_model(model, optimizer, "./model-v7-5-full.pth")


if __name__ == "__main__":
    train(
        model,
        cap_train_dl,
        cap_val_dl,
        optimizer,
        criterion,
        "./model-v7-5.pth",
        # "./model-v7-2.pth",
        # True,
    )
