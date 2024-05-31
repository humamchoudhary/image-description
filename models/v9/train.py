import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from const import *
from hyperparm import *
import torch
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import gc
import multiprocessing as mp

# Adjust import paths as needed
from v9 import model, criterion, optimizer
from dataloader import cap_train_dl, cap_val_dl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(output, target):
    """Calculate accuracy based on predictions and target labels."""
    _, pred = torch.max(output, dim=1)
    correct = (pred == target).sum().item()
    total = target.size(0)
    return correct / total


def train_val_fn(model, train_loader, val_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
        imgs, trg = batch
        imgs, trg = imgs.to(device, non_blocking=True), trg.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()
        output = model(imgs, trg)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += calculate_accuracy(output, trg)

        del imgs, trg, output, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss = epoch_loss / len(train_loader)
    train_acc = epoch_acc / len(train_loader)
    print(
        f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_acc:.2f}"
    )

    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs, trg = batch
            imgs, trg = imgs.to(device, non_blocking=True), trg.to(
                device, non_blocking=True
            )

            output = model(imgs, trg)
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            loss = criterion(output, trg)
            val_loss += loss.item()
            val_acc += calculate_accuracy(output, trg)

            del imgs, trg, output, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)
    print(
        f"\tVal Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f} | Val Acc: {val_acc:.2f}"
    )
    return train_loss, val_loss, train_acc, val_acc


mp.set_start_method("fork", force=True)


def save_model(model, optimizer, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train(model, train_dl, val_dl, optim, crit, model_path, n_epochs=10):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, factor=0.5
    )

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, val_loss, train_acc, val_acc = train_val_fn(
            model,
            train_dl,
            val_dl,
            optim,
            crit,
            device,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Train Acc: {train_acc:.2f} "
            f"| Val Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f} | Val Acc: {val_acc:.2f}"
        )
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optim, model_path)

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
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Validation Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()
        plt.savefig("model-v9-t-v-accu.png")
    save_model(model, optim, "./model-v9-full.pth")


if __name__ == "__main__":
    model = model.to(device)
    train(model, cap_train_dl, cap_val_dl, optimizer, criterion, "./model-v9.pth")
