import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
import torch
import numpy as np
from const import *
from hyperparm import *
import tqdm
from matplotlib import pyplot as plt
from v4 import model, criterion, optimizer
from dataloader import cap_train_dl, cap_val_dl
import gc
import multiprocessing as mp


def train_val_fn(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    teacher_forcing_ratio,
    device,
):
    print("train")
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(train_loader):
        print(i)
        imgs, trg = batch
        imgs, trg = imgs.to(device, non_blocking=True), trg.to(
            device, non_blocking=True
        )

        optimizer.zero_grad()
        output = model(imgs, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1, 1)  # Add an extra dimension to trg
        loss = criterion(output, trg.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        del imgs, trg, output, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss = epoch_loss / len(train_loader)
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, trg = batch
            imgs, trg = imgs.to(device, non_blocking=True), trg.to(
                device, non_blocking=True
            )

            output = model(imgs, trg, 0.0)  # Turn off teacher forcing for validation
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1, 1)  # Add an extra dimension to trg
            loss = criterion(output, trg.float())
            val_loss += loss.item()

            del imgs, trg, output, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss = val_loss / len(val_loader)
    print(f"\tVal Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f}")

    return train_loss, val_loss


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

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, val_loss = train_val_fn(
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
        print(
            f"Epoch: {epoch+1} | Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f} | Val Loss: {val_loss:7.3f} | Val PPL: {np.exp(val_loss):7.3f}"
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
    save_model(model, optimizer, "./model-v4-full.pth")


if __name__ == "__main__":
    train(model, cap_train_dl, cap_val_dl, optimizer, criterion, "./model-v4.pth")
