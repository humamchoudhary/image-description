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

from const import *
from hyperparm import *
from v11 import model, optimizer, criterion
from dataloader import cap_train_dl, cap_val_dl


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


def calculate_accuracy(output, target):
    _, pred = torch.max(output, dim=1)
    correct = (pred == target).sum().item()
    total = target.size(0)
    return correct / total


# Training and validation function
def train_val_fn(
    model, train_loader, val_loader, optimizer, criterion, teacher_forcing_ratio, device
):
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
        epoch_acc += calculate_accuracy(output, trg)

        del imgs, trg, output, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss = epoch_loss / len(train_loader)
    train_acc = epoch_acc / len(train_loader)
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}")

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
            val_acc += calculate_accuracy(output, trg)

            del imgs, trg, output, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)
    print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}")
    return train_loss, val_loss, train_acc, val_acc


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
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    patience = 5
    epochs_no_improve = 0

    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, val_loss, train_acc, val_acc = train_val_fn(
            model, train_dl, val_dl, optim, crit, teacher_forcing_ratio, device
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}"
        )

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
        plt.savefig("model-v11-1-t-v-loss.png")

        plt.figure(figsize=(10, 6))
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()
        plt.savefig("model-v11-1-t-v-accu.png")
        if epochs_no_improve == patience:
            print("Early stopping")
            break

    save_model(model, optimizer, "./model-v11-1-final.pth")


# Main function
if __name__ == "__main__":
    train(model, cap_train_dl, cap_val_dl, optimizer, criterion, "./model-v11-1.pth")
