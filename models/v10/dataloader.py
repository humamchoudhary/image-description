import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
sys.path.insert(0, parent_dir)
from const import *
from hyperparm import *

# from models.v10.const import *
# from models.v10.hyperparm import *
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import requests
from io import BytesIO
import torchvision.transforms as transforms
from transformers import BertTokenizer

no_workers = 8


class CaptionDataset(Dataset):
    def __init__(self, df, transform=None, tokenizer=None):
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer
        self.w, self.h = 224, 224

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        caption = row["text"]

        if image_path.startswith("http"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Tokenize the caption
        tokenized_caption = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=30,
        )["input_ids"].squeeze(0)

        return image, tokenized_caption

    def collate_fn(self, batch):
        images, captions = zip(*batch)
        captions = nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        images = torch.stack(images)
        return images, captions


# Define transformations
transform = transforms.Compose(
    [
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.3),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.3),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                )
            ],
            p=0.3,
        ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def dataloader(
    dataset,
    dataset_class,
    batch_size,
    train_transform=None,
    val_transform=None,
    tokenizer=None,
    train_size=0.8,
):
    if type(dataset) == pd.DataFrame:
        train_ds = dataset.sample(frac=train_size, random_state=42)
        val_ds = dataset.drop(train_ds.index)

    train_dataset = dataset_class(
        train_ds, transform=train_transform, tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=no_workers,
        pin_memory=True,
    )

    val_dataset = dataset_class(val_ds, transform=val_transform, tokenizer=tokenizer)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=no_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Assuming df_caption is your dataframe with 'image_path' and 'text' columns
cap_train_dl, cap_val_dl = dataloader(
    df_caption, CaptionDataset, batch_size, transform, transform, tokenizer=tokenizer
)

if __name__ == "__main__":
    for images, captions in cap_train_dl:
        print(images.shape)
        print(captions.shape)
        print(captions)
        break
