from const import *
from hyperparm import *
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import requests
from io import BytesIO
import torchvision.transforms as transforms

no_workers = 64


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
        caption = torch.tensor(caption)
        # print(idx, end=" ")

        if image_path.startswith("http"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        # print(image)
        return image, caption

    def collate_fn(self, batch):
        images, caption = zip(*batch)
        caption = nn.utils.rnn.pad_sequence(caption, padding_value=pad_index)
        images = torch.stack(images)
        return images, caption


transform = transforms.Compose(
    [
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
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=no_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


cap_train_dl, cap_val_dl = dataloader(
    df_caption, CaptionDataset, batch_size, transform, transform
)
if __name__ == "__main__":
    print(cap_train_dl)
    print(len(cap_train_dl))
