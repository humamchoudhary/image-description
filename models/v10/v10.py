import torch
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_dir)
sys.path.insert(0, parent_dir)
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from transformers import BertModel
from models.v10.dataloader import tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        image_model_name="efficientnet-b0",
        bert_model_name="bert-base-uncased",
        embed_size=256,
        hidden_size=512,
        vocab_size=30522,
        num_layers=1,
    ):
        super(ImageCaptioningModel, self).__init__()

        # Load pre-trained EfficientNet
        self.image_model = EfficientNet.from_pretrained(image_model_name)
        self.image_model._fc = nn.Linear(self.image_model._fc.in_features, embed_size)

        # Load pre-trained BERT
        self.bert_model = BertModel.from_pretrained(bert_model_name)

        # Linear layer to transform BERT embeddings to the same size as image features
        self.linear_transform = nn.Linear(
            self.bert_model.config.hidden_size, embed_size
        )

        # Define LSTM for generating captions
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # Extract features from images
        features = self.image_model(images)

        # Get embeddings for captions
        embeddings = self.bert_model.embeddings(captions)

        # Transform BERT embeddings to match the size of image features
        embeddings = self.linear_transform(embeddings)

        # Concatenate image features and caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        # Generate captions
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


# Initialize the model
model = ImageCaptioningModel()

# Move the model to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

if __name__ == "__main__":
    print(model)
    from torchsummary import summary

    dummy_images = torch.randn(1, 3, 224, 224).to(
        device
    )  # Batch size of 1, 3 color channels, 224x224 image
    dummy_captions = ["This is a dummy caption"]  # A dummy caption
    dummy_embeddings = model.bert_model.embeddings(
        tokenizer(dummy_captions, return_tensors="pt", padding=True, truncation=True)[
            "input_ids"
        ].to(device)
    )

    # Print the model summary
    summary(model, [(3, 224, 224), dummy_embeddings.size()[1:]], device=str(device))
