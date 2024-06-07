from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import os
import numpy as np

# from models.model import model, vocab, device, sos_token, eos_token
import torch
from models.v11.const import *
from models.v11.hyperparm import *
from models.v11.v11 import model
from transformers import BertTokenizer

print(device)
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

checkpoint_path = "./model-v11-2.pth"
# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Load the state dictionaries into the model and optimizer
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
from PIL import Image


# def generate_caption(img, max_output_length=15):
#     global vocab
#     model.eval()
#     with torch.no_grad():
#         img = img.unsqueeze(0).to(device)
#         encoder_outputs, hidden = model.encoder(img)
#         inputs = [vocab.lookup_indices([sos_token])[0]]
#         for _ in range(max_output_length):
#             inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
#             output, hidden = model.decoder(inputs_tensor, encoder_outputs, hidden)
#             predicted_token = output.argmax(-1).item()
#             inputs.append(predicted_token)
#             if predicted_token == vocab.lookup_indices([eos_token])[0]:
#                 break
#         tokens = vocab.lookup_tokens(inputs)
#     return tokens


# # For transformers
# def generate_caption(img, max_output_length=15):
#     global vocab
#     model.eval()
#     with torch.no_grad():
#         img = img.unsqueeze(0).to(device)
#         encoder_outputs = model.encoder(img)
#         inputs = [vocab.lookup_indices([sos_token])[0]]
#         for _ in range(max_output_length):
#             inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)

#             output = model.decoder(
#                 inputs_tensor.unsqueeze(0), encoder_outputs.permute(1, 0, 2)
#             )
#             predicted_token = output.argmax(-1).item()
#             inputs.append(predicted_token)
#             if predicted_token == vocab.lookup_indices([eos_token])[0]:
#                 break
#         tokens = vocab.lookup_tokens(inputs)
#     return tokens


def tokens_to_sentence(tokens):
    sentence = " ".join(
        token for token in tokens if token not in ["<sos>", "<eos>", "<pad>"]
    )
    sentence = sentence.capitalize()
    return sentence


# ==================================================================== V10


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def generate_caption(image, max_length=30):
    model.eval()
    # Start with the [CLS] token (BERT's start token)
    input_ids = torch.tensor([tokenizer.cls_token_id], device=device).unsqueeze(0)
    print(input_ids)
    for _ in range(max_length):
        # Get the model output
        with torch.no_grad():
            output = model(image, input_ids)

        # Get the most likely next token
        next_token_id = output[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        # Append the next token to the input_ids
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

        # Stop if we generate the [SEP] token (BERT's end token)
        if next_token_id == tokenizer.sep_token_id:
            break

    return input_ids


def untokenize(token_ids):
    return tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)


from torchvision import transforms

# Define image transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# Load and transform the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)


def prediction(image_path):
    image = load_image(image_path)
    predicted_tokens = generate_caption(image)
    # predicted_caption = tokens_to_sentence(predicted_tokens)
    # This is a dummy prediction function. Replace it with your actual prediction logic.
    return untokenize(predicted_tokens)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            prediction = prediction(filepath)
            image_url = url_for("uploaded_file", filename=filename)
            return render_template(
                "result.html", image_url=image_url, prediction=prediction
            )
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
