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
from models.const import *
from models.hyperparm import *
from models.v7.v7 import model

print(device)
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

checkpoint_path = "model-v7-2.pth"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Load the state dictionaries into the model and optimizer
model.load_state_dict(checkpoint["model_state_dict"])
from PIL import Image


def generate_caption(img, max_output_length=15):
    global vocab
    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        encoder_outputs, hidden = model.encoder(img)
        inputs = [vocab.lookup_indices([sos_token])[0]]
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden = model.decoder(inputs_tensor, encoder_outputs, hidden)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == vocab.lookup_indices([eos_token])[0]:
                break
        tokens = vocab.lookup_tokens(inputs)
    return tokens


def tokens_to_sentence(tokens):
    sentence = " ".join(
        token for token in tokens if token not in ["<sos>", "<eos>", "<pad>"]
    )
    sentence = sentence.capitalize()
    return sentence


def dummy_prediction_function(image_path):
    img = Image.open(image_path).convert("RGB")
    image = np.array(img.resize((224, 224))) / 255.0
    image = torch.tensor(image).permute(2, 0, 1)
    image = image.float()
    predicted_tokens = generate_caption(image)
    predicted_caption = tokens_to_sentence(predicted_tokens)
    # This is a dummy prediction function. Replace it with your actual prediction logic.
    return predicted_caption


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
            prediction = dummy_prediction_function(filepath)
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
