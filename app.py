from flask import Flask, request, jsonify
import os, requests, torch, pickle
from nanoGPT.model import GPT, GPTConfig

# Auto-download ckpt.pt from Google Drive if missing
def download_model():
    url = "https://drive.google.com/uc?id=1-8qve0TOFVn46pRaqeDfxIkn_o_KBa--&export=download"
    out_path = "out_ai_gf/ckpt.pt"
    os.makedirs("out_ai_gf", exist_ok=True)
    if not os.path.exists(out_path):
        print("Downloading ckpt.pt from Google Drive...")
        r = requests.get(url)
        with open(out_path, "wb") as f:
            f.write(r.content)
        print("Download complete.")

download_model()

# Load model config
with open("out_ai_gf/config.pkl", "rb") as f:
    model_args = pickle.load(f)

model = GPT(GPTConfig(**model_args))

# ✅ Fix for PyTorch 2.6+ — set weights_only=False
checkpoint = torch.load("out_ai_gf/ckpt.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

# Load tokenizer info
with open("out_ai_gf/meta.pkl", "rb") as f:
    meta = pickle.load(f)

itos = meta["itos"]
stoi = meta["stoi"]

def encode(text): return [stoi.get(c, 0) for c in text]
def decode(tokens): return ''.join([itos.get(i, '') for i in tokens])

# Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    x = torch.tensor([encode(user_input)], dtype=torch.long)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100)[0].tolist()
    reply = decode(y[len(x[0]):])
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
