from flask import Flask, request, jsonify
import os, torch, pickle
from nanoGPT.model import GPT, GPTConfig

# Auto-download ckpt.pt from Google Drive using gdown
def download_model():
    import gdown
    url = "https://drive.google.com/uc?id=1-8qve0TOFVn46pRaqeDfxIkn_o_KBa--"
    out_path = "out_ai_gf/ckpt.pt"
    os.makedirs("out_ai_gf", exist_ok=True)
    if not os.path.exists(out_path):
        print("Downloading ckpt.pt from Google Drive...")
        gdown.download(url, out_path, quiet=False)
        print("✅ Download complete.")

download_model()

# Load model config
with open("out_ai_gf/config.pkl", "rb") as f:
    model_args = pickle.load(f)

model = GPT(GPTConfig(**model_args))

# ✅ Fix for PyTorch 2.6+ and key mismatch (ln_1 → ln1)
checkpoint = torch.load("out_ai_gf/ckpt.pt", map_location="cpu", weights_only=False)

fixed_state_dict = {}
for k, v in checkpoint["model"].items():
    new_key = k.replace("ln_1", "ln1").replace("ln_2", "ln2")
    fixed_state_dict[new_key] = v

# 🔥 THIS LINE FIXES THE ERROR!
model.load_state_dict(fixed_state_dict, strict=False)


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
    port = int(os.environ.get("PORT", 10000))  # Use PORT env var (Render default)
    app.run(host="0.0.0.0", port=port)
