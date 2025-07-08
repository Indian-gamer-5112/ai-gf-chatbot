from flask import Flask, request, jsonify
import os, torch, pickle, requests
from nanoGPT.model import GPT, GPTConfig

# Automatically download ckpt.pt if missing
def download_model():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"
    out_path = "out_ai_gf/ckpt.pt"
    os.makedirs("out_ai_gf", exist_ok=True)
    if not os.path.exists(out_path):
        print("Downloading ckpt.pt...")
        r = requests.get(url)
        with open(out_path, "wb") as f:
            f.write(r.content)
        print("Downloaded.")

download_model()

# Load model and config
with open("out_ai_gf/config.pkl", "rb") as f:
    model_args = pickle.load(f)

model = GPT(GPTConfig(**model_args))
model.load_state_dict(torch.load("out_ai_gf/ckpt.pt", map_location="cpu"))
model.eval()

# Token mappings
with open("out_ai_gf/meta.pkl", "rb") as f:
    meta = pickle.load(f)
itos = meta["itos"]
stoi = meta["stoi"]

def encode(text): return [stoi.get(c, 0) for c in text]
def decode(tokens): return ''.join([itos.get(i, '') for i in tokens])

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.json.get("message", "")
    x = torch.tensor([encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100)[0].tolist()
    reply = decode(y[len(x[0]):])
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
