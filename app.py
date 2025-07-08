from flask import Flask, request, jsonify
import torch
import pickle
from nanoGPT.model import GPT, GPTConfig

# Load model config
with open("out_ai_gf/config.pkl", "rb") as f:
    model_args = pickle.load(f)

model = GPT(GPTConfig(**model_args))
model.load_state_dict(torch.load("out_ai_gf/ckpt.pt", map_location="cpu"))
model.eval()

# Load token mappings
with open("out_ai_gf/meta.pkl", "rb") as f:
    meta = pickle.load(f)

itos = meta["itos"]
stoi = meta["stoi"]

def encode(text):
    return [stoi.get(ch, 0) for ch in text]

def decode(indices):
    return ''.join([itos.get(i, '') for i in indices])

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    x = torch.tensor([encode(user_input)], dtype=torch.long)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100)[0].tolist()
    generated = decode(y[len(x[0]):])  # skip prompt
    return jsonify({"response": generated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
