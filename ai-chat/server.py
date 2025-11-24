from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gpt_model import GPTModel
from tokenizer import BPETokenizer
import json
import time
import os
import re
import signal
import sys
from background_trainer import BackgroundTrainer

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# Serve Frontend
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path in ['style.css', 'app.js', 'manifest.json']:
        return send_from_directory(BASE_DIR, path)
    return jsonify({"error": "File not found"}), 404

# ----------------------------------------------------------------------
# Load the pre-trained model (once)
# ----------------------------------------------------------------------
print("=" * 70)
print("🧠 GPT CHAT SERVER - Transformer Architecture + BPE")
print("=" * 70)

# 1. Load Tokenizer
tokenizer = BPETokenizer(vocab_size=1000)
if os.path.exists('tokenizer.json'):
    tokenizer.load('tokenizer.json')
    print(f"✅ Tokenizer loaded. Vocab: {tokenizer.vocab_size}")
else:
    print("⚠️ No tokenizer found! Please run pretrain.py first.")
    # Fallback to untrained tokenizer (byte-level)
    tokenizer.vocab_size = 256

# 2. Initialize GPT
gpt = GPTModel(tokenizer, d_model=64, n_layer=4, n_head=4, block_size=32)

MODEL_FILE = "gpt_model.pkl"

# Check if we can load the model (vocab must match)
if os.path.exists(MODEL_FILE):
    # Peek at the saved vocab size without fully loading
    import pickle
    try:
        with open(MODEL_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Check saved vocab size
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            saved_params = checkpoint['params']
        else:
            saved_params = checkpoint
        
        saved_lm_head = saved_params.get('lm_head_w')
        if saved_lm_head is not None:
            saved_vocab_size = saved_lm_head.shape[1]  # (d_model, vocab_size)
            
            if saved_vocab_size != tokenizer.vocab_size:
                print(f"⚠️ Vocab size mismatch! Saved: {saved_vocab_size}, Current: {tokenizer.vocab_size}")
                print(f"🔄 Starting fresh with new tokenizer...")
                # Don't load - start from scratch
            else:
                if gpt.load(MODEL_FILE):
                    print(f"✅ Model loaded successfully")
                    print(f"📊 Vocabulary size: {gpt.vocab_size} tokens")
                else:
                    print(f"⚠️ Failed to load {MODEL_FILE}! Starting fresh.")
        else:
            print(f"⚠️ Invalid model file. Starting fresh.")
    except Exception as e:
        print(f"⚠️ Error checking model: {e}. Starting fresh.")
else:
    print(f"ℹ️ No saved model found. Training from scratch.")

# Initialize Background Trainer
trainer = BackgroundTrainer(gpt, MODEL_FILE)
# Start training automatically on server start (Render deployment)
if os.environ.get('RENDER'): 
    print("☁️ Detected Render environment: Starting background training...")
    trainer.start()

# ----------------------------------------------------------------------
# Training metadata (persisted between restarts)
# ----------------------------------------------------------------------
META_PATH = "training_meta.json"
try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        training_meta = json.load(f)
except Exception:
    training_meta = {"epochs": 0, "last_loss": None}

# ----------------------------------------------------------------------
# /chat – generate a response from the user message
# ----------------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    """Generate response using user message as seed."""
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Always use the user's message as the seed
        seed = user_message
        
        # Capitalize first character for consistency
        if seed and seed[0].islower():
            seed = seed[0].upper() + seed[1:]

        print(f"\n💬 User: '{user_message}'")
        print(f"🌱 Seed used: '{seed}'")

        # Generate
        # Temperature 0.7 is good for Transformers
        generated = gpt.generate(seed_text=seed, length=50, temperature=0.7)
        print(f"✨ Generated (full): '{generated}'")
        
        # Post-processing
        response = generated
        
        # Basic cleanup: keep only printable ASCII characters, collapse whitespace
        response = re.sub(r"[^\x20-\x7E]", "", response)
        response = re.sub(r"\s+", " ", response).strip()

        print(f"📤 Response: '{response}'")
        return jsonify({"response": response, "seed_used": seed, "model": "GPT-Transformer"})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"response": "Error generating response", "error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    """Check training status."""
    return jsonify({
        "is_training": trainer.is_training,
        "current_epoch": trainer.current_epoch,
        "target_epochs": trainer.target_epochs,
        "last_loss": trainer.last_loss,
        "vocab_size": gpt.vocab_size
    })

@app.route("/start_training", methods=["POST"])
def start_training():
    """Manually start background training."""
    if not trainer.is_training:
        trainer.start()
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})

@app.route("/stop_training", methods=["POST"])
def stop_training():
    """Manually stop background training."""
    if trainer.is_training:
        trainer.stop()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})

# ----------------------------------------------------------------------
# /model_info – expose how many epochs the model has been trained
# ----------------------------------------------------------------------
@app.route("/model_info", methods=["GET"])
def model_info():
    """Return training metadata for the loaded model."""
    return jsonify({
        "trained_epochs": training_meta.get("epochs", 0),
        "last_loss": training_meta.get("last_loss"),
        "vocab_size": gpt.vocab_size,
        "model_loaded": True,
        "architecture": "Transformer"
    })

# ----------------------------------------------------------------------
# /stats – simple model statistics
# ----------------------------------------------------------------------
@app.route("/stats", methods=["GET"])
def stats():
    """Return model statistics."""
    return jsonify({
        "vocab_size": gpt.vocab_size,
        "model_loaded": True,
        "training_mode": "static",
        "background_training": False,
        "architecture": "Transformer"
    })

# ----------------------------------------------------------------------
# /health – health check endpoint
# ----------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "healthy", "model_loaded": gpt.vocab_size is not None})

def graceful_shutdown(signum, frame):
    print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
    if trainer.is_training:
        trainer.stop()
    # Force save one last time to be sure
    gpt.save(MODEL_FILE)
    print("✅ Model saved. Goodbye!")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# ----------------------------------------------------------------------
# Run the server
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Server starting on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
