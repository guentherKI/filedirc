# 1. Load Tokenizer
tokenizer = BPETokenizer(vocab_size=1000)
if os.path.exists('tokenizer.json'):
    tokenizer.load('tokenizer.json')
    print(f"✅ Tokenizer loaded. Vocab: {tokenizer.vocab_size}")
else:
    print("⚠️ No tokenizer found! Please run pretrain.py first.")
    # Fallback to untrained tokenizer (byte-level)
    tokenizer.vocab_size = 256

from background_trainer import BackgroundTrainer

# ... (Previous imports and setup) ...

# 2. Initialize GPT
gpt = GPTModel(tokenizer, d_model=64, n_layer=4, n_head=4, block_size=32)

MODEL_FILE = "gpt_model.pkl"

if gpt.load(MODEL_FILE):
    print(f"✅ Model loaded successfully")
    print(f"📊 Vocabulary size: {gpt.vocab_size} tokens")
else:
    print(f"⚠️ Failed to load {MODEL_FILE}! Starting fresh.")

# Initialize Background Trainer
trainer = BackgroundTrainer(gpt, MODEL_FILE)
# Start training automatically on server start (Render deployment)
if os.environ.get('RENDER'): 
    print("☁️ Detected Render environment: Starting background training...")
    trainer.start()

# ... (Chat endpoint remains same) ...

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
        # Remove the seed from the beginning if it's repeated (GPT usually continues, but sometimes repeats)
        # Actually, my generate function returns the FULL text including seed.
        # We want to return just the new part or the whole thing? 
        # Usually chat interfaces want the whole thing or just the response. 
        # Let's return the whole thing but clean it up.
        
        response = generated
        
        # Basic cleanup: keep only printable ASCII characters, collapse whitespace
        response = re.sub(r"[^\x20-\x7E]", "", response)
        response = re.sub(r"\s+", " ", response).strip()

        print(f"📤 Response: '{response}'")
        return jsonify({"response": response, "seed_used": seed, "model": "GPT-Transformer"})
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"response": "Error generating response", "error": str(e)}), 500

# Old /train endpoint removed in favor of background trainer


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

import signal
import sys

# ... (Previous code) ...

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
