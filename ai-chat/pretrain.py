"""
Simple Pre-training with BPE Tokenizer
"""

from gpt_model import GPTModel
from tokenizer import BPETokenizer
from training_corpus import get_training_corpus, get_shakespeare_corpus, get_science_corpus
import time
import os

print("🔥 GPT PRE-TRAINING (Transformer + BPE)")

# Auto-download corpus if needed
corpus_files = [f for f in os.listdir('extra_corpus') if f.endswith('.txt')]
if len(corpus_files) < 10:  # Should have ~14 files after download
    print("📥 Corpus files missing. Downloading...")
    import subprocess
    subprocess.run(['python', 'download_corpus.py'], check=True)
else:
    print(f"✅ Found {len(corpus_files)} corpus files")

# Load corpus
corpus = get_training_corpus()
shakespeare = get_shakespeare_corpus()  
science = get_science_corpus()
full_text = corpus + "\n\n" + shakespeare + "\n\n" + science

print(f"✅ Loaded {len(full_text)} characters")

# 1. Train Tokenizer
tokenizer = BPETokenizer(vocab_size=1000) # Target 1000 tokens
if os.path.exists('tokenizer.json'):
    print("⏳ Loading existing tokenizer...")
    tokenizer.load('tokenizer.json')
    print(f"✅ Loaded Tokenizer. Vocab: {tokenizer.vocab_size}")
else:
    print("⏳ Training BPE Tokenizer...")
    tokenizer.train(full_text)
    tokenizer.save('tokenizer.json')

# 2. Initialize GPT
# d_model=64, n_layer=4, n_head=4 is a small but capable Transformer
gpt = GPTModel(tokenizer, d_model=64, n_layer=4, n_head=4, block_size=32)
print(f"✅ Model Initialized. Vocab: {gpt.vocab_size}")

# Training
epochs = 5000 
print(f"\n🔥 Training {epochs} epochs...")

start_time = time.time()
for epoch in range(epochs):
    loss = gpt.train_step(full_text, learning_rate=0.001)
    
    if (epoch + 1) % 100 == 0:
        elapsed = time.time() - start_time
        remaining = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | {remaining/60:.1f}m remaining")
    
    if (epoch + 1) % 1000 == 0:
        test = gpt.generate("Hello", 20, 0.7)
        print(f"Test: '{test}'\n")

print(f"\n✅ Training done! Final loss: {loss:.2f}")

# ALWAYS SAVE
print("\n💾 Saving model...")
gpt.save('gpt_model.pkl')

# Verify
test_gpt = GPTModel(tokenizer, d_model=64, n_layer=4, n_head=4, block_size=32)
test_gpt.load('gpt_model.pkl')
print(f"✅ Saved! Vocab: {test_gpt.vocab_size}")
print(f"Test: {test_gpt.generate('Hello', 20, 0.7)}")
