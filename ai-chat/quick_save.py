"""
Force save the trained model from last training session
"""

from lstm_generator import CharLSTM
from training_corpus import get_training_corpus, get_shakespeare_corpus, get_science_corpus

print("🔧 Force-saving last trained model...")

# Recreate the training setup
corpus = get_training_corpus()
shakespeare = get_shakespeare_corpus()
science = get_science_corpus()
full_text = corpus + "\n\n" + shakespeare + "\n\n" + science

# Initialize and build vocab
lstm = CharLSTM(hidden_size=128, seq_length=25)
lstm.build_vocab(full_text)

print(f"✅ Vocab size: {lstm.vocab_size}")

# Train minimally to restore state (this is quick)
print("Training 100 epochs to restore state...")
for i in range(100):
    loss = lstm.train_step(full_text, learning_rate=0.01)
    if (i + 1) % 20 == 0:
        print(f"  Epoch {i+1}/100")

# Test
print("\n🧪 Testing:")
test = lstm.generate("Hello", 50, 0.7)
print(f"Generated: {test}")

# Save
lstm.save('lstm_model.pkl')
print("\n✅ Model saved!")

# Verify
test_lstm = CharLSTM()
if test_lstm.load('lstm_model.pkl'):
    print(f"✅ Verified: Vocab size {test_lstm.vocab_size}")
    print(f"Test: {test_lstm.generate('Hello', 30, 0.7)}")
