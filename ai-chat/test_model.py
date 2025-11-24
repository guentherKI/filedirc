"""
Quick test of the trained LSTM model
"""

from lstm_generator import CharLSTM

# Load the pre-trained model
lstm = CharLSTM()
if lstm.load('lstm_model.pkl'):
    print("✅ Model loaded successfully")
    print(f"Vocabulary size: {lstm.vocab_size}")
    
    # Test different seeds
    seeds = ["Hello", "What is", "I am", "The ", "Tell me"]
    
    print("\n" + "="*70)
    print("TESTING GENERATION WITH DIFFERENT SEEDS")
    print("="*70)
    
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 70)
        
        # Try different temperatures
        for temp in [0.5, 0.7, 1.0]:
            generated = lstm.generate(seed, length=60, temperature=temp)
            print(f"Temp {temp}: {generated}")
        print()
else:
    print("❌ Failed to load model!")
