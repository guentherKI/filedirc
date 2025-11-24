"""
Smart Startup Script for LSTM Chat Server
Handles pre-training, background training, and chat serving
"""

import os
import sys

def check_model_exists():
    """Check if pre-trained model exists"""
    return os.path.exists('lstm_model.pkl')

def main():
    print("="*70)
    print("🧠 LSTM CHAT SERVER - SMART STARTUP")
    print("="*70)
    
    if check_model_exists():
        print("\n✅ Pre-trained model found!")
        print("🚀 Starting server with existing model...")
        print("   → Background training will continue")
        print("   → Learning from every chat")
        print("\n")
        
        # Start server directly
        os.system('python server.py')
        
    else:
        print("\n⚠️  No pre-trained model found!")
        print("\n📋 Options:")
        print("   1. Quick start (use minimal training)")
        print("   2. Pre-train first (recommended, 15-30 min)")
        print("   3. Exit")
        
        choice = input("\nYour choice (1/2/3): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting with minimal training...")
            print("⚠️  Responses will be gibberish initially")
            print("✅ Will improve as it learns from chats\n")
            os.system('python server.py')
            
        elif choice == '2':
            print("\n🎓 Starting pre-training...")
            print("⏱️  This will take 15-30 minutes")
            print("☕ Perfect time for a coffee break!\n")
            
            epochs = input("Epochs (default 2000, more=better): ").strip()
            if not epochs:
                epochs = "2000"
            
            # Run pre-training
            os.system(f'python pretrain.py --epochs {epochs}')
            
            print("\n✅ Pre-training complete!")
            print("🚀 Starting server...\n")
            os.system('python server.py')
            
        else:
            print("\n👋 Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        sys.exit(0)
