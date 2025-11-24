"""
24/7 Background Training Thread for LSTM
Continuously trains the model on conversation data
"""

import threading
import time
import queue
from datetime import datetime
from lstm_generator import CharLSTM

class ContinuousTrainer:
    """Manages 24/7 background training of LSTM model"""
    
    def __init__(self, model_path='lstm_model.pkl'):
        self.model_path = model_path
        self.lstm = CharLSTM(hidden_size=128, seq_length=25)
        
        # Training queue for new conversations
        self.training_queue = queue.Queue()
        
        # Training state
        self.is_training = False
        self.total_iterations = 0
        self.current_loss = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        
        # Thread management
        self.training_thread = None
        self.should_stop = False
        
        # Conversation storage
        self.all_conversations = []
        self.load_conversations()
        
        # Load or initialize model
        if not self.lstm.load(model_path):
            print("No saved model found. Initializing new LSTM...")
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize model with starter training data"""
        starter_text = """Hello! How are you? I am learning to generate text.
I can chat and learn from our conversations.
The more we talk, the smarter I become.
I understand questions and try to give helpful answers.
What would you like to talk about?
I love learning new things from you!
This is amazing! I'm generating text character by character.
"""
        self.lstm.build_vocab(starter_text)
        self.all_conversations.append(starter_text)
        
        # Initial training
        print("Initial training...")
        for i in range(100):
            loss = self.lstm.train_step(starter_text, learning_rate=0.01)
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/100, Loss: {loss:.4f}")
        
        self.lstm.save(self.model_path)
        self.save_conversations()
    
    def add_conversation(self, user_input, ai_response):
        """Add new conversation to training queue"""
        conversation = f"User: {user_input}\nAI: {ai_response}\n"
        self.training_queue.put(conversation)
        self.all_conversations.append(conversation)
        print(f"📚 Added conversation to training queue (queue size: {self.training_queue.qsize()})")
    
    def training_loop(self):
        """Main 24/7 training loop"""
        print("🔄 Starting 24/7 training loop...")
        
        while not self.should_stop:
            try:
                self.is_training = True
                
                # Check for new conversations
                new_conversations = []
                while not self.training_queue.empty():
                    try:
                        conv = self.training_queue.get_nowait()
                        new_conversations.append(conv)
                    except queue.Empty:
                        break
                
                # If we have new data, retrain
                if new_conversations:
                    print(f"\n📖 Training on {len(new_conversations)} new conversations...")
                    
                    # Only train on NEW conversations, not old corrupted ones
                    training_text = "\n".join(new_conversations)
                    
                    # Training iterations on new data only
                    for i in range(20):  # Reduced from 50 to avoid overfitting
                        loss = self.lstm.train_step(training_text, learning_rate=0.001)
                        self.current_loss = loss
                        self.total_iterations += 1
                        
                        if (i + 1) % 5 == 0:
                            print(f"  Iteration {i+1}/20, Loss: {loss:.4f}")
                
                # DON'T train on old conversations - they corrupt the model!
                
                # Periodic saving
                if time.time() - self.last_save_time > self.save_interval:
                    print("💾 Auto-saving model...")
                    self.lstm.save(self.model_path)
                    self.save_conversations()
                    self.last_save_time = time.time()
                
                # Small sleep to not max out CPU
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Training error: {e}")
                time.sleep(1)
            
            finally:
                self.is_training = False
        
        print("🛑 Training loop stopped")
    
    def start_training(self):
        """Start the background training thread"""
        if self.training_thread and self.training_thread.is_alive():
            print("Training already running!")
            return
        
        self.should_stop = False
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        print("✅ Background training started!")
    
    def stop_training(self):
        """Stop the background training thread"""
        print("Stopping training...")
        self.should_stop = True
        if self.training_thread:
            self.training_thread.join(timeout=5)
        
        # Final save
        self.lstm.save(self.model_path)
        self.save_conversations()
        print("✅ Training stopped and model saved")
    
    def generate_response(self, user_input, max_length=150):
        """Generate AI response using LSTM"""
        try:
            print(f"\n🔍 DEBUG: Generating response for '{user_input}'")
            print(f"   Vocab size: {self.lstm.vocab_size}")
            
            # Use generic seeds that are well-trained
            seeds = ["Hello", "I ", "The ", "What "]
            
            # Pick seed based on user input keywords
            user_lower = user_input.lower()
            if any(word in user_lower for word in ['hi', 'hello', 'hey']):
                seed = "Hello"
            elif any(word in user_lower for word in ['what', 'how', 'why', 'when', 'where']):
                seed = "The "
            elif any(word in user_lower for word in ['joke', 'funny', 'laugh']):
                seed = "What "
            else:
                seed = "I "
            
            print(f"   Using seed: '{seed}'")
            
            # Generate response
            generated = self.lstm.generate(
                seed_text=seed,
                length=100,
                temperature=0.7
            )
            
            print(f"   Generated: '{generated}'")
            
            # Extract just the AI's response (after the seed)
            response = generated[len(seed):].strip()
            
            # Clean up response - take first sentence
            if '\n' in response:
                response = response.split('\n')[0]
            if '.' in response and len(response) > 20:
                response = response.split('.')[0] + '.'
            
            if not response or len(response) < 5:
                response = "I'm still learning!"
            
            print(f"   Final response: '{response}'")
            
            return response
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return "I'm having trouble generating a response right now!"
    
    def get_stats(self):
        """Get training statistics"""
        return {
            'is_training': self.is_training,
            'total_iterations': self.total_iterations,
            'current_loss': float(self.current_loss),
            'vocab_size': self.lstm.vocab_size if self.lstm.vocab_size else 0,
            'conversations_count': len(self.all_conversations),
            'queue_size': self.training_queue.qsize()
        }
    
    def save_conversations(self):
        """Save conversation history"""
        try:
            with open('conversations.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.all_conversations))
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def load_conversations(self):
        """Load conversation history"""
        try:
            with open('conversations.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    self.all_conversations = content.split('\n')
                    print(f"Loaded {len(self.all_conversations)} previous conversations")
        except FileNotFoundError:
            print("No previous conversations found")
        except Exception as e:
            print(f"Error loading conversations: {e}")


if __name__ == "__main__":
    # Test the continuous trainer
    print("Testing Continuous Trainer...")
    
    trainer = ContinuousTrainer()
    trainer.start_training()
    
    # Simulate conversations
    conversations = [
        ("hello", "Hi there! How are you?"),
        ("how are you", "I'm doing great! Thanks for asking!"),
        ("what can you do", "I can chat and learn from our conversations!")
    ]
    
    for user_msg, ai_msg in conversations:
        trainer.add_conversation(user_msg, ai_msg)
        time.sleep(1)
    
    # Wait for training
    print("\nWaiting for training...")
    time.sleep(10)
    
    # Test generation
    print("\nTesting generation:")
    response = trainer.generate_response("hello")
    print(f"Generated: {response}")
    
    # Stats
    print("\nStats:", trainer.get_stats())
    
    trainer.stop_training()
