import threading
import time
import os
from training_corpus import get_training_corpus, get_shakespeare_corpus, get_science_corpus

class BackgroundTrainer:
    def __init__(self, gpt_model, save_path='gpt_model.pkl'):
        self.gpt = gpt_model
        self.save_path = save_path
        self.is_training = False
        self.stop_event = threading.Event()
        self.thread = None
        self.current_epoch = 0
        self.target_epochs = 30000
        self.last_loss = 0.0

    def start(self):
        if self.is_training:
            return
        self.is_training = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._train_loop)
        self.thread.daemon = True # Daemon thread dies when main program dies
        self.thread.start()
        print("🚀 Background training started!")

    def stop(self):
        if not self.is_training:
            return
        print("🛑 Stopping background training...")
        self.stop_event.set()
        self.thread.join()
        self.is_training = False
        print("✅ Background training stopped.")

    def _train_loop(self):
        # Load corpus once
        corpus = get_training_corpus()
        shakespeare = get_shakespeare_corpus()
        science = get_science_corpus()
        full_text = corpus + "\n\n" + shakespeare + "\n\n" + science
        
        print(f"📚 Background Trainer loaded {len(full_text)} chars")

        while not self.stop_event.is_set() and self.current_epoch < self.target_epochs:
            try:
                # Train one step
                loss = self.gpt.train_step(full_text, learning_rate=0.0005) # Lower LR for stability
                self.last_loss = loss
                self.current_epoch += 1
                
                # Log progress
                if self.current_epoch % 100 == 0:
                    print(f"[Background] Epoch {self.current_epoch}/{self.target_epochs} | Loss: {loss:.4f}")
                
                # Save periodically (every 500 epochs or 5 minutes)
                if self.current_epoch % 500 == 0:
                    self.gpt.save(self.save_path)
                    print(f"💾 [Background] Model saved at epoch {self.current_epoch}")
                    
                # Small sleep to prevent CPU hogging on shared hosting
                time.sleep(0.01) 
                
            except Exception as e:
                print(f"❌ Background training error: {e}")
                break
        
        # Final save
        self.gpt.save(self.save_path)
        self.is_training = False
        print("✅ Background training finished!")
