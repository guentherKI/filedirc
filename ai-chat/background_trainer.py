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
        try:
            corpus = get_training_corpus()
            shakespeare = get_shakespeare_corpus()
            science = get_science_corpus()
            full_text = corpus + "\n\n" + shakespeare + "\n\n" + science
            
            print(f"📚 Background Trainer loaded {len(full_text)} chars", flush=True)
            print(f"🔤 Starting training loop...", flush=True)
        except Exception as e:
            print(f"❌ Failed to load corpus: {e}", flush=True)
            self.is_training = False
            return

        epoch_count = 0
        while not self.stop_event.is_set() and self.current_epoch < self.target_epochs:
            try:
                # Train one step
                loss = self.gpt.train_step(full_text, learning_rate=0.0005)
                
                if loss == 0:
                    print(f"⚠️ train_step returned 0 (text too short?)", flush=True)
                    self.is_training = False
                    break
                    
                self.last_loss = loss
                self.current_epoch += 1
                epoch_count += 1
                
                # First epoch always logs
                if epoch_count == 1:
                    print(f"✅ [Background] First epoch complete! Loss: {loss:.4f}", flush=True)
                
                # Log progress
                if self.current_epoch % 10 == 0:  # More frequent logging
                    print(f"[Background] Epoch {self.current_epoch}/{self.target_epochs} | Loss: {loss:.4f}", flush=True)
                
                # Save periodically
                if self.current_epoch % 500 == 0:
                    self.gpt.save(self.save_path)
                    print(f"💾 [Background] Model saved at epoch {self.current_epoch}", flush=True)
                
            except Exception as e:
                print(f"❌ Background training error at epoch {self.current_epoch}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                break
        
        # Final save
        try:
            self.gpt.save(self.save_path)
            print(f"✅ Background training finished! Final epoch: {self.current_epoch}", flush=True)
        except Exception as e:
            print(f"❌ Failed to save final model: {e}", flush=True)
        
        self.is_training = False
