import os
import json
import re

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {} # token -> id
        self.inverse_vocab = {} # id -> token
        self.merges = {} # (p1, p2) -> merged_token
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_ids(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        print(f"🔤 Training BPE Tokenizer (Target: {self.vocab_size} tokens)...")
        
        # 1. Initialize with characters
        # We use bytes to be safe for any input
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers 0-255
        
        # Base vocabulary: 0-255 are the bytes
        current_vocab_size = 256
        num_merges = self.vocab_size - 256
        
        # Iteratively merge
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
                
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            
            # Mint new token ID
            idx = current_vocab_size + i
            
            # Record merge
            self.merges[pair] = idx
            
            # Apply merge
            ids = self.merge_ids(ids, pair, idx)
            
            if (i+1) % 100 == 0:
                print(f"BPE Merge {i+1}/{num_merges}: {pair} -> {idx}")
                
        self.trained_vocab_size = 256 + len(self.merges)
        print(f"✅ BPE Training Complete. Final Vocab Size: {self.trained_vocab_size}")

    def encode(self, text):
        if not self.merges:
            # Fallback to byte encoding if not trained
            return list(text.encode("utf-8"))
            
        ids = list(text.encode("utf-8"))
        
        # Apply merges in the same order they were learned
        # Actually, we just need to apply them greedily. 
        # The correct way is to apply them in order of priority (which is implicit if we iterate through merges)
        # But for efficiency, usually we iterate.
        # For this simple implementation, we can just loop through our stored merges.
        # Optimization: In a real BPE, we'd prioritize. Here, let's just loop until no changes.
        
        while True:
            stats = self.get_stats(ids)
            if not stats: break
            
            # Find the pair that was merged earliest (lowest ID)
            # This ensures we reconstruct the same tokens
            pair_to_merge = None
            min_merge_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_merge_idx:
                        min_merge_idx = self.merges[pair]
                        pair_to_merge = pair
            
            if pair_to_merge is None:
                break
                
            ids = self.merge_ids(ids, pair_to_merge, min_merge_idx)
            
        return ids

    def decode(self, ids):
        # Reverse the merges? 
        # Actually, simpler: we need a map from ID -> bytes
        # We can build this map from the merges.
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Reconstruct vocab from merges
        # We must do this in order of creation (256, 257, ...)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        for (p0, p1), idx in sorted_merges:
            vocab[idx] = vocab[p0] + vocab[p1]
            
        # Now decode
        tokens = b"".join(vocab[idx] for idx in ids if idx in vocab)
        return tokens.decode("utf-8", errors="replace")

    def save(self, filename):
        data = {
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab_size": self.vocab_size
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
            
    def load(self, filename):
        if not os.path.exists(filename): return False
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        # Convert keys back to tuples
        self.merges = {}
        for k, v in data["merges"].items():
            p0, p1 = map(int, k.split(','))
            self.merges[(p0, p1)] = v
        return True
