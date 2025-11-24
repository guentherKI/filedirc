"""
Character-Level LSTM Text Generator - Built from Scratch
Generates text character-by-character using LSTM neural network
"""

import numpy as np
import pickle
import os

class LSTMCell:
    """Single LSTM cell with forget, input, and output gates"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for gates (Xavier initialization)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Forget gate
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))
        
        # Input gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        
        # Candidate gate
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        
        # Output gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through LSTM cell
        x: input (input_size, 1)
        h_prev: previous hidden state (hidden_size, 1)
        c_prev: previous cell state (hidden_size, 1)
        """
        # Concatenate input and previous hidden state
        concat = np.vstack((x, h_prev))
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        
        # Candidate cell state
        c_candidate = self.tanh(np.dot(self.Wc, concat) + self.bc)
        
        # New cell state
        c = f * c_prev + i * c_candidate
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # New hidden state
        h = o * self.tanh(c)
        
        # Cache for backward pass
        cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'concat': concat, 'f': f, 'i': i, 'c_candidate': c_candidate,
            'c': c, 'o': o, 'h': h
        }
        
        return h, c, cache
    
    def backward(self, dh, dc, cache):
        """Backward pass through LSTM cell"""
        x, h_prev, c_prev = cache['x'], cache['h_prev'], cache['c_prev']
        concat, f, i, c_candidate, c, o, h = cache['concat'], cache['f'], cache['i'], cache['c_candidate'], cache['c'], cache['o'], cache['h']
        
        # Backprop through output gate
        do = dh * self.tanh(c)
        do = do * o * (1 - o)  # sigmoid derivative
        
        # Backprop through cell state
        dc = dc + dh * o * (1 - self.tanh(c)**2)
        
        # Backprop through forget gate
        df = dc * c_prev
        df = df * f * (1 - f)
        
        # Backprop through input gate
        di = dc * c_candidate
        di = di * i * (1 - i)
        
        # Backprop through candidate
        dc_candidate = dc * i
        dc_candidate = dc_candidate * (1 - c_candidate**2)
        
        # Gradients for weights
        dWf = np.dot(df, concat.T)
        dbf = df
        
        dWi = np.dot(di, concat.T)
        dbi = di
        
        dWc = np.dot(dc_candidate, concat.T)
        dbc = dc_candidate
        
        dWo = np.dot(do, concat.T)
        dbo = do
        
        # Backprop to previous hidden and cell state
        dconcat = (np.dot(self.Wf.T, df) + np.dot(self.Wi.T, di) + 
                   np.dot(self.Wc.T, dc_candidate) + np.dot(self.Wo.T, do))
        
        dx = dconcat[:self.input_size]
        dh_prev = dconcat[self.input_size:]
        dc_prev = dc * f
        
        gradients = {
            'dWf': dWf, 'dbf': dbf,
            'dWi': dWi, 'dbi': dbi,
            'dWc': dWc, 'dbc': dbc,
            'dWo': dWo, 'dbo': dbo
        }
        
        return dx, dh_prev, dc_prev, gradients


class CharLSTM:
    """Word-level LSTM text generator with Embeddings and 2 Layers"""
    
    def __init__(self, vocab_size=None, hidden_size=128, seq_length=25, embedding_dim=64):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        if vocab_size:
            # Embedding layer
            self.We = np.random.randn(vocab_size, embedding_dim) * 0.1
            
            # Layer 1: Takes embeddings
            self.lstm1 = LSTMCell(embedding_dim, hidden_size)
            # Layer 2: Takes output of Layer 1
            self.lstm2 = LSTMCell(hidden_size, hidden_size)
            
            # Output layer
            self.Wy = np.random.randn(vocab_size, hidden_size) * 0.01
            self.by = np.zeros((vocab_size, 1))
        
    def build_vocab(self, text):
        """Build word-level vocabulary from text"""
        # Simple tokenization: split by whitespace, keeping punctuation as separate tokens
        import re
        # Add spaces around punctuation so they become separate tokens
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        words = text.split()
        
        # Create vocabulary
        unique_words = sorted(list(set(words)))
        # Add special tokens
        if "<UNK>" not in unique_words:
            unique_words.insert(0, "<UNK>")
            
        self.vocab_size = len(unique_words)
        self.char_to_idx = {word: i for i, word in enumerate(unique_words)}
        self.idx_to_char = {i: word for i, word in enumerate(unique_words)}
        
        # Re-initialize weights with new vocab size
        self.We = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        self.lstm1 = LSTMCell(self.embedding_dim, self.hidden_size)
        self.lstm2 = LSTMCell(self.hidden_size, self.hidden_size)
        self.Wy = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.by = np.zeros((self.vocab_size, 1))
        
        print(f"Vocabulary built: {self.vocab_size} unique words")
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def forward(self, inputs, h_prev, c_prev):
        """
        Forward pass through entire sequence
        inputs: list of word indices
        """
        # h_prev and c_prev are now lists [layer1, layer2]
        h1, h2 = h_prev[0], h_prev[1]
        c1, c2 = c_prev[0], c_prev[1]
        
        outputs = []
        caches = []
        
        for idx in inputs:
            # Embedding lookup
            x = self.We[idx].reshape(-1, 1)
            
            # Layer 1
            h1, c1, cache1 = self.lstm1.forward(x, h1, c1)
            
            # Layer 2 (input is h1)
            h2, c2, cache2 = self.lstm2.forward(h1, h2, c2)
            
            # Output layer (uses h2)
            y = np.dot(self.Wy, h2) + self.by
            probs = self.softmax(y)
            
            outputs.append(probs)
            # Store embedding index and both caches
            cache = {'emb_idx': idx, 'l1': cache1, 'l2': cache2}
            caches.append(cache)
        
        return outputs, [h1, h2], [c1, c2], caches
    
    def backward(self, inputs, targets, outputs, caches, learning_rate=0.001):
        """Backpropagation through time"""
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dWe = np.zeros_like(self.We)
        
        dh2_next = np.zeros((self.hidden_size, 1))
        dc2_next = np.zeros((self.hidden_size, 1))
        dh1_next = np.zeros((self.hidden_size, 1))
        dc1_next = np.zeros((self.hidden_size, 1))
        
        loss = 0
        
        # Backward pass
        for t in reversed(range(len(inputs))):
            # Output layer gradient
            dy = outputs[t].copy()
            dy[targets[t]] -= 1  # Cross-entropy gradient
            
            loss += -np.log(outputs[t][targets[t], 0] + 1e-8)
            
            # Gradient flowing into h2 from output layer
            # Use cache['l2']['h'] which is h2 at time t
            dWy += np.dot(dy, caches[t]['l2']['h'].T)
            dby += dy
            
            dh2 = np.dot(self.Wy.T, dy) + dh2_next
            
            # Layer 2 backward
            dx2, dh2_next, dc2_next, grads2 = self.lstm2.backward(dh2, dc2_next, caches[t]['l2'])
            
            # Gradient flowing into h1 (dx2 is the gradient w.r.t input of layer 2, which is h1)
            dh1 = dx2 + dh1_next
            
            # Layer 1 backward
            dx1, dh1_next, dc1_next, grads1 = self.lstm1.backward(dh1, dc1_next, caches[t]['l1'])
            
            # Embedding gradient
            emb_idx = caches[t]['emb_idx']
            dWe[emb_idx] += dx1.ravel()
            
            # Update Layer 2 weights
            for key in grads2: grads2[key] = np.clip(grads2[key], -5, 5)
            self.lstm2.Wf -= learning_rate * grads2['dWf']
            self.lstm2.bf -= learning_rate * grads2['dbf']
            self.lstm2.Wi -= learning_rate * grads2['dWi']
            self.lstm2.bi -= learning_rate * grads2['dbi']
            self.lstm2.Wc -= learning_rate * grads2['dWc']
            self.lstm2.bc -= learning_rate * grads2['dbc']
            self.lstm2.Wo -= learning_rate * grads2['dWo']
            self.lstm2.bo -= learning_rate * grads2['dbo']
            
            # Update Layer 1 weights
            for key in grads1: grads1[key] = np.clip(grads1[key], -5, 5)
            self.lstm1.Wf -= learning_rate * grads1['dWf']
            self.lstm1.bf -= learning_rate * grads1['dbf']
            self.lstm1.Wi -= learning_rate * grads1['dWi']
            self.lstm1.bi -= learning_rate * grads1['dbi']
            self.lstm1.Wc -= learning_rate * grads1['dWc']
            self.lstm1.bc -= learning_rate * grads1['dbc']
            self.lstm1.Wo -= learning_rate * grads1['dWo']
            self.lstm1.bo -= learning_rate * grads1['dbo']
        
        # Clip output layer gradients
        dWy = np.clip(dWy, -5, 5)
        dby = np.clip(dby, -5, 5)
        dWe = np.clip(dWe, -5, 5)
        
        # Update weights
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby
        self.We -= learning_rate * dWe
        
        return loss
    
    def train_step(self, text, learning_rate=0.001):
        """Single training step on text"""
        # Tokenize text
        import re
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        tokens = text.split()
        
        if len(tokens) < self.seq_length + 1:
            return 0
        
        # Random starting position
        start = np.random.randint(0, len(tokens) - self.seq_length - 1)
        inputs = [self.char_to_idx.get(token, self.char_to_idx["<UNK>"]) for token in tokens[start:start + self.seq_length]]
        targets = [self.char_to_idx.get(token, self.char_to_idx["<UNK>"]) for token in tokens[start + 1:start + self.seq_length + 1]]
        
        # Initialize hidden states for both layers
        h = [np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))]
        c = [np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))]
        
        # Forward pass
        outputs, h, c, caches = self.forward(inputs, h, c)
        
        # Backward pass
        loss = self.backward(inputs, targets, outputs, caches, learning_rate)
        
        return loss
    
    def generate(self, seed_text, length=50, temperature=0.8):
        """
        Generate text starting from seed
        temperature: higher = more random, lower = more conservative
        """
        # Tokenize seed
        import re
        seed_text = re.sub(r'([.,!?;:])', r' \1 ', seed_text)
        seed_tokens = seed_text.split()
        
        if not seed_tokens:
            seed_tokens = [list(self.char_to_idx.keys())[1]] # Skip UNK
        
        # Initialize hidden states for both layers
        h = [np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))]
        c = [np.zeros((self.hidden_size, 1)), np.zeros((self.hidden_size, 1))]
        
        generated_tokens = list(seed_tokens)
        
        # Process seed text
        for token in seed_tokens[:-1]:
            idx = self.char_to_idx.get(token, self.char_to_idx["<UNK>"])
            x = self.We[idx].reshape(-1, 1)
            
            # Manual forward for generation (simplified)
            h[0], c[0], _ = self.lstm1.forward(x, h[0], c[0])
            h[1], c[1], _ = self.lstm2.forward(h[0], h[1], c[1])
        
        # Generate new tokens
        current_token = seed_tokens[-1]
        
        for _ in range(length):
            idx = self.char_to_idx.get(current_token, self.char_to_idx["<UNK>"])
            x = self.We[idx].reshape(-1, 1)
            
            # Forward pass through both layers
            h[0], c[0], _ = self.lstm1.forward(x, h[0], c[0])
            h[1], c[1], _ = self.lstm2.forward(h[0], h[1], c[1])
            
            # Output probabilities (from layer 2)
            y = np.dot(self.Wy, h[1]) + self.by
            probs = self.softmax(y / temperature)
            
            # Sample next token
            idx = np.random.choice(range(self.vocab_size), p=probs.ravel())
            current_token = self.idx_to_char[idx]
            
            # Don't output UNK
            if current_token == "<UNK>":
                probs[0] = 0 
                probs = probs / np.sum(probs)
                idx = np.random.choice(range(self.vocab_size), p=probs.ravel())
                current_token = self.idx_to_char[idx]
                
            generated_tokens.append(current_token)
            
            # Stop at period if long enough
            if current_token == '.' and len(generated_tokens) > 20:
                break
        
        # Detokenize
        text = ' '.join(generated_tokens)
        text = re.sub(r' ([.,!?;:])', r'\1', text)
        return text
    
    def save(self, filepath='lstm_model.pkl'):
        """Save model to file"""
        model_data = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'seq_length': self.seq_length,
            'embedding_dim': self.embedding_dim,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            
            # Layer 1
            'lstm1_Wf': self.lstm1.Wf, 'lstm1_bf': self.lstm1.bf,
            'lstm1_Wi': self.lstm1.Wi, 'lstm1_bi': self.lstm1.bi,
            'lstm1_Wc': self.lstm1.Wc, 'lstm1_bc': self.lstm1.bc,
            'lstm1_Wo': self.lstm1.Wo, 'lstm1_bo': self.lstm1.bo,
            
            # Layer 2
            'lstm2_Wf': self.lstm2.Wf, 'lstm2_bf': self.lstm2.bf,
            'lstm2_Wi': self.lstm2.Wi, 'lstm2_bi': self.lstm2.bi,
            'lstm2_Wc': self.lstm2.Wc, 'lstm2_bc': self.lstm2.bc,
            'lstm2_Wo': self.lstm2.Wo, 'lstm2_bo': self.lstm2.bo,
            
            'Wy': self.Wy,
            'by': self.by,
            'We': self.We
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='lstm_model.pkl'):
        """Load model from file"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.hidden_size = model_data['hidden_size']
        self.seq_length = model_data['seq_length']
        self.embedding_dim = model_data.get('embedding_dim', 64)
        
        self.char_to_idx = model_data['char_to_idx']
        self.idx_to_char = model_data['idx_to_char']
        
        # Initialize layers
        self.lstm1 = LSTMCell(self.embedding_dim, self.hidden_size)
        self.lstm2 = LSTMCell(self.hidden_size, self.hidden_size)
        
        # Load Layer 1
        if 'lstm1_Wf' in model_data:
            self.lstm1.Wf = model_data['lstm1_Wf']; self.lstm1.bf = model_data['lstm1_bf']
            self.lstm1.Wi = model_data['lstm1_Wi']; self.lstm1.bi = model_data['lstm1_bi']
            self.lstm1.Wc = model_data['lstm1_Wc']; self.lstm1.bc = model_data['lstm1_bc']
            self.lstm1.Wo = model_data['lstm1_Wo']; self.lstm1.bo = model_data['lstm1_bo']
            
            self.lstm2.Wf = model_data['lstm2_Wf']; self.lstm2.bf = model_data['lstm2_bf']
            self.lstm2.Wi = model_data['lstm2_Wi']; self.lstm2.bi = model_data['lstm2_bi']
            self.lstm2.Wc = model_data['lstm2_Wc']; self.lstm2.bc = model_data['lstm2_bc']
            self.lstm2.Wo = model_data['lstm2_Wo']; self.lstm2.bo = model_data['lstm2_bo']
        else:
            # Backward compatibility for single layer models (load into layer 1)
            print("⚠️ Loading single-layer model into 2-layer architecture. Layer 2 initialized randomly.")
            self.lstm1.Wf = model_data['lstm_Wf']; self.lstm1.bf = model_data['lstm_bf']
            self.lstm1.Wi = model_data['lstm_Wi']; self.lstm1.bi = model_data['lstm_bi']
            self.lstm1.Wc = model_data['lstm_Wc']; self.lstm1.bc = model_data['lstm_bc']
            self.lstm1.Wo = model_data['lstm_Wo']; self.lstm1.bo = model_data['lstm_bo']
        
        self.Wy = model_data['Wy']
        self.by = model_data['by']
        self.We = model_data.get('We', np.random.randn(self.vocab_size, self.embedding_dim) * 0.1)
        
        print(f"Model loaded from {filepath}")
        return True


if __name__ == "__main__":
    # Test the LSTM
    print("Testing LSTM Text Generator...")
    
    training_text = """Hello! How are you? I am learning to generate text.
This is amazing! I can create new sentences character by character.
The more I train, the better I get at understanding language patterns."""
    
    lstm = CharLSTM(hidden_size=64, seq_length=20)
    lstm.build_vocab(training_text)
    
    print("\nTraining for 100 iterations...")
    for i in range(100):
        loss = lstm.train_step(training_text, learning_rate=0.01)
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}, Loss: {loss:.4f}")
    
    print("\nGenerating text:")
    generated = lstm.generate("Hello", length=50, temperature=0.7)
    print(generated)
