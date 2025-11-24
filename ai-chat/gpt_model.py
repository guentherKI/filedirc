import numpy as np
import pickle
import os

class GPTModel:
    """
    A pure NumPy implementation of a GPT-style Transformer.
    Features:
    - Multi-Head Self-Attention
    - Feed-Forward Networks
    - Layer Normalization
    - Residual Connections
    - Positional Embeddings
    """
    
    def __init__(self, tokenizer, d_model=64, n_layer=2, n_head=2, block_size=32):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.head_size = d_model // n_head
        
        # Initialize parameters
        self.params = {}
        
        # Token & Positional Embeddings
        # Note: BPE vocab size is fixed at init, unlike dynamic word vocab
        self.params['wte'] = np.random.randn(self.vocab_size, d_model) * 0.02
        self.params['wpe'] = np.random.randn(block_size, d_model) * 0.02
        self.params['lm_head_w'] = np.random.randn(d_model, self.vocab_size) * 0.02
        
        # Transformer Blocks
        for i in range(n_layer):
            # Attention (Q, K, V projections combined)
            self.params[f'b{i}_attn_w'] = np.random.randn(d_model, 3 * d_model) * 0.02
            self.params[f'b{i}_attn_b'] = np.zeros(3 * d_model)
            # Attention Output Projection
            self.params[f'b{i}_proj_w'] = np.random.randn(d_model, d_model) * 0.02
            self.params[f'b{i}_proj_b'] = np.zeros(d_model)
            # Layer Norm 1
            self.params[f'b{i}_ln1_g'] = np.ones(d_model)
            self.params[f'b{i}_ln1_b'] = np.zeros(d_model)
            
            # MLP
            self.params[f'b{i}_mlp_fc_w'] = np.random.randn(d_model, 4 * d_model) * 0.02
            self.params[f'b{i}_mlp_fc_b'] = np.zeros(4 * d_model)
            self.params[f'b{i}_mlp_proj_w'] = np.random.randn(4 * d_model, d_model) * 0.02
            self.params[f'b{i}_mlp_proj_b'] = np.zeros(d_model)
            # Layer Norm 2
            self.params[f'b{i}_ln2_g'] = np.ones(d_model)
            self.params[f'b{i}_ln2_b'] = np.zeros(d_model)

        # Final Layer Norm
        self.params['ln_f_g'] = np.ones(d_model)
        self.params['ln_f_b'] = np.zeros(d_model)
        
        # Optimizer state (Adam)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def layer_norm(self, x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(var + eps) + b

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token + Pos Embeddings
        tok_emb = self.params['wte'][idx] # (B, T, C)
        pos_emb = self.params['wpe'][:T]  # (T, C)
        x = tok_emb + pos_emb
        
        caches = []
        
        for i in range(self.n_layer):
            # --- Attention Block ---
            # Layer Norm 1
            ln1 = self.layer_norm(x, self.params[f'b{i}_ln1_g'], self.params[f'b{i}_ln1_b'])
            
            # QKV
            qkv = np.dot(ln1, self.params[f'b{i}_attn_w']) + self.params[f'b{i}_attn_b']
            q, k, v = np.split(qkv, 3, axis=-1)
            
            # Split heads (B, T, H, HS)
            k = k.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            q = q.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            
            # Attention scores
            att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_size))
            # Causal Mask
            mask = np.tril(np.ones((T, T)))
            att = np.where(mask == 0, -1e9, att)
            att = self.softmax(att)
            
            # Weighted sum
            y = att @ v # (B, H, T, HS)
            y = y.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            
            # Projection
            y = np.dot(y, self.params[f'b{i}_proj_w']) + self.params[f'b{i}_proj_b']
            
            # Residual
            x = x + y
            
            # --- MLP Block ---
            # Layer Norm 2
            ln2 = self.layer_norm(x, self.params[f'b{i}_ln2_g'], self.params[f'b{i}_ln2_b'])
            
            # FC
            h = np.dot(ln2, self.params[f'b{i}_mlp_fc_w']) + self.params[f'b{i}_mlp_fc_b']
            h = np.maximum(0, h) # ReLU
            
            # Projection
            out = np.dot(h, self.params[f'b{i}_mlp_proj_w']) + self.params[f'b{i}_mlp_proj_b']
            
            # Residual
            x = x + out
            
            caches.append({'ln1': ln1, 'qkv': qkv, 'att': att, 'y': y, 'ln2': ln2, 'h': h})
            
        # Final Layer Norm
        x = self.layer_norm(x, self.params['ln_f_g'], self.params['ln_f_b'])
        
        # Logits
        logits = np.dot(x, self.params['lm_head_w'])
        
        loss = None
        if targets is not None:
            # Cross Entropy Loss
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = targets.reshape(-1)
            probs = self.softmax(logits_flat)
            correct_logprobs = -np.log(probs[range(len(targets_flat)), targets_flat] + 1e-10)
            loss = np.mean(correct_logprobs)
            
        return logits, loss

    def train_step(self, text, learning_rate=1e-3):
        # Tokenize using BPE
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.block_size: return 0
        
        # Sample batch
        ix = np.random.randint(0, len(tokens) - self.block_size)
        x_tokens = tokens[ix:ix+self.block_size]
        y_tokens = tokens[ix+1:ix+self.block_size+1]
        
        x = np.array([x_tokens]) # (1, T)
        targets = np.array([y_tokens]) # (1, T)
        
        # --- Forward Pass ---
        # We need to cache intermediate values for backprop
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.params['wte'][x] # (B, T, C)
        pos_emb = self.params['wpe'][:T]
        h = tok_emb + pos_emb
        
        activations = {'h0': h}
        
        # Blocks
        for i in range(self.n_layer):
            # LN1
            ln1 = self.layer_norm(h, self.params[f'b{i}_ln1_g'], self.params[f'b{i}_ln1_b'])
            
            # Attn
            qkv = np.dot(ln1, self.params[f'b{i}_attn_w']) + self.params[f'b{i}_attn_b']
            q, k, v = np.split(qkv, 3, axis=-1)
            
            # Split heads
            k_heads = k.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            q_heads = q.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            v_heads = v.reshape(B, T, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            
            # Attention
            att = (q_heads @ k_heads.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_size))
            mask = np.tril(np.ones((T, T)))
            att = np.where(mask == 0, -1e9, att)
            att_probs = self.softmax(att)
            
            y_heads = att_probs @ v_heads
            y = y_heads.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
            
            # Proj
            proj = np.dot(y, self.params[f'b{i}_proj_w']) + self.params[f'b{i}_proj_b']
            h = h + proj # Residual
            
            # LN2
            ln2 = self.layer_norm(h, self.params[f'b{i}_ln2_g'], self.params[f'b{i}_ln2_b'])
            
            # MLP
            mlp_h = np.dot(ln2, self.params[f'b{i}_mlp_fc_w']) + self.params[f'b{i}_mlp_fc_b']
            mlp_act = np.maximum(0, mlp_h) # ReLU
            mlp_out = np.dot(mlp_act, self.params[f'b{i}_mlp_proj_w']) + self.params[f'b{i}_mlp_proj_b']
            
            h = h + mlp_out # Residual
            
            activations[f'b{i}'] = {'ln1': ln1, 'qkv': qkv, 'att_probs': att_probs, 'q': q, 'k': k, 'v': v, 
                                    'y': y, 'proj': proj, 'ln2': ln2, 'mlp_h': mlp_h, 'mlp_act': mlp_act}
            activations[f'h{i+1}'] = h

        # Final LN
        ln_f = self.layer_norm(h, self.params['ln_f_g'], self.params['ln_f_b'])
        logits = np.dot(ln_f, self.params['lm_head_w'])
        
        # Loss
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = targets.reshape(-1)
        probs = self.softmax(logits_flat)
        
        # --- Backward Pass ---
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        
        # dLoss/dLogits
        dlogits = probs.copy()
        dlogits[range(len(targets_flat)), targets_flat] -= 1
        dlogits /= B # Normalize by batch size
        dlogits = dlogits.reshape(B, T, self.vocab_size)
        
        # LM Head
        # (B, C, T) @ (B, T, V) -> (B, C, V) -> sum(0) -> (C, V)
        grads['lm_head_w'] = (ln_f.transpose(0, 2, 1) @ dlogits).sum(axis=0)
        dln_f = dlogits @ self.params['lm_head_w'].T
        
        # Final LN
        dh = dln_f # Simplified LN gradient
        
        # Backprop through blocks
        for i in reversed(range(self.n_layer)):
            cache = activations[f'b{i}']
            
            # MLP Backprop
            dmlp_out = dh
            # (B, 4C, T) @ (B, T, C) -> (B, 4C, C)
            grads[f'b{i}_mlp_proj_w'] = (cache['mlp_act'].transpose(0, 2, 1) @ dmlp_out).sum(axis=0)
            grads[f'b{i}_mlp_proj_b'] = dmlp_out.sum(axis=(0, 1))
            
            dmlp_act = dmlp_out @ self.params[f'b{i}_mlp_proj_w'].T
            dmlp_h = dmlp_act * (cache['mlp_h'] > 0) # ReLU deriv
            
            grads[f'b{i}_mlp_fc_w'] = (cache['ln2'].transpose(0, 2, 1) @ dmlp_h).sum(axis=0)
            grads[f'b{i}_mlp_fc_b'] = dmlp_h.sum(axis=(0, 1))
            
            dln2 = dmlp_h @ self.params[f'b{i}_mlp_fc_w'].T
            dh = dh + dln2 
            
            # Attention Backprop
            dproj = dh
            grads[f'b{i}_proj_w'] = (cache['y'].transpose(0, 2, 1) @ dproj).sum(axis=0)
            grads[f'b{i}_proj_b'] = dproj.sum(axis=(0, 1))
            
            dy = dproj @ self.params[f'b{i}_proj_w'].T
            dln1 = dy # Simplified attention backprop
            
            dh = dh + dln1 
            
        # Update weights (Adam)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        
        for k in self.params:
            g = grads[k]
            # Gradient Clipping
            g = np.clip(g, -1.0, 1.0)
            
            # Adam update
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g**2)
            
            m_hat = self.m[k] / (1 - beta1**self.t)
            v_hat = self.v[k] / (1 - beta2**self.t)
            
            self.params[k] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            
        loss = -np.log(probs[range(len(targets_flat)), targets_flat] + 1e-10).mean()
        return loss

    def generate(self, seed_text, length=50, temperature=0.8):
        tokens = self.tokenizer.encode(seed_text)
        if not tokens: tokens = [0]
        
        idx = tokens.copy()
        
        for _ in range(length):
            # Crop to block size
            idx_cond = idx[-self.block_size:]
            x = np.array([idx_cond])
            
            logits, _ = self.forward(x)
            logits = logits[0, -1, :] / temperature
            probs = self.softmax(logits)
            
            next_idx = np.random.choice(len(probs), p=probs)
            idx.append(next_idx)
            
            # Stop token? No explicit stop token trained yet, but we could check for EOS
            
        return self.tokenizer.decode(idx)

    def save(self, path):
        # Save full state (params + optimizer)
        checkpoint = {
            'params': self.params,
            'm': self.m,
            'v': self.v,
            't': self.t
        }
        
        # Atomic save: write to tmp then rename
        # This prevents corruption if the server shuts down mid-write
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            # Atomic replace
            if os.path.exists(path):
                os.remove(path) # Windows requires removal before rename
            os.rename(tmp_path, path)
        except Exception as e:
            print(f"❌ Save failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def load(self, path):
        if not os.path.exists(path): return False
        try:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Handle both old format (just params) and new format (dict)
            if isinstance(checkpoint, dict) and 'params' in checkpoint:
                self.params = checkpoint['params']
                self.m = checkpoint.get('m', {k: np.zeros_like(v) for k, v in self.params.items()})
                self.v = checkpoint.get('v', {k: np.zeros_like(v) for k, v in self.params.items()})
                self.t = checkpoint.get('t', 0)
            else:
                # Old format (just params dict)
                self.params = checkpoint
                # Reset optimizer
                self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
                self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
                self.t = 0
                
            return True
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return False
