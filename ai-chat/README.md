# LSTM Text Generator - Self-Learning Chat AI 🧠

A **true text generation AI** built from scratch that generates responses character-by-character using LSTM neural networks!

## 🌟 Features

- **Character-Level LSTM** - Generates text from scratch (not canned responses!)
- **24/7 Background Training** - Continuously learns from conversations
- **From-Scratch Implementation** - Pure Python/NumPy, no TensorFlow/PyTorch
- **Self-Learning** - Improves with every conversation
- **Model Persistence** - Saves progress automatically
- **Web Interface** - Beautiful chat UI

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy flask flask-cors
```

### 2. Start the LSTM Server

```bash
python server.py
```

Server starts on `http://localhost:5000`

### 3. Open Chat Interface

Open `index.html` in your browser!

## 🧠 How It Works

### LSTM Architecture

```
Input (Character) → LSTM Cell → Output (Next Character)
                       ↓
            [Forget Gate | Input Gate | Output Gate]
                       ↓
               Hidden State → Next Prediction
```

### Character-Level Generation

Instead of word-level (like simple chatbots), this AI:
1. Encodes text character-by-character
2. Learns patterns in the character sequences
3. Generates NEW text one character at a time
4. Can create responses it's never seen before!

### 24/7 Training Loop

```python
Background Thread:
  while True:
    - Check for new conversations
    - Retrain on updated data
    - Save model checkpoints
    - Continuously improve
```

## 📁 Project Structure

```
ai-chat/
├── lstm_generator.py        # LSTM implementation from scratch
├── continuous_trainer.py    # 24/7 training system
├── server.py                # Flask API server
├── app.js                   # Frontend JavaScript
├── index.html              # Chat UI
├── style.css               # Styling
├── lstm_model.pkl          # Saved model (auto-generated)
└── conversations.txt       # Training data (auto-generated)
```

## 🎯 API Endpoints

- `POST /chat` - Send message, get AI response
- `GET /stats` - View training statistics
- `POST /generate` - Custom text generation
- `GET /history` - Conversation history
- `POST /save` - Manual model save

## 🔧 Technical Details

### LSTM Implementation

- **Input Size**: Character vocabulary size
- **Hidden Size**: 128 neurons
- **Sequence Length**: 25 characters
- **Gates**: Forget, Input, Output, Candidate
- **Training**: Backpropagation Through Time (BPTT)

### Training Process

1. One-hot encode characters
2. Forward pass through LSTM
3. Calculate loss (cross-entropy)
4. Backward pass (gradients)
5. Update weights
6. Repeat continuously!

## 📊 Stats Dashboard

View real-time stats:
- Training status (Active/Idle)
- Total training iterations
- Current loss
- Vocabulary size
- Conversations learned
- Training queue size

## 💡 Tips

### First Run
- Model trains ~5 minutes on startup
- Starts with basic conversation knowledge
- Improves dramatically after 10+ conversations

### Better Responses
- Talk more! AI learns from every chat
- Use complete sentences
- Vary your questions
- After 50+ conversations, quality improves significantly

### Model Management
- Auto-saves every 5 minutes
- Delete `lstm_model.pkl` to retrain from scratch
- `conversations.txt` stores all training data

## 🎨 What Makes This Special?

### vs. Traditional Chatbots:
❌ Traditional: Match pattern → Return canned response  
✅ LSTM: Learn patterns → **Generate NEW text**

### vs. Pre-trained Models:
❌ GPT/BERT: Use pre-trained weights  
✅ This: **Built from scratch**, learns YOUR data

### Real Text Generation:
```
User: "Hello!"
Bot: Doesn't just return "Hi there!"
Bot: GENERATES: "Hello! I'm learning..." 
     (created character-by-character!)
```

## 🔬 Advanced Usage

### Custom Generation

```python
from lstm_generator import CharLSTM

lstm = CharLSTM()
lstm.load('lstm_model.pkl')

# Generate text
text = lstm.generate(
    seed_text="Hello",
    length=100,
    temperature=0.8  # Higher = more creative
)
```

### Manual Training

```python
from continuous_trainer import ContinuousTrainer

trainer = ContinuousTrainer()
trainer.add_conversation("user input", "ai response")
# Automatically trains in background!
```

## 📈 Monitoring Progress

Watch the console:
```
📚 Added conversation to training queue
📖 Training on 3 new conversations...
  Iteration 10/50, Loss: 0.2314
💾 Auto-saving model...
💪 Background training - Iteration 1000, Loss: 0.1847
```

## ⚠️ Limitations

- **Not as smart as GPT** (needs millions more parameters!)
- **Requires training time** (gets better over days)
- **CPU-intensive** (best on decent machines)
- **Small vocabulary initially** (expands with use)

## 🎓 Learning Resources

This implements:
- **LSTM networks** (Long Short-Term Memory)
- **Backpropagation Through Time** (BPTT)
- **Character-level language modeling**
- **Continuous learning systems**

Perfect for learning how real AI text generation works!

---

**Built with ❤️ using Pure Python & NumPy**  
*No TensorFlow, No PyTorch - Just Math!*
