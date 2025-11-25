// LSTM Chat Frontend - Connects to Python LSTM Server

// Auto-detect: use same domain in production, localhost in development
const API_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:5000'
    : window.location.origin;
let isGenerating = false;

async function sendMessage(message) {
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        if (!response.ok) throw new Error('Server error');

        const data = await response.json();
        return { text: data.response };
    } catch (error) {
        console.error('Chat error:', error);
        return { text: 'Server connection failed! Is Python server running?' };
    }
}

async function getStats() {
    try {
        const response = await fetch(`${API_URL}/stats`);
        return await response.json();
    } catch (error) {
        return null;
    }
}

function addMessage(text, isUser = false) {
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeMessage = messagesContainer.querySelector('.welcome-message');
    if (welcomeMessage) welcomeMessage.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    contentDiv.appendChild(timeDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showTypingIndicator() {
    const messagesContainer = document.getElementById('messagesContainer');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai';
    typingDiv.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    contentDiv.appendChild(typingIndicator);
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(contentDiv);
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) typingIndicator.remove();
}

async function handleSendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!message || isGenerating) return;

    addMessage(message, true);
    input.value = '';
    updateCharCount();
    autoResizeTextarea();

    isGenerating = true;
    showTypingIndicator();

    const response = await sendMessage(message);

    hideTypingIndicator();
    addMessage(response.text, false);
    isGenerating = false;
}

function updateCharCount() {
    const input = document.getElementById('messageInput');
    const charCount = document.getElementById('charCount');
    charCount.textContent = `${input.value.length} / 500`;
}

function autoResizeTextarea() {
    const textarea = document.getElementById('messageInput');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

function clearChat() {
    const messagesContainer = document.getElementById('messagesContainer');
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon"><i class="fas fa-robot"></i></div>
            <h2>LSTM Text Generator 🧠</h2>
            <p>Generates original text character-by-character!</p>
            <p class="welcome-info">
                <i class="fas fa-fire"></i> True text generation (not canned responses!)<br>
                <i class="fas fa-check-circle"></i> Learns from every conversation<br>
                <i class="fas fa-sync"></i> Training 24/7 in background<br>
                <i class="fas fa-brain"></i> Built from scratch with LSTM!
            </p>
            <div class="quick-prompts">
                <button class="prompt-btn" data-prompt="Hello!">👋 Say Hi</button>
                <button class="prompt-btn" data-prompt="Tell me about yourself">🤖 About You</button>
                <button class="prompt-btn" data-prompt="What can you do?">💭 Capabilities</button>
                <button class="prompt-btn" data-prompt="How do you learn?">📚 Learning</button>
            </div>
        </div>
    `;
    attachQuickPromptListeners();
}

function toggleTheme() {
    document.body.classList.toggle('light-theme');
}

async function showSettings() {
    const stats = await getStats();
    if (stats) {
        const trainingStatus = stats.is_training ? '🟢 ACTIVE' : '⚪ Idle';
        alert(`🧠 LSTM Text Generator\n\n` +
            `Training Status: ${trainingStatus}\n` +
            `Total Iterations: ${stats.total_iterations.toLocaleString()}\n` +
            `Current Loss: ${stats.current_loss.toFixed(4)}\n` +
            `Vocabulary: ${stats.vocab_size} characters\n` +
            `Learned Conversations: ${stats.conversations_count}\n` +
            `Training Queue: ${stats.queue_size}\n\n` +
            `✨ This AI generates text from scratch!\n` +
            `🔄 Trains 24/7 on all conversations`);
    }
}

function attachQuickPromptListeners() {
    document.querySelectorAll('.prompt-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('messageInput').value = btn.dataset.prompt;
            handleSendMessage();
        });
    });
}

async function checkServerConnection() {
    const loadingText = document.getElementById('loadingText');
    const progressFill = document.getElementById('progressFill');

    try {
        loadingText.textContent = 'Connecting to LSTM server...';
        progressFill.style.width = '50%';

        const stats = await getStats();

        if (stats) {
            progressFill.style.width = '100%';
            loadingText.textContent = `LSTM Ready! ${stats.vocab_size} char vocab 🧠`;

            setTimeout(() => {
                document.getElementById('loadingScreen').classList.add('fade-out');
                document.getElementById('chatContainer').classList.add('visible');

                setTimeout(() => {
                    document.getElementById('loadingScreen').style.display = 'none';
                }, 400);
            }, 1000);
        } else {
            throw new Error('Server not responding');
        }
    } catch (error) {
        loadingText.textContent = '❌ Server offline! Run: python server.py';
        progressFill.style.width = '0%';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    checkServerConnection();

    const messageInput = document.getElementById('messageInput');
    messageInput.addEventListener('input', () => {
        updateCharCount();
        autoResizeTextarea();
    });

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    document.getElementById('sendBtn').addEventListener('click', handleSendMessage);
    document.getElementById('clearBtn').addEventListener('click', () => {
        if (confirm('Clear messages?')) clearChat();
    });
    document.getElementById('settingsBtn').addEventListener('click', showSettings);
    document.getElementById('themeBtn').addEventListener('click', toggleTheme);

    attachQuickPromptListeners();
});

console.log('%c🧠 LSTM Text Generator!', 'color: #10b981; font-size: 24px; font-weight: bold;');
console.log('%cGenerates text character-by-character!', 'color: #667eea; font-size: 16px;');
console.log('%c24/7 Self-Learning System', 'color: #f59e0b; font-size: 14px;');
