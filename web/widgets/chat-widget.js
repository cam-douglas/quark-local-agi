/**
 * Meta-Model AI Assistant - Embeddable Chat Widget
 */

class MetaModelWidget {
    constructor(config = {}) {
        this.config = {
            serverUrl: config.serverUrl || 'http://localhost:8000',
            apiKey: config.apiKey || '',
            position: config.position || 'bottom-right',
            theme: config.theme || 'light',
            ...config
        };
        
        this.isOpen = false;
        this.init();
    }

    init() {
        this.createWidget();
        this.attachEventListeners();
    }

    createWidget() {
        // Create container
        this.container = document.createElement('div');
        this.container.id = 'meta-model-widget';
        this.container.style.cssText = `
            position: fixed;
            z-index: 10000;
            ${this.getPositionStyles()}
        `;

        // Create toggle button
        this.toggleButton = document.createElement('button');
        this.toggleButton.innerHTML = '🤖';
        this.toggleButton.style.cssText = `
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            background: #6366f1;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        `;

        // Create chat window
        this.chatWindow = document.createElement('div');
        this.chatWindow.style.cssText = `
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            display: flex;
            flex-direction: column;
            opacity: 0;
            transform: scale(0.8);
            pointer-events: none;
            transition: all 0.3s ease;
        `;

        this.createChatContent();

        this.container.appendChild(this.toggleButton);
        this.container.appendChild(this.chatWindow);
        document.body.appendChild(this.container);
    }

    createChatContent() {
        // Header
        const header = document.createElement('div');
        header.style.cssText = `
            padding: 16px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;
        header.innerHTML = '<strong>Meta-Model AI</strong><button id="close-btn">×</button>';

        // Messages container
        this.messagesContainer = document.createElement('div');
        this.messagesContainer.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        `;

        // Input area
        const inputArea = document.createElement('div');
        inputArea.style.cssText = `
            padding: 16px;
            border-top: 1px solid #e9ecef;
            display: flex;
            gap: 8px;
        `;

        this.inputField = document.createElement('textarea');
        this.inputField.placeholder = 'Type your message...';
        this.inputField.style.cssText = `
            flex: 1;
            padding: 8px;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            resize: none;
            height: 40px;
        `;

        this.sendButton = document.createElement('button');
        this.sendButton.textContent = 'Send';
        this.sendButton.style.cssText = `
            padding: 8px 16px;
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        `;

        inputArea.appendChild(this.inputField);
        inputArea.appendChild(this.sendButton);
        this.chatWindow.appendChild(header);
        this.chatWindow.appendChild(this.messagesContainer);
        this.chatWindow.appendChild(inputArea);

        // Add welcome message
        this.addMessage('Hello! How can I help you?', 'ai');
    }

    attachEventListeners() {
        this.toggleButton.addEventListener('click', () => this.toggleChat());
        document.getElementById('close-btn').addEventListener('click', () => this.closeChat());
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    toggleChat() {
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        this.isOpen = true;
        this.chatWindow.style.opacity = '1';
        this.chatWindow.style.transform = 'scale(1)';
        this.chatWindow.style.pointerEvents = 'auto';
        this.toggleButton.style.display = 'none';
        this.inputField.focus();
    }

    closeChat() {
        this.isOpen = false;
        this.chatWindow.style.opacity = '0';
        this.chatWindow.style.transform = 'scale(0.8)';
        this.chatWindow.style.pointerEvents = 'none';
        this.toggleButton.style.display = 'block';
    }

    async sendMessage() {
        const message = this.inputField.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        this.inputField.value = '';

        try {
            const response = await fetch(`${this.config.serverUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` })
                },
                body: JSON.stringify({
                    message: message,
                    capability: 'chat'
                })
            });

            const data = await response.json();
            this.addMessage(data.response, 'ai');
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('Sorry, I encountered an error.', 'ai');
        }
    }

    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.style.cssText = `
            margin-bottom: 12px;
            display: flex;
            ${sender === 'user' ? 'justify-content: flex-end;' : ''}
        `;

        const messageContent = document.createElement('div');
        messageContent.style.cssText = `
            max-width: 80%;
            padding: 8px 12px;
            border-radius: 12px;
            background: ${sender === 'user' ? '#6366f1' : '#f1f3f4'};
            color: ${sender === 'user' ? 'white' : 'black'};
        `;
        messageContent.textContent = text;

        messageDiv.appendChild(messageContent);
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    getPositionStyles() {
        const positions = {
            'bottom-right': 'bottom: 20px; right: 20px;',
            'bottom-left': 'bottom: 20px; left: 20px;',
            'top-right': 'top: 20px; right: 20px;',
            'top-left': 'top: 20px; left: 20px;'
        };
        return positions[this.config.position] || positions['bottom-right'];
    }

    destroy() {
        if (this.container && this.container.parentNode) {
            this.container.parentNode.removeChild(this.container);
        }
    }
}

// Global initialization
window.MetaModelWidget = MetaModelWidget; 