// Meta-Model AI Assistant Frontend
// ChatGPT-like interface with real-time capabilities

class MetaModelApp {
    constructor() {
        this.currentCapability = 'chat';
        this.conversationHistory = [];
        this.isStreaming = false;
        this.websocket = null;
        this.settings = {
            serverUrl: 'http://localhost:8000',
            apiKey: '',
            theme: 'dark',
            autoScroll: true,
            streamingEnabled: true
        };
        
        this.init();
    }

    async init() {
        this.loadSettings();
        this.setupEventListeners();
        this.setupWebSocket();
        this.loadConversationHistory();
        this.updateUI();
        
        // Initialize with default capability
        this.switchCapability('chat');
    }

    loadSettings() {
        const saved = localStorage.getItem('metaModelSettings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
        this.applyTheme();
    }

    saveSettings() {
        localStorage.setItem('metaModelSettings', JSON.stringify(this.settings));
    }

    setupEventListeners() {
        // New chat button
        document.getElementById('new-chat-btn').addEventListener('click', () => {
            this.clearChat();
        });

        // Capability switching
        document.querySelectorAll('.capability-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const capability = e.currentTarget.dataset.capability;
                this.switchCapability(capability);
            });
        });

        // Chat input
        const chatInput = document.getElementById('chat-input');
        const sendButton = document.getElementById('send-btn');
        
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        chatInput.addEventListener('input', () => {
            // Auto-resize textarea
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
            
            // Enable/disable send button
            sendButton.disabled = !chatInput.value.trim() || this.isStreaming;
        });

        sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Settings
        document.getElementById('settings-btn').addEventListener('click', () => {
            this.showSettings();
        });

        // Help
        document.getElementById('help-btn').addEventListener('click', () => {
            this.showHelp();
        });

        // Clear chat
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearChat();
        });

        // Export chat
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportChat();
        });
    }

    setupWebSocket() {
        try {
            this.websocket = new WebSocket(`ws://${this.settings.serverUrl.replace('http', 'ws')}/ws`);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                // Attempt to reconnect
                setTimeout(() => this.setupWebSocket(), 5000);
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.updateConnectionStatus(false);
        }
    }

    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.className = connected ? 'status-connected' : 'status-disconnected';
            statusEl.innerHTML = connected ? 
                '<i class="fas fa-circle"></i> Connected' : 
                '<i class="fas fa-circle"></i> Disconnected';
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'stream_start':
                this.startStreamingResponse();
                break;
            case 'stream_chunk':
                this.appendStreamingChunk(data.content);
                break;
            case 'stream_end':
                this.endStreamingResponse();
                break;
            case 'error':
                this.showError(data.message);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    switchCapability(capability) {
        this.currentCapability = capability;
        
        // Update UI
        document.querySelectorAll('.capability-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-capability="${capability}"]`).classList.add('active');
        
        // Update chat placeholder
        const chatInput = document.getElementById('chat-input');
        const placeholders = {
            chat: 'Message Meta-Model AI...',
            memory: 'Search or store memories...',
            reasoning: 'Ask for logical reasoning...',
            planning: 'Request a plan or strategy...',
            web: 'Search the web for information...',
            metrics: 'Check performance metrics...'
        };
        chatInput.placeholder = placeholders[capability] || 'Message Meta-Model AI...';
        
        // Update capability display
        document.getElementById('current-capability').textContent = capability.charAt(0).toUpperCase() + capability.slice(1);
        
        // Load capability-specific history
        this.loadCapabilityHistory(capability);
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message || this.isStreaming) return;
        
        // Clear input and reset height
        input.value = '';
        input.style.height = 'auto';
        
        // Disable send button
        document.getElementById('send-btn').disabled = true;
        
        // Add user message to chat
        this.addMessageToChat('user', message);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            if (this.settings.streamingEnabled && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                // Use WebSocket for streaming
                this.websocket.send(JSON.stringify({
                    type: 'chat',
                    message: message,
                    capability: this.currentCapability
                }));
            } else {
                // Fallback to REST API
                await this.sendViaRestAPI(message);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.showError('Failed to send message. Please try again.');
            this.hideTypingIndicator();
        }
    }

    async sendViaRestAPI(message) {
        try {
            const response = await fetch(`${this.settings.serverUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.settings.apiKey && { 'Authorization': `Bearer ${this.settings.apiKey}` })
                },
                body: JSON.stringify({
                    message: message,
                    capability: this.currentCapability,
                    stream: false
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.hideTypingIndicator();
            
            if (data.error) {
                this.showError(data.error);
            } else {
                this.addMessageToChat('assistant', data.response || data.message);
            }
        } catch (error) {
            console.error('REST API error:', error);
            this.showError('Failed to get response. Please check your connection.');
            this.hideTypingIndicator();
        }
    }

    startStreamingResponse() {
        this.isStreaming = true;
        this.hideTypingIndicator();
        this.addMessageToChat('assistant', '', true);
    }

    appendStreamingChunk(content) {
        const lastMessage = document.querySelector('.message.assistant:last-child .message-content');
        if (lastMessage) {
            lastMessage.textContent += content;
            this.scrollToBottom();
        }
    }

    endStreamingResponse() {
        this.isStreaming = false;
        const streamingMessage = document.querySelector('.message.assistant:last-child');
        if (streamingMessage) {
            streamingMessage.classList.remove('streaming');
        }
        this.scrollToBottom();
    }

    addMessageToChat(role, content, isStreaming = false) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}${isStreaming ? ' streaming' : ''}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="message-role">${role === 'user' ? 'You' : 'Meta-Model AI'}</span>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-content">${this.escapeHtml(content)}</div>
            <div class="message-actions">
                ${role === 'assistant' ? `
                    <button class="action-btn" onclick="app.copyMessage(this)" title="Copy">
                        <i class="fas fa-copy"></i>
                    </button>
                    <button class="action-btn" onclick="app.regenerateResponse(this)" title="Regenerate">
                        <i class="fas fa-redo"></i>
                    </button>
                ` : ''}
            </div>
        `;
        
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Save to conversation history
        this.conversationHistory.push({
            role,
            content,
            capability: this.currentCapability,
            timestamp: new Date().toISOString()
        });
        
        this.saveConversationHistory();
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = `
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <span>Meta-Model AI is thinking...</span>
        `;
        
        const chatContainer = document.getElementById('chat-container');
        chatContainer.appendChild(indicator);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const indicator = document.querySelector('.typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message error';
        errorDiv.innerHTML = `
            <div class="message-header">
                <span class="message-role">Error</span>
                <span class="message-time">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${this.escapeHtml(message)}</div>
        `;
        
        const chatContainer = document.getElementById('chat-container');
        chatContainer.appendChild(errorDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        if (this.settings.autoScroll) {
            const chatContainer = document.getElementById('chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    copyMessage(button) {
        const messageContent = button.closest('.message').querySelector('.message-content').textContent;
        navigator.clipboard.writeText(messageContent).then(() => {
            this.showToast('Message copied to clipboard');
        });
    }

    async regenerateResponse(button) {
        const message = button.closest('.message');
        const userMessage = this.getLastUserMessage();
        
        if (userMessage) {
            message.remove();
            await this.sendMessage(userMessage);
        }
    }

    getLastUserMessage() {
        const userMessages = document.querySelectorAll('.message.user .message-content');
        return userMessages.length > 0 ? userMessages[userMessages.length - 1].textContent : null;
    }

    clearChat() {
        if (confirm('Are you sure you want to start a new chat?')) {
            document.getElementById('chat-container').innerHTML = `
                <div class="message assistant">
                    <div class="message-header">
                        <span class="message-role">Meta-Model AI</span>
                        <span class="message-time">Just now</span>
                    </div>
                    <div class="message-content">
                        Hello! I'm the Meta-Model AI Assistant. I can help you with various tasks including:

                        • **General conversation** and assistance
                        • **Memory management** and information retrieval
                        • **Logical reasoning** and problem solving
                        • **Planning** and strategy development
                        • **Web search** for current information
                        • **Performance metrics** and analysis

                        How can I assist you today?
                    </div>
                </div>
            `;
            this.conversationHistory = [];
            this.saveConversationHistory();
        }
    }

    exportChat() {
        const chatData = {
            conversation: this.conversationHistory,
            exportDate: new Date().toISOString(),
            capability: this.currentCapability
        };
        
        const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `meta-model-chat-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showToast('Chat exported successfully');
    }

    showSettings() {
        const settings = `
            <div class="settings-modal">
                <h3>Settings</h3>
                <div class="setting-group">
                    <label>Server URL:</label>
                    <input type="text" id="server-url" value="${this.settings.serverUrl}">
                </div>
                <div class="setting-group">
                    <label>API Key:</label>
                    <input type="password" id="api-key" value="${this.settings.apiKey}">
                </div>
                <div class="setting-group">
                    <label>Theme:</label>
                    <select id="theme-select">
                        <option value="dark" ${this.settings.theme === 'dark' ? 'selected' : ''}>Dark</option>
                        <option value="light" ${this.settings.theme === 'light' ? 'selected' : ''}>Light</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>
                        <input type="checkbox" id="auto-scroll" ${this.settings.autoScroll ? 'checked' : ''}>
                        Auto-scroll to new messages
                    </label>
                </div>
                <div class="setting-group">
                    <label>
                        <input type="checkbox" id="streaming-enabled" ${this.settings.streamingEnabled ? 'checked' : ''}>
                        Enable streaming responses
                    </label>
                </div>
                <div class="setting-actions">
                    <button onclick="app.saveSettingsFromModal()">Save</button>
                    <button onclick="app.closeModal()">Cancel</button>
                </div>
            </div>
        `;
        
        this.showModal(settings);
    }

    saveSettingsFromModal() {
        this.settings.serverUrl = document.getElementById('server-url').value;
        this.settings.apiKey = document.getElementById('api-key').value;
        this.settings.theme = document.getElementById('theme-select').value;
        this.settings.autoScroll = document.getElementById('auto-scroll').checked;
        this.settings.streamingEnabled = document.getElementById('streaming-enabled').checked;
        
        this.saveSettings();
        this.applyTheme();
        this.setupWebSocket();
        this.closeModal();
        this.showToast('Settings saved');
    }

    showHelp() {
        const help = `
            <div class="help-modal">
                <h3>Help & Usage</h3>
                <div class="help-section">
                    <h4>Capabilities</h4>
                    <ul>
                        <li><strong>Chat:</strong> General conversation and assistance</li>
                        <li><strong>Memory:</strong> Store and retrieve information</li>
                        <li><strong>Reasoning:</strong> Logical analysis and problem solving</li>
                        <li><strong>Planning:</strong> Create strategies and plans</li>
                        <li><strong>Web Search:</strong> Search the internet for information</li>
                        <li><strong>Metrics:</strong> View performance and usage statistics</li>
                    </ul>
                </div>
                <div class="help-section">
                    <h4>Keyboard Shortcuts</h4>
                    <ul>
                        <li><strong>Enter:</strong> Send message</li>
                        <li><strong>Shift+Enter:</strong> New line</li>
                        <li><strong>Ctrl/Cmd+C:</strong> Copy response</li>
                        <li><strong>Ctrl/Cmd+L:</strong> Clear chat</li>
                    </ul>
                </div>
                <div class="help-section">
                    <h4>Features</h4>
                    <ul>
                        <li>Real-time streaming responses</li>
                        <li>Conversation history</li>
                        <li>Export chat data</li>
                        <li>Dark/Light theme</li>
                        <li>WebSocket connection</li>
                    </ul>
                </div>
                <button onclick="app.closeModal()">Close</button>
            </div>
        `;
        
        this.showModal(help);
    }

    showModal(content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal">
                ${content}
            </div>
        `;
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal();
            }
        });
        
        document.body.appendChild(modal);
    }

    closeModal() {
        const modal = document.querySelector('.modal-overlay');
        if (modal) {
            modal.remove();
        }
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    applyTheme() {
        document.body.className = `theme-${this.settings.theme}`;
    }

    loadConversationHistory() {
        const saved = localStorage.getItem('metaModelConversationHistory');
        if (saved) {
            this.conversationHistory = JSON.parse(saved);
        }
    }

    saveConversationHistory() {
        localStorage.setItem('metaModelConversationHistory', JSON.stringify(this.conversationHistory));
    }

    loadCapabilityHistory(capability) {
        // Filter conversation history by capability
        const capabilityHistory = this.conversationHistory.filter(msg => msg.capability === capability);
        
        // Update conversation history display
        const historyContainer = document.getElementById('conversation-history');
        historyContainer.innerHTML = '';
        
        capabilityHistory.slice(-5).forEach(msg => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.textContent = msg.content.substring(0, 50) + (msg.content.length > 50 ? '...' : '');
            historyItem.addEventListener('click', () => {
                this.addMessageToChat(msg.role, msg.content);
            });
            historyContainer.appendChild(historyItem);
        });
    }

    updateUI() {
        // Update connection status
        this.updateConnectionStatus(this.websocket && this.websocket.readyState === WebSocket.OPEN);
        
        // Update current capability
        document.getElementById('current-capability').textContent = 
            this.currentCapability.charAt(0).toUpperCase() + this.currentCapability.slice(1);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MetaModelApp();
});

// Global functions for button actions
function copyMessage(button) {
    window.app.copyMessage(button);
}

function regenerateResponse(button) {
    window.app.regenerateResponse(button);
} 