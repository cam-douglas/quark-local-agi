import { App, Editor, MarkdownView, Plugin, PluginSettingTab, Setting } from 'obsidian';

interface MetaModelSettings {
    serverUrl: string;
    apiKey: string;
    defaultCapability: string;
}

const DEFAULT_SETTINGS: MetaModelSettings = {
    serverUrl: 'http://localhost:8000',
    apiKey: '',
    defaultCapability: 'chat'
}

export default class MetaModelPlugin extends Plugin {
    settings: MetaModelSettings;

    async onload() {
        await this.loadSettings();

        // Add ribbon icon
        const ribbonIconEl = this.addRibbonIcon('brain', 'Meta-Model AI', (evt: MouseEvent) => {
            this.showChatModal();
        });

        // Add commands
        this.addCommand({
            id: 'meta-model-chat',
            name: 'Open Chat',
            callback: () => {
                this.showChatModal();
            }
        });

        this.addCommand({
            id: 'meta-model-explain',
            name: 'Explain Selected Text',
            editorCallback: (editor: Editor, view: MarkdownView) => {
                this.explainSelectedText(editor);
            }
        });

        this.addCommand({
            id: 'meta-model-improve',
            name: 'Improve Selected Text',
            editorCallback: (editor: Editor, view: MarkdownView) => {
                this.improveSelectedText(editor);
            }
        });

        this.addCommand({
            id: 'meta-model-summarize',
            name: 'Summarize Selected Text',
            editorCallback: (editor: Editor, view: MarkdownView) => {
                this.summarizeSelectedText(editor);
            }
        });

        this.addCommand({
            id: 'meta-model-generate',
            name: 'Generate Content',
            editorCallback: (editor: Editor, view: MarkdownView) => {
                this.generateContent(editor);
            }
        });

        // Add settings tab
        this.addSettingTab(new MetaModelSettingTab(this.app, this));
    }

    onunload() {
        console.log('Meta-Model AI plugin unloaded');
    }

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        await this.saveData(this.settings);
    }

    async showChatModal() {
        const modal = new ChatModal(this.app, this);
        modal.open();
    }

    async explainSelectedText(editor: Editor) {
        const selectedText = this.getSelectedText(editor);
        if (!selectedText) {
            new Notice('No text selected');
            return;
        }

        const prompt = `Please explain this text in detail:\n\n${selectedText}`;
        await this.sendToMetaModel(prompt, 'reasoning', editor);
    }

    async improveSelectedText(editor: Editor) {
        const selectedText = this.getSelectedText(editor);
        if (!selectedText) {
            new Notice('No text selected');
            return;
        }

        const prompt = `Please improve and enhance this text:\n\n${selectedText}`;
        await this.sendToMetaModel(prompt, 'reasoning', editor);
    }

    async summarizeSelectedText(editor: Editor) {
        const selectedText = this.getSelectedText(editor);
        if (!selectedText) {
            new Notice('No text selected');
            return;
        }

        const prompt = `Please provide a concise summary of this text:\n\n${selectedText}`;
        await this.sendToMetaModel(prompt, 'reasoning', editor);
    }

    async generateContent(editor: Editor) {
        const modal = new GenerateModal(this.app, this, editor);
        modal.open();
    }

    getSelectedText(editor: Editor): string {
        const selection = editor.getSelection();
        return selection || editor.getValue();
    }

    async sendToMetaModel(prompt: string, capability: string = 'chat', editor?: Editor) {
        try {
            const response = await fetch(`${this.settings.serverUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.settings.apiKey && { 'Authorization': `Bearer ${this.settings.apiKey}` })
                },
                body: JSON.stringify({
                    message: prompt,
                    capability: capability,
                    stream: false
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.safety_validated) {
                if (editor) {
                    this.insertResponse(editor, data.response);
                } else {
                    new Notice(data.response);
                }
            } else {
                new Notice('Response was blocked by safety system');
            }
        } catch (error) {
            console.error('Error communicating with Meta-Model AI:', error);
            new Notice('Failed to communicate with Meta-Model AI Assistant');
        }
    }

    insertResponse(editor: Editor, response: string) {
        const cursor = editor.getCursor();
        editor.replaceRange(`\n\n${response}\n\n`, cursor);
    }
}

class ChatModal extends Modal {
    plugin: MetaModelPlugin;
    inputEl: HTMLTextAreaElement;
    outputEl: HTMLDivElement;

    constructor(app: App, plugin: MetaModelPlugin) {
        super(app);
        this.plugin = plugin;
    }

    onOpen() {
        const { contentEl } = this;
        contentEl.createEl('h2', { text: 'Meta-Model AI Chat' });

        // Input area
        const inputContainer = contentEl.createDiv('input-container');
        this.inputEl = inputContainer.createEl('textarea', {
            attr: { placeholder: 'Ask Meta-Model AI Assistant...' }
        });
        this.inputEl.style.width = '100%';
        this.inputEl.style.height = '100px';
        this.inputEl.style.marginBottom = '10px';

        // Send button
        const sendButton = inputContainer.createEl('button', { text: 'Send' });
        sendButton.addEventListener('click', () => this.sendMessage());

        // Output area
        this.outputEl = contentEl.createDiv('output-container');
        this.outputEl.style.marginTop = '20px';
        this.outputEl.style.padding = '10px';
        this.outputEl.style.border = '1px solid var(--background-modifier-border)';
        this.outputEl.style.borderRadius = '4px';
        this.outputEl.style.minHeight = '200px';
        this.outputEl.style.maxHeight = '400px';
        this.outputEl.style.overflowY = 'auto';
    }

    async sendMessage() {
        const message = this.inputEl.value.trim();
        if (!message) return;

        this.outputEl.innerHTML = '<p>Thinking...</p>';
        
        try {
            await this.plugin.sendToMetaModel(message, this.plugin.settings.defaultCapability);
            this.outputEl.innerHTML = '<p>Message sent successfully!</p>';
        } catch (error) {
            this.outputEl.innerHTML = '<p>Error sending message</p>';
        }
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}

class GenerateModal extends Modal {
    plugin: MetaModelPlugin;
    editor: Editor;
    inputEl: HTMLTextAreaElement;

    constructor(app: App, plugin: MetaModelPlugin, editor: Editor) {
        super(app);
        this.plugin = plugin;
        this.editor = editor;
    }

    onOpen() {
        const { contentEl } = this;
        contentEl.createEl('h2', { text: 'Generate Content' });

        // Input area
        const inputContainer = contentEl.createDiv('input-container');
        this.inputEl = inputContainer.createEl('textarea', {
            attr: { placeholder: 'Describe what you want to generate...' }
        });
        this.inputEl.style.width = '100%';
        this.inputEl.style.height = '100px';
        this.inputEl.style.marginBottom = '10px';

        // Generate button
        const generateButton = inputContainer.createEl('button', { text: 'Generate' });
        generateButton.addEventListener('click', () => this.generateContent());
    }

    async generateContent() {
        const prompt = this.inputEl.value.trim();
        if (!prompt) return;

        try {
            await this.plugin.sendToMetaModel(prompt, 'reasoning', this.editor);
            this.close();
        } catch (error) {
            new Notice('Error generating content');
        }
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}

class MetaModelSettingTab extends PluginSettingTab {
    plugin: MetaModelPlugin;

    constructor(app: App, plugin: MetaModelPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();

        containerEl.createEl('h2', { text: 'Meta-Model AI Settings' });

        new Setting(containerEl)
            .setName('Server URL')
            .setDesc('Meta-Model AI Assistant server URL')
            .addText(text => text
                .setPlaceholder('http://localhost:8000')
                .setValue(this.plugin.settings.serverUrl)
                .onChange(async (value) => {
                    this.plugin.settings.serverUrl = value;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('API Key')
            .setDesc('Optional API key for authentication')
            .addText(text => text
                .setPlaceholder('Enter API key')
                .setValue(this.plugin.settings.apiKey)
                .onChange(async (value) => {
                    this.plugin.settings.apiKey = value;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Default Capability')
            .setDesc('Default AI capability to use')
            .addDropdown(dropdown => dropdown
                .addOption('chat', 'Chat')
                .addOption('reasoning', 'Reasoning')
                .addOption('planning', 'Planning')
                .addOption('memory', 'Memory')
                .addOption('web', 'Web Search')
                .addOption('metrics', 'Metrics')
                .setValue(this.plugin.settings.defaultCapability)
                .onChange(async (value) => {
                    this.plugin.settings.defaultCapability = value;
                    await this.plugin.saveSettings();
                }));
    }
} 