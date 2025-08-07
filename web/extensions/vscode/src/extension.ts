import * as vscode from 'vscode';
import axios from 'axios';

interface MetaModelResponse {
    response: string;
    category: string;
    safety_validated: boolean;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('Meta-Model AI Assistant extension is now active!');

    // Register commands
    const chatCommand = vscode.commands.registerCommand('meta-model.chat', () => {
        showChatPanel();
    });

    const explainCommand = vscode.commands.registerCommand('meta-model.explain', () => {
        explainSelectedCode();
    });

    const improveCommand = vscode.commands.registerCommand('meta-model.improve', () => {
        improveSelectedCode();
    });

    const debugCommand = vscode.commands.registerCommand('meta-model.debug', () => {
        debugSelectedCode();
    });

    const documentCommand = vscode.commands.registerCommand('meta-model.document', () => {
        documentSelectedCode();
    });

    const testCommand = vscode.commands.registerCommand('meta-model.test', () => {
        generateTestsForCode();
    });

    context.subscriptions.push(
        chatCommand,
        explainCommand,
        improveCommand,
        debugCommand,
        documentCommand,
        testCommand
    );
}

async function showChatPanel() {
    const message = await vscode.window.showInputBox({
        prompt: 'Ask Meta-Model AI Assistant:',
        placeHolder: 'Enter your question or request...'
    });

    if (message) {
        await sendToMetaModel(message, 'chat');
    }
}

async function explainSelectedCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const selection = editor.selection;
    const text = editor.document.getText(selection);
    
    if (!text) {
        vscode.window.showErrorMessage('No text selected');
        return;
    }

    const prompt = `Please explain this code:\n\n${text}`;
    await sendToMetaModel(prompt, 'reasoning');
}

async function improveSelectedCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const selection = editor.selection;
    const text = editor.document.getText(selection);
    
    if (!text) {
        vscode.window.showErrorMessage('No text selected');
        return;
    }

    const prompt = `Please improve and optimize this code:\n\n${text}`;
    await sendToMetaModel(prompt, 'reasoning');
}

async function debugSelectedCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const selection = editor.selection;
    const text = editor.document.getText(selection);
    
    if (!text) {
        vscode.window.showErrorMessage('No text selected');
        return;
    }

    const prompt = `Please help debug this code and identify potential issues:\n\n${text}`;
    await sendToMetaModel(prompt, 'reasoning');
}

async function documentSelectedCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const selection = editor.selection;
    const text = editor.document.getText(selection);
    
    if (!text) {
        vscode.window.showErrorMessage('No text selected');
        return;
    }

    const prompt = `Please generate comprehensive documentation for this code:\n\n${text}`;
    await sendToMetaModel(prompt, 'reasoning');
}

async function generateTestsForCode() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const selection = editor.selection;
    const text = editor.document.getText(selection);
    
    if (!text) {
        vscode.window.showErrorMessage('No text selected');
        return;
    }

    const prompt = `Please generate comprehensive unit tests for this code:\n\n${text}`;
    await sendToMetaModel(prompt, 'reasoning');
}

async function sendToMetaModel(message: string, capability: string = 'chat'): Promise<void> {
    const config = vscode.workspace.getConfiguration('metaModel');
    const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');
    const apiKey = config.get<string>('apiKey');

    try {
        const response = await axios.post(`${serverUrl}/chat`, {
            message: message,
            capability: capability,
            stream: false
        }, {
            headers: {
                'Content-Type': 'application/json',
                ...(apiKey && { 'Authorization': `Bearer ${apiKey}` })
            },
            timeout: 30000
        });

        const data: MetaModelResponse = response.data;
        
        if (data.safety_validated) {
            await showResponse(data.response, data.category);
        } else {
            vscode.window.showWarningMessage('Response was blocked by safety system');
        }
    } catch (error) {
        console.error('Error communicating with Meta-Model AI:', error);
        vscode.window.showErrorMessage('Failed to communicate with Meta-Model AI Assistant');
    }
}

async function showResponse(response: string, category: string): Promise<void> {
    // Create and show a new webview panel
    const panel = vscode.window.createWebviewPanel(
        'metaModelResponse',
        'Meta-Model AI Response',
        vscode.ViewColumn.One,
        {
            enableScripts: true,
            retainContextWhenHidden: true
        }
    );

    // Set the webview content
    panel.webview.html = getWebviewContent(response, category);

    // Handle messages from the webview
    panel.webview.onDidReceiveMessage(
        message => {
            switch (message.command) {
                case 'insertCode':
                    insertCodeIntoEditor(message.code);
                    return;
                case 'copyToClipboard':
                    vscode.env.clipboard.writeText(message.text);
                    vscode.window.showInformationMessage('Copied to clipboard');
                    return;
            }
        },
        undefined,
        []
    );
}

function getWebviewContent(response: string, category: string): string {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Meta-Model AI Response</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                    padding: 20px;
                    margin: 0;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
                .category {
                    background-color: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    text-transform: uppercase;
                }
                .response {
                    background-color: var(--vscode-editor-background);
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 6px;
                    padding: 16px;
                    margin-bottom: 20px;
                    white-space: pre-wrap;
                    font-family: var(--vscode-editor-font-family);
                    line-height: 1.5;
                }
                .actions {
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }
                button {
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }
                button:hover {
                    background-color: var(--vscode-button-hoverBackground);
                }
                .code-block {
                    background-color: var(--vscode-textCodeBlock-background);
                    border: 1px solid var(--vscode-textCodeBlock-border);
                    border-radius: 4px;
                    padding: 12px;
                    margin: 10px 0;
                    font-family: var(--vscode-editor-font-family);
                    overflow-x: auto;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Meta-Model AI Response</h2>
                <span class="category">${category}</span>
            </div>
            
            <div class="response">${escapeHtml(response)}</div>
            
            <div class="actions">
                <button onclick="copyToClipboard()">Copy Response</button>
                <button onclick="extractAndInsertCode()">Insert Code</button>
                <button onclick="openInChat()">Open in Chat</button>
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                
                function copyToClipboard() {
                    const response = document.querySelector('.response').textContent;
                    vscode.postMessage({
                        command: 'copyToClipboard',
                        text: response
                    });
                }
                
                function extractAndInsertCode() {
                    const response = document.querySelector('.response').textContent;
                    const codeBlocks = extractCodeBlocks(response);
                    if (codeBlocks.length > 0) {
                        vscode.postMessage({
                            command: 'insertCode',
                            code: codeBlocks[0]
                        });
                    }
                }
                
                function openInChat() {
                    const response = document.querySelector('.response').textContent;
                    vscode.postMessage({
                        command: 'openInChat',
                        text: response
                    });
                }
                
                function extractCodeBlocks(text) {
                    const codeBlockRegex = /\`\`\`[\s\S]*?\`\`\`/g;
                    const matches = text.match(codeBlockRegex);
                    if (matches) {
                        return matches.map(block => 
                            block.replace(/```[\w]*\n?/, '').replace(/```$/, '')
                        );
                    }
                    return [];
                }
                
                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }
            </script>
        </body>
        </html>
    `;
}

function insertCodeIntoEditor(code: string): void {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        editor.edit(editBuilder => {
            if (editor.selection.isEmpty) {
                editBuilder.insert(editor.selection.active, code);
            } else {
                editBuilder.replace(editor.selection, code);
            }
        });
    }
}

export function deactivate() {
    console.log('Meta-Model AI Assistant extension is now deactivated!');
} 