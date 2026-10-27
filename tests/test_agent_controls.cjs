// Offline UI state tests: node --test tests/test_agent_controls.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');

const sandbox = {
    console,
    setTimeout: resolve => { resolve(); },
    document: {
        createElement: () => ({}),
        head: {appendChild() {}},
        addEventListener() {},
        querySelectorAll: () => []
    }
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('static/app.js', 'utf8') + '\nglobalThis.App = SuperBizAgentApp;', sandbox);

function snapshot(status, version, answer = '', mode = 'stream') {
    return {id: 'run', session_id: 'session', mode, status, version, answer, events: [], error: null};
}

function makeApp() {
    const app = Object.create(sandbox.App.prototype);
    Object.assign(app, {
        sessionId: 'session', currentMode: 'stream', currentChatHistory: [],
        activeRun: null, sessionRuns: new Map(), runControlBusy: false, isStreaming: false,
        sendButton: {setAttribute(name, value) { this[name] = value; }}, messageInput: {}, agentRunStatus: {},
        renderMarkdown: text => text, highlightCodeBlocks() {}, scrollToBottom() {},
        notifications: [], showNotification(text) { this.notifications.push(text); },
        addMessage() {
            const content = {innerHTML: ''};
            return {content, querySelector: () => content, classList: {toggle() {}}};
        }
    });
    return app;
}

async function settle(app) {
    for (let i = 0; i < 20 && app.activeRun.watching; i++) {
        await new Promise(resolve => setImmediate(resolve));
    }
    assert.equal(app.activeRun.watching, false);
}

test('stopped output remains, continue replaces it, history has one assistant entry', async () => {
    const app = makeApp();
    const responses = [snapshot('running', 1), snapshot('paused', 2, 'partial'),
        snapshot('running', 3), snapshot('completed', 4, 'full answer')];
    const paths = [];
    app.runApi = async path => { paths.push(path); return responses.shift(); };
    await app.startManagedRun('stream', 'question');
    assert.equal(app.sendButton.title, '继续');
    assert.equal(app.sendButton.disabled, false);
    assert.equal(app.messageInput.disabled, true);
    assert.match(app.activeRun.element.content.innerHTML, /partial/);
    await app.controlAgentRun();
    await settle(app);
    assert.equal(app.currentChatHistory.length, 1);
    assert.equal(app.currentChatHistory[0].content, 'full answer');
    assert.equal(app.activeRun.element.content.innerHTML, 'full answer');
    assert.equal(app.sendButton.disabled, false);
    assert.equal(app.sendButton.title, '发送');
    assert.equal(app.sendButton['data-action'], 'send');
    assert.ok(paths.includes('/run/resume'));
    app.renderManagedRun(app.activeRun, snapshot('paused', 2, 'stale response'));
    assert.equal(app.activeRun.snapshot.status, 'completed');
});

test('quick mode hides partial output until stopped or complete', async () => {
    const app = makeApp();
    app.runApi = async () => snapshot('completed', 4, 'final', 'quick');
    await app.startManagedRun('quick', 'question');
    app.renderManagedRun(app.activeRun, snapshot('running', 5, 'partial', 'quick'));
    assert.equal(app.activeRun.element.content.innerHTML, '正在思考…');
    assert.equal(app.sendButton.title, '停止');
    assert.equal(app.sendButton.disabled, false);
});

test('network loss offers reconnect without starting another agent', async () => {
    const app = makeApp();
    let calls = 0;
    const paths = [];
    app.runApi = async path => {
        paths.push(path);
        calls++;
        if (calls === 1) return snapshot('running', 1);
        if (calls === 2) throw new Error('offline');
        return snapshot('paused', 2, 'saved');
    };
    await app.startManagedRun('stream', 'question');
    assert.equal(app.sendButton.title, '重连');
    await app.controlAgentRun();
    await settle(app);
    assert.equal(app.sendButton.title, '继续');
    assert.equal(paths.filter(path => path === '').length, 1);
    assert.equal(paths[2], '/run');
});

test('AIOps final report is shown once', async () => {
    const app = makeApp();
    const final = snapshot('completed', 2, 'report', 'aiops');
    final.events = [{type: 'report', report: 'report'}, {type: 'complete'}];
    app.runApi = async path => path ? final : snapshot('running', 1, '', 'aiops');
    await app.startManagedRun('aiops');
    assert.equal(app.activeRun.element.content.innerHTML, 'report');
});

test('one send button dispatches send, stop and resume and waits for acknowledgement', async () => {
    const app = makeApp();
    const actions = [];
    app.sendMessage = () => actions.push('send');
    app.controlAgentRun = () => actions.push(app.activeRun.snapshot.status === 'paused' ? 'resume' : 'stop');
    app.handleSendButton();
    app.activeRun = {snapshot: snapshot('running', 1)};
    app.updateUI();
    assert.equal(app.sendButton['aria-label'], '停止');
    assert.match(app.sendButton.innerHTML, /<rect/);
    app.handleSendButton();
    app.activeRun.snapshot = snapshot('stopping', 2);
    app.updateUI();
    assert.equal(app.sendButton.disabled, true);
    assert.equal(app.sendButton.title, '正在停止…');
    app.activeRun.snapshot = snapshot('paused', 3);
    app.isStreaming = false;
    app.updateUI();
    assert.equal(app.sendButton.disabled, false);
    app.handleSendButton();
    assert.deepEqual(actions, ['send', 'stop', 'resume']);
});
