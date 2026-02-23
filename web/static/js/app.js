/**
 * app.js – Main UI controller for Lily.
 *
 * Manages:
 *  - WebSocket connection to the FastAPI backend
 *  - Button (send / mic) event handling
 *  - Chat message rendering (with expression tags styled inline)
 *  - Audio recording and streaming to the backend
 *  - Receiving TTS audio and playing it
 *  - Receiving expression events and forwarding to avatar.js
 *  - Wake-word integration (backend pushes "wake_word" events over WS)
 */

import { applyExpression } from './avatar.js';

// ── WebSocket ─────────────────────────────────────────────────────────────

const WS_URL = `ws://${location.host}/ws`;
let ws        = null;
let wsReady   = false;

const connStatus = document.getElementById('connection-status');
const sysStatus  = document.getElementById('system-status');

function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    wsReady = true;
    connStatus.textContent = '🟢 Connected';
    fetchStatus();
  };

  ws.onclose = () => {
    wsReady = false;
    connStatus.textContent = '🔴 Disconnected – retrying…';
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => {
    connStatus.textContent = '🔴 Connection error';
  };

  ws.onmessage = evt => handleServerMessage(JSON.parse(evt.data));
}

connectWS();

// ── Status ────────────────────────────────────────────────────────────────

async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    sysStatus.textContent = [
      s.llm  ? '🧠 LLM'  : '⚠ no LLM',
      s.stt  ? '🎙 STT'  : '',
      s.tts  ? '🔊 TTS'  : '',
      s.wake_word ? '👂 Wake' : '',
    ].filter(Boolean).join('  ');
  } catch (_) {}
}

// ── Message rendering ─────────────────────────────────────────────────────

const messagesEl = document.getElementById('chat-messages');
let _currentLilyBubble = null;

function appendUserMessage(text) {
  _currentLilyBubble = null;
  const div = document.createElement('div');
  div.className = 'message user';
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function startLilyMessage() {
  _currentLilyBubble = document.createElement('div');
  _currentLilyBubble.className = 'message lily';
  messagesEl.appendChild(_currentLilyBubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendLilyToken(text) {
  if (!_currentLilyBubble) startLilyMessage();
  _currentLilyBubble.appendChild(document.createTextNode(text));
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendLilyExpression(raw) {
  if (!_currentLilyBubble) startLilyMessage();
  const span = document.createElement('span');
  span.className = 'expression-tag';
  span.textContent = ` *${raw}* `;
  _currentLilyBubble.appendChild(span);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendSystemMessage(text) {
  const div = document.createElement('div');
  div.className = 'message system';
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Server message handler ────────────────────────────────────────────────

function handleServerMessage(msg) {
  switch (msg.type) {
    case 'token':
      appendLilyToken(msg.text);
      break;

    case 'expression':
      appendLilyExpression(msg.raw);
      applyExpression(msg.name, 1.0, 2500);
      break;

    case 'tts_audio':
      playAudio(msg.data, msg.mime || 'audio/wav');
      break;

    case 'stt_result':
      if (msg.text) appendUserMessage(`🎙 ${msg.text}`);
      break;

    case 'wake_word':
      appendSystemMessage('Wake word detected – listening…');
      startRecording();
      break;

    case 'done':
      _currentLilyBubble = null;
      break;

    case 'history_cleared':
      appendSystemMessage('Conversation history cleared.');
      break;

    case 'error':
      appendSystemMessage(`⚠ ${msg.message}`);
      break;
  }
}

// ── TTS audio playback ────────────────────────────────────────────────────

function playAudio(base64Data, mime) {
  const binary = atob(base64Data);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const blob = new Blob([bytes], { type: mime });
  const url  = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.onended = () => URL.revokeObjectURL(url);
  audio.play().catch(console.warn);
}

// ── Text input / send button ──────────────────────────────────────────────

const textInput = document.getElementById('text-input');
const sendBtn   = document.getElementById('send-btn');

function sendText() {
  const text = textInput.value.trim();
  if (!text || !wsReady) return;
  appendUserMessage(text);
  startLilyMessage();
  ws.send(JSON.stringify({ type: 'text', content: text }));
  textInput.value = '';
}

sendBtn.addEventListener('click', sendText);
textInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendText(); });

// ── Mic button / audio recording ─────────────────────────────────────────

const micBtn = document.getElementById('mic-btn');
let _mediaRecorder = null;
let _audioChunks   = [];
let _recording     = false;

async function startRecording() {
  if (_recording) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    _mediaRecorder = new MediaRecorder(stream);
    _audioChunks   = [];
    _recording     = true;
    micBtn.classList.add('recording');

    _mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) _audioChunks.push(e.data);
    };

    _mediaRecorder.onstop = async () => {
      micBtn.classList.remove('recording');
      _recording = false;
      const blob = new Blob(_audioChunks, { type: 'audio/webm' });
      const buf  = await blob.arrayBuffer();
      const b64  = btoa(String.fromCharCode(...new Uint8Array(buf)));
      startLilyMessage();
      ws.send(JSON.stringify({ type: 'audio', data: b64 }));
      // Stop all tracks
      stream.getTracks().forEach(t => t.stop());
    };

    _mediaRecorder.start();
  } catch (err) {
    appendSystemMessage(`Microphone error: ${err.message}`);
  }
}

function stopRecording() {
  if (_mediaRecorder && _recording) {
    _mediaRecorder.stop();
  }
}

// Hold-to-talk behavior
micBtn.addEventListener('mousedown',  startRecording);
micBtn.addEventListener('mouseup',    stopRecording);
micBtn.addEventListener('mouseleave', stopRecording);
micBtn.addEventListener('touchstart', e => { e.preventDefault(); startRecording(); });
micBtn.addEventListener('touchend',   stopRecording);
