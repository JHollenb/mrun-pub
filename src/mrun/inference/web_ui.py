# ruff: noqa: E501
"""Self-contained browser chat shell for the native inference data plane."""

from __future__ import annotations

import html

_CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mrun native chat</title>
  <style nonce="__NONCE__">
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #0b1020; color: #e6edf7; }
    button, input, textarea, select { font: inherit; }
    button { cursor: pointer; }
    .shell { display: grid; grid-template-columns: minmax(230px, 290px) 1fr; min-height: 100vh; }
    aside { border-right: 1px solid #26324d; padding: 1rem; background: #10172a; overflow: auto; }
    main { display: grid; grid-template-rows: auto 1fr auto; min-width: 0; }
    h1 { margin: 0 0 .25rem; font-size: 1.2rem; }
    h2 { margin: 1.2rem 0 .55rem; font-size: .9rem; color: #a9bad5; }
    .muted { color: #8fa1be; font-size: .82rem; line-height: 1.4; }
    .field { display: grid; gap: .25rem; margin: .55rem 0; }
    .field label { color: #b7c5da; font-size: .78rem; }
    input, textarea, select { width: 100%; color: #eef4ff; background: #0a1120; border: 1px solid #344362; border-radius: .5rem; padding: .55rem; }
    input:focus, textarea:focus, button:focus { outline: 2px solid #67b0ff; outline-offset: 1px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; }
    .actions { display: flex; gap: .45rem; flex-wrap: wrap; }
    button { color: #eff6ff; background: #253555; border: 1px solid #3d5278; border-radius: .5rem; padding: .5rem .7rem; }
    button.primary { background: #1268c4; border-color: #2383e2; }
    button.danger { background: #6d2835; border-color: #984052; }
    button:disabled { cursor: not-allowed; opacity: .45; }
    header { display: flex; justify-content: space-between; gap: 1rem; align-items: center; padding: .75rem 1rem; border-bottom: 1px solid #26324d; }
    #status { color: #9eb2cf; font-size: .82rem; text-align: right; }
    #transcript { padding: 1.2rem max(1rem, calc((100% - 850px) / 2)); overflow: auto; }
    .message { margin: 0 0 1rem; border: 1px solid #293755; border-radius: .8rem; overflow: hidden; }
    .message .role { padding: .42rem .7rem; color: #9eb2cf; background: #141d33; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; }
    .message .content { padding: .8rem; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.55; }
    .message.user { border-color: #345a80; }
    .message.assistant { border-color: #365343; }
    .composer { border-top: 1px solid #26324d; padding: .8rem max(1rem, calc((100% - 850px) / 2)); background: #0d1425; }
    #prompt { min-height: 5.5rem; resize: vertical; }
    .composer-row { display: flex; justify-content: space-between; gap: .5rem; margin-top: .5rem; }
    .empty { color: #7689a8; text-align: center; margin-top: 18vh; }
    code { color: #a9d4ff; }
    @media (max-width: 760px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid #26324d; }
      main { min-height: 70vh; }
    }
  </style>
</head>
<body>
<div class="shell">
  <aside>
    <h1>mrun native chat</h1>
    <div class="muted">Direct browser client for the loaded compute-host model. No external assets or telemetry.</div>
    <h2>Connection</h2>
    <div class="field"><label for="api-key">Bearer token (kept only in this page)</label><input id="api-key" type="password" autocomplete="off" spellcheck="false"></div>
    <div class="actions"><button id="connect">Load model info</button></div>
    <div id="model-info" class="muted" aria-live="polite">Not connected.</div>
    <h2>Sampling</h2>
    <div class="grid">
      <div class="field"><label for="max-tokens">Max tokens</label><input id="max-tokens" type="number" min="1" value="256"></div>
      <div class="field"><label for="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="0.05" value="0"></div>
      <div class="field"><label for="top-p">Top p</label><input id="top-p" type="number" min="0.01" max="1" step="0.01" value="1"></div>
      <div class="field"><label for="top-k">Top k</label><input id="top-k" type="number" min="0" step="1" value="0"></div>
      <div class="field"><label for="frequency">Frequency penalty</label><input id="frequency" type="number" min="-2" max="2" step="0.1" value="0"></div>
      <div class="field"><label for="presence">Presence penalty</label><input id="presence" type="number" min="-2" max="2" step="0.1" value="0"></div>
    </div>
    <div class="field"><label for="seed">Seed (blank = random)</label><input id="seed" type="number" step="1"></div>
    <div class="field"><label for="stop-sequence">Stop string (optional)</label><input id="stop-sequence" type="text"></div>
    <div class="field"><label for="system-prompt">System instruction (optional)</label><textarea id="system-prompt" rows="3"></textarea></div>
    <div class="actions"><button id="clear">Clear conversation</button></div>
  </aside>
  <main>
    <header><strong id="model-name">No model selected</strong><span id="status" role="status" aria-live="polite">Idle</span></header>
    <section id="transcript" aria-label="Conversation"><div class="empty">Load model info, then start a conversation.</div></section>
    <section class="composer">
      <label class="muted" for="prompt">Message</label>
      <textarea id="prompt" placeholder="Ask the local model…"></textarea>
      <div class="composer-row">
        <div class="actions"><button id="stop" class="danger" disabled>Stop</button><button id="regenerate" disabled>Regenerate</button></div>
        <button id="send" class="primary">Send</button>
      </div>
    </section>
  </main>
</div>
<script nonce="__NONCE__">
(() => {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const state = { model: null, messages: [], draft: "", controller: null, active: null };
  const authHeaders = () => {
    const headers = {"content-type": "application/json"};
    const token = byId("api-key").value;
    if (token) headers.authorization = `Bearer ${token}`;
    return headers;
  };
  const setStatus = (value) => { byId("status").textContent = value; };
  const running = () => state.controller !== null;
  const setRunning = (value) => {
    byId("send").disabled = value;
    byId("stop").disabled = !value;
    const tailRole = state.messages.at(-1)?.role;
    byId("regenerate").disabled = value || !["user", "assistant"].includes(tailRole);
    byId("clear").disabled = value;
  };
  const messageNode = (message, draft = false) => {
    const card = document.createElement("article");
    card.className = `message ${message.role}`;
    const role = document.createElement("div");
    role.className = "role";
    role.textContent = draft ? "assistant · generating" : message.role;
    const content = document.createElement("div");
    content.className = "content";
    content.textContent = message.content;
    card.append(role, content);
    return card;
  };
  const render = () => {
    const transcript = byId("transcript");
    transcript.replaceChildren();
    for (const message of state.messages) transcript.append(messageNode(message));
    if (state.draft || running()) transcript.append(messageNode({role: "assistant", content: state.draft}, true));
    if (!state.messages.length && !state.draft) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "Your conversation stays in this browser page; the server logs metadata only.";
      transcript.append(empty);
    }
    transcript.scrollTop = transcript.scrollHeight;
    setRunning(running());
  };
  const publicError = async (response) => {
    try {
      const body = await response.json();
      return body?.error?.message || `HTTP ${response.status}`;
    } catch (_) {
      return `HTTP ${response.status}`;
    }
  };
  const loadModel = async () => {
    setStatus("Loading model info…");
    const response = await fetch("/v1/models", {headers: authHeaders(), cache: "no-store"});
    if (!response.ok) throw new Error(await publicError(response));
    const body = await response.json();
    state.model = body.data[0];
    const info = state.model.mrun;
    const limit = info.rate_limit ? ` · burst ${info.rate_limit.burst_tokens.toLocaleString()}` : "";
    byId("model-name").textContent = state.model.id;
    byId("model-info").textContent = `${state.model.id} · context ${info.context_size.toLocaleString()} · ${info.sessions ? "sessions on" : "sessions off"}${limit}`;
    setStatus("Ready");
  };
  const processFrame = (frame, onData) => {
    const lines = frame.replaceAll("\r\n", "\n").split("\n");
    const payload = lines.filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()).join("\n");
    if (payload) onData(payload);
  };
  const readSse = async (response, onData) => {
    if (!response.body) throw new Error("Streaming response has no body");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        processFrame(buffer.slice(0, boundary), onData);
        buffer = buffer.slice(boundary + 2);
      }
      if (done) break;
    }
    if (buffer.trim()) processFrame(buffer, onData);
  };
  const numberValue = (id) => Number(byId(id).value);
  const payload = () => {
    const request = {
      model: state.model.id,
      messages: state.messages,
      stream: true,
      stream_options: {include_usage: true},
      max_tokens: numberValue("max-tokens"),
      temperature: numberValue("temperature"),
      top_p: numberValue("top-p"),
      top_k: numberValue("top-k"),
      frequency_penalty: numberValue("frequency"),
      presence_penalty: numberValue("presence")
    };
    const seed = byId("seed").value.trim();
    const stop = byId("stop-sequence").value;
    if (seed) request.seed = Number(seed);
    if (stop) request.stop = stop;
    return request;
  };
  const execute = async () => {
    state.controller = new AbortController();
    state.draft = "";
    render();
    setStatus("Generating…");
    let completed = false;
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify(payload()),
        signal: state.controller.signal,
        cache: "no-store"
      });
      if (!response.ok) throw new Error(await publicError(response));
      await readSse(response, (raw) => {
        if (raw === "[DONE]") { completed = true; return; }
        const event = JSON.parse(raw);
        if (event.error) throw new Error(event.error.message || "Generation failed");
        const choice = event.choices?.[0];
        if (choice?.delta?.content) {
          state.draft += choice.delta.content;
          render();
        }
        if (choice?.finish_reason) setStatus(`Finished: ${choice.finish_reason}`);
      });
      if (!completed) throw new Error("Stream ended without [DONE]");
      state.messages.push({role: "assistant", content: state.draft});
      state.draft = "";
      if (!byId("status").textContent.startsWith("Finished:")) setStatus("Finished");
    } catch (error) {
      if (error?.name === "AbortError") setStatus("Stopped");
      else setStatus(`Error: ${error?.message || "request failed"}`);
      if (state.draft) state.messages.push({role: "assistant", content: state.draft});
      state.draft = "";
    } finally {
      state.controller = null;
      render();
    }
  };
  const send = async () => {
    if (running()) return;
    if (!state.model) await loadModel();
    const content = byId("prompt").value.trim();
    if (!content) { setStatus("Enter a message first"); return; }
    if (state.messages.at(-1)?.role === "user") {
      setStatus("Regenerate the unanswered message before adding another");
      return;
    }
    if (!state.messages.length) {
      const system = byId("system-prompt").value.trim();
      if (system) state.messages.push({role: "system", content: system});
    }
    state.messages.push({role: "user", content});
    byId("prompt").value = "";
    render();
    state.active = execute();
    await state.active;
    state.active = null;
  };
  const stop = () => { if (state.controller) state.controller.abort(); };
  const regenerate = async () => {
    if (running()) { stop(); await state.active; }
    if (state.messages.at(-1)?.role === "assistant") state.messages.pop();
    if (state.messages.at(-1)?.role !== "user") return;
    render();
    state.active = execute();
    await state.active;
    state.active = null;
  };
  byId("connect").addEventListener("click", () => loadModel().catch((error) => setStatus(`Error: ${error.message}`)));
  byId("send").addEventListener("click", () => send().catch((error) => setStatus(`Error: ${error.message}`)));
  byId("stop").addEventListener("click", stop);
  byId("regenerate").addEventListener("click", () => regenerate().catch((error) => setStatus(`Error: ${error.message}`)));
  byId("clear").addEventListener("click", () => { state.messages = []; state.draft = ""; render(); setStatus("Cleared"); });
  byId("prompt").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); byId("send").click(); }
  });
  render();
  loadModel().catch(() => setStatus("Enter bearer token, then load model info"));
})();
</script>
</body>
</html>
"""


def browser_chat_html(nonce: str) -> str:
    if type(nonce) is not str or not nonce or not nonce.isascii():
        raise ValueError("browser CSP nonce must be a non-empty ASCII string")
    return _CHAT_HTML.replace("__NONCE__", html.escape(nonce, quote=True))


__all__ = ["browser_chat_html"]
