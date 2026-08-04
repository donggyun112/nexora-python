const $ = (id) => document.getElementById(id);
const state = { runId: null, callId: null, pendingId: null, model: "", busy: false, nextFault: false, recoverable: false, rounds: new Set(), effectIds: new Set(), events: 0, assistant: null, toolCalls: new Map() };

function setStatus(value, label = value.toUpperCase()) {
  const el = $("status");
  el.className = `status ${value}`;
  el.textContent = label;
}

function message(role, content = "") {
  $("messages").querySelector(".empty-state")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "OPERATOR" : "NEXORA AGENT";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  wrap.append(who, bubble);
  $("messages").append(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return bubble;
}

function json(value) {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function toolCall(event) {
  $("messages").querySelector(".empty-state")?.remove();
  const card = document.createElement("div");
  card.className = "tool-exchange pending";

  const head = document.createElement("div");
  head.className = "tool-head";
  const title = document.createElement("strong");
  title.textContent = `TOOL REQUEST · ${event.name || "unknown"}`;
  const status = document.createElement("span");
  status.className = "tool-status";
  status.textContent = "REQUESTED";
  head.append(title, status);

  const id = document.createElement("div");
  id.className = "tool-id";
  id.textContent = event.id || "NO CALL ID";

  const inputLabel = document.createElement("div");
  inputLabel.className = "tool-label";
  inputLabel.textContent = "INPUT";
  const input = document.createElement("pre");
  input.className = "tool-payload";
  input.textContent = json(event.input);

  const resultLabel = document.createElement("div");
  resultLabel.className = "tool-label result-label";
  resultLabel.textContent = "EXECUTION";
  const result = document.createElement("pre");
  result.className = "tool-payload tool-result";
  result.textContent = "Awaiting pre-tool gate…";

  card.append(head, id, inputLabel, input, resultLabel, result);
  $("messages").append(card);
  $("messages").scrollTop = $("messages").scrollHeight;
  state.toolCalls.set(event.id, { card, status, resultLabel, result });
  state.assistant = null;
  return state.toolCalls.get(event.id);
}

function toolResult(event) {
  const entry = state.toolCalls.get(event.id) || toolCall(event);
  entry.card.classList.remove("pending", "waiting", "success", "failure");
  entry.card.classList.add(event.is_error ? "failure" : "success");
  entry.status.textContent = event.is_error ? "FAILED" : "DONE";
  entry.resultLabel.textContent = "RESULT";
  entry.result.textContent = json(event.result);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function toolPermission(payload) {
  const entry = state.toolCalls.get(payload.call_id) || toolCall({
    id: payload.call_id,
    name: payload.name,
    input: payload.input,
  });
  entry.card.classList.remove("pending", "success", "failure");
  entry.card.classList.add("waiting");
  entry.status.textContent = "WAITING FOR PERMISSION";
  entry.resultLabel.textContent = "PERMISSION REQUEST";
  entry.result.textContent = json(payload.request);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function toolDenied(payload) {
  const entry = state.toolCalls.get(payload.call_id) || toolCall({
    id: payload.call_id,
    name: payload.name,
    input: payload.input,
  });
  entry.card.classList.remove("pending", "waiting", "success");
  entry.card.classList.add("failure");
  entry.status.textContent = "PERMISSION DENIED";
  entry.resultLabel.textContent = "PERMISSION DECISION";
  entry.result.textContent = json(payload.reason);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function toolRequestCancelled(payload) {
  const entry = state.toolCalls.get(payload.call_id) || toolCall({
    id: payload.call_id,
    name: payload.name,
    input: payload.input,
  });
  entry.card.classList.remove("pending", "waiting", "success");
  entry.card.classList.add("failure");
  entry.status.textContent = "CANCELLED";
  entry.resultLabel.textContent = "CONTROL DECISION";
  entry.result.textContent = json(payload.reason);
  state.callId = null;
  state.pendingId = null;
  state.recoverable = false;
  $("approval").classList.add("hidden");
  $("recovery").classList.add("hidden");
  $("messages").scrollTop = $("messages").scrollHeight;
}

function addEvent(name, detail = "", tone = "") {
  $("events").querySelector(".rail-empty")?.remove();
  const item = document.createElement("div");
  item.className = `event ${tone}`;
  const top = document.createElement("div");
  top.className = "event-top";
  const eventName = document.createElement("span");
  eventName.className = "event-name";
  eventName.textContent = name;
  const time = document.createElement("span");
  time.className = "event-time";
  time.textContent = new Date().toLocaleTimeString();
  top.append(eventName, time);
  const body = document.createElement("div");
  body.className = "event-detail";
  body.textContent = detail;
  item.append(top, body);
  $("events").append(item);
  $("events").scrollTop = $("events").scrollHeight;
  state.events += 1;
  $("event-count").textContent = state.events;
}

function resetRun(prompt) {
  state.callId = null;
  state.pendingId = null;
  state.recoverable = false;
  state.assistant = null;
  state.rounds = new Set();
  state.effectIds = new Set();
  state.toolCalls = new Map();
  state.events = 0;
  $("events").innerHTML = '<div class="rail-empty">Starting execution…</div>';
  $("round-count").textContent = "0";
  $("effect-count").textContent = "0";
  $("event-count").textContent = "0";
  $("approval").classList.add("hidden");
  $("recovery").classList.add("hidden");
  message("user", prompt);
  setStatus("running");
}

function handle(frame) {
  if (frame.kind === "meta") {
    state.runId = frame.run_id;
    $("run-id").textContent = frame.run_id;
    return;
  }
  if (frame.kind === "lifecycle") {
    const p = frame.payload || {};
    if (p.turn !== undefined) {
      state.rounds.add(p.turn);
      $("round-count").textContent = state.rounds.size;
    }
    if ((frame.type === "post_tool_use" || frame.type === "post_tool_use_failure") && p.call_id) {
      state.effectIds.add(p.call_id);
      $("effect-count").textContent = state.effectIds.size;
      toolResult({ id: p.call_id, name: p.name, result: p.result, is_error: frame.type === "post_tool_use_failure" });
    }
    if (frame.type === "permission_request" && p.source !== "tool_result") toolPermission(p);
    if (frame.type === "permission_denied") toolDenied(p);
    if (frame.type === "tool_request_cancelled") toolRequestCancelled(p);
    const detail = p.name || p.call_id || p.reason || JSON.stringify(p);
    const tone = frame.type.includes("failure") || frame.type.includes("denied") ? "fail" : frame.type.includes("permission") ? "wait" : "";
    addEvent(frame.type, detail, tone);
    return;
  }
  if (frame.kind === "agent") {
    const event = frame.event;
    if (event.type === "text") {
      if (!state.assistant) state.assistant = message("assistant");
      state.assistant.textContent += event.text;
      $("messages").scrollTop = $("messages").scrollHeight;
    }
    if (event.type === "tool_call") toolCall(event);
    if (event.type === "tool_result") toolResult(event);
    return;
  }
  if (frame.kind === "suspended") {
    state.callId = frame.tool_call_id;
    state.pendingId = frame.pending_id;
    $("approval").classList.remove("hidden");
    setStatus("suspended", "WAITING FOR SIGNAL");
    return;
  }
  if (frame.kind === "outcome") {
    state.callId = null;
    state.pendingId = null;
    $("approval").classList.add("hidden");
    $("recovery").classList.add("hidden");
    state.recoverable = false;
    setStatus("idle", "COMPLETED");
    return;
  }
  if (frame.kind === "recoverable") {
    state.recoverable = true;
    const entry = state.toolCalls.get(frame.tool_call_id);
    if (entry) {
      entry.card.classList.remove("pending", "success", "failure");
      entry.card.classList.add("waiting");
      entry.status.textContent = "STEP DONE · WORKER CRASHED";
      entry.resultLabel.textContent = "RECOVERY STATE";
      entry.result.textContent = "Result committed in StepLog. ToolMessage was not appended.";
    }
    $("recovery").classList.remove("hidden");
    setStatus("suspended", "RECOVERABLE CRASH");
    return;
  }
  if (frame.kind === "error") {
    setStatus("failed");
    if (!state.assistant) state.assistant = message("assistant");
    state.assistant.textContent = `Error: ${frame.message}`;
  }
}

async function consume(url, body) {
  state.busy = true;
  $("send").disabled = true;
  try {
    const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) throw new Error(await response.text());
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) if (line.trim()) handle(JSON.parse(line));
      if (done) break;
    }
    if (buffer.trim()) handle(JSON.parse(buffer));
  } catch (error) {
    handle({ kind: "error", message: error.message });
  } finally {
    state.busy = false;
    $("send").disabled = false;
  }
}

async function run() {
  const prompt = $("prompt").value.trim();
  if (!prompt || state.busy) return;
  state.model = $("model").value.trim();
  const switching = Boolean(state.callId && state.runId);
  const runId = switching ? state.runId : null;
  if (switching) {
    state.assistant = null;
    $("approval").classList.add("hidden");
    message("user", prompt);
    setStatus("running", "CANCELLING & SWITCHING");
  } else {
    resetRun(prompt);
  }
  $("prompt").value = "";
  const faultAfterStepCommit = state.nextFault;
  state.nextFault = false;
  await consume("/api/run", {
    prompt,
    model: state.model,
    run_id: runId,
    permission_gate: $("policy").value === "approval",
    fault_after_step_commit: faultAfterStepCommit,
  });
}

async function recover() {
  if (!state.runId || !state.recoverable || state.busy) return;
  state.assistant = null;
  $("recovery").classList.add("hidden");
  setStatus("running", "RECOVERING FROM STEP");
  await consume("/api/recover", { run_id: state.runId, model: state.model });
}

async function resume(approved) {
  if (!state.runId || !state.pendingId || state.busy) return;
  $("approval").classList.add("hidden");
  state.assistant = null;
  setStatus("running", "RESUMING");
  await consume("/api/resume", { run_id: state.runId, pending_id: state.pendingId, approved, model: state.model });
}

$("send").addEventListener("click", run);
$("prompt").addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") run(); });
$("approve").addEventListener("click", () => resume(true));
$("deny").addEventListener("click", () => resume(false));
$("recover").addEventListener("click", recover);
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => {
  $("prompt").value = button.dataset.prompt;
  $("policy").value = button.dataset.policy || "allow";
  state.nextFault = button.dataset.fault === "after-step";
  $("prompt").focus();
}));

fetch("/api/health").then((r) => r.json()).then((health) => {
  $("health-dot").classList.add("ok");
  $("engine-label").textContent = health.engine;
  $("model").value = health.default_model;
}).catch(() => { $("engine-label").textContent = "backend unavailable"; });
