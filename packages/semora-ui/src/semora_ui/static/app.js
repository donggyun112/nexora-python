const $ = (id) => document.getElementById(id);
const state = { runId: null, parentRunId: null, attachTo: null, callId: null, pendingId: null, pending: [], model: "", busy: false, nextFault: false, recoverable: false, rounds: new Set(), effectIds: new Set(), events: 0, assistant: null, toolCalls: new Map(), children: new Map(), watching: null };

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
  who.textContent = role === "user" ? "OPERATOR" : "SEMORA AGENT";
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
  state.assistant = state.thinking = null;
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
  state.pending = [];
  state.recoverable = false;
  $("approval").classList.add("hidden");
  $("recovery").classList.add("hidden");
  $("messages").scrollTop = $("messages").scrollHeight;
}

// A child's own stream. One card per child, indented under the round that launched it — the
// operator is watching a second agent work, not reading more of the first one's output.
function childCard(agent) {
  const known = state.children.get(agent);
  if (known) return known;
  $("messages").querySelector(".empty-state")?.remove();
  const card = document.createElement("div");
  card.className = "child-exchange";
  const head = document.createElement("div");
  head.className = "child-head";
  const title = document.createElement("strong");
  title.textContent = `SUBAGENT · ${agent}`;
  const status = document.createElement("span");
  status.className = "child-status";
  status.textContent = "WORKING";
  head.append(title, status);
  const run = document.createElement("div");
  run.className = "child-run";
  run.textContent = "";
  const body = document.createElement("pre");
  body.className = "child-body";
  body.textContent = "";
  const steps = document.createElement("div");
  steps.className = "child-step";
  card.append(head, run, body, steps);
  $("messages").append(card);
  $("messages").scrollTop = $("messages").scrollHeight;
  const entry = { card, status, run, body, steps };
  state.children.set(agent, entry);
  return entry;
}

function childEvent(frame) {
  const entry = childCard(frame.agent);
  const event = frame.event || {};
  if (event.type === "text") entry.body.textContent += event.text;
  if (event.type === "tool_call") {
    const line = document.createElement("div");
    line.innerHTML = "→ <b></b>";
    line.querySelector("b").textContent = `${event.name} ${json(event.input)}`;
    entry.steps.append(line);
  }
  if (event.type === "tool_result") {
    const line = document.createElement("div");
    line.innerHTML = "  ← <b></b>";
    line.querySelector("b").textContent = json(event.result);
    entry.steps.append(line);
  }
  if (event.type === "done") {
    entry.card.classList.add("done");
    entry.status.textContent = `DONE · ${event.stop_reason || "completed"}`;
  }
  if (event.type === "error") {
    entry.card.classList.add("failure");
    entry.status.textContent = "FAILED";
    entry.body.textContent += `\n${event.message}`;
  }
  $("messages").scrollTop = $("messages").scrollHeight;
}

function childStarted(payload) {
  const entry = childCard(String(payload.agent_id || "subagent"));
  entry.run.textContent = payload.run_id || "";
}

function childStopped(payload) {
  const entry = state.children.get(String(payload.agent_id || "subagent"));
  if (!entry) return;
  const failed = payload.reason === "error";
  entry.card.classList.add(failed ? "failure" : "done");
  entry.status.textContent = failed ? "FAILED" : "ANSWERED";
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
  state.parentRunId = null;
  state.attachTo = null;
  $("attached").classList.add("hidden");
  $("agents").innerHTML = '<div class="rail-empty">No children yet</div>';
  state.callId = null;
  state.pendingId = null;
  state.pending = [];
  state.recoverable = false;
  state.assistant = state.thinking = null;
  state.rounds = new Set();
  state.effectIds = new Set();
  state.toolCalls = new Map();
  state.children = new Map();
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
    if (frame.type === "subagent_start") childStarted(p);
    if (frame.type === "subagent_stop") childStopped(p);
    const detail = p.name || p.call_id || p.reason || JSON.stringify(p);
    const tone = frame.type.includes("failure") || frame.type.includes("denied") ? "fail" : frame.type.includes("permission") ? "wait" : "";
    addEvent(frame.type, detail, tone);
    return;
  }
  if (frame.kind === "child") {
    childEvent(frame);
    return;
  }
  if (frame.kind === "agent") {
    const event = frame.event;
    if (event.type === "thinking") {
      // Its own bubble, above the answer: reasoning is not the answer, and a turn that thinks
      // before every tool call would otherwise splice its notes into the middle of a sentence.
      if (!state.thinking) {
        state.thinking = message("assistant");
        state.thinking.className = "bubble thinking";
      }
      state.thinking.textContent += event.content;
      $("messages").scrollTop = $("messages").scrollHeight;
    }
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
    state.pending = Array.isArray(frame.pending) && frame.pending.length
      ? frame.pending
      : [[frame.pending_id, frame.tool_call_id]];
    [state.pendingId, state.callId] = state.pending[0];
    $("approval").classList.remove("hidden");
    setStatus("suspended", `WAITING FOR ${state.pending.length} SIGNAL${state.pending.length === 1 ? "" : "S"}`);
    return;
  }
  if (frame.kind === "outcome") {
    state.callId = null;
    state.pendingId = null;
    state.pending = [];
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
      entry.result.textContent = `${frame.message}\nResult committed in StepLog. ToolMessage was not appended.`;
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

// The ledger, not the rail. Events say what was announced; this says what recovery will read —
// a crash after a commit leaves `done` here, and a crash before one leaves `running`.
async function refreshLedger() {
  if (!state.runId) return;
  let steps = [];
  try {
    const response = await fetch(`/api/steps/${encodeURIComponent(state.runId)}`);
    if (!response.ok) return;
    ({ steps } = await response.json());
  } catch { return; }
  const box = $("ledger");
  box.textContent = "";
  if (!steps.length) {
    const empty = document.createElement("div");
    empty.className = "rail-empty";
    empty.textContent = "No steps yet";
    box.append(empty);
    return;
  }
  for (const step of steps) {
    const item = document.createElement("div");
    item.className = `event ${step.status === "running" ? "wait" : ""}`;
    const top = document.createElement("div");
    top.className = "event-top";
    const name = document.createElement("span");
    name.className = "event-name";
    name.textContent = step.key;
    const status = document.createElement("span");
    status.className = "event-time";
    status.textContent = step.status.toUpperCase();
    top.append(name, status);
    const detail = document.createElement("div");
    detail.className = "event-detail";
    detail.textContent = step.value || (step.status === "running" ? "started, never finished — Indeterminate on the next attempt" : "");
    item.append(top, detail);
    box.append(item);
  }
}

// Two lists, not one. `background` is what `cancel_task` can reach; `independent` is a run id and
// nothing else, which is exactly what opening an independent agent hands back.
async function refreshTasks() {
  const runId = state.parentRunId || state.runId;
  if (!runId) return;
  let data;
  try {
    const response = await fetch(`/api/tasks/${encodeURIComponent(runId)}`);
    if (!response.ok) return;
    data = await response.json();
  } catch { return; }
  const box = $("agents");
  box.textContent = "";
  const background = data.background || [];
  const independent = data.independent || [];
  if (!background.length && !independent.length) {
    const empty = document.createElement("div");
    empty.className = "rail-empty";
    empty.textContent = "No children yet";
    box.append(empty);
    return;
  }
  if (background.length) {
    box.append(group("ON THE PARENT'S LEASH"));
    for (const task of background) box.append(taskRow(task));
  }
  if (independent.length) {
    box.append(group("INDEPENDENT · ADDRESS ONLY"));
    for (const agent of independent) box.append(independentRow(agent));
  }
}

function group(text) {
  const label = document.createElement("div");
  label.className = "group";
  label.textContent = text;
  return label;
}

function taskRow(task) {
  const row = document.createElement("div");
  row.className = `task ${task.status}`;
  const name = document.createElement("div");
  name.className = "task-name";
  name.textContent = task.label;
  const status = document.createElement("span");
  status.className = "task-state";
  status.textContent = task.status.toUpperCase();
  const meta = document.createElement("div");
  meta.className = "task-meta";
  meta.textContent = task.task_id;
  row.append(name, status, meta);
  if (task.status === "running") {
    const stop = document.createElement("button");
    stop.className = "secondary";
    stop.textContent = "Cancel";
    stop.addEventListener("click", () => cancelTask(task.task_id));
    row.append(stop);
  }
  return row;
}

function independentRow(agent) {
  const row = document.createElement("div");
  row.className = "task done";
  const name = document.createElement("div");
  name.className = "task-name";
  name.textContent = agent.agent || "agent";
  const talk = document.createElement("button");
  talk.textContent = "Talk to it";
  talk.addEventListener("click", () => attach(agent.run_id));
  const meta = document.createElement("div");
  meta.className = "task-meta";
  meta.textContent = agent.run_id;
  row.append(name, talk, meta);
  return row;
}

async function cancelTask(taskId) {
  const runId = state.parentRunId || state.runId;
  if (!runId) return;
  try {
    await fetch("/api/tasks/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, task_id: taskId }),
    });
  } finally {
    await refreshTasks();
  }
}

function attach(runId) {
  state.parentRunId = state.parentRunId || state.runId;
  state.attachTo = runId;
  $("attached-id").textContent = runId;
  $("attached").classList.remove("hidden");
  $("prompt").focus();
}

function detach() {
  state.attachTo = null;
  $("attached").classList.add("hidden");
}

async function consume(url, body) {
  state.busy = true;
  $("send").disabled = true;
  // Polled while the round is open: a handed-off child settles on its own schedule, and a rail
  // that only refreshed at the end would never show one running.
  if (!state.watching) state.watching = setInterval(refreshTasks, 1200);
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
    if (state.watching) { clearInterval(state.watching); state.watching = null; }
    await refreshLedger();
    await refreshTasks();
  }
}

async function run() {
  const prompt = $("prompt").value.trim();
  if (!prompt || state.busy) return;
  state.model = $("model").value.trim();
  if (state.attachTo) {
    // Same durable queue a steer crosses, aimed at the child's run. This is the whole of what an
    // independent agent's address buys, so the console has to be able to spend it.
    message("user", prompt);
    $("prompt").value = "";
    state.assistant = state.thinking = null;
    setStatus("running", "TALKING TO A CHILD");
    await consume("/api/attach", { run_id: state.attachTo, prompt, model: state.model });
    return;
  }
  const switching = Boolean(state.callId && state.runId);
  const runId = switching ? state.runId : null;
  if (switching) {
    state.assistant = state.thinking = null;
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
  state.assistant = state.thinking = null;
  $("recovery").classList.add("hidden");
  setStatus("running", "RECOVERING FROM STEP");
  await consume("/api/recover", { run_id: state.runId, model: state.model });
}

async function resume(approved) {
  if (!state.runId || !state.pendingId || state.busy) return;
  $("approval").classList.add("hidden");
  state.assistant = state.thinking = null;
  setStatus("running", "RESUMING");
  await consume("/api/resume", { run_id: state.runId, pending_id: state.pendingId, approved, model: state.model });
}

$("send").addEventListener("click", run);
$("prompt").addEventListener("keydown", (event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") run(); });
$("approve").addEventListener("click", () => resume(true));
$("deny").addEventListener("click", () => resume(false));
$("recover").addEventListener("click", recover);
$("detach").addEventListener("click", detach);
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
