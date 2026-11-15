// main.ts — free-model-aggregator-02
//
// One server that serves the OpenAI Chat Completions wire and the Anthropic
// Messages wire, backed by the fourteen free OpenRouter models. Every prompt
// is answered by a real completion dialled through the grader's gateway on
// Chat Completions; the entry picks the best-rated free model for the prompt,
// fails over to the next-best on a refusal, and reports the refusal trail on
// the x-makefun-upstream-failures header.
//
// Run it:
//   node main.ts --list     print the catalog and exit 0
//   node main.ts            serve on 0.0.0.0:<a free port>, register with the
//                           env at $MAKEFUN_ENV_URL, poll GET /work and exit
//                           0 when the trial is done.
//
// The gateway is $OPENAI_BASE_URL (the grader's forwarder). The trial's key
// is already in OPENAI_API_KEY; the gateway swaps it for the entrant's own
// OpenRouter key, which never enters this process. Only the OpenAI route of
// the gateway is used: every free model is dialled on Chat Completions,
// never on Messages, and an Anthropic-shaped request is translated into a
// Chat Completions body first, then the answer is translated back.

import { createServer } from "node:http";
import type { AddressInfo, IncomingMessage, ServerResponse } from "node:http";

// ---------------------------------------------------------------------------
// The free catalog
// ---------------------------------------------------------------------------

interface FreeModel {
  id: string;
  rating: number;
  context: number;
  tools: boolean;
  json: boolean;
}

// The house's sealed free catalog, in rating order (highest first). Ratings
// are the house's quality ratings, 0-2000. `json` says the model takes a
// JSON-mode response_format; every free model takes tools.
const FREE_MODELS: FreeModel[] = [
  { id: "openrouter/thinkingmachines/inkling-2-small:free", rating: 1475, context: 1048576, tools: true, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-super-vl:free", rating: 1303, context: 1000000, tools: true, json: true },
  { id: "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", rating: 1285, context: 1000000, tools: true, json: false },
  { id: "openrouter/nvidia/nemotron-3.5-lightning:free", rating: 1221, context: 1000000, tools: true, json: false },
  { id: "openrouter/google/gemma-4-31b-it:free", rating: 1210, context: 262144, tools: true, json: true },
  { id: "openrouter/cohere/north-mini-code:free", rating: 1200, context: 256000, tools: true, json: false },
  { id: "openrouter/baidu/ernie-5.5-flash-preview:free", rating: 1150, context: 1024000, tools: true, json: true },
  { id: "openrouter/dots-studio/dots-3-note-preview:free", rating: 1150, context: 512000, tools: true, json: true },
  { id: "openrouter/google/gemma-4-26b-a4b-it:free", rating: 1150, context: 262144, tools: true, json: true },
  { id: "openrouter/inclusionai/ling-3.2-flash:free", rating: 1150, context: 1048576, tools: true, json: true },
  { id: "openrouter/liquid/lfm-3-1.7b:free", rating: 1150, context: 262144, tools: true, json: true },
  { id: "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", rating: 1150, context: 256000, tools: true, json: false },
  { id: "openrouter/particle-ai/particle-1:free", rating: 1150, context: 1048576, tools: true, json: true },
  { id: "openrouter/recursive/flint-v1.0:free", rating: 1150, context: 262144, tools: true, json: true },
];

const PROVIDER_ID = "openrouter";
const PROVIDER_BASE_URL = "https://openrouter.ai/api/v1";
const PROVIDER_KEY_ENV = "OPENROUTER_API_KEY";

// The catalog document: the same object --list prints and the entry
// registers. It names exactly the fourteen free models, once each.
function catalogDocument(): Record<string, unknown> {
  return {
    providers: [
      {
        id: PROVIDER_ID,
        base_url: PROVIDER_BASE_URL,
        key_env: PROVIDER_KEY_ENV,
        wire: "openai",
        models: FREE_MODELS.map((m) => ({ id: m.id, free: true, context_length: m.context })),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Routing: rating-ranked, capability-filtered, deterministic
// ---------------------------------------------------------------------------

interface PromptNeeds {
  json: boolean;
  tools: boolean;
}

// Order the models for a prompt, best first. A JSON-mode prompt may only use
// a model that takes JSON mode; a tools prompt may only use one that takes
// tools. The catalog is already in rating order, so filtering keeps the
// ranking. No randomness: the same prompt always gets the same order.
function routeOrder(needs: PromptNeeds): FreeModel[] {
  return FREE_MODELS.filter((m) => (!needs.json || m.json) && (!needs.tools || m.tools));
}

// ---------------------------------------------------------------------------
// Cooldown: a model that refuses sits out the next few dial attempts
// ---------------------------------------------------------------------------

// How many later gateway attempts a refused model skips. The counter ticks
// once per gateway attempt, never on the wall clock, so a trial replays
// exactly: the route is a function of the requests and the gateway answers
// seen so far, and nothing else.
const COOLDOWN_ATTEMPTS = 6;
const cooled = new Map<string, number>();

function tickCooldowns(): void {
  for (const [id, left] of Array.from(cooled.entries())) {
    if (left <= 1) cooled.delete(id);
    else cooled.set(id, left - 1);
  }
}

function isCooled(id: string): boolean {
  return cooled.has(id);
}

function coolModel(id: string): void {
  cooled.set(id, COOLDOWN_ATTEMPTS);
}

// ---------------------------------------------------------------------------
// The gateway
// ---------------------------------------------------------------------------

function gatewayBase(): string {
  return process.env.OPENAI_BASE_URL ?? PROVIDER_BASE_URL;
}

function gatewayKey(): string {
  return process.env.OPENAI_API_KEY ?? "";
}

interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

interface Completion {
  content: string;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  finishReason: string;
  toolCalls: Array<Record<string, unknown>>;
}

// Read the error code out of a refused gateway answer. A refusal that is not
// JSON, or holds no code, is reported as an upstream error.
async function errorCodeOf(res: Response): Promise<string> {
  try {
    const text = await res.text();
    const value = JSON.parse(text) as Record<string, unknown>;
    const err = value.error as Record<string, unknown> | undefined;
    if (err && typeof err.code === "string" && err.code !== "") return err.code;
    if (err && typeof err.type === "string" && err.type !== "") return err.type;
    if (typeof value.code === "string" && value.code !== "") return value.code;
  } catch {
    // Not JSON: fall through to the generic code.
  }
  return "upstream_error";
}

// Pull the text out of a Chat Completions message content field, which may
// be a plain string or an array of content parts.
function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p): p is Record<string, unknown> => typeof p === "object" && p !== null)
      .filter((p) => p.type === "text" && typeof p.text === "string")
      .map((p) => p.text as string)
      .join("");
  }
  return "";
}

// A usage object the wire checks accept. The gateway always sends usage; the
// fallback only keeps the shape valid if an answer somehow carries none.
function usageOrFallback(usage: unknown, content: string): Completion["usage"] {
  if (usage && typeof usage === "object") {
    const u = usage as Record<string, unknown>;
    const input = typeof u.prompt_tokens === "number" ? u.prompt_tokens : 0;
    const output = typeof u.completion_tokens === "number" ? u.completion_tokens : 0;
    if (output >= 1) return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
  }
  const guessed = Math.max(1, Math.ceil(content.length / 4));
  return { prompt_tokens: 0, completion_tokens: guessed, total_tokens: guessed };
}

// Parse a non-streaming Chat Completions answer into a Completion.
function parseChatAnswer(data: Record<string, unknown>): Completion {
  const choices = Array.isArray(data.choices) ? data.choices : [];
  const first = (choices[0] ?? {}) as Record<string, unknown>;
  const message = (first.message ?? {}) as Record<string, unknown>;
  const content = textOfContent(message.content);
  const toolCalls = Array.isArray(message.tool_calls)
    ? (message.tool_calls as Array<Record<string, unknown>>)
    : [];
  return {
    content,
    usage: usageOrFallback(data.usage, content),
    finishReason: typeof first.finish_reason === "string" ? first.finish_reason : "stop",
    toolCalls,
  };
}

// Parse a streaming Chat Completions answer (server-sent events) into one
// Completion, collecting every content delta, the finish reason and the
// usage chunk. The gateway's own model field is ignored: the caller puts the
// dialled id back on the answer.
async function parseChatStream(res: Response): Promise<Completion> {
  const text = await res.text();
  let content = "";
  let finishReason = "stop";
  let usage: unknown = null;
  const toolCalls: Array<Record<string, unknown>> = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === "[DONE]" || payload === "") continue;
    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (chunk.usage !== undefined) usage = chunk.usage;
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const first = (choices[0] ?? {}) as Record<string, unknown>;
    const delta = (first.delta ?? {}) as Record<string, unknown>;
    if (typeof delta.content === "string") content += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls as Array<Record<string, unknown>>) toolCalls.push(call);
    }
    if (typeof first.finish_reason === "string") finishReason = first.finish_reason;
  }
  return { content, usage: usageOrFallback(usage, content), finishReason, toolCalls };
}

interface Attempt {
  ok: boolean;
  failover: boolean;
  status: number;
  code: string;
  completion: Completion | null;
}

// Dial one model on the gateway's Chat Completions route. A refusal is
// returned, never thrown, so the caller can fail over from it.
async function tryModel(
  model: FreeModel,
  chatBody: Record<string, unknown>,
  stream: boolean,
): Promise<Attempt> {
  const body: Record<string, unknown> = { ...chatBody, model: model.id, stream };
  if (stream) {
    body.stream_options = { ...(body.stream_options as Record<string, unknown> | undefined), include_usage: true };
  }
  let res: Response;
  try {
    res = await fetch(`${gatewayBase()}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${gatewayKey()}` },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(25000),
    });
  } catch {
    return { ok: false, failover: true, status: 502, code: "network_error", completion: null };
  }
  if (!res.ok) {
    const code = await errorCodeOf(res);
    const failover = res.status === 402 || res.status === 429 || res.status >= 500;
    return { ok: false, failover, status: res.status, code, completion: null };
  }
  const completion = stream ? await parseChatStream(res) : parseChatAnswer((await res.json()) as Record<string, unknown>);
  return { ok: true, failover: false, status: 200, code: "", completion };
}

interface Served {
  ok: boolean;
  status: number;
  code: string;
  model: string | null;
  completion: Completion | null;
  failures: UpstreamFailure[];
}

// Serve one prompt: walk the rating order, failing over on a refusal, and
// cool each refused model down. A status that failover cannot fix (a 400, a
// 401) ends the walk at once, because no other model would answer it either.
async function servePrompt(
  chatBody: Record<string, unknown>,
  stream: boolean,
  order: FreeModel[],
): Promise<Served> {
  const failures: UpstreamFailure[] = [];
  for (const model of order) {
    if (isCooled(model.id)) continue;
    tickCooldowns();
    const attempt = await tryModel(model, chatBody, stream);
    if (attempt.ok && attempt.completion) {
      return { ok: true, status: 200, code: "", model: model.id, completion: attempt.completion, failures };
    }
    failures.push({ status: attempt.status, code: attempt.code, model: model.id });
    if (!attempt.failover) {
      return { ok: false, status: attempt.status, code: attempt.code, model: null, completion: null, failures };
    }
    coolModel(model.id);
  }
  return { ok: false, status: 502, code: "all_models_failed", model: null, completion: null, failures };
}

// ---------------------------------------------------------------------------
// Request translation
// ---------------------------------------------------------------------------

// Build the Chat Completions body for an OpenAI-wire request. Messages,
// tools, temperature and response_format carry through unchanged; a missing
// max_tokens defaults to 1024.
function openAiChatBody(body: Record<string, unknown>, stream: boolean): Record<string, unknown> {
  const out: Record<string, unknown> = { messages: body.messages };
  out.max_tokens = typeof body.max_tokens === "number" ? body.max_tokens : 1024;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (body.response_format !== undefined) out.response_format = body.response_format;
  if (stream) out.stream = true;
  return out;
}

// Translate an Anthropic Messages request into a Chat Completions body. The
// system prompt becomes a system message, text blocks flatten to strings,
// and Anthropic tools become OpenAI function tools. JSON mode exists only
// on the OpenAI wire, so no response_format is ever set here.
function anthropicChatBody(body: Record<string, unknown>, stream: boolean): Record<string, unknown> {
  const messages: Array<Record<string, unknown>> = [];
  if (body.system !== undefined) {
    const system = textOfContent(body.system);
    if (system !== "") messages.push({ role: "system", content: system });
  }
  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const m of incoming as Array<Record<string, unknown>>) {
    const role = typeof m.role === "string" ? m.role : "user";
    messages.push({ role, content: textOfContent(m.content) });
  }
  const out: Record<string, unknown> = { messages };
  out.max_tokens = typeof body.max_tokens === "number" ? body.max_tokens : 1024;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = (body.tools as Array<Record<string, unknown>>).map((t) => {
      if (t.type === "function") return t;
      return {
        type: "function",
        function: {
          name: t.name,
          description: t.description,
          parameters: t.input_schema ?? { type: "object", properties: {} },
        },
      };
    });
  }
  if (stream) out.stream = true;
  return out;
}

// Map a Chat Completions finish reason onto the Anthropic stop reason.
function anthropicStop(finishReason: string): string {
  switch (finishReason) {
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
    default:
      return "end_turn";
  }
}

// The Anthropic content blocks for an answer: its text, then one tool_use
// block per tool call the model made.
function anthropicBlocks(completion: Completion): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  blocks.push({ type: "text", text: completion.content });
  for (const call of completion.toolCalls) {
    const fn = (call.function ?? {}) as Record<string, unknown>;
    let input: Record<string, unknown> = {};
    if (typeof fn.arguments === "string") {
      try {
        input = JSON.parse(fn.arguments) as Record<string, unknown>;
      } catch {
        input = {};
      }
    }
    blocks.push({
      type: "tool_use",
      id: typeof call.id === "string" ? call.id : "toolu_1",
      name: typeof fn.name === "string" ? fn.name : "",
      input,
    });
  }
  return blocks;
}

let messageCounter = 0;

function nextMessageId(): string {
  messageCounter += 1;
  return `msg_agg_${messageCounter}`;
}

// ---------------------------------------------------------------------------
// Response rendering
// ---------------------------------------------------------------------------

function trailHeaders(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { "x-makefun-upstream-failures": JSON.stringify(failures.slice(0, 8)) };
}

function sendJson(res: ServerResponse, status: number, value: unknown, failures: UpstreamFailure[]): void {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(body)),
    ...trailHeaders(failures),
  });
  res.end(body);
}

function sendError(res: ServerResponse, served: Served): void {
  sendJson(
    res,
    served.status,
    { error: { message: `the gateway refused every model (${served.code})`, type: "upstream_error", code: served.code } },
    served.failures,
  );
}

// Render a served completion on the OpenAI wire, streamed or not. The model
// on every answer and every chunk is the id actually dialled.
function renderOpenAi(res: ServerResponse, served: Served, stream: boolean): void {
  const completion = served.completion as Completion;
  const model = served.model as string;
  if (!stream) {
    const message: Record<string, unknown> = { role: "assistant", content: completion.content };
    if (completion.toolCalls.length > 0) message.tool_calls = completion.toolCalls;
    sendJson(
      res,
      200,
      {
        id: "chatcmpl_agg",
        object: "chat.completion",
        created: 0,
        model,
        choices: [{ index: 0, message, finish_reason: completion.finishReason }],
        usage: completion.usage,
      },
      served.failures,
    );
    return;
  }
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...trailHeaders(served.failures),
  });
  const first = {
    id: "chatcmpl_agg",
    object: "chat.completion.chunk",
    created: 0,
    model,
    choices: [{ index: 0, delta: { role: "assistant", content: completion.content }, finish_reason: null }],
  };
  const last = {
    id: "chatcmpl_agg",
    object: "chat.completion.chunk",
    created: 0,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: completion.finishReason }],
    usage: completion.usage,
  };
  res.write(`data: ${JSON.stringify(first)}\n\n`);
  res.write(`data: ${JSON.stringify(last)}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
}

// Render a served completion on the Anthropic wire, streamed or not.
function renderAnthropic(res: ServerResponse, served: Served, stream: boolean): void {
  const completion = served.completion as Completion;
  const model = served.model as string;
  const stop = anthropicStop(completion.finishReason);
  if (!stream) {
    sendJson(
      res,
      200,
      {
        id: nextMessageId(),
        type: "message",
        role: "assistant",
        content: anthropicBlocks(completion),
        model,
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: completion.usage.prompt_tokens, output_tokens: completion.usage.completion_tokens },
      },
      served.failures,
    );
    return;
  }
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...trailHeaders(served.failures),
  });
  const writeEvent = (event: string, data: unknown): void => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };
  writeEvent("message_start", {
    type: "message_start",
    message: {
      id: nextMessageId(),
      type: "message",
      role: "assistant",
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: completion.usage.prompt_tokens, output_tokens: 0 },
    },
  });
  let index = 0;
  writeEvent("content_block_start", { type: "content_block_start", index, content_block: { type: "text", text: "" } });
  writeEvent("content_block_delta", { type: "content_block_delta", index, delta: { type: "text_delta", text: completion.content } });
  writeEvent("content_block_stop", { type: "content_block_stop", index });
  index += 1;
  for (const call of completion.toolCalls) {
    const fn = (call.function ?? {}) as Record<string, unknown>;
    let input: Record<string, unknown> = {};
    if (typeof fn.arguments === "string") {
      try {
        input = JSON.parse(fn.arguments) as Record<string, unknown>;
      } catch {
        input = {};
      }
    }
    writeEvent("content_block_start", {
      type: "content_block_start",
      index,
      content_block: { type: "tool_use", id: typeof call.id === "string" ? call.id : "toolu_1", name: fn.name ?? "", input },
    });
    writeEvent("content_block_delta", {
      type: "content_block_delta",
      index,
      delta: { type: "input_json_delta", partial_json: JSON.stringify(input) },
    });
    writeEvent("content_block_stop", { type: "content_block_stop", index });
    index += 1;
  }
  writeEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stop },
    usage: { output_tokens: completion.usage.completion_tokens },
  });
  writeEvent("message_stop", { type: "message_stop" });
  res.end();
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > 2_000_000) throw new Error("request body too large");
    chunks.push(buffer);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  if (text.trim() === "") return {};
  return JSON.parse(text) as Record<string, unknown>;
}

// GET /v1/models: the OpenAI shape for an Authorization: Bearer header, the
// Anthropic shape for x-api-key. Both list exactly the fourteen free ids.
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const anthropic = req.headers["x-api-key"] !== undefined;
  if (anthropic) {
    sendJson(
      res,
      200,
      {
        data: FREE_MODELS.map((m) => ({ id: m.id, type: "model", display_name: m.id, created_at: "2026-11-15T00:00:00Z" })),
        has_more: false,
        first_id: FREE_MODELS[0].id,
        last_id: FREE_MODELS[FREE_MODELS.length - 1].id,
      },
      [],
    );
    return;
  }
  sendJson(
    res,
    200,
    {
      object: "list",
      data: FREE_MODELS.map((m) => ({ id: m.id, object: "model", created: 0, owned_by: "openrouter" })),
    },
    [],
  );
}

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: Record<string, unknown>;
  try {
    body = await readJsonBody(req);
  } catch {
    sendJson(res, 400, { error: { message: "the request body is not JSON", type: "invalid_request_error", code: "invalid_request" } }, []);
    return;
  }
  const stream = body.stream === true;
  const jsonMode =
    (body.response_format as Record<string, unknown> | undefined)?.type === "json_object" ||
    (body.response_format as Record<string, unknown> | undefined)?.type === "json_schema";
  const hasTools = Array.isArray(body.tools) && (body.tools as unknown[]).length > 0;
  const order = routeOrder({ json: jsonMode, tools: hasTools });
  const served = await servePrompt(openAiChatBody(body, stream), stream, order);
  if (!served.ok) {
    sendError(res, served);
    return;
  }
  console.error(`chat ${stream ? "stream" : "json"} -> ${served.model}${served.failures.length > 0 ? ` (failed over ${served.failures.length}x)` : ""}`);
  renderOpenAi(res, served, stream);
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  let body: Record<string, unknown>;
  try {
    body = await readJsonBody(req);
  } catch {
    sendJson(res, 400, { error: { message: "the request body is not JSON", type: "invalid_request_error", code: "invalid_request" } }, []);
    return;
  }
  const stream = body.stream === true;
  const hasTools = Array.isArray(body.tools) && (body.tools as unknown[]).length > 0;
  const order = routeOrder({ json: false, tools: hasTools });
  const served = await servePrompt(anthropicChatBody(body, stream), stream, order);
  if (!served.ok) {
    sendError(res, served);
    return;
  }
  console.error(`messages ${stream ? "stream" : "json"} -> ${served.model}${served.failures.length > 0 ? ` (failed over ${served.failures.length}x)` : ""}`);
  renderAnthropic(res, served, stream);
}

// ---------------------------------------------------------------------------
// The trial loop: register, then poll /work until the env says done
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runTrial(envUrl: string, port: number): Promise<void> {
  try {
    const reg = await fetch(`${envUrl}/register`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog: catalogDocument() }),
    });
    const answer = (await reg.json()) as Record<string, unknown>;
    console.error(`registered: ok=${answer.ok} requests=${answer.requests}`);
  } catch (e) {
    console.error(`register failed: ${e}`);
  }
  for (;;) {
    await sleep(500);
    let work: Record<string, unknown> | null = null;
    try {
      const res = await fetch(`${envUrl}/work`);
      work = (await res.json()) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (work && work.done === true) return;
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  if (process.argv.includes("--list")) {
    console.log(JSON.stringify(catalogDocument()));
    return;
  }
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://localhost");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      handleModels(req, res);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      void handleChat(req, res);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      void handleMessages(req, res);
      return;
    }
    sendJson(res, 404, { error: { message: "not found", type: "not_found", code: "not_found" } }, []);
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const port = (server.address() as AddressInfo).port;
  console.error(`aggregator-02: listening on 0.0.0.0:${port} with ${FREE_MODELS.length} free models`);
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (envUrl) {
    await runTrial(envUrl, port);
    server.close();
  }
}

void main();
