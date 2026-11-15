// aggregator-02: one server, two client wires, live free models behind every answer.
//
// ROUTING — the brief ranks every free model by rating_milli. For each prompt the server dials
// the best-rated free model that meets the prompt's requirements (JSON mode, tools, vision,
// context length), and on a 402/429/5xx — or a broken call — it fails over to the next best,
// recording each refusal in the x-makefun-upstream-failures trail. A model that refuses is
// cooled down for the rest of the trial, so one rate-limited provider never takes the later
// prompts down with it. The plan is a pure function of the brief and the gateway's answers so
// far (no clock, no unseeded randomness), so a replay that serves the recorded gateway answers
// makes the same calls and leaves the same state export, byte for byte.
//
// CATALOG — exactly the brief's fourteen free ids, in the brief's order, in --list, in the
// register body and in both GET /v1/models answers. Every response names the model really
// dialled: the gateway's own answer spells it without the openrouter/ prefix, and this server
// puts the dialled id back on every answer and every streamed chunk.
//
// WIRES — the client arrives as OpenAI (POST /v1/chat/completions) or Anthropic (POST
// /v1/messages). Every prompt is dialled on the gateway's Chat Completions route, the only one
// these ids have: an Anthropic-shaped request is translated into a Chat Completions body first,
// and the gateway's answer is translated back into the wire the client used, streamed or not,
// with its usage and stop reason whole. JSON mode is never sent to the Anthropic wire — the
// brief's JSON-mode prompts are OpenAI-only, and the translator never adds a response_format.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The register body's protocol version. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The output cap named when a client request carries none: the gateway refuses an uncapped call. */
const FALLBACK_MAX_TOKENS = 1024;
/** The most refusal rows reported on one response's trail header. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which manifest row the prompt now arriving is: the brief lists them in arrival order. */
let manifestCursor = 0;
/**
 * Models that refused this trial, by id. A cooled-down model is never dialled again, so a
 * provider that rate-limits once cannot sink every later prompt. The set is a function of the
 * gateway's answers so far, never of the clock, so a replay rebuilds it identically.
 */
const coolingDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator-02: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root; a base
 * that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function gatewayChatUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function listedModels(): ServedModel[] {
  return freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));
}

/**
 * Free models meeting the requirements, best rating first, id as the deterministic tie-break.
 * Quality is the rating of the model that really serves the prompt over the best rating for a
 * free model meeting its requirements, so the head of this list is the prompt's best serve.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((m) => meetsRequirements(requires, capabilitiesOf(m)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * The ordered plan for one prompt: its ranked candidates with the cooling-down models taken
 * out. The plan is computed per prompt from the brief and the refusals so far, which is the
 * pre-planned assignment: no provider is ever retried after it refuses, and the trial never
 * spends a call on a model that is already cooling down.
 */
function planFor(requires: Requirement): ModelRow[] {
  return rankedCandidates(requires).filter((m) => !coolingDown.has(m.id));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestCursor];
  manifestCursor += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One refusal, as the x-makefun-upstream-failures trail reports it. */
interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

/** The error code of a refused gateway body: its own code, its type, else the HTTP status. */
function codeOf(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
    if (isRecord(parsed) && typeof parsed.code === 'string' && parsed.code !== '') return parsed.code;
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** A Chat Completions body that always names an output cap: the gateway refuses an uncapped call. */
function withOutputCap(body: Json): Json {
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : 0;
  if (cap > 0) return body;
  return { ...body, max_tokens: FALLBACK_MAX_TOKENS };
}

/** A stream request that always asks for its usage chunk: the wire check needs the whole usage. */
function withStreamUsage(body: Json): Json {
  if (body.stream !== true) return body;
  const options = isRecord(body.stream_options) ? body.stream_options : {};
  return { ...body, stream_options: { ...options, include_usage: true } };
}

/** POST one Chat Completions request to the gateway, naming the model this entry dialled. */
async function dialGateway(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(gatewayChatUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(withStreamUsage(withOutputCap({ ...body, model }))),
  });
}

/**
 * One prompt served by the best free model that meets it, failing over down the ranked plan.
 * Returns the dialled model and its gateway response, or null when every candidate refused.
 * Each refusal cools its model down and joins the trail the response reports.
 */
async function serveWithFailover(
  plan: ModelRow[],
  buildBody: (model: string) => Json,
  trail: Refusal[],
): Promise<{ model: string; upstream: Response } | null> {
  for (const candidate of plan) {
    let upstream: Response;
    try {
      upstream = await dialGateway(candidate.id, buildBody(candidate.id));
    } catch (err) {
      coolingDown.add(candidate.id);
      trail.push({ status: 502, code: 'fetch_failed', model: candidate.id });
      log(`dial ${candidate.id} threw: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { model: candidate.id, upstream };
    const text = await upstream.text();
    coolingDown.add(candidate.id);
    trail.push({ status: upstream.status, code: codeOf(upstream.status, text), model: candidate.id });
    log(`dial ${candidate.id} refused HTTP ${upstream.status}: ${text.slice(0, 200)}`);
  }
  return null;
}

/** The trail header's value, at most eight rows, in the order the calls refused. */
function trailHeader(trail: readonly Refusal[]): string {
  return JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJsonResponse(res: ServerResponse, status: number, body: unknown, trail: readonly Refusal[]): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
  };
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
  res.writeHead(status, headers);
  res.end(text);
}

function sendErrorResponse(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') {
    sendJsonResponse(res, status, anthropicError(type, message), []);
  } else {
    sendJsonResponse(res, status, openAIError(message, type, null), []);
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function listModels(req: IncomingMessage, res: ServerResponse): void {
  const list = listedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJsonResponse(res, 200, anthropicModelList(list), []);
  else sendJsonResponse(res, 200, openAIModelList(list), []);
}

/** A refused prompt as a 502 in the client's wire: every candidate refused, the prompt is lost. */
function allRefused(res: ServerResponse, wire: 'openai' | 'anthropic', trail: readonly Refusal[]): void {
  const last = trail[trail.length - 1];
  const detail = last !== undefined ? `the gateway refused every free model (last HTTP ${last.status})` : 'no free model in the brief meets this request';
  if (wire === 'anthropic') sendJsonResponse(res, 502, anthropicError('api_error', detail), trail);
  else sendJsonResponse(res, 502, openAIError(detail, 'upstream_error', null), trail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function serveChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendErrorResponse(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const plan = planFor(requirementsFor(body));
  if (plan.length === 0) return allRefused(res, 'openai', []);
  const trail: Refusal[] = [];
  const served = await serveWithFailover(plan, (model) => ({ ...body, model }), trail);
  if (served === null) return allRefused(res, 'openai', trail);
  const { model, upstream } = served;
  if (body.stream === true) {
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJsonResponse(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function serveMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendErrorResponse(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const plan = planFor(requirementsFor(body));
  if (plan.length === 0) return allRefused(res, 'anthropic', []);
  const trail: Refusal[] = [];
  // JSON mode is never sent to the Anthropic wire: the translator builds a plain Chat
  // Completions body, and the brief's JSON-mode prompts are OpenAI-only by construction.
  const served = await serveWithFailover(plan, (model) => anthropicToOpenAI(body, model), trail);
  if (served === null) return allRefused(res, 'anthropic', trail);
  const { model, upstream } = served;
  if (body.stream === true) {
    const headers: Record<string, string> = {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJsonResponse(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trail);
}

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    listModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveMessages(req, res);
    return;
  }
  sendErrorResponse(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listenOn(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function registerWithEnv(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object
    // and stop before any server starts. A process whose stdout is not flushed loses the tail.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (gatewayChatUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handleRequest(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendErrorResponse(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listenOn(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${listedModels().length} free models`);
  await registerWithEnv(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
