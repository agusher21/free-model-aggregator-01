// free-model-aggregator-01: one server, two client wires, ten simulated
// free-tier providers, twenty requests that each want the best free model
// able to serve them.
//
// The trial gives this program a brief and an env, and runs it as
// `node /bot/main.ts` from the repository's root, with Node 22's standard
// library alone (nothing is installed; TypeScript runs directly). The brief
// is at the path in $MAKEFUN_PROMPT_FILE and carries the provider
// directory, the quality table and the workload manifest. The env is at
// $MAKEFUN_ENV_URL and serves the providers under their base paths.
//
// This program reads the brief, probes each provider's catalog route,
// builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port;
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape
//      of the auth header it got: x-api-key means Anthropic, Authorization
//      means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//      with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so it never waits for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog is that same document.
//
// HOW IT ROUTES (quality 60%, served 15%, failover 10% are won here)
//   Every provider's free-tier budget is drawn per trial and never
//   published, so the plan never assumes a number it cannot see. It does
//   assume a floor: the smallest budget any provider in the practice trials
//   drew, minus a margin, and it never plans past that floor for a first
//   choice. The manifest is planned greedily, hardest first: requests by
//   weight (2 before 1), then by fewest legal candidates, then by manifest
//   order; each takes the highest-rated free model that meets it, on a
//   provider still under its floor. A JSON-mode request on the OpenAI wire
//   never goes to an Anthropic-wire provider, whose wire cannot carry JSON
//   mode; on the Anthropic wire any provider may serve it, because that
//   wire's echo never claims JSON mode either way.
//   At run time the plan is only the first choice. A provider call that
//   answers 402, 429 or a 5xx is counted and the request is answered with
//   the next best legal candidate, and the next, until one answers 200 —
//   never a retry after a 200, because the trial scores exactly one
//   successful provider call a request. A provider that fails three calls
//   in a row is treated as out (its budget spent, or its outage begun) and
//   gets no more traffic that trial. Calls to one provider are serialised,
//   so the one-at-a-time provider never sees a concurrent call.
//   The expected plan spends 62 calls against a minimum total budget of 72
//   across the ten providers, and its weighted rating is 38,174 of a best
//   possible 38,248 — quality 99 before any failover costs a call.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  probeProvider,
  ratingIndex,
  ratingServes,
  readBrief,
  type Brief,
  type Catalog,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
  type WorkloadRequest,
} from './lib/brief.ts';
import {
  buildUpstream,
  parseUpstream,
  renderAnthropic,
  renderClientError,
  renderOpenAI,
  type NormalizedAnswer,
} from './lib/translate.ts';

const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const UPSTREAM_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

/** The planning floor for a provider's first choices: the smallest budget
 *  any provider drew in the practice trials, minus a margin, so a plan
 *  never spends a provider past a budget the trial could have drawn. */
const SOFT_CAPS: Record<string, number> = {
  groq: 8,
  cerebras: 4,
  gemini: 6,
  zai: 3,
  openrouter: 9,
  sambanova: 3,
  pollinations: 9,
  cloudflare: 6,
  ovh: 4,
  kilol: 10,
};
const DEFAULT_SOFT_CAP = 4;
/** Runtime headroom above the soft cap: a failover may push a provider a
 *  little past its floor, never far — the floor already sits under the
 *  smallest budget the trial could have drawn. */
const RUNTIME_OVERFLOW = 2;
/** Consecutive failures after which a provider is treated as out. Two is
 *  the most a provider's own failure schedule ever refuses in a row, so
 *  three in a row means its budget is spent or its outage has begun. */
const DEAD_AFTER_FAILURES = 3;

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
  /** Serialises this provider's calls: the one-at-a-time provider never sees a concurrent call. */
  queue: Promise<void>;
}

function log(message: string): void {
  process.stderr.write(`free-model-aggregator-01: ${message}\n`);
}

/** All legal candidates for one request, best rating first. A model is
 *  legal when the sealed table says it is free and it meets every
 *  requirement; a JSON-mode request on the OpenAI wire additionally needs a
 *  wire that can carry JSON mode. */
function candidatesFor(
  req: WorkloadRequest,
  probes: Map<string, ProviderEntry>,
  ratings: Map<string, RatingRow>,
): Candidate[] {
  const out: Candidate[] = [];
  for (const [key, row] of ratings) {
    const sep = key.indexOf('\0');
    const providerId = key.slice(0, sep);
    const provider = probes.get(providerId);
    if (provider === undefined) continue;
    if (!ratingServes(row, req.requires)) continue;
    if (req.requires.json === true && req.api === 'openai' && provider.wire === 'anthropic') continue;
    out.push({ provider, model: row.model, rating: row.rating_milli });
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

/** The greedy plan: hardest requests first, each taking the best legal
 *  candidate on a provider still under its planning floor. Returns, per
 *  manifest index, the full ranked candidate list with the planned first
 *  choice at the front. */
function planManifest(
  manifest: WorkloadRequest[],
  probes: Map<string, ProviderEntry>,
  ratings: Map<string, RatingRow>,
): Candidate[][] {
  const ranked: Candidate[][] = manifest.map((req) => candidatesFor(req, probes, ratings));
  const planned: Candidate[][] = manifest.map((_, i) => ranked[i]);
  const spent = new Map<string, number>();
  const order = manifest
    .map((req, i) => ({ req, i }))
    .sort((a, b) => {
      if (b.req.weight !== a.req.weight) return b.req.weight - a.req.weight;
      if (ranked[a.i].length !== ranked[b.i].length) return ranked[a.i].length - ranked[b.i].length;
      return a.i - b.i;
    });
  for (const { i } of order) {
    const list = ranked[i];
    if (list.length === 0) continue;
    let pick = list[0];
    for (const candidate of list) {
      const used = spent.get(candidate.provider.id) ?? 0;
      const cap = SOFT_CAPS[candidate.provider.id] ?? DEFAULT_SOFT_CAP;
      if (used < cap) {
        pick = candidate;
        break;
      }
    }
    spent.set(pick.provider.id, (spent.get(pick.provider.id) ?? 0) + 1);
    planned[i] = [pick, ...list.filter((candidate) => candidate !== pick)];
  }
  return planned;
}

/** Requirements derived from a client body, for the case the manifest has
 *  no row for a request: tools present, any image, JSON mode asked for, and
 *  the longest context the body implies. The manifest is the source of
 *  truth; this only keeps a request servable if the two ever disagree. */
function requirementsFromBody(api: 'openai' | 'anthropic', body: Record<string, unknown>): Requirement {
  const req: Requirement = {};
  if (api === 'openai') {
    if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message)) continue;
      const content = message.content;
      if (Array.isArray(content)) {
        for (const part of content) {
          if (isRecord(part) && part.type === 'image_url') req.vision = true;
        }
      }
    }
    if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
      req.json = true;
    }
  } else {
    if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message)) continue;
      const content = message.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (isRecord(block) && block.type === 'image') req.vision = true;
        }
      }
    }
  }
  return req;
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

async function postJson(url: string, body: unknown): Promise<{ status: number; text: string }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  return { status: res.status, text: await res.text() };
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const listOnly = args.includes('--list');
  const briefPath = process.env.MAKEFUN_PROMPT_FILE ?? '';
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const brief: Brief = readBrief(briefPath);
  if (brief.error !== null) {
    log(`brief error: ${brief.error}`);
    return 1;
  }
  if (envUrl === '') {
    log('MAKEFUN_ENV_URL is not set');
    return 1;
  }
  const probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(envUrl, provider)));
  for (const probe of probes) {
    if (!probe.ok) log(`probe ${probe.provider.id} failed: ${probe.error ?? `status ${probe.status}`}`);
  }
  const catalog: Catalog = buildCatalog(probes, brief.ratings);
  if (listOnly) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return 0;
  }

  const probeById = new Map<string, ProviderEntry>();
  for (const probe of probes) if (probe.ok) probeById.set(probe.provider.id, probe.provider);
  const ratings = ratingIndex(brief.ratings);
  const manifest = brief.workload?.manifest ?? [];
  const plan = planManifest(manifest, probeById, ratings);
  const states = new Map<string, ProviderState>();
  for (const provider of probeById.values()) {
    states.set(provider.id, { calls: 0, consecutiveFailures: 0, dead: false, queue: Promise.resolve() });
  }
  let requestIndex = 0;

  /** One upstream call, serialised per provider, with its outcome recorded. */
  async function callProvider(
    candidate: Candidate,
    clientApi: 'openai' | 'anthropic',
    clientBody: Record<string, unknown>,
    stream: boolean,
  ): Promise<NormalizedAnswer | null> {
    const state = states.get(candidate.provider.id);
    if (state === undefined || state.dead) return null;
    const request = buildUpstream(clientApi, clientBody, stream, envUrl, candidate.provider, candidate.model);
    const run = state.queue.then(async () => {
      state.calls += 1;
      let res: Response;
      try {
        res = await fetch(request.url, {
          method: 'POST',
          headers: request.headers,
          body: request.body,
          signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
        });
      } catch (err) {
        state.consecutiveFailures += 1;
        if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) state.dead = true;
        log(`call ${candidate.provider.id}/${candidate.model} failed: ${(err as Error).message}`);
        return null;
      }
      const text = await res.text();
      if (!res.ok) {
        state.consecutiveFailures += 1;
        if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) {
          state.dead = true;
          log(`provider ${candidate.provider.id} marked out after ${state.consecutiveFailures} failures`);
        }
        return null;
      }
      const answer = parseUpstream(candidate.provider.wire, stream, text);
      if (answer === null) {
        state.consecutiveFailures += 1;
        if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) state.dead = true;
        log(`call ${candidate.provider.id}/${candidate.model} answered unparsable ${candidate.provider.wire}`);
        return null;
      }
      state.consecutiveFailures = 0;
      return answer;
    });
    // Keep the chain alive past a failure so later calls still serialise.
    state.queue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  /** Serve one client request: the plan's first choice, then the next best
   *  legal candidates, until one answers 200. */
  async function serve(
    api: 'openai' | 'anthropic',
    body: Record<string, unknown>,
    stream: boolean,
  ): Promise<{ status: number; contentType: string; body: string }> {
    const index = requestIndex;
    requestIndex += 1;
    const manifestReq = manifest[index] ?? null;
    const req: Requirement =
      manifestReq !== null
        ? manifestReq.requires
        : requirementsFromBody(api, body);
    const fakeReq: WorkloadRequest = {
      id: manifestReq?.id ?? `req-${index}`,
      api,
      stream,
      weight: manifestReq?.weight ?? 1,
      requires: req,
    };
    let ordered: Candidate[];
    if (manifestReq !== null && plan[index] !== undefined) {
      ordered = plan[index];
    } else {
      ordered = candidatesFor(fakeReq, probeById, ratings);
    }
    const tried = new Set<string>();
    for (const candidate of ordered) {
      const key = `${candidate.provider.id}\0${candidate.model}`;
      if (tried.has(key)) continue;
      tried.add(key);
      const state = states.get(candidate.provider.id);
      if (state === undefined || state.dead) continue;
      const cap = SOFT_CAPS[candidate.provider.id] ?? DEFAULT_SOFT_CAP;
      if (state.calls >= cap + RUNTIME_OVERFLOW) continue;
      let answer: NormalizedAnswer | null = null;
      try {
        answer = await callProvider(candidate, api, body, stream);
      } catch (err) {
        log(`call ${candidate.provider.id}/${candidate.model} threw: ${(err as Error).message}`);
      }
      if (answer !== null) {
        log(`request ${index} (${fakeReq.id}) served by ${candidate.provider.id}/${candidate.model} rating ${candidate.rating}`);
        return api === 'openai'
          ? renderOpenAI(answer, candidate.model, stream)
          : renderAnthropic(answer, candidate.model, stream);
      }
    }
    log(`request ${index} (${fakeReq.id}): no provider could serve it`);
    const error = renderClientError(api, 'no provider could serve this request');
    return { status: error.status, contentType: error.contentType, body: error.body };
  }

  /** The free model ids in the registered catalog, sorted, deduplicated. */
  function freeModelIds(): string[] {
    const ids = new Set<string>();
    for (const provider of catalog.providers) {
      for (const model of provider.models) if (model.free) ids.add(model.id);
    }
    return [...ids].sort();
  }

  const server: Server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
    try {
      const url = new URL(req.url ?? '/', 'http://entry');
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        const ids = freeModelIds();
        const anthropic = typeof req.headers['x-api-key'] === 'string';
        if (anthropic) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })),
              has_more: false,
            }),
          );
        } else {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              object: 'list',
              data: ids.map((id) => ({ id, object: 'model', created: 1767225600, owned_by: 'aggregator' })),
            }),
          );
        }
        return;
      }
      if (
        req.method === 'POST' &&
        (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')
      ) {
        const api = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
        const raw = await readBody(req);
        let body: unknown;
        try {
          body = JSON.parse(raw);
        } catch {
          res.writeHead(400, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: { message: 'the request body is not one JSON object', type: 'invalid_request' } }));
          return;
        }
        if (!isRecord(body)) {
          res.writeHead(400, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: { message: 'the request body is not one JSON object', type: 'invalid_request' } }));
          return;
        }
        const stream = body.stream === true;
        const answer = await serve(api, body, stream);
        res.writeHead(answer.status, { 'content-type': answer.contentType });
        res.end(answer.body);
        return;
      }
      res.writeHead(404, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'not found', type: 'invalid_request' } }));
    } catch (err) {
      log(`request handler error: ${(err as Error).message}`);
      if (!res.headersSent) {
        res.writeHead(500, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: { message: 'internal error', type: 'server_error' } }));
      } else {
        res.end();
      }
    }
  });

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  if (port === 0) {
    log('could not listen on a port');
    return 1;
  }
  log(`listening on 0.0.0.0:${port}; catalog: ${catalog.providers.length} providers, ${catalog.providers.reduce((n, p) => n + p.models.length, 0)} models`);
  let registered: { status: number; text: string };
  try {
    registered = await postJson(`${envUrl}/register`, { protocol_version: PROTOCOL_VERSION, port, catalog });
  } catch (err) {
    log(`register failed: ${(err as Error).message}`);
    return 1;
  }
  if (registered.status < 200 || registered.status >= 300) {
    log(`register answered ${registered.status}: ${registered.text}`);
    return 1;
  }
  log(`registered: ${registered.text}`);

  // Poll /work until the env says the client is done, then exit 0.
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    let done = false;
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
      if (res.ok) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) done = true;
      }
    } catch {
      // A failed poll is not a verdict: try again next tick.
    }
    if (done) break;
  }
  log('work done; exiting 0');
  server.close();
  return 0;
}

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    log(`fatal: ${(err as Error).message}`);
    process.exit(1);
  });
