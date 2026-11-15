// The trial brief: three JSON documents in the prompt file, the provider
// probe, and the catalog this entry registers.
//
// A trial hands this program a brief at the path in $MAKEFUN_PROMPT_FILE.
// The prose around the documents does not matter: each document is found by
// its `schema` member.
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): one
//     row a provider, with its id, wire, auth scheme, the environment
//     variable a real deployment would read the key from, the key the
//     simulated service takes, and the base path the entry joins to
//     $MAKEFUN_ENV_URL. A directory entry may give an absolute base_url
//     instead of a path; joinBase takes either.
//   - the quality table (`free-model-aggregator-01/ratings/v1`): each
//     model's public rating in milli-points, what it can do (tools, vision,
//     JSON mode, context length) and whether it is free. This is the sealed
//     truth the scorer grades against, so routing reads the rating, the
//     free flag and the capability columns from here, never from a guess.
//   - the workload manifest (`free-model-aggregator-01/workload/v1`): the
//     requests the client will send, in order, each with its api, whether
//     it streams, its weight and the requirements a model must meet.
//
// The probe then asks each provider's catalog route, in the provider's own
// wire:
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails (a refused connection, a non-2xx, a body that is not
// JSON) leaves its provider out of the catalog and gets no traffic; the
// trial goes on. Everything here is deterministic: no clock, no randomness,
// so `node main.ts --list` and the register body build the same document.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. `base` is the base path (or absolute URL) its routes hang off. */
export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderEntry[];
}

/** One row of the sealed quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as this entry reads it. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** One model in the registered catalog. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface Catalog {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)
        ? { min_context: requires.min_context }
        : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') {
    return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return {
      directory,
      ratings: null,
      workload: null,
      error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document`,
    };
  }
  for (const row of directoryDoc.providers) {
    const provider = providerOf(row);
    if (provider !== null) directory.providers.push(provider);
  }
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  let ratings: RatingsFile | null = null;
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    const rows: RatingRow[] = [];
    for (const row of ratingsDoc.ratings) {
      const parsed = ratingOf(row);
      if (parsed !== null) rows.push(parsed);
    }
    const best: { id: string; rating_milli: number }[] = [];
    if (Array.isArray(ratingsDoc.best)) {
      for (const row of ratingsDoc.best) {
        if (isRecord(row) && typeof row.id === 'string' && typeof row.rating_milli === 'number') {
          best.push({ id: row.id, rating_milli: row.rating_milli });
        }
      }
    }
    ratings = { schema: RATINGS_SCHEMA, ratings: rows, best };
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  let workload: WorkloadFile | null = null;
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    const manifest: WorkloadRequest[] = [];
    for (const row of workloadDoc.manifest) {
      const parsed = workloadRequestOf(row);
      if (parsed !== null) manifest.push(parsed);
    }
    workload = { schema: WORKLOAD_SCHEMA, manifest };
  }
  return { directory, ratings, workload, error: null };
}

/** The ratings table as a lookup: `${provider}\0${model}` -> row. */
export function ratingIndex(ratings: RatingsFile | null): Map<string, RatingRow> {
  const index = new Map<string, RatingRow>();
  if (ratings === null) return index;
  for (const row of ratings.ratings) index.set(`${row.provider}\0${row.model}`, row);
  return index;
}

/** Whether a rated model may serve a request: free, and every requirement met. */
export function ratingServes(row: RatingRow, req: Requirement): boolean {
  if (!row.free) return false;
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (typeof req.min_context === 'number' && (row.context_length === null || row.context_length < req.min_context)) {
    return false;
  }
  return true;
}

/** The catalog route's URL for a provider, in its own wire. */
export function catalogUrl(envUrl: string, provider: ProviderEntry): string {
  const base = joinBase(envUrl, provider.base);
  return provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
}

/** One model row of a catalog listing, in whichever shape the wire returns it. */
function listedModelOf(v: unknown, wire: Wire): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id =
    wire === 'gemini'
      ? stringOf(v.name).replace(/^models\//, '') !== ''
        ? stringOf(v.name).replace(/^models\//, '')
        : stringOf(v.id)
      : stringOf(v.id);
  if (id === '') return null;
  const context =
    wire === 'gemini'
      ? numberOrNull(v.inputTokenLimit) ?? numberOrNull(v.context_length)
      : numberOrNull(v.context_length);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName),
    free: boolOrNull(v.free),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Probe one provider's catalog route. Never throws: a failed probe is a probe with ok=false. */
export async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const url = catalogUrl(envUrl, provider);
  const failed = (status: number, error: string): Probe => ({ provider, ok: false, status, models: [], error });
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'GET',
      headers: { ...authHeaders(provider.auth, provider.key), accept: 'application/json' },
      signal: AbortSignal.timeout(5000),
    });
  } catch (err) {
    return failed(0, `probe ${provider.id} failed: ${(err as Error).message}`);
  }
  if (!res.ok) return failed(res.status, `probe ${provider.id} answered ${res.status}`);
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return failed(res.status, `probe ${provider.id} answered non-JSON`);
  }
  if (!isRecord(body)) return failed(res.status, `probe ${provider.id} answered a non-object`);
  const rows: unknown[] = provider.wire === 'gemini' ? (Array.isArray(body.models) ? body.models : []) : Array.isArray(body.data) ? body.data : [];
  const models: ProbedModel[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    const model = listedModelOf(row, provider.wire);
    if (model === null || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push(model);
  }
  return { provider, ok: true, status: res.status, models, error: null };
}

/** Build the registered catalog from the probes and the sealed quality table.
 *  The free flag and the context length come from the ratings table where it
 *  has a row (the sealed truth the catalog check grades against); a model
 *  the table never heard of keeps what its listing said. Models are sorted
 *  by id and deduplicated, so `--list` and the register body are the same
 *  document byte for byte. */
export function buildCatalog(probes: Probe[], ratings: RatingsFile | null): Catalog {
  const index = ratingIndex(ratings);
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const models: CatalogModel[] = [];
    const seen = new Set<string>();
    for (const listed of probe.models) {
      if (seen.has(listed.id)) continue;
      seen.add(listed.id);
      const row = index.get(`${probe.provider.id}\0${listed.id}`) ?? null;
      models.push({
        id: listed.id,
        free: row !== null ? row.free : listed.free === true,
        context_length: row !== null ? row.context_length : listed.context_length,
      });
    }
    models.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
