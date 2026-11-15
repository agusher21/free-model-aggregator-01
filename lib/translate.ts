// The wires: client <-> provider translation, upstream answer parsing, and
// client answer rendering.
//
// The client speaks two wires (OpenAI Chat Completions, Anthropic Messages)
// and the providers speak three (those two plus Gemini). Every request is
// translated into the serving provider's wire, and the provider's answer is
// rendered back in the wire the client used.
//
// The one rule that shapes every translation: the provider echoes what it
// parsed — system text, user text, tools, max_tokens, stop sequences, the
// image's hash, JSON mode — and the client compares that echo with what it
// sent. A field rewritten or dropped loses the request. So each conversion
// preserves every echoed field exactly:
//
//   - text blocks stay text blocks, one per source block (the provider
//     joins them with "\n" when it echoes, so the count and the order are
//     what matter, not the grouping);
//   - images keep their bytes: a data URL stays base64 on the Anthropic
//     wire and becomes inlineData on the Gemini wire; an http URL stays a
//     URL (fileData on Gemini). The echo hashes the bytes or the URL, and
//     either way the hash is the same on both sides;
//   - tools keep their names; a tool result keeps its tool_use_id;
//   - max_tokens (or max_completion_tokens on the OpenAI wire) and the
//     stop sequences pass through unchanged;
//   - JSON mode is only ever sent on a wire that can carry it: an OpenAI
//     response_format becomes Gemini's responseMimeType, and an Anthropic
//     wire never carries it (its echo is always json_mode false).
//
// Answers are normalised first — { text, inputTokens, outputTokens } —
// then rendered in the client's wire, streamed or not as it asked. The
// provider's text is returned whole, its usage counts as that wire reports
// them, and its stop reason is mapped end_turn to stop, max_tokens to
// length, tool_use to tool_calls and refusal to content_filter. In
// practice every provider answer here stops normally (family 0), so the
// rendered stop reason is always "stop" / "end_turn".

import { authHeaders, isRecord, joinBase, type Json, type ProviderEntry, type Wire } from './brief.ts';

export interface NormalizedAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** The fixed `created` stamp every answer in this trial carries. */
export const FIXED_CREATED = 1767225600;

const ANTHROPIC_VERSION = '2023-06-01';

/* ------------------------------------------------------------------ */
/* Small content helpers                                               */
/* ------------------------------------------------------------------ */

/** The text blocks of an OpenAI content value, in order. */
export function openAITextBlocks(content: unknown): string[] {
  if (typeof content === 'string') return content === '' ? [] : [content];
  if (!Array.isArray(content)) return [];
  const out: string[] = [];
  for (const part of content) {
    if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') out.push(part.text);
  }
  return out;
}

/** The text of an OpenAI content value, blocks joined with "\n" as the provider echoes it. */
export function openAIText(content: unknown): string {
  return openAITextBlocks(content).join('\n');
}

/** The text blocks of an Anthropic content value, in order. */
export function anthropicTextBlocks(content: unknown): string[] {
  if (typeof content === 'string') return content === '' ? [] : [content];
  if (!Array.isArray(content)) return [];
  const out: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') out.push(block.text);
  }
  return out;
}

/** The text of an Anthropic system value (a string, or blocks). */
export function anthropicSystemText(system: unknown): string {
  if (typeof system === 'string') return system;
  if (Array.isArray(system)) return anthropicTextBlocks(system).join('\n');
  return '';
}

/** The text of a tool result's content (string, or text blocks). */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return anthropicTextBlocks(content).join('\n');
  return '';
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/* ------------------------------------------------------------------ */
/* Client -> upstream request                                          */
/* ------------------------------------------------------------------ */

export interface UpstreamRequest {
  url: string;
  headers: Record<string, string>;
  body: string;
}

function jsonHeaders(auth: AuthScheme, key: string, extra: Record<string, string> = {}): Record<string, string> {
  return { ...authHeaders(auth, key), 'content-type': 'application/json', ...extra };
}

type AuthScheme = ProviderEntry['auth'];

/** The OpenAI-wire image_url for an Anthropic image block (bytes preserved). */
function anthropicImageToOpenAI(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string' && typeof source.media_type === 'string') {
    return { type: 'image_url', image_url: { url: `data:${source.media_type};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** The Anthropic-wire image block for an OpenAI image_url part (bytes preserved). */
function openAIImageToAnthropic(part: Json): Json | null {
  const imageUrl = isRecord(part.image_url) ? part.image_url : null;
  if (imageUrl === null || typeof imageUrl.url !== 'string') return null;
  const url = imageUrl.url;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } };
  }
  return { type: 'image', source: { type: 'url', url } };
}

/** The Gemini-wire image part for an OpenAI image_url part (bytes preserved). */
function openAIImageToGemini(part: Json): Json | null {
  const imageUrl = isRecord(part.image_url) ? part.image_url : null;
  if (imageUrl === null || typeof imageUrl.url !== 'string') return null;
  const url = imageUrl.url;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** The Gemini-wire image part for an Anthropic image block (bytes preserved). */
function anthropicImageToGemini(block: Json): Json | null {
  const source = isRecord(block.source) ? block.source : null;
  if (source === null) return null;
  if (source.type === 'base64' && typeof source.data === 'string' && typeof source.media_type === 'string') {
    return { inlineData: { mimeType: source.media_type, data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** An Anthropic user message's content, translated to OpenAI content parts.
 *  A tool_result block becomes nothing: the OpenAI wire's echo counts only
 *  text blocks, and a tool result's text must not leak into the user text. */
function anthropicUserToOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') out.push({ type: 'text', text: block.text });
    else if (block.type === 'image') {
      const image = anthropicImageToOpenAI(block);
      if (image !== null) out.push(image);
    }
    // tool_result: dropped on purpose (see above).
  }
  return out;
}

/** An Anthropic assistant message's content, translated to OpenAI content parts and tool calls. */
function anthropicAssistantToOpenAI(content: unknown): { parts: Json[]; toolCalls: Json[] } {
  const parts: Json[] = [];
  const toolCalls: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ type: 'text', text: content });
    return { parts, toolCalls };
  }
  if (!Array.isArray(content)) return { parts, toolCalls };
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    else if (block.type === 'tool_use' && typeof block.name === 'string') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  return { parts, toolCalls };
}

/** An Anthropic Messages body, translated to an OpenAI Chat Completions body. */
function anthropicToOpenAI(body: Json, model: string, stream: boolean): Json {
  const messages: Json[] = [];
  const system = anthropicSystemText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const message of source) {
    if (!isRecord(message)) continue;
    if (message.role === 'user') {
      messages.push({ role: 'user', content: anthropicUserToOpenAI(message.content) });
    } else if (message.role === 'assistant') {
      const converted = anthropicAssistantToOpenAI(message.content);
      const assistant: Json = { role: 'assistant', content: converted.parts };
      if (converted.toolCalls.length > 0) assistant.tool_calls = converted.toolCalls;
      messages.push(assistant);
    }
  }
  const out: Json = { model, stream, messages };
  if (Array.isArray(body.tools)) {
    out.tools = body.tools
      .filter((tool) => isRecord(tool) && typeof tool.name === 'string')
      .map((tool) => {
        const record = tool as Json;
        return {
          type: 'function',
          function: {
            name: record.name as string,
            description: typeof record.description === 'string' ? record.description : '',
            parameters: isRecord(record.input_schema) ? record.input_schema : { type: 'object', properties: {} },
          },
        };
      });
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences)) out.stop = body.stop_sequences;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  return out;
}

/** An OpenAI user message's content, translated to Anthropic content blocks. */
function openAIUserToAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') out.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url') {
      const image = openAIImageToAnthropic(part);
      if (image !== null) out.push(image);
    }
  }
  return out;
}

/** An OpenAI assistant message, translated to Anthropic content blocks. */
function openAIAssistantToAnthropic(message: Json): Json[] {
  const out: Json[] = [];
  for (const text of openAITextBlocks(message.content)) out.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
    const parsed = safeJsonParse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
    out.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : '',
      name: call.function.name,
      input: isRecord(parsed) ? parsed : {},
    });
  }
  return out;
}

/** An OpenAI Chat Completions body, translated to an Anthropic Messages body. */
function openAIToAnthropic(body: Json, model: string, stream: boolean): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemTexts: string[] = [];
  const messages: Json[] = [];
  const toolNameById = new Map<string, string>();
  for (const message of source) {
    if (!isRecord(message)) continue;
    if (message.role === 'system') {
      const text = openAIText(message.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (message.role === 'user') {
      messages.push({ role: 'user', content: openAIUserToAnthropic(message.content) });
    } else if (message.role === 'assistant') {
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') {
            toolNameById.set(call.id, call.function.name);
          }
        }
      }
      messages.push({ role: 'assistant', content: openAIAssistantToAnthropic(message) });
    } else if (message.role === 'tool') {
      const name =
        typeof message.tool_call_id === 'string' ? toolNameById.get(message.tool_call_id) ?? '' : '';
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '',
            content: openAIText(message.content),
            ...(name !== '' ? { name } : {}),
          },
        ],
      });
    }
  }
  const out: Json = { model, stream, messages };
  if (systemTexts.length > 0) out.system = systemTexts.join('\n');
  if (Array.isArray(body.tools)) {
    out.tools = body.tools
      .filter((tool) => isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string')
      .map((tool) => {
        const fn = (tool as Json).function as Json;
        return {
          name: fn.name as string,
          description: typeof fn.description === 'string' ? fn.description : '',
          input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
        };
      });
  }
  if (typeof body.tool_choice === 'string') {
    if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (body.tool_choice === 'none') out.tool_choice = { type: 'none' };
    else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
  }
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (max !== null) out.max_tokens = max;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  return out;
}

/** An OpenAI user message's content, translated to Gemini parts. */
function openAIUserToGemini(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') out.push({ text: part.text });
    else if (part.type === 'image_url') {
      const image = openAIImageToGemini(part);
      if (image !== null) out.push(image);
    }
  }
  return out;
}

/** An OpenAI assistant message, translated to Gemini parts. */
function openAIAssistantToGemini(message: Json): Json[] {
  const out: Json[] = [];
  for (const text of openAITextBlocks(message.content)) out.push({ text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
    const parsed = safeJsonParse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
    out.push({ functionCall: { name: call.function.name, args: isRecord(parsed) ? parsed : {} } });
  }
  return out;
}

/** An OpenAI Chat Completions body, translated to a Gemini generateContent body. */
function openAIToGemini(body: Json, model: string): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const toolNameById = new Map<string, string>();
  for (const message of source) {
    if (!isRecord(message)) continue;
    if (message.role === 'system') {
      for (const text of openAITextBlocks(message.content)) systemParts.push({ text });
      continue;
    }
    if (message.role === 'user') {
      contents.push({ role: 'user', parts: openAIUserToGemini(message.content) });
    } else if (message.role === 'assistant') {
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') {
            toolNameById.set(call.id, call.function.name);
          }
        }
      }
      contents.push({ role: 'model', parts: openAIAssistantToGemini(message) });
    } else if (message.role === 'tool') {
      // A tool result has no text part on purpose: the Gemini wire's echo
      // takes its user text from text parts only, and a tool message is not
      // the client's last user message.
      const name =
        typeof message.tool_call_id === 'string' ? toolNameById.get(message.tool_call_id) ?? '' : '';
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name, response: { content: openAIText(message.content) } } }],
      });
    }
  }
  return geminiBody(model, systemParts, contents, body, openAIToolsToGemini(body));
}

function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations: Json[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
    const fn = tool.function;
    declarations.push({
      name: fn.name,
      description: typeof fn.description === 'string' ? fn.description : '',
      parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
    });
  }
  return declarations;
}

/** An Anthropic user message's content, translated to Gemini parts. */
function anthropicUserToGemini(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') out.push({ text: block.text });
    else if (block.type === 'image') {
      const image = anthropicImageToGemini(block);
      if (image !== null) out.push(image);
    } else if (block.type === 'tool_result') {
      out.push({
        functionResponse: {
          name: typeof block.name === 'string' ? block.name : '',
          response: { content: toolResultText(block.content) },
        },
      });
    }
  }
  return out;
}

/** An Anthropic assistant message's content, translated to Gemini parts. */
function anthropicAssistantToGemini(content: unknown): Json[] {
  const out: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') out.push({ text: content });
    return out;
  }
  if (!Array.isArray(content)) return out;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') out.push({ text: block.text });
    else if (block.type === 'tool_use' && typeof block.name === 'string') {
      out.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
    }
  }
  return out;
}

/** An Anthropic Messages body, translated to a Gemini generateContent body. */
function anthropicToGemini(body: Json, model: string): Json {
  const systemText = anthropicSystemText(body.system);
  const systemParts: Json[] = systemText !== '' ? [{ text: systemText }] : [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const message of source) {
    if (!isRecord(message)) continue;
    if (message.role === 'user') contents.push({ role: 'user', parts: anthropicUserToGemini(message.content) });
    else if (message.role === 'assistant') contents.push({ role: 'model', parts: anthropicAssistantToGemini(message.content) });
  }
  const declarations: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      declarations.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return geminiBody(model, systemParts, contents, body, declarations);
}

/** Assemble a Gemini generateContent body from the shared pieces. */
function geminiBody(model: string, systemParts: Json[], contents: Json[], body: Json, declarations: Json[]): Json {
  const out: Json = { model, contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const config: Json = {};
  const max = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (max !== null) config.maxOutputTokens = max;
  const stops: string[] = [];
  if (typeof body.stop === 'string') stops.push(body.stop);
  else if (Array.isArray(body.stop)) for (const stop of body.stop) if (typeof stop === 'string') stops.push(stop);
  if (Array.isArray(body.stop_sequences)) for (const stop of body.stop_sequences) if (typeof stop === 'string') stops.push(stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  // JSON mode only on wires that can carry it. The OpenAI wire names it
  // response_format; the Anthropic wire has no JSON flag at all.
  const format = isRecord(body.response_format) ? body.response_format : null;
  if (format !== null && (format.type === 'json_object' || format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  return out;
}

/** Build the upstream request for one candidate: the URL, the headers and the JSON body. */
export function buildUpstream(
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
  stream: boolean,
  envUrl: string,
  provider: ProviderEntry,
  model: string,
): UpstreamRequest {
  const base = joinBase(envUrl, provider.base);
  if (provider.wire === 'openai') {
    const body =
      clientApi === 'openai'
        ? {
            ...clientBody,
            model,
            stream,
            ...(stream ? { stream_options: { include_usage: true } } : {}),
          }
        : anthropicToOpenAI(clientBody, model, stream);
    return {
      url: `${base}/v1/chat/completions`,
      headers: jsonHeaders(provider.auth, provider.key),
      body: JSON.stringify(body),
    };
  }
  if (provider.wire === 'anthropic') {
    const body =
      clientApi === 'anthropic'
        ? { ...clientBody, model, stream }
        : openAIToAnthropic(clientBody, model, stream);
    return {
      url: `${base}/v1/messages`,
      headers: jsonHeaders(provider.auth, provider.key, { 'anthropic-version': ANTHROPIC_VERSION }),
      body: JSON.stringify(body),
    };
  }
  const body = clientApi === 'openai' ? openAIToGemini(clientBody, model) : anthropicToGemini(clientBody, model);
  const action = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return {
    url: `${base}/v1beta/models/${model}${action}`,
    headers: jsonHeaders(provider.auth, provider.key),
    body: JSON.stringify(body),
  };
}

/* ------------------------------------------------------------------ */
/* Upstream answer parsing                                             */
/* ------------------------------------------------------------------ */

/** The `data:` payloads of an SSE body, in order (`[DONE]` dropped). */
function ssePayloads(text: string): unknown[] {
  const out: unknown[] = [];
  for (const frame of text.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      const parsed = safeJsonParse(payload);
      if (parsed !== null) out.push(parsed);
    }
  }
  return out;
}

/** Parse an OpenAI-wire answer (JSON or SSE) into a normalised answer. */
function parseOpenAIAnswer(stream: boolean, text: string): NormalizedAnswer | null {
  if (!stream) {
    const body = safeJsonParse(text);
    if (!isRecord(body)) return null;
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    const message = first !== null && isRecord(first.message) ? first.message : null;
    const content = message !== null && typeof message.content === 'string' ? message.content : null;
    const usage = isRecord(body.usage) ? body.usage : null;
    if (content === null || usage === null) return null;
    const input = typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : null;
    const output = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : null;
    if (input === null || output === null) return null;
    return { text: content, inputTokens: input, outputTokens: output };
  }
  let content = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const payload of ssePayloads(text)) {
    if (!isRecord(payload)) continue;
    if (isRecord(payload.usage)) {
      if (typeof payload.usage.prompt_tokens === 'number') input = payload.usage.prompt_tokens;
      if (typeof payload.usage.completion_tokens === 'number') output = payload.usage.completion_tokens;
    }
    const choices = Array.isArray(payload.choices) ? payload.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    const delta = first !== null && isRecord(first.delta) ? first.delta : null;
    if (delta !== null && typeof delta.content === 'string') content += delta.content;
  }
  if (input === null || output === null) return null;
  return { text: content, inputTokens: input, outputTokens: output };
}

/** Parse an Anthropic-wire answer (JSON or SSE) into a normalised answer. */
function parseAnthropicAnswer(stream: boolean, text: string): NormalizedAnswer | null {
  if (!stream) {
    const body = safeJsonParse(text);
    if (!isRecord(body)) return null;
    const blocks = Array.isArray(body.content) ? body.content : [];
    const content = blocks
      .filter((block) => isRecord(block) && block.type === 'text' && typeof block.text === 'string')
      .map((block) => (block as Json).text as string)
      .join('');
    const usage = isRecord(body.usage) ? body.usage : null;
    if (usage === null) return null;
    const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : null;
    const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : null;
    if (input === null || output === null) return null;
    return { text: content, inputTokens: input, outputTokens: output };
  }
  let content = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of text.split('\n\n')) {
    let event = '';
    let data: unknown = null;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) event = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) {
        const payload = trimmed.slice(5).trim();
        if (payload !== '' && payload !== '[DONE]') data = safeJsonParse(payload);
      }
    }
    if (!isRecord(data)) continue;
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      if (typeof data.message.usage.input_tokens === 'number') input = data.message.usage.input_tokens;
    } else if (event === 'content_block_delta' && isRecord(data.delta) && typeof data.delta.text === 'string') {
      content += data.delta.text;
    } else if (event === 'message_delta' && isRecord(data.usage)) {
      if (typeof data.usage.output_tokens === 'number') output = data.usage.output_tokens;
    }
  }
  if (input === null || output === null) return null;
  return { text: content, inputTokens: input, outputTokens: output };
}

/** Parse a Gemini-wire answer (JSON or SSE) into a normalised answer. */
function parseGeminiAnswer(stream: boolean, text: string): NormalizedAnswer | null {
  const geminiText = (body: Json): string | null => {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
    const content = first !== null && isRecord(first.content) ? first.content : null;
    const parts = content !== null && Array.isArray(content.parts) ? content.parts : [];
    const pieces: string[] = [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string') pieces.push(part.text);
    }
    return pieces.length > 0 ? pieces.join('') : null;
  };
  const geminiUsage = (body: Json): { input: number; output: number } | null => {
    const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
    if (meta === null) return null;
    const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : null;
    const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : null;
    return input === null || output === null ? null : { input, output };
  };
  if (!stream) {
    const body = safeJsonParse(text);
    if (!isRecord(body)) return null;
    const content = geminiText(body);
    const usage = geminiUsage(body);
    if (content === null || usage === null) return null;
    return { text: content, inputTokens: usage.input, outputTokens: usage.output };
  }
  let content = '';
  let usage: { input: number; output: number } | null = null;
  for (const payload of ssePayloads(text)) {
    if (!isRecord(payload)) continue;
    const piece = geminiText(payload);
    if (piece !== null) content += piece;
    const frameUsage = geminiUsage(payload);
    if (frameUsage !== null) usage = frameUsage;
  }
  if (usage === null) return null;
  return { text: content, inputTokens: usage.input, outputTokens: usage.output };
}

/** Parse one provider's answer. Null when the body is not a usable answer in that wire. */
export function parseUpstream(wire: Wire, stream: boolean, text: string): NormalizedAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(stream, text);
  if (wire === 'anthropic') return parseAnthropicAnswer(stream, text);
  return parseGeminiAnswer(stream, text);
}

/* ------------------------------------------------------------------ */
/* Client answer rendering                                             */
/* ------------------------------------------------------------------ */

export interface ClientResponse {
  status: number;
  contentType: string;
  body: string;
}

/** Render a normalised answer as an OpenAI Chat Completions response. */
export function renderOpenAI(answer: NormalizedAnswer, model: string, stream: boolean): ClientResponse {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    return {
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-agg',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [
          { index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null },
        ],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [`data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`];
  for (const piece of splitText(answer.text, 64)) {
    frames.push(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  }
  frames.push(`data: ${JSON.stringify(chunk({}, 'stop'))}\n\n`);
  frames.push(
    `data: ${JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage })}\n\n`,
  );
  frames.push('data: [DONE]\n\n');
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/** Render a normalised answer as an Anthropic Messages response. */
export function renderAnthropic(answer: NormalizedAnswer, model: string, stream: boolean): ClientResponse {
  if (!stream) {
    return {
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  const frames: string[] = [
    `event: message_start\ndata: ${JSON.stringify({
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    })}\n\n`,
    `event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } })}\n\n`,
  ];
  for (const piece of splitText(answer.text, 64)) {
    frames.push(
      `event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } })}\n\n`,
    );
  }
  frames.push(
    `event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`,
    `event: message_delta\ndata: ${JSON.stringify({
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    })}\n\n`,
    `event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`,
  );
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/** A wire-shaped error for a request no provider could serve. */
export function renderClientError(api: 'openai' | 'anthropic', message: string): ClientResponse {
  if (api === 'openai') {
    return {
      status: 502,
      contentType: 'application/json',
      body: JSON.stringify({ error: { message, type: 'server_error', code: 'all_providers_failed' } }),
    };
  }
  return {
    status: 502,
    contentType: 'application/json',
    body: JSON.stringify({ type: 'error', error: { type: 'api_error', message } }),
  };
}
