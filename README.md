# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/agusher21/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/agusher21/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator: one server, two client wires, ten simulated free-tier providers. It reads the trial brief, probes each provider's catalog in its own wire, and routes every request to the highest-rated free model that meets it — under a per-provider planning floor, since each provider's free-tier budget is drawn per trial and never published. A provider that refuses (402, 429, 5xx) is counted and the request is answered with the next best legal model; a provider that fails three calls in a row is treated as out. All three provider wires are spoken (OpenAI, Anthropic, Gemini) and both client wires are served, streamed or not, with the provider's text whole and its usage counts as that wire reports them.

## Run it

```sh
node main.ts
```

The trial runs it as `node /bot/main.ts` with Node 22's standard library alone, the brief at `$MAKEFUN_PROMPT_FILE` and the env at `$MAKEFUN_ENV_URL`. `node main.ts --list` prints the registered catalog as one JSON document and exits 0 without starting the server.

---

<sub>Entered on [make.fun](https://make.fun/?ref=w513bxaz&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
