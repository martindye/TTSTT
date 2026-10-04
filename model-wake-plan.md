# Plan: DSH starts the local Qwen model on demand (per dropdown)

Status: PLAN ONLY — nothing implemented, no changes made.
Date: 25/09/2026. DSH checkout version: 0.1.2-alpha.3 (this checkout).

## Requirement (Martin's words, 25/09/2026)

- Start **only** the DeepSeek harness. It should start the model itself.
- The model started must match the **dropdown selection** in the GUI.
- Trigger: the **first text entry** (no keypress/alias mechanics; no typing
  "qwen" in a command window any more).
- If the dropdown is changed and a new entry is typed in and sent, **that** is
  when the harness checks whether the dropdown changed and reloads the model
  with the matching context.
- **No auto-stop** (the model keeps running once started, until the machine
  or the server is stopped).
- **No changes to DSH code** — the hook must live in the right place
  (DSH's own extension point).

## Facts found (all verified on this machine)

1. **The dropdown** is the provider `qwen3827bq6` in
   `C:\Users\press\.dsh\settings.yaml` — one local server
   (`http://127.0.0.1:8080/v1`) offered as three entries:

   | dropdown entry            | contextWindow | maxTokens |
   |---------------------------|---------------|-----------|
   | qwen3.8-27B-Q6            | 350000        | 300000    |
   | qwen3.8-27B-Q6-small      | 225000       | 220000    |
   | qwen3.8-27B-Q6-smaller    | 75000        | 70000     |

   Same weights; only the context (and therefore KV/VRAM) differs.
   Default agent model: `qwen3.8-27B-Q6-small`.
2. **The context size is fixed at server start** (`-c` flag). The running
   server is `-c 225000` (verified via its command line; `/props` reports
   `n_ctx: 225024` — llama rounds up). Changing the dropdown alone does
   nothing to the server; that is today's manual pain.
3. **The start script already supports the size as a parameter**:
   `C:\Users\press\OneDrive\Projects\TurboQuantLlamaCppSetup\scripts\`
   `start-qwen3.8-27b-q6_0-turboquant-vision.ps1 -ContextSize N`
   (defaults: 225000 = "1GB spare for speech", 75000 = "2GB spare",
   350000 = "just fits in 32GB" with no speech headroom).
   Companion `stop-server.ps1` exists too.
4. **DSH's sanctioned hook point is the plugin system** (the same seam
   `dsh-searxng` uses): a plugin package with a `cordis.patch.yml` row
   installed into the profile; host-side code runs as plain Node inside the
   DSH host (`apply(ctx, config)`), so it can spawn processes and run an
   HTTP server with zero DSH source changes.
5. **The LLM call path**: agent (host, Node) → pi-ai `openai-completions`
   provider → `baseURL` from settings.yaml (currently 8080/v1) → the
   request body carries the **dropdown's model id** (`"model": ...`) and
   `maxTokens` per entry. So the dropdown choice already travels in every
   LLM request — a front-end for that URL sees exactly what we need.
6. llama-server exposes `/props` (has `n_ctx`) and `/v1/models` — enough to
   (a) know a server is up and (b) verify which context it was started with,
   even one started by hand.

## Design — one small DSH plugin, zero DSH code changes

New local plugin package (separate small project, **not** inside the
deepseek-harness checkout), e.g. `dsh-local-model-manager`:

1. **Host side (`apply`)**: start a tiny HTTP proxy on `127.0.0.1:8081`
   (in-process Node `http`; no new service to manage — it dies with the
   harness, which is what we want).
2. **Provider config (the only config edit)**: in `settings.yaml`,
   `baseURL: http://127.0.0.1:8081/v1` (was `:8080`). Everything else —
   model ids, contextWindows, maxTokens — stays exactly as it is.
3. **Proxy behaviour** (all of it in the plugin):
   - On any `/v1/*` request, read the model id from the body (or path).
   - Map model id → context size by **reading `settings.yaml` itself**
     (the `contextWindow` of that entry) — one source of truth, no
     duplicated table in the plugin.
   - Ensure the server:
     - **not running** (probe `:8080/v1/models`) → launch
       `start-qwen3.8-27b-q6_0-turboquant-vision.ps1 -ContextSize <size>`
       (powershell, detached child), poll `:8080/v1/models` until ready
       (generous timeout, ~5 min; startup log to
       `~/.dsh/logs/local-model.log`).
     - **running, but `/props` n_ctx ≠ requested** (allow llama's round-up:
       match when `requested <= n_ctx < requested + 4096`) → stop it
       (child PID when we started it; `stop-server.ps1`/port-lookup for a
       hand-started one) and start with the new size.
     - **running and matching** → forward.
   - **Forwarding**: byte/stream passthrough of `/v1/*`, including
     SSE streaming. While waiting for a cold start, open the SSE response
     immediately and send SSE comment keep-alives (`: ...` lines, legal in
     the stream) so the OpenAI client never times out; relay the real
     stream once the server is ready.
   - Log every start/stop/forward decision to `~/.dsh/logs/local-model.log`.
4. **Install**: `dsh plugin add <local path>` + the profile row in
   `~/.dsh/profiles/web/cordis.patch.yml` (the same file already carries
   the searxng row). No DSH files touched.

## Resulting behaviour

- Start DSH (`pnpm dsh web` / start.bat). Model is NOT running.
- Type first message, send → first LLM request hits the proxy → proxy
  starts the server with the dropdown's context (tens of seconds) → request
  is forwarded. No "qwen" in any command window any more.
- Change the dropdown, send a new entry → next request carries the new
  model id → proxy sees the size mismatch → restarts the server with the
  new `-c`, then forwards.
- Nothing ever stops the model (no auto-stop, per requirement). The
  `stop-server.ps1` habit remains for the odd manual shutdown.
- Vision (`describe-image` → 8080 directly) is untouched.

## Risks / notes

- **VRAM**: 350k context "just fits in 32 GB" with no speech headroom
  (his own script comment). While the voice stack is loaded on the GPU,
  starting 350k can OOM; 225k/75k leave 1–2 GB. The plugin will surface a
  server-start failure with the server log tail instead of hanging — but
  the 350k entry simply may not coexist with the voice stack.
- **Cold-start wait**: the first send waits out the model load (tens of
  seconds). Accepted explicitly (no keystroke pre-warm requested).
  If it's ever wanted, the client-plugin seam exists
  (`packages/client/AGENTS.md`: `dsh.client` packages, slot registration) —
  that would be a separate, later project.
- **Slot cache**: `--slot-save-path` persists KV across restarts, but a
  context-size change invalidates it — expected.
- **pi-ai client timeout**: verify its HTTP timeout during
  implementation; the SSE keep-alive is the belt-and-braces.
- If DSH later ships first-party model-lifecycle management, this plugin
  retires and the baseURL reverts to 8080.

## Open questions (for Martin)

1. Proxy port 8081 OK (anything free works)?
2. Mismatch policy confirmed as "always restart to match the dropdown" —
   including a hand-started server? (Plan says yes, per the requirement.)
3. Plugin name/location: e.g. `C:\Users\press\OneDrive\Projects\dsh-local-model-manager`?
