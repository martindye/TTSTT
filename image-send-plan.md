# Image sending from DSH to the vision model — plan (no changes made)

Date: 2026-09-25. Companion to `model-wake-plan.md`. Plan only — nothing was
changed. Phase 0 verification is partly in flight (format-matrix probe running
as a background job); Phases 1–2 need Martin's go-ahead.

## 1. Goal

Use DSH's built-in image upload facility (paste/drag an image into the
composer) so images reach the Qwen vision model reliably — replacing the
manual describe-image skill for the common case.

## 2. Current state (verified 2026-09-25)

### Architecture — one server does it all

- One `llama-server` on `127.0.0.1:8080`, started by
  `TurboQuantLlamaCppSetup\scripts\start-qwen3.8-27b-q6_0-turboquant-vision.ps1`:
  - model `Qwen3.8-27B-Q6_K_L.gguf` **plus the vision projector**
    (`mmproj-Qwen3.8-27B-bf16.gguf`, `--mmproj-offload`),
  - aliases `qwen3.8-27b-q6_k_l-vision, qwen3.8-27b-vision, qwen3.8-27b`,
  - `-c 225000` ("1–2 GB spare for speech"), `--parallel 1`, `--jinja`.
- DSH (`~/.dsh/settings.yaml`, provider `qwen3827bq6`) points this session at
  `http://127.0.0.1:8080/v1` — i.e. **the chat session and the vision skill
  talk to the same server**. The vision model is not separate: it is what
  this session's server already runs.
- DSH upload path (read in source): composer paste/drag → attachment store
  (validated + normalised, below) → `llm-pi-ai` openai-completions adapter →
  `POST /v1/chat/completions` with standard OpenAI `image_url` data-URI parts
  → llama.cpp `mtmd` decodes, runs the projector, feeds the Qwen-VL template.
  This is the canonical llama.cpp multimodal flow (see
  [tools/mtmd README](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md)).

### DSH already does the normalisation (attachment-local package)

Read in `packages/attachment/attachment-local` (this is what runs today):

- Accepted input types: **PNG, JPEG, WebP, GIF** (nothing else is accepted —
  AVIF/HEIC/TIFF/BMP are refused at the attachment layer).
- On submit, every image is validated and **normalised** (`normalizeImage`,
  sharp-based): EXIF orientation applied, converted to sRGB, resized to a
  **2048×2048 total-pixel budget** (long edge ≤ 8192), re-encoded through a
  quality ladder (85/75/60) to a **≤ 4 MiB** target:
  - **JPEG** when there is no alpha, **WebP** when there is (alpha kept),
  - pass-through (byte-identical) only for already-clean single-frame
    8-bit sRGB sources inside all limits — so a clean PNG stays a PNG.
- Limits: 20 MiB per submitted image, 20 images/message, 200 MiB
  aggregate/message, 64 MP, 8192 px/side.
- The **normalised** image (not the original paste) is what rides every
  later model request; the 20 MiB `maxRequestImageBytes` route cap with
  oldest-first text offload sits on top (llm-pi-ai) — so the base64
  session-bloat problem known from
  [Open WebUI #13122](https://github.com/open-webui/open-webui/discussions/13122)
  is already mitigated two layers deep.
- Consequence: **the server can only ever receive PNG, JPEG or WebP from the
  DSH path** (JPEG for opaque images, WebP for anything with alpha — the
  usual case for screenshots —, PNG for clean pass-throughs). Animated GIFs
  are flattened to one frame before they reach the server.

### Evidence collected today

| Event | Result |
|---|---|
| 4× PNG tests (1×1, 320×240, 4K, 24 MB noise) via the raw vision endpoint | decoded OK |
| 0-byte file | clean HTTP 400 "Failed to load image or audio file" |
| Animated GIF via the **raw API** (bypassing DSH) | decode failure: `ffprobe failed` → `failed to decode buffer as either image/audio/video` → request rejected (log 10-45-08, 11:29:17). Via DSH this case cannot occur — the normaliser flattens GIFs first. |
| 4× `mtmd_tokenize: media markers (1) > bitmaps (0)` on the 09-58:37 server (10:43:50–10:44:47) | **Not from this session** — this session's history contains no image parts (full scan); the micro-turns at 10:44:07/41/47 all completed normally. The failing requests came from *another session whose history holds an image part whose bitmap the server can no longer build* — i.e. a poisoned session. Every turn of such a session fails until the bad message is removed. |

### Format matrix (probe job, 13:16, raw endpoint, 512² samples)

Measured by `prompt_tokens` delta (base prompt = 63 tokens) plus HTTP status:

| Format | Status | Image tokens @512² | Verdict |
|---|---|---|---|
| PNG 8-bit | 200 | 256 | **works** |
| PNG 16-bit | 200 | 256 | **works** (decoded) |
| JPEG | 200 | 256 | **works** |
| BMP | 200 | 256 | works (DSH never sends BMP) |
| GIF single-frame | 200 | 256 | works (DSH never sends multi-frame) |
| Animated GIF | 400 | — | rejected (DSH flattens first) |
| **WebP (±alpha), any size** | **200** | **0** | **silently ignored — the model never sees it** |

The WebP result is the headline: the request succeeds with **no error
anywhere** (no mtmd line in the server log), yet 1×1, 512² and 2048² WebPs
all contribute **zero** image tokens — the bitmap is dropped before the
template runs.

**Behavioural confirmation** (`%TEMP%\see-probe.txt`, thinking off, "what
colour is the background?"):

| Input | prompt tokens | Model's answer |
|---|---|---|
| PNG, solid red | 289 (= 33 text + 256 image) | **"Red"** |
| WebP, solid green | 33 (text only) | *(empty)* |
| WebP w/ alpha, solid blue | 33 (text only) | `"<?"` (gibberish) |

The model sees the PNG and is blind to the WebPs — no error anywhere.

### So the remaining gaps are

1. **WebP is silently invisible to the model** (confirmed, not a hazard):
   DSH sends every alpha image — the normal case for screenshots — as
   WebP, and this server build drops it without complaint. The user and the
   model both believe the image was processed. **This is the bug to fix.**
2. **Poisoned-session hazard** — see above. Need: what DSH shows the user
   when an image request 400s, whether the offending message stays in the
   session, and the recovery procedure.
3. **Image token budget unset** — the start script sets no
   `--image-max-tokens`. Measured: a 512² image costs 256 tokens; at the
   normalisation max (2048², 16× the pixels) the uncapped cost is ≈ 4000
   tokens per image.

## 3. How other people do it (research)

- **Native single-server multimodal** — llama.cpp's own design
  ([mtmd README](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md)):
  projector file + data-URI `image_url` parts on `/v1/chat/completions`;
  third-party OpenAI-compatible servers do the same
  ([llama-cpp-connector](https://github.com/fidecastro/llama-cpp-connector):
  "accepts multimodal requests (text and data: image URLs)";
  [llama.cpp WebUI discussion #16938](https://github.com/ggml-org/llama.cpp/discussions/16938)).
  **This is what we already have.**
- **Image-to-text sidecar** — Open WebUI's documented fallback when the
  active model can't see: images go to a configured OpenAI-compatible vision
  endpoint, the description is injected into the prompt. SillyTavern has the
  same "image-to-text pipeline" for blind backends
  ([docs](https://docs.sillytavern.app/extensions/captioning/)).
  Only relevant when the main model is *not* vision-capable.
- **Content-based routing proxies** — routers that swap in a vision model on
  image content ([NadirClaw](https://github.com/NadirRouter/NadirClaw):
  "Vision routing — if image content is detected, swaps to a vision…").
  Overkill here: the "router" would have exactly one endpoint.
- **Known pitfalls** in the wild: base64 session bloat (Open WebUI #13122 —
  DSH already handles it, twice), providers that don't accept `image_url` at
  all ([hermes-webui #2297](https://github.com/nesquena/hermes-webui/issues/2297)),
  and decode failures that kill the whole request (our GIF case — now
  unreachable via DSH).

**Verdict:** the best method for this machine is the native path we already
have — one server, projector loaded, standard OpenAI image parts, DSH
normalising the payload. The work is *verification + a small reliability
layer*, not new architecture. The sidecar (Martin's "temp folder + location"
idea) is retained as Phase 2 for a future text-only main model.

## 4. Phased plan

### Phase 0 — verification

1. **Format matrix** — *done* (13:16, table in §2), incl. the behavioural
   see-probe (`%TEMP%\see-probe.txt`): WebP blindness confirmed.
2. **400 behaviour in DSH** (needs Martin present, 5 min, scratch session):
   paste an image that the server will reject (candidate: a 16-bit PNG or a
   > 20 MiB file to trip the DSH limits) and observe:
   a. what the user sees (error? silent drop?),
   b. whether the message (with the image part) remains in the session,
   c. whether the next plain-text turn then fails (marker/bitmap mismatch in
      the server log) — i.e. poisoned-session confirmation + the recovery
      procedure (edit/delete the message? new session?).

### Phase 1 — reliability (after go-ahead)

1. **Fix the WebP path (the actual bug).** DSH's normaliser hard-codes
   "alpha → WebP" (`packages/attachment/attachment-local/src/encoding.ts`,
   `encodingLadder`), and this server build silently drops WebP.
   a. **Local DSH patch — WRITTEN and green (2026-09-25, dormant until
      rebuild + gateway restart).** Tagged `dsh-webp-jpeg` in every comment:
      - `encoding.ts`: `encodingLadder` always emits JPEG (alpha flattened
        upstream);
      - `normalization.ts`: `preparedPipeline` flattens alpha over white;
        verification expects alpha-less output;
      - `request-image.ts`: same flatten on the request pipeline, WebP
        pass-through blocked (pre-patch attachments re-encode), transform
        version bumped v5→v6 to invalidate stale WebP cache entries;
      - test expectations updated (normalization/request-image/index).
      All 89 attachment tests pass; host build running. Activation =
      `pnpm build:lib:host` (done) + gateway restart (brief session
      blip; voice bridge reconnects).
   b. A DSH config knob, if one ever appears — none today, the codec choice
      is hard-coded.
   c. Server-side: nothing practical — the mtmd decoder set is fixed in the
      build (stb-class loaders, no WebP); an llama.cpp upgrade is a bigger
      move with the same likely outcome.
2. **Explicit token budget** in the start script: `--image-max-tokens <N>`.
   Measured 256 tokens @ 512²; uncapped 2048² ≈ 4000. Suggest **1024**
   (full 2048² screenshots still fit with mild downscale) — Martin's call
   between 1024 and 2048.
3. **Poison guard** — whatever Phase 0.2 shows: document (and, if cheap,
   implement) the rule that a message whose image failed to decode must not
   linger in history, plus the recovery steps.
4. **Optional**: `ffprobe` on PATH — only if video attachments are ever
   wanted (mtmd's video path needs it). Recommend: skip.

Note: client-side normalisation is **not** a task — DSH's attachment layer
already down-scales, re-encodes, flattens animation and caps size, better
than the original sketch of this plan proposed.

### Phase 2 — sidecar routing (only if the main model ever stops being
vision-capable, or the vision model moves to another server/port)

Open-WebUI-style image-to-text shim in front of the text model:

1. Detect `image_url` parts in the outgoing request.
2. Persist each image to a temp folder (Martin's "drop in a temp folder"
   step — also gives the image a stable path, e.g. for re-running
   describe-image with thinking mode).
3. Call the vision endpoint (model-wake pattern from `model-wake-plan.md`
   if that server isn't running).
4. Replace the image parts with the returned description + the temp path.

Not recommended today: the projector already lives in this session's own
server, so routing would add a hop, a wake-up, and description loss for no
gain.

### Decisions needed from Martin

1. **Go for Phase 1.1**: the ~10-line local DSH patch (alpha → composite
   over white → always JPEG). Needs a DSH restart to take effect.
2. `--image-max-tokens` value — measured data in hand now (256 tokens @
   512²; 2048² uncapped ≈ 4000). Suggest **1024**; 2048 if he prefers
   sharper.
3. Go/no-go for Phase 0.2 (the scratch-session 400 test, ~5 min) — lower
   priority once the WebP fix lands, since most 400s become unreachable.
4. Keep the describe-image skill after the native path is reliable?
   (Suggestion: keep it for on-demand deep descriptions with thinking mode.)

## 5. Risks

- **WebP blindness (confirmed)** — the worst kind of failure: no error, no
  400, the model simply never sees the image and answers anyway. Until the
  Phase 1.1 fix lands, every alpha screenshot pasted through DSH is
  invisible to the model while looking like a normal success.
- A poisoned *real* session may need its history edited — Phase 0.2 defines
  the recovery.
- Downscaling changes what the model sees (small code in screenshots) —
  2048² keeps screenshots legible; the normalisation policy is DSH-default,
  not per-deployment, so changing it is a DSH config change, not a local one.
- VRAM: projector already resident; per-image encoder work is transient — no
  new risk.

## 6. Appendix — probe recipe (already written)

`%TEMP%\format-probe.py` (background job): generates 512² samples per
format, POSTs to `/v1/chat/completions` with model
`qwen3.8-27b-vision`, records HTTP status + reply + usage per sample, and
writes `%TEMP%\format-matrix.txt`. Sample images from the earlier manual
tests (`dsh-test-*.png`, `dsh-test-anim.gif`) remain in `%TEMP%`.

Manual raw probe (for later use):

```powershell
$body = @{
  model = "qwen3.8-27b-vision"; max_tokens = 8
  messages = @(@{ role = "user"; content = @(
    @{ type = "text"; text = "One word: what is the background colour?" },
    @{ type = "image_url"; image_url = @{ url = "data:$mime;base64,$b64" } }
  )})
} | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
# 200 + a colour word = decodable; 400 "Failed to load image or audio file" = not
```
