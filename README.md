# ComfyUI-H3LookSheets

Reference sheets of a person wearing a specific look, built on **MiniMax H3**'s
multi-reference conditioning (`ref2va`).

Two reference images go in — one of the person, one of the look (an outfit to
start, with hairstyle, makeup and accessories planned) — and H3 renders that
person, in that look, from every angle the sheet needs.

Built around `MiniMaxH3ReferenceToVideo`: describe the two references, write
the shot-by-shot prompt, generate, pick the frames worth keeping out of the
rendered move, lay them out as one sheet.

📺 [Demo video](https://www.youtube.com/watch?v=G1Gbli-9nFY)

Person + outfit references in, look sheet out:

| Picture 1 — Person | Picture 2 — Outfit |
|---|---|
| ![Person reference](example_results/1_person.webp) | ![Outfit reference](example_results/1_outfit.webp) |

![Result look sheet](example_results/1_result.webp)

---

## The pipeline, node by node

Each step feeds the next; only step 3 is a core ComfyUI/MiniMax H3 node, not
part of this pack.

1. **Load the two reference photos** (`LoadImage` ×2 — one person, one look).
2. **Describe each one** with a separate **Describe Reference** node
   (`target=person` on the first photo, `target=outfit` on the second) →
   `person_description` / `outfit_description`.
3. **Write the prompt**: feed both descriptions into **Look Sheet Prompt**
   (or `Shot Config` ×N → **Look Sheet Prompt - Custom Shots** for a freely
   chosen shot list instead of the fixed 6-shot turnaround) → `prompt`.
4. **Generate**: `prompt`, the two reference images, `clip`, `vae` and
   `audio_vae` go into `MiniMaxH3ReferenceToVideo` (core node), then through
   your usual sampler and VAE decode → a batch of decoded frames.
5. **Pick the keepers**: the decoded frames go into **Select Frames** →
   `frames`.
6. **Lay them out**: `frames` goes into **Datasheet Settings** → `sheet`
   (IMAGE), the finished contact sheet.

```
LoadImage (person) ─► Describe Reference (target=person) ─► person_description ─┐
LoadImage (outfit) ─► Describe Reference (target=outfit) ─► outfit_description ─┤
                                                                                  ▼
                                                                     Look Sheet Prompt ─► prompt
                                                                                  │
                                                                                  ▼
                                                                     MiniMaxH3ReferenceToVideo
                                                                                  │
                                                                                  ▼
                                                                   sampler ─► VAE decode ─► decoded frames
                                                                                  │
                                                                                  ▼
                                                                          Select Frames ─► frames
                                                                                  │
                                                                                  ▼
                                                                       Datasheet Settings ─► sheet (IMAGE)
```

---

## Nodes

| Node | Category | What it does |
|---|---|---|
| **Look Sheet Prompt (H3)** | `H3LookSheets` | Fixed 6-shot, all-neutral turnaround prompt (`H3LookSheetsPrompt`) |
| **Look Sheet Prompt - Custom Shots (H3)** | `H3LookSheets` | Same idea, but 1–15 freely described shots (`H3LookSheetsCustomPrompt`) |
| **Shot Config (H3 Look Sheet)** | `H3LookSheets` | One shot's angle/framing/expression, packed for the node above (`H3LookSheetsShotConfig`) |
| **Describe Reference (H3 Look Sheet)** | `H3LookSheets` | One-sentence vision description of a reference photo, with retry (`H3LookSheetsDescribe`) |
| **Select Frames (H3 Look Sheet)** | `H3LookSheets` | Picks reference frames out of the rendered take (`H3LookSheetsSelectFrames`) |
| **Datasheet Settings (H3 Look Sheet)** | `H3LookSheets` | Lays the picked frames out as one contact-sheet image (`H3LookSheetsDatasheetSettings`) |

---

### Look Sheet Prompt (H3)

Writes the `subject_definitions` / `summary` / `retention_analysis` /
`detailed_description` prompt `MiniMaxH3ReferenceToVideo` expects, referencing
`<Picture 1>` (the person) and `<Picture 2>` (the look) per H3's ref2va guide.
Always 6 shots, always neutral: full body → face close-up → left profile →
right profile → back → a second, wider medium close-up (chest-up).

| Input | Type | Notes |
|---|---|---|
| `person_description` | STRING | From `Describe Reference` (`target=person`) or written by hand |
| `outfit_description` | STRING | From `Describe Reference` (`target=outfit`) |
| `picture_1_gender` | `auto` / `female` / `male` | Auto-detects from `person_description`'s wording; override when needed |
| `picture_2_subject_type` | `auto` / `female` / `male` / `mannequin` | Who/what `<Picture 2>`'s original wearer is |
| `backdrop` | STRING (multiline) | Default: plain light neutral grey studio backdrop |
| `video_duration_seconds` | FLOAT | The take's real length — feed it from whatever sets `length` on `MiniMaxH3ReferenceToVideo`. Per-shot duration is `duration / 6`, truncated to one decimal |

For expressions other than neutral, or a shot list that isn't this exact
six, use **Look Sheet Prompt - Custom Shots** instead.

---

### Look Sheet Prompt - Custom Shots (H3)

Same `<Picture 1>`/`<Picture 2>` identity+outfit logic, but the shot list
itself is not fixed. Up to 15 `shot_N` sockets (plug one, the next appears),
each fed by a **Shot Config** node — however many are actually connected
becomes the shot count. `<Picture 1>`/Shot 1 anchors to whatever the *first*
connected shot describes.

| Input | Type | Notes |
|---|---|---|
| `person_description` / `outfit_description` | STRING | Same as above |
| `picture_1_gender` / `picture_2_subject_type` | Combo | Same as above |
| `backdrop` | STRING (multiline) | Same as above |
| `video_duration_seconds` | FLOAT | Same truncation rule, divided by however many shots are connected |
| `shot_0` … `shot_14` | STRING (Autogrow) | Each fed by a **Shot Config** node |

### Shot Config (H3 Look Sheet)

Three inline dropdowns, packed into one string for the node above.

| Combo | Options |
|---|---|
| `angle` | front, front 3/4 left, front 3/4 right, left profile, right profile, back 3/4 left, back 3/4 right, back |
| `framing` | extreme wide shot, wide shot (full body), medium wide shot (knees-up), medium shot (waist-up), medium close-up (chest-up), close-up (shoulders/face), extreme close-up (eyes/detail) |
| `expression` | neutral, happy, smiling, sad, angry, surprised, scared, disgusted, shy/embarrassed, confident, serious, laughing, crying, smirking, confused |

Pick angle and framing together with the subject visible — e.g. `back` +
`close-up (shoulders/face)` shows no face for the expression to read on.

---

### Describe Reference (H3 Look Sheet)

Wraps core's `TextGenerate` node with one of two fixed system prompts
selected by `target`:

| `target` | Asks the vision model to… |
|---|---|
| `person` | Describe face, hair, eyes, skin, body shape; ignore outfit and clothes |
| `outfit` | Describe the outfit and accessories only; skip hair |

One node per reference photo. Plug its `generated_text` output straight into
`person_description`/`outfit_description` on either prompt node above.

Retries automatically (up to 3 times) on an empty reply, a refusal, or a
leaked chat-template role tag. If every attempt fails, returns whatever the
last attempt produced — check the output before trusting it downstream.

Inherited from `TextGenerate`: `max_length`, `sampling_mode` (on/off +
temperature/top_k/top_p/seed/…), `thinking`, `use_default_template`. `video`
and `audio` inputs are dropped — this node only ever describes one still
image.

---

### Select Frames (H3 Look Sheet)

Picks `saved_frame_count` reference frames out of the rendered take, via a
fixed three-tier cascade:

1. **Tier 1 — cluster by content.** Groups the leftover frames into
   `tier1_shots_clusters` visually distinct clusters, takes the sharpest
   frame of each. Deterministic, always runs.
2. **Tier 2 — vision-informed diversity fill.** If slots remain
   (`saved_frame_count > tier1_shots_clusters`) and `tier2_clip` is wired,
   builds a second content-diverse shortlist and fills the remaining slots
   with whichever of those are most different from what's already picked.
   The vision model judges the shortlist and its reply is shown in `debug`,
   but does not decide which frames are kept.
3. **Tier 3 — sharpness + diversity fallback.** Fills whatever tier 2
   didn't, the same diversity-first way, so the node always returns exactly
   `saved_frame_count` frames (or every frame, if the video has fewer).

| Input | Tier | Notes |
|---|---|---|
| `images` | — | The rendered take (decoded frames) |
| `saved_frame_count` | — | Total frames returned |
| `tier1_shots_clusters` | 1 | Distinct views to guarantee. Set equal to `saved_frame_count` for a fully deterministic run — tiers 2/3 never trigger |
| `tier2_clip` | 2 | In-graph vision CLIP. Leave unplugged to skip straight to tier 3 |
| `tier2_candidates` | 2 | Clusters formed from the leftover pool for the shortlist |
| `tier2_prompt_subject_description` / `tier2_prompt_how_to_select_frames` | 2 | Text shown to the vision model, logged in `debug` |
| `tier2_free_vram_first` | 2 | Unload other resident models before the tier-2 vision call |
| `tier2_max_token_length` / `tier2_temperature` / `tier2_thinking` | 2 | Passed to the vision call |
| `tier3_sharpness_diversity` | 3 | Off returns fewer than `saved_frame_count` instead of filling |
| `tier3_sharpness_weight` | 3 | Balance between sharpness and diversity when filling |

`debug` (STRING output) lists, per returned frame, which tier picked it:

```
8/120 frames — 6 Tier 1: cluster (by content), 2 Tier 2: vision (clip, diversity-picked)
  frame 13: Tier 1: cluster (by content)
  ...
  frame 71: Tier 2: vision (clip, diversity-picked)
tier 2 shortlist shown to model (16): [...]
tier 2 raw reply (json_ok=True, 153 chars): {"picks": [1, 2, ...], "why": "..."}
```

---

### Datasheet Settings (H3 Look Sheet)

Lays the frames from **Select Frames** out as one contact-sheet image. No
per-tile labels, no vision call.

| Input | Notes |
|---|---|
| `images` | The frames to lay out |
| `columns` | Per row (default 3) |
| `columns_width` | Per-tile width in pixels; height follows the first frame's aspect ratio (default 384) |
| `padding` | Gap in pixels, on every side and between tiles (default 8) |

---

## Requirements

**Models**: MiniMax H3 in `ref2va` mode (`MiniMaxH3ReferenceToVideo`) plus a
vision-language CLIP that supports `.generate()` (Qwen3-VL) for **Describe
Reference** and (optionally) tier 2 of **Select Frames**.

**Python**: nothing outside what ComfyUI already ships (`torch`, `numpy`,
`Pillow`) — no `requirements.txt` needed.

---

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/shisa84/ComfyUI-H3LookSheets
```

Restart ComfyUI. The nodes appear under the **H3LookSheets** category.

Load a graph from `example_workflows/` (canvas format, drag onto the canvas)
to see a complete pipeline wired up.

---

## License

MIT
