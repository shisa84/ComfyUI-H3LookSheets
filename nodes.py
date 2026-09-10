"""Prompt construction for MiniMax H3's ref2va (reference-to-video) mode.

Builds the multi-reference prompt H3's guide expects when conditioning
`MiniMaxH3ReferenceToVideo` on two images: a person (<Picture 1>) and a look
to dress them in (<Picture 2>). The vision-language description of each image
stays outside this node — plug in two `TextGenerate` nodes (one prompted to
describe the person while ignoring clothes, one prompted to describe the look
while ignoring hair) and feed their generated text in here. This node only
assembles the `subject_definitions` / `summary` / `retention_analysis` /
`detailed_description` structure ref2va expects around those two sentences.
"""

from __future__ import annotations

import json
import logging
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerate

#: English word for small shot counts (5 or 6 in practice) — reads naturally
#: in the intro sentence ("across all six shots") instead of a bare digit.
_COUNT_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen",
}

#: Only two manual choices — each reference photo holds exactly one person,
#: never an ambiguous group, so there's no case a they/them override serves.
_PRONOUNS = {
    "female": {"subj": "she", "poss": "her"},
    "male": {"subj": "he", "poss": "his"},
}

#: Grammar for the one case detection can't call — never offered as a manual
#: choice, only ever reached when auto-detect ties or finds nothing.
_NEUTRAL_PRONOUN = {"subj": "they", "poss": "their"}

_AUTO_PRONOUN = "auto"
_PRONOUN_OPTIONS = [_AUTO_PRONOUN, "female", "male"]

#: Word-boundary matches, cheapest thing that works on the one-sentence
#: descriptions `person_description` is meant to hold — no need for a real
#: gender-classifier model over a phrase this short.
_SHE_WORDS = re.compile(r"\b(woman|female|girl|she|her|hers)\b", re.IGNORECASE)
_HE_WORDS = re.compile(r"\b(man|male|boy|he|him|his)\b", re.IGNORECASE)


def _detect_pronoun(description: str) -> str | None:
    """Guess "female"/"male" from a short physical-description sentence.

    Ties or no match return None — a one-sentence VLM description often
    skips gendered words entirely ("long dark hair, oval face, slender
    build"), or a styling word (a "boy-cut" on a woman) creates a false tie.
    The caller falls back to neutral grammar rather than guessing wrong.
    """
    she_hits = len(_SHE_WORDS.findall(description))
    he_hits = len(_HE_WORDS.findall(description))
    if she_hits > he_hits:
        return "female"
    if he_hits > she_hits:
        return "male"
    return None


#: <Picture 2> can also be a headless dress-form shot, which is neither
#: "female" nor "male" and needs its own noun rather than a pronoun at all.
_MANNEQUIN_WORDS = re.compile(r"\b(mannequin|mannikin|dress form|dressform)\b", re.IGNORECASE)

_SUBJECT_TYPE_OPTIONS = [_AUTO_PRONOUN, "female", "male", "mannequin"]

_SUBJECT_NOUNS = {
    "female": "the woman",
    "male": "the man",
    "mannequin": "the mannequin",
}
#: Reached only when detection can't tell — same role as _NEUTRAL_PRONOUN.
_NEUTRAL_SUBJECT_NOUN = "the person"


def _detect_subject_type(description: str) -> str | None:
    """Guess "female"/"male"/"mannequin" from the outfit-photo description."""
    if _MANNEQUIN_WORDS.search(description):
        return "mannequin"
    return _detect_pronoun(description)


def _timecode(seconds: float) -> str:
    """Seconds -> H3's MM:SS.mmm cut marker."""
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{rest:06.3f}"


class H3LookSheetsPrompt:
    """Write H3's ref2va prompt for a 6-shot, all-neutral look turnaround.

    Two references go in — a person (<Picture 1>) and a look to dress them in
    (<Picture 2>) — and the prompt asks H3 for a 6-shot turnaround that keeps
    only <Picture 1>'s identity (hair, body shape, skin, facial structure) and
    only <Picture 2>'s outfit, discarding everything else either reference
    carries (<Picture 2>'s own hair, body shape and proportions in
    particular): [Shot 1] full body -> [Shot 2] face close-up -> [Shot 3] left
    profile -> [Shot 4] right profile -> [Shot 5] back -> [Shot 6] a second,
    closer face shot. Every shot is neutral; for other expressions or a
    freely chosen shot list, use H3LookSheetsCustomPrompt instead.

    Cuts, not a continuous move: a cut forces the model to re-establish
    <Subject 1> at each angle, which is what a turnaround needs, and identity
    stays locked because every shot reuses the same two references.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_description": ("STRING", {"multiline": True, "default": ""}),
                "outfit_description": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                # Default guesses female/male from `person_description`'s own
                # wording (a "woman"/"man"/etc. in there is usually enough);
                # override only when it gets it wrong. Only two manual
                # choices because exactly one person is ever in the photo.
                "picture_1_gender": (_PRONOUN_OPTIONS,),
                # Who/what <Picture 2>'s original wearer is, for the clause
                # saying their hair/body do NOT carry over — auto-detected
                # from `outfit_description` the same way as picture_1_gender,
                # plus a mannequin option for a headless dress-form shot.
                "picture_2_subject_type": (_SUBJECT_TYPE_OPTIONS,),
                "backdrop": ("STRING", {"default": "plain light neutral grey studio backdrop", "multiline": True}),
                # The take's actual length (seconds) — feed this from whatever
                # sets `length` on MiniMaxH3ReferenceToVideo (e.g. a Duration
                # node) so the 6 shots always divide up the real take instead
                # of an assumed fixed 5s.
                "video_duration_seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "H3LookSheets"

    def build(self, person_description, outfit_description,
              picture_1_gender=_AUTO_PRONOUN, picture_2_subject_type=_AUTO_PRONOUN,
              backdrop="plain light neutral grey studio backdrop",
              video_duration_seconds=5.0):
        resolved_pronoun = (
            _detect_pronoun(person_description)
            if picture_1_gender == _AUTO_PRONOUN else picture_1_gender
        )
        p = _PRONOUNS.get(resolved_pronoun, _NEUTRAL_PRONOUN)
        resolved_subject_type = (
            _detect_subject_type(outfit_description)
            if picture_2_subject_type == _AUTO_PRONOUN else picture_2_subject_type
        )
        subject_noun = _SUBJECT_NOUNS.get(resolved_subject_type, _NEUTRAL_SUBJECT_NOUN)
        person = person_description.strip().rstrip(".") or "the same figure"
        outfit = outfit_description.strip().rstrip(".") or "the outfit shown"
        set_dressing = backdrop.strip().rstrip(".") or "plain light neutral grey studio backdrop"

        total_shots = 6
        duration = float(video_duration_seconds)
        # Truncated to 1 decimal, not rounded: 0.833s -> 0.8s, never 0.83s
        # rounding up to overshoot the take's real length.
        step = max(0.1, math.floor((duration / total_shots) * 10) / 10)
        at = [_timecode(i * step) for i in range(total_shots)]
        shot_tags = ", ".join(f"[Shot {i + 1}]" for i in range(total_shots))

        header = (
            f"subject_definitions:\n<Subject 1> {person}, and the same figure "
            f"and proportions established in <Picture 1>. {outfit} "
            f"from the <Picture 2>, the garment tailored to {p['poss']} own "
            "body shape and proportions from <Picture 1>, not the proportions "
            "in <Picture 2>. <Picture 1> hairstyle is transferred to <Subject "
            f"1> — none of the hair, hairstyle, or hair texture of "
            f"{subject_noun} shown in <Picture 2> is carried over.\n"
            "<Picture 1> is the first frame of [Shot 1], showing <Subject 1> "
            f"full-body, facing the camera with a neutral expression against "
            f"{set_dressing}."
        )

        summary = (
            "summary:\n[reference generation] The target video presents "
            f"<Subject 1> in a {duration:g}-second sequence of "
            f"{total_shots} static shots, retaining only {p['poss']} hair, "
            "body shape, skin, and facial structure from <Picture 1>, while "
            f"preserving the outfit from <Picture 2>. The background remains "
            f"{set_dressing}. All shots maintain identical framing, lighting, "
            "and pose, every expression neutral throughout."
        )

        retention = (
            "retention_analysis:\n"
            f"<Subject 1> (appears in {shot_tags}): partially_preserved - "
            "retains only hair, hairstyle, body shape, skin, and facial "
            "structure; no clothing or movement preserved.\n"
            f"<Picture 2> (appears in {shot_tags}): partially_preserved - "
            "retains only the outfit; no other visual elements or props "
            "preserved."
        )

        shots_word = _COUNT_WORDS.get(total_shots, str(total_shots))
        # The explicit "only the camera moves, <Subject 1> never does, hair
        # stays at rest" clause is load-bearing: without it H3 tends to
        # render windswept/flying hair between cuts, as if carrying motion
        # over from a turn that never actually happened.
        intro = (
            f"The target video uses a static studio lighting setup against "
            f"{set_dressing}. All shots are framed with generous empty "
            "margins on every side, ensuring the entire figure and any "
            "extensions remain fully within the frame without touching or "
            f"crossing edges. {set_dressing[0].upper()}{set_dressing[1:]} "
            "and its bright, even lighting stay completely identical across "
            f"all {shots_word} shots. No text, watermark, logo, caption, or "
            "writing of any kind appears anywhere in the video. Between every "
            "shot, only the camera's position around <Subject 1> changes to "
            "capture each new angle — <Subject 1> remains completely "
            "motionless throughout, holding one exact pose without turning, "
            "walking, gesturing, or otherwise moving between cuts, and hair "
            "stays at rest in that pose, undisturbed by any camera movement."
        )

        detail = (
            "detailed_description:\n"
            f"{intro}\n"
            "[Shot 1] A full-body shot shows <Subject 1> standing still in a "
            f"neutral upright pose with arms relaxed at {p['poss']} sides.\n"
            f"[Shot 2] At {at[1]}, a tight close-up of <Subject 1>'s face, "
            f"framed from the top of {p['poss']} hair to {p['poss']} chin, "
            f"captures {p['poss']} neutral expression. Eyes are calm, brows "
            f"are relaxed, mouth is closed, and {p['subj']} faces directly "
            "toward the camera. Hair, skin tone, and facial structure are "
            "fully visible.\n"
            f"[Shot 3] At {at[2]}, a left side profile of <Subject 1>'s full "
            "figure, head turned to show the left profile. Framing margin "
            "remains consistent. Expression is still neutral. Hair rests "
            "naturally over the shoulder in this pose, undisturbed by any "
            "camera movement. Body shape and proportions are unchanged. "
            "Clothing and outfit are fully visible.\n"
            f"[Shot 4] At {at[3]}, a right side profile of <Subject 1>'s full "
            "figure, head turned to show the right profile. Same framing, "
            "lighting, and neutral expression. Hair, skin, and facial "
            "structure are clearly visible from this angle. Outfit remains "
            "identical.\n"
            f"[Shot 5] At {at[4]}, a rear view of <Subject 1>'s full figure, "
            "back facing the camera. Head is still upright, mouth closed, "
            "expression neutral. Hair is visible from behind. Body shape and "
            "proportions are preserved. Outfit is fully visible from the back.\n"
            f"[Shot 6] At {at[5]}, a medium close-up of <Subject 1> from the "
            "chest up, still facing the camera, the expression neutral: "
            "eyes calm, brows relaxed, mouth closed, the head held still and "
            "upright. <Subject 1> does not flinch, recoil, turn away or "
            "move, and the framing and lighting stay as before."
        )

        prompt = (
            f"{header}\n\n{summary}\n\n{retention}\n\n{detail}\n\n"
            "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
        )
        return (prompt,)


#: Kept as code, not graph text, so both prompts stay in sync with each
#: other and with H3LookSheetsPrompt's expectations without relying on a
#: workflow to wire the right PrimitiveStringMultiline to the right image.
#: Verbatim wording from the working manual setup — an earlier version added
#: a "not a sensitive or sexual context" reassurance line to cut down on
#: refusals, but that backfired: naming "sexual"/"sensitive" at all, even in
#: a negation, primed the model's safety filter instead of calming it. The
#: refusal retry below is the safety net now, not the wording.
_DESCRIBE_SYSTEM_PROMPTS = {
    "person": (
        "Analyzes the person's physical appearance face, hair, eyes, skin, "
        "body shape from the provided image, ignore outfit and clothes. \n"
        "Very short description, only 1 sentence."
    ),
    "outfit": (
        "Analyzes the outfit, only describe the outfit, specifically "
        "detailing the clothing and accessories, skip hair description. "
        "Very short description, only 1 sentence"
    ),
}

#: A genuine one-sentence physical/outfit description never contains any of
#: these — catches refusals ("I can't fulfill this request...", "I apologize,
#: but...", copyright/watermark objections) that empty-string detection alone
#: lets straight through, since refusal text isn't empty. Searched anywhere
#: in the text, not just at the start: a refusal often opens with an apology
#: ("I apologize, but I cannot...") before the actual "I can't" clause.
_REFUSAL_PATTERN = re.compile(
    r"\b(i can'?t|i cannot|i'?m (unable|sorry|not able)|i am (unable|not able)|"
    r"as an ai|i must (decline|respect)|i won'?t|i apologize|unfortunately,? i|"
    r"cannot fulfill|can'?t fulfill|due to copyright|intellectual property|"
    r"copyrighted content)\b",
    re.IGNORECASE,
)


def _is_bad_output(text: str) -> bool:
    text = text.strip()
    return not text or bool(_REFUSAL_PATTERN.search(text))


#: Small chat-tuned models occasionally echo the chat-template's own role
#: tag ("assistant") as literal generated text instead of stopping right
#: after it — a known quirk, not something specific to this node's prompt.
_LEADING_ROLE_TAG = re.compile(r"^\s*(assistant|system|user)\s*[:\n]+\s*", re.IGNORECASE)


def _strip_role_leak(text: str) -> str:
    return _LEADING_ROLE_TAG.sub("", text, count=1)


class H3LookSheetsDescribe(TextGenerate):
    """Describe a Look Sheet reference photo in one sentence.

    Wraps core's `TextGenerate` (same tokenize/generate/decode call, reused
    via `super().execute()`) with the fixed system prompt H3LookSheetsPrompt
    expects for whichever reference this is — `target=person` for <Picture
    1>, `target=outfit` for <Picture 2>. One node per image, same as before,
    just without a `PrimitiveStringMultiline` to keep in sync by hand.
    """

    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        inputs = []
        for inp in parent.inputs:
            if inp.id == "prompt":
                inputs.append(io.Combo.Input(
                    "target", options=list(_DESCRIBE_SYSTEM_PROMPTS.keys()),
                    tooltip="Which reference this image is: the person "
                    "(<Picture 1>) or the outfit (<Picture 2>).",
                ))
            elif inp.id == "image":
                # Required here — there is nothing to describe without it,
                # unlike bare TextGenerate which can run text-only.
                inputs.append(io.Image.Input("image"))
            elif inp.id in ("video", "audio"):
                # This node only ever describes one still reference photo.
                continue
            else:
                inputs.append(inp)
        return io.Schema(
            node_id="H3LookSheetsDescribe",
            display_name="Describe Reference (H3 Look Sheet)",
            category="H3LookSheets",
            inputs=inputs,
            outputs=parent.outputs,
        )

    #: Retried on: an empty decode (first sampled token landed on
    #: end-of-sequence — bad luck of that seed draw) or a refusal (the vision
    #: model treating an ordinary photo as sensitive content). Neither is
    #: tied to a specific seed value; a different seed often just works.
    MAX_EMPTY_RETRIES = 3
    _RETRY_SEED_STRIDE = 104729  # an arbitrary large prime, just to jump seeds

    @classmethod
    def execute(cls, clip, target, max_length, sampling_mode, image,
                thinking=False, use_default_template=True) -> io.NodeOutput:
        system_prompt = _DESCRIBE_SYSTEM_PROMPTS[target]
        mode = sampling_mode
        base_seed = mode.get("seed") if isinstance(mode, dict) else None

        for attempt in range(cls.MAX_EMPTY_RETRIES + 1):
            out = super().execute(
                clip, system_prompt, max_length, mode, image=image,
                thinking=thinking, use_default_template=use_default_template,
            )
            text = _strip_role_leak(out.args[0] if out.args else "")
            if not _is_bad_output(text):
                return io.NodeOutput(text)
            if base_seed is None or attempt == cls.MAX_EMPTY_RETRIES:
                # No seed to vary (sampling is off), or retries exhausted.
                break
            mode = dict(mode)
            mode["seed"] = (base_seed + (attempt + 1) * cls._RETRY_SEED_STRIDE) % 0xffffffffffffffff
            logging.warning(
                "[H3LookSheetsDescribe] %s for target=%s (seed %s), "
                "retrying with seed %s (attempt %d/%d)",
                "empty output" if not text.strip() else "refusal",
                target, base_seed, mode["seed"], attempt + 1, cls.MAX_EMPTY_RETRIES,
            )

        if _is_bad_output(text):
            logging.warning(
                "[H3LookSheetsDescribe] still bad output for target=%s after "
                "retries — check the image and system prompt.", target,
            )
        return io.NodeOutput(text)


# --------------------------------------------------------------------------
# H3LookSheetsSelectFrames
#
# A fixed three-tier cascade, always run in this order, never picked by a
# mode switch: (1) cluster by visual content — one sharp frame per distinct
# view, deterministic; (2) vision judgement fills whatever's left, only if a
# `clip` is wired; (3) sharpness+diversity fills whatever tier 2 still
# couldn't (no clip connected, the call failed, or it under-returned). No
# HTTP fallback (`llm_url`) — in-graph `clip` or nothing. `debug` names, for
# every frame in the output, which tier picked it.
# --------------------------------------------------------------------------


def _sf_sharpness(images: torch.Tensor) -> torch.Tensor:
    """Variance of the Laplacian per frame — the standard blur detector."""
    gray = images.mean(dim=3).unsqueeze(1)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=gray.dtype, device=gray.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.view(lap.shape[0], -1).var(dim=1)


def _sf_descriptors(images: torch.Tensor, size: int) -> torch.Tensor:
    """A per-frame signature: colour thumbnail, per-frame normalised.

    Colour and `size=32` (not a coarser grey `size=16`) so a left profile
    doesn't merge with its mirror-image right profile, and a front doesn't
    merge with a back — both share the same silhouette in grey and low res.
    """
    small = images.permute(0, 3, 1, 2)
    small = F.interpolate(small, size=(size, size), mode="area")
    small = small.reshape(small.shape[0], -1)
    small = small - small.mean(dim=1, keepdim=True)
    return small / (small.std(dim=1, keepdim=True) + 1e-6)


def _sf_cluster_by_content(images: torch.Tensor, k: int) -> list[list[int]]:
    """K-means over `_sf_descriptors`, farthest-point seeded for determinism.

    Splitting the timeline by time only works if each slice happens to hold
    a different view, and it does not: H3 gives its shots wildly different
    lengths from run to run. Grouping by appearance cannot make that mistake
    — k distinct views are k clusters wherever the video's cuts (if any)
    happen to fall.
    """
    total = int(images.shape[0])
    k = max(1, min(k, total))
    desc = _sf_descriptors(images, 32)

    seeds = [0]
    while len(seeds) < k:
        dist = torch.cdist(desc, desc[seeds]).min(dim=1).values
        dist[torch.tensor(seeds, device=dist.device)] = -1.0
        seeds.append(int(torch.argmax(dist).item()))
    centres = desc[seeds].clone()

    labels = torch.zeros(total, dtype=torch.long)
    for _ in range(25):
        labels = torch.cdist(desc, centres).argmin(dim=1)
        moved = False
        for c in range(k):
            members = labels == c
            if not bool(members.any()):
                worst = int(torch.cdist(desc, centres).min(dim=1).values.argmax())
                centres[c] = desc[worst]
                moved = True
                continue
            mean = desc[members].mean(dim=0)
            if not torch.allclose(mean, centres[c]):
                centres[c] = mean
                moved = True
        if not moved:
            break

    labels = torch.cdist(desc, centres).argmin(dim=1)
    return [[i for i in range(total) if int(labels[i]) == c] for c in range(k)]


def _sf_cluster_pool(images: torch.Tensor, pool: list[int], k: int) -> list[list[int]]:
    """`_sf_cluster_by_content`, restricted to and indexed by `pool`."""
    if not pool:
        return []
    sub_groups = _sf_cluster_by_content(images[pool], k)
    return [[pool[i] for i in g] for g in sub_groups]


def _sf_fill_by_diversity(images: torch.Tensor, pool: list[int], chosen: list[int],
                           want: int, sharpness_weight: float) -> list[int]:
    """Greedy farthest-point fill from `pool`: sharp, and far from `chosen`.

    `chosen` seeds the distance calculation (frames already picked by an
    earlier tier still count against this tier's diversity) but is never
    itself a candidate — `pool` excludes anything already picked.
    """
    remaining = list(pool)
    chosen = list(chosen)
    result: list[int] = []
    if not remaining or want <= 0:
        return result

    sharp = _sf_sharpness(images)
    desc = _sf_descriptors(images, 16)
    sharp_min, sharp_max = float(sharp.min()), float(sharp.max())
    span = sharp_max - sharp_min + 1e-9

    for _ in range(min(want, len(remaining))):
        idx_t = torch.tensor(remaining)
        sharp_norm = (sharp[idx_t] - sharp_min) / span
        if chosen:
            dist = torch.cdist(desc[idx_t], desc[torch.tensor(chosen)]).min(dim=1).values
            dist = dist / (float(dist.max()) + 1e-9)
        else:
            dist = torch.ones(len(remaining))
        score = sharpness_weight * sharp_norm + (1.0 - sharpness_weight) * dist
        best = int(torch.argmax(score).item())
        picked_idx = remaining.pop(best)
        chosen.append(picked_idx)
        result.append(picked_idx)
    return result


def _sf_shortlist_diverse(images: torch.Tensor, sharp: torch.Tensor,
                          pool: list[int], wanted: int) -> list[int]:
    """`wanted` candidates spread across `pool`'s visual variety, not its time.

    Sampling evenly by index and then keeping only the sharpest few can wipe
    out an entire shot: if one angle is structurally softer than the others
    (mid-turn vs a held pose), sharpness-only gating removes every candidate
    from that angle, not just its blurriest frames — clustering first
    guarantees the shortlist still spans whatever variety is left in `pool`.
    """
    wanted = max(1, min(wanted, len(pool)))
    groups = [g for g in _sf_cluster_pool(images, pool, wanted) if g]
    return sorted({max(g, key=lambda f: float(sharp[f])) for g in groups})


def _sf_tensor_to_pils(images: torch.Tensor) -> list:
    from PIL import Image
    arr = (images.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


def _sf_pils_to_tensor(pils: list) -> torch.Tensor:
    stack = [np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(stack, axis=0))


def _sf_load_font(size: int):
    from PIL import ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _sf_montage(pils: list, columns: int, cell_width: int, padding: int,
                 labels: list[str] | None = None, background: tuple = (18, 18, 20)):
    from PIL import Image, ImageDraw

    columns = max(1, min(columns, len(pils)))
    rows = (len(pils) + columns - 1) // columns
    ratio = pils[0].height / pils[0].width
    cell_h = max(1, int(round(cell_width * ratio)))
    sheet_w = columns * cell_width + padding * (columns + 1)
    sheet_h = rows * cell_h + padding * (rows + 1)
    sheet = Image.new("RGB", (sheet_w, sheet_h), background)

    draw = ImageDraw.Draw(sheet)
    font = _sf_load_font(max(14, cell_width // 16))
    for index, pil in enumerate(pils):
        row, col = divmod(index, columns)
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_h + padding)
        sheet.paste(pil.convert("RGB").resize((cell_width, cell_h)), (x, y))
        text = labels[index] if labels else ""
        if not text:
            continue
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x + 4, y + 4, x + 4 + tw + 8, y + 4 + th + 8], fill=(0, 0, 0))
        draw.text((x + 8, y + 6), text, fill=(255, 255, 0), font=font)
    return sheet


def _sf_strip_thinking(text: str) -> str:
    """Drop a `<think>...</think>` reasoning block, including one left
    unclosed by max_length — the JSON reply comes after it, not inside it."""
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()


def _sf_json_slice(text: str) -> str:
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start:end + 1] if start >= 0 and end > start else cleaned


def _sf_parse_picks(text: str, total: int) -> list[int]:
    """Pull frame numbers out of the reply, tolerating chatty models."""
    numbers: list[int] = []
    try:
        payload = json.loads(_sf_json_slice(text))
        raw = payload.get("picks", [])
        numbers = [int(n) for n in raw if isinstance(n, (int, float, str))
                   and str(n).strip().lstrip("-").isdigit()]
    except Exception:
        numbers = [int(n) for n in re.findall(r"\b\d{1,3}\b", text)]

    seen, out = set(), []
    for number in numbers:
        index = number - 1  # labels are 1-based
        if 0 <= index < total and index not in seen:
            seen.add(index)
            out.append(index)
    return out


_SF_DEFAULT_BRIEF = (
    "Prefer: sharp, well-exposed frames; clearly distinct angles that each "
    "reveal a different side or aspect.\n"
    "Reject: motion-blurred or smeared frames; near-duplicates of a frame "
    "you already chose; frames where the subject is cropped or occluded."
)


def _sf_clip_pick(clip, pils: list, count: int, hint: str, brief: str,
                   max_length: int, temperature: float, thinking: bool = False) -> str:
    """Ask the in-graph CLIP to choose, comparing every candidate at once."""
    labels = [str(i + 1) for i in range(len(pils))]
    board = _sf_montage(pils, columns=4, cell_width=384, padding=10, labels=labels)
    subject = hint.strip() or "the subject"
    instruction = (
        f"This is a numbered contact sheet of {len(pils)} candidate frames. "
        f"Choose exactly {count} frames that together best document {subject}.\n"
        f"{brief.strip() or _SF_DEFAULT_BRIEF}\n"
        "Look at every tile individually and compare it against the others "
        "before deciding — do not default to the first frames in numeric "
        f"order (1, 2, 3, ...{count}); that is almost never the right answer "
        "and gets rejected.\n"
        'Reply with JSON only, no prose: {"picks": [numbers], "why": "one short sentence"}'
    )
    image = _sf_pils_to_tensor([board])
    tokens = clip.tokenize(instruction, image=image, min_length=1, thinking=thinking)
    ids = clip.generate(
        tokens, do_sample=float(temperature) > 0.0, max_length=int(max_length),
        temperature=float(temperature), top_k=64, top_p=0.95, min_p=0.05,
        repetition_penalty=1.05, seed=0,
    )
    return _sf_strip_thinking(clip.decode(ids))


_SF_LEVEL_LABELS = {
    "cluster": "Tier 1: cluster (by content)",
    "vision": "Tier 2: vision (clip, diversity-picked)",
    "sharpness_diversity": "Tier 3: sharpness+diversity (fallback)",
    "all": "kept (fewer frames in the video than requested)",
}


class H3LookSheetsSelectFrames:
    """Pick reference frames via a fixed cascade: cluster, then vision, then spread.

    No `mode` switch — every run does, in this fixed order:
    1. Cluster by visual content into `shots` groups, take the sharpest frame
       of each. Deterministic, always runs, guarantees one frame per
       distinct view regardless of whether a vision model is available.
    2. If slots remain (`count > shots`) and `clip` is wired, build a
       content-diverse shortlist from the untouched frames and fill the rest
       by picking whichever of those are most different from what's already
       chosen. `clip` is still asked to judge the shortlist (logged for
       visibility) but its answer does not drive the pick: testing showed it
       reliably defaults to "the first N candidates" regardless of prompt
       wording, candidate count, or thinking mode — not real comparison — so
       trusting it would mean never recovering frames from later in the clip.
    3. Whatever step 2 didn't fill — no `clip` connected, the call failed,
       or it under-returned — gets filled by sharpness+diversity, so the
       node always returns exactly `count` frames (or every frame, if the
       video has fewer than `count`).

    `debug` lists, for every returned frame, which of the three tiers chose
    it — so a run with `count == shots` reads "cluster" on every line, and
    you know at a glance whether `clip` ever got exercised at all.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "saved_frame_count": ("INT", {"default": 6, "min": 1, "max": 32,
                    "tooltip": "Total frames returned. count > shots leaves slots for vision/spread."}),
                # --- tier 1: cluster by content ---
                "tier1_shots_clusters": ("INT", {"default": 6, "min": 1, "max": 32,
                    "tooltip": "Distinct views to guarantee via clustering — tier 1."}),
            },
            "optional": {
                # --- tier 2: vision judge (only if clip is wired) ---
                "tier2_clip": ("CLIP", {"tooltip": "In-graph vision-language CLIP for tier 2. Leave unplugged to skip straight to tier 3 for any leftover slots."}),
                "tier2_candidates": ("INT", {"default": 16, "min": 4, "max": 32,
                    "tooltip": "Clusters formed from the leftover pool — also the shortlist size shown to the vision model. Each is represented by its sharpest frame, so this is both the diversity and the readability knob."}),
                "tier2_prompt_subject_description": ("STRING", {"default": "the subject", "multiline": True}),
                "tier2_prompt_how_to_select_frames": ("STRING", {"default": _SF_DEFAULT_BRIEF, "multiline": True}),
                "tier2_free_vram_first": ("BOOLEAN", {"default": True,
                    "tooltip": "Unload other resident models before the tier-2 vision call."}),
                "tier2_max_token_length": ("INT", {"default": 400, "min": 64, "max": 4096}),
                "tier2_temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "tier2_thinking": ("BOOLEAN", {"default": False,
                    "tooltip": "Let the vision model reason before answering. The reasoning block is stripped before parsing picks either way."}),
                # --- tier 3: sharpness + diversity fallback ---
                "tier3_sharpness_diversity": ("BOOLEAN", {"default": True,
                    "tooltip": "Fill leftover slots by sharpness+diversity when tier 2 doesn't cover them. Off returns fewer than saved_frame_count instead."}),
                "tier3_sharpness_weight": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "debug")
    FUNCTION = "select"
    CATEGORY = "H3LookSheets"

    @staticmethod
    def _free_vram():
        try:
            import comfy.model_management
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        except Exception as exc:
            logging.debug("[H3LookSheetsSelectFrames] could not free VRAM: %s", exc)

    @staticmethod
    def _debug(picked_sorted, origin, total, notes):
        counts: dict[str, int] = {}
        for i in picked_sorted:
            level = origin.get(i, "?")
            counts[level] = counts.get(level, 0) + 1
        summary = ", ".join(
            f"{n} {_SF_LEVEL_LABELS.get(lvl, lvl)}" for lvl, n in counts.items()
        )
        lines = [f"{len(picked_sorted)}/{total} frames — {summary}"]
        for i in picked_sorted:
            level = origin.get(i, "?")
            lines.append(f"  frame {i}: {_SF_LEVEL_LABELS.get(level, level)}")
        lines.extend(notes)
        return "\n".join(lines)

    def select(self, images, saved_frame_count, tier1_shots_clusters, tier2_clip=None,
               tier2_candidates=16, tier2_prompt_subject_description="the subject", tier2_prompt_how_to_select_frames=_SF_DEFAULT_BRIEF,
               tier2_free_vram_first=True,
               tier2_max_token_length=400, tier2_temperature=0.2, tier2_thinking=False,
               tier3_sharpness_diversity=True, tier3_sharpness_weight=0.35):
        total = int(images.shape[0])
        count = max(1, min(int(saved_frame_count), total))
        shots = max(1, min(int(tier1_shots_clusters), total))
        notes: list[str] = []

        if total <= count:
            picked = list(range(total))
            origin = {i: "all" for i in picked}
            return (images, self._debug(picked, origin, total, notes))

        origin: dict[int, str] = {}
        picked: list[int] = []

        def take(indices, level):
            for i in indices:
                if i not in origin and len(picked) < count:
                    origin[i] = level
                    picked.append(i)

        # Tier 1 — cluster by content, sharpest frame per cluster
        sharp = _sf_sharpness(images)
        groups = [g for g in _sf_cluster_by_content(images, shots) if g]
        forced = sorted({max(g, key=lambda f: float(sharp[f])) for g in groups})
        take(forced, "cluster")

        # Tier 2 — vision judge fills what's left, only if clip is wired
        remaining = count - len(picked)
        if remaining > 0:
            if tier2_clip is None:
                notes.append("tier 2 skipped: no clip connected")
            else:
                if tier2_free_vram_first:
                    self._free_vram()
                pool = [i for i in range(total) if i not in origin]
                shortlist = _sf_shortlist_diverse(images, sharp, pool, max(remaining, int(tier2_candidates)))
                notes.append(f"tier 2 shortlist shown to model ({len(shortlist)}): {shortlist}")
                try:
                    pils = _sf_tensor_to_pils(images[shortlist])
                    text = _sf_clip_pick(tier2_clip, pils, remaining, tier2_prompt_subject_description,
                                         tier2_prompt_how_to_select_frames, int(tier2_max_token_length), float(tier2_temperature),
                                         thinking=bool(tier2_thinking))
                    try:
                        json.loads(_sf_json_slice(text))
                        json_ok = True
                    except Exception:
                        json_ok = False
                    preview = text.strip().replace("\n", " ")[:300]
                    notes.append(
                        f"tier 2 raw reply (json_ok={json_ok}, {len(text)} chars): {preview}"
                        + ("..." if len(text) > 300 else "")
                    )
                    local_picks = _sf_parse_picks(text, len(pils))
                    if local_picks:
                        notes.append(f"tier 2: model picks (not used to choose — see below): {[shortlist[i] for i in local_picks]}")

                    # The model's own pick order is not trusted to choose which
                    # frames to take — testing showed it reliably defaults to
                    # "the first N candidates", regardless of prompt wording,
                    # candidate count, or thinking mode: not real comparison.
                    # Picking the most-different-from-what's-chosen frames out
                    # of this same (already content-diverse) shortlist is what
                    # actually recovers frames from later in the clip.
                    diverse_picks = _sf_fill_by_diversity(
                        images, shortlist, picked, remaining, tier3_sharpness_weight)
                    take(diverse_picks, "vision")
                except Exception as exc:
                    notes.append(f"tier 2 failed: {type(exc).__name__}")

        # Tier 3 — sharpness+diversity fills whatever tier 2 didn't
        remaining = count - len(picked)
        if remaining > 0:
            if not tier3_sharpness_diversity:
                notes.append(f"tier 3 disabled: returning {len(picked)}/{count} frames")
            else:
                pool = [i for i in range(total) if i not in origin]
                fill = _sf_fill_by_diversity(images, pool, picked, remaining, tier3_sharpness_weight)
                take(fill, "sharpness_diversity")

        picked_sorted = sorted(picked)
        return (images[picked_sorted], self._debug(picked_sorted, origin, total, notes))


class H3LookSheetsDatasheetSettings:
    """Lay the selected reference frames out as one contact-sheet image.

    Plain grid, no per-tile labels — `columns` per row, each tile resized to
    `columns_width` (aspect ratio kept from the first frame), `padding`
    pixels of gap on every side and between tiles.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 3, "min": 1, "max": 12}),
                "columns_width": ("INT", {"default": 384, "min": 64, "max": 2048}),
                "padding": ("INT", {"default": 8, "min": 0, "max": 128}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "build"
    CATEGORY = "H3LookSheets"

    def build(self, images, columns, columns_width, padding):
        pils = _sf_tensor_to_pils(images)
        sheet = _sf_montage(pils, columns=int(columns), cell_width=int(columns_width),
                            padding=int(padding))
        return (_sf_pils_to_tensor([sheet]),)


# --------------------------------------------------------------------------
# H3LookSheetsShotConfig / H3LookSheetsCustomPrompt
#
# A custom-shot alternative to H3LookSheetsPrompt's fixed 5/6-shot
# turnaround: any number of shots (up to 15), each with its own angle,
# framing and expression, instead of a hardcoded per-shot template.
#
# Autogrow (the "plug one, the next slot appears" mechanism) only works on
# socket inputs, and forces every widget it wraps into one too — three
# Autogrow'd combos per shot would mean 3x sockets and no ready-made node to
# feed a bare combo value into a socket. H3LookSheetsShotConfig sidesteps
# that: its own angle/framing/expression combos stay ordinary inline
# widgets (it isn't itself using Autogrow), and it packs all three into one
# string — so H3LookSheetsCustomPrompt only needs ONE Autogrow socket per
# shot, fed by one of these nodes, not three.
# --------------------------------------------------------------------------

_ANGLE_PHRASES = {
    "front": "facing the camera directly",
    "front 3/4 left": "turned three-quarters toward camera-left, most of the face and body visible",
    "front 3/4 right": "turned three-quarters toward camera-right, most of the face and body visible",
    "left profile": "turned to show the left profile",
    "right profile": "turned to show the right profile",
    "back 3/4 left": "turned mostly away, seen three-quarters from the back on the left side",
    "back 3/4 right": "turned mostly away, seen three-quarters from the back on the right side",
    "back": "facing away from the camera, back to camera",
}
_ANGLE_OPTIONS = list(_ANGLE_PHRASES.keys())

_FRAMING_PHRASES = {
    "extreme wide shot": "an extreme wide shot",
    "wide shot (full body)": "a full-body wide shot, the entire figure visible from head to toe",
    "medium wide shot (knees-up)": "a medium-wide shot from the knees up",
    "medium shot (waist-up)": "a medium shot from the waist up",
    "medium close-up (chest-up)": "a medium close-up from the chest up",
    "close-up (shoulders/face)": "a close-up of the shoulders and face",
    "extreme close-up (eyes/detail)": "an extreme close-up on the eyes and fine detail",
}
_FRAMING_OPTIONS = list(_FRAMING_PHRASES.keys())

_EXPRESSION_PHRASES = {
    "neutral": "neutral",
    "happy": "happy",
    "smiling": "warmly smiling",
    "sad": "sad",
    "angry": "angry",
    "surprised": "surprised, eyes wide",
    "scared": "frightened, eyes wide and brows raised",
    "disgusted": "disgusted",
    "shy/embarrassed": "shy, embarrassed",
    "confident": "confident",
    "serious": "serious",
    "laughing": "laughing, genuinely amused",
    "crying": "tearful",
    "smirking": "playfully smirking",
    "confused": "confused",
}
_EXPRESSION_OPTIONS = list(_EXPRESSION_PHRASES.keys())

#: Separator packed between the three combo values — chosen because none of
#: the option strings above contain it, so splitting is unambiguous.
_SHOT_CONFIG_SEP = "||"

#: How many shot_N sockets H3LookSheetsCustomPrompt exposes at once.
_MAX_CUSTOM_SHOTS = 15


class H3LookSheetsShotConfig:
    """One shot's angle, framing and expression, packed into one string.

    Feeds a single `shot_N` socket on H3LookSheetsCustomPrompt — see the
    module note above for why this exists instead of three plain combos.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "angle": (_ANGLE_OPTIONS,),
                "framing": (_FRAMING_OPTIONS,),
                "expression": (_EXPRESSION_OPTIONS,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("shot_config",)
    FUNCTION = "build"
    CATEGORY = "H3LookSheets"

    def build(self, angle, framing, expression):
        return (_SHOT_CONFIG_SEP.join((angle, framing, expression)),)


class H3LookSheetsCustomPrompt(io.ComfyNode):
    """Write H3's ref2va prompt for a custom, freely-shot-configured turnaround.

    Same <Picture 1>/<Picture 2> identity+outfit logic as H3LookSheetsPrompt,
    but the shot list itself is not fixed — each `shot_N` socket (fed by an
    H3LookSheetsShotConfig node) supplies its own angle, framing and
    expression, and however many are actually connected becomes the shot
    count (1 to 15). <Picture 1> is always anchored to whatever the first
    connected shot describes, so the header stays consistent with it even
    when shot 1 isn't the usual full-body/front/neutral default.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LookSheetsCustomPrompt",
            display_name="Look Sheet Prompt - Custom Shots (H3)",
            category="H3LookSheets",
            inputs=[
                io.String.Input("person_description", multiline=True, default=""),
                io.String.Input("outfit_description", multiline=True, default=""),
                io.Combo.Input("picture_1_gender", options=_PRONOUN_OPTIONS, optional=True),
                io.Combo.Input("picture_2_subject_type", options=_SUBJECT_TYPE_OPTIONS, optional=True),
                io.String.Input("backdrop", default="plain light neutral grey studio backdrop", multiline=True, optional=True),
                io.Float.Input("video_duration_seconds", default=5.0, min=1.0, max=60.0, step=0.1, optional=True,
                    tooltip="The take's actual length in seconds — feed this from whatever sets "
                    "`length` on MiniMaxH3ReferenceToVideo (e.g. a Duration node)."),
                io.Autogrow.Input(
                    "shots",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.String.Input("shot"),
                        prefix="shot_", min=1, max=_MAX_CUSTOM_SHOTS,
                    ),
                ),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, person_description, outfit_description, shots,
                picture_1_gender=_AUTO_PRONOUN, picture_2_subject_type=_AUTO_PRONOUN,
                backdrop="plain light neutral grey studio backdrop",
                video_duration_seconds=5.0) -> io.NodeOutput:
        resolved_pronoun = (
            _detect_pronoun(person_description)
            if picture_1_gender == _AUTO_PRONOUN else picture_1_gender
        )
        p = _PRONOUNS.get(resolved_pronoun, _NEUTRAL_PRONOUN)
        resolved_subject_type = (
            _detect_subject_type(outfit_description)
            if picture_2_subject_type == _AUTO_PRONOUN else picture_2_subject_type
        )
        subject_noun = _SUBJECT_NOUNS.get(resolved_subject_type, _NEUTRAL_SUBJECT_NOUN)
        person = person_description.strip().rstrip(".") or "the same figure"
        outfit = outfit_description.strip().rstrip(".") or "the outfit shown"
        set_dressing = backdrop.strip().rstrip(".") or "plain light neutral grey studio backdrop"

        parsed: list[tuple[str, str, str]] = []
        for value in (shots or {}).values():
            if not value:
                continue
            parts = value.split(_SHOT_CONFIG_SEP)
            if len(parts) != 3:
                continue
            parsed.append((parts[0], parts[1], parts[2]))

        total_shots = max(1, len(parsed))
        duration = float(video_duration_seconds)
        # Truncated to 1 decimal, not rounded: 0.833s -> 0.8s, never 0.83s
        # rounding up to overshoot the take's real length.
        step = max(0.1, math.floor((duration / total_shots) * 10) / 10)
        at = [_timecode(i * step) for i in range(total_shots)]
        shot_tags = ", ".join(f"[Shot {i + 1}]" for i in range(total_shots))
        shots_word = _COUNT_WORDS.get(total_shots, str(total_shots))

        if parsed:
            a0, f0, e0 = parsed[0]
        else:
            a0, f0, e0 = "front", "wide shot (full body)", "neutral"
        anchor_angle = _ANGLE_PHRASES.get(a0, a0)
        anchor_framing = _FRAMING_PHRASES.get(f0, f0)
        anchor_expression = _EXPRESSION_PHRASES.get(e0, e0)

        header = (
            f"subject_definitions:\n<Subject 1> {person}, and the same figure "
            f"and proportions established in <Picture 1>. {outfit} "
            f"from the <Picture 2>, the garment tailored to {p['poss']} own "
            "body shape and proportions from <Picture 1>, not the proportions "
            "in <Picture 2>. <Picture 1> hairstyle is transferred to <Subject "
            f"1> — none of the hair, hairstyle, or hair texture of "
            f"{subject_noun} shown in <Picture 2> is carried over.\n"
            "<Picture 1> is the first frame of [Shot 1], showing <Subject 1> "
            f"in {anchor_framing}, {anchor_angle}, with a {anchor_expression} "
            f"expression, against {set_dressing}."
        )

        summary = (
            "summary:\n[reference generation] The target video presents "
            f"<Subject 1> in a {duration:g}-second sequence of "
            f"{total_shots} static shots, retaining only {p['poss']} hair, "
            "body shape, skin, and facial structure from <Picture 1>, while "
            f"preserving the outfit from <Picture 2>. The background remains "
            f"{set_dressing}. Framing, angle and expression change shot to "
            "shot as described below; pose is otherwise held still within "
            "each shot."
        )

        retention = (
            "retention_analysis:\n"
            f"<Subject 1> (appears in {shot_tags}): partially_preserved - "
            "retains only hair, hairstyle, body shape, skin, and facial "
            "structure; no clothing or movement preserved.\n"
            f"<Picture 2> (appears in {shot_tags}): partially_preserved - "
            "retains only the outfit; no other visual elements or props "
            "preserved."
        )

        # Same load-bearing stillness clause as H3LookSheetsPrompt — without
        # it H3 tends to render windswept/flying hair between cuts.
        intro = (
            f"The target video uses a static studio lighting setup against "
            f"{set_dressing}. All shots are framed with generous empty "
            "margins on every side, ensuring the entire figure and any "
            "extensions remain fully within the frame without touching or "
            f"crossing edges. {set_dressing[0].upper()}{set_dressing[1:]} "
            "and its bright, even lighting stay completely identical across "
            f"all {shots_word} shots. No text, watermark, logo, caption, or "
            "writing of any kind appears anywhere in the video. Between every "
            "shot, only the camera's position around <Subject 1> changes to "
            "capture each new angle — <Subject 1> remains completely "
            "motionless throughout, holding one exact pose without turning, "
            "walking, gesturing, or otherwise moving between cuts, and hair "
            "stays at rest in that pose, undisturbed by any camera movement."
        )

        shot_lines = []
        for i, (angle, framing, expression) in enumerate(parsed):
            framing_text = _FRAMING_PHRASES.get(framing, framing)
            angle_text = _ANGLE_PHRASES.get(angle, angle)
            expr_text = _EXPRESSION_PHRASES.get(expression, expression)
            if i == 0:
                opener = "[Shot 1] "
            else:
                opener = f"[Shot {i + 1}] At {at[i]}, the shot cuts to "
            shot_lines.append(
                f"{opener}{framing_text} of <Subject 1>, {angle_text}, "
                f"with a {expr_text} expression."
            )
        detail = "detailed_description:\n" + intro + "\n" + "\n".join(shot_lines)

        prompt = (
            f"{header}\n\n{summary}\n\n{retention}\n\n{detail}\n\n"
            "overall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A"
        )
        return io.NodeOutput(prompt)


NODE_CLASS_MAPPINGS = {
    "H3LookSheetsPrompt": H3LookSheetsPrompt,
    "H3LookSheetsDescribe": H3LookSheetsDescribe,
    "H3LookSheetsSelectFrames": H3LookSheetsSelectFrames,
    "H3LookSheetsDatasheetSettings": H3LookSheetsDatasheetSettings,
    "H3LookSheetsShotConfig": H3LookSheetsShotConfig,
    "H3LookSheetsCustomPrompt": H3LookSheetsCustomPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LookSheetsPrompt": "Look Sheet Prompt (H3)",
    "H3LookSheetsDescribe": "Describe Reference (H3 Look Sheet)",
    "H3LookSheetsSelectFrames": "Select Frames (H3 Look Sheet)",
    "H3LookSheetsDatasheetSettings": "Datasheet Settings (H3 Look Sheet)",
    "H3LookSheetsShotConfig": "Shot Config (H3 Look Sheet)",
    "H3LookSheetsCustomPrompt": "Look Sheet Prompt - Custom Shots (H3)",
}
