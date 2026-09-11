"""Prompt construction for MiniMax H3's ref2va (reference-to-video) mode.

Builds the multi-reference prompt H3's guide expects when conditioning
`MiniMaxH3ReferenceToVideo` on a person (<Picture 1>) and a look to dress
them in — one photo (<Picture 2>) or several (<Picture 2>, <Picture 3>...
for a look shown across front/back/shoes etc). The vision-language
description of each image stays outside this node — plug in
`H3LookSheetsDescribe` nodes (target=person for the one person photo,
target=outfit for one or several outfit photos) and feed their generated
text in here, along with `outfit_image_count` when the outfit spans more
than one photo. This node only assembles the `subject_definitions` /
`summary` / `retention_analysis` / `detailed_description` structure ref2va
expects around those descriptions.
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


def _picture_tags(start: int, count: int) -> list[str]:
    """`<Picture start>`, `<Picture start+1>`, ... `count` of them.

    <Picture 1> is always the person; the outfit starts at <Picture 2> and
    spans `count` consecutive tags when it's described across several
    photos (front, back, shoes...). `count` must match, in order, the
    outfit images wired into MiniMaxH3ReferenceToVideo's ref_image sockets
    — the person's photo first, then the outfit photos in the same order
    they were described here. H3 numbers <Picture i> by connection order on
    that node, not by anything this node knows about, so the two have to be
    kept in sync by hand.
    """
    return [f"<Picture {start + i}>" for i in range(count)]


def _join_tags_english(tags: list[str]) -> str:
    """Oxford-style join for a list of `<Picture N>` tags in running prose."""
    if len(tags) == 1:
        return tags[0]
    if len(tags) == 2:
        return f"{tags[0]} and {tags[1]}"
    return ", ".join(tags[:-1]) + f" and {tags[-1]}"


def _parse_multi_description(text: str) -> list[str]:
    """Unwrap H3LookSheetsDescribe's `{"outfit_0": "...", "outfit_1": "..."}`
    JSON into an ordered list of per-picture description strings.

    Falls back to `[text]` for anything that isn't a JSON object — a plain,
    manually typed description (or a single H3LookSheetsDescribe entry that
    was pasted in bare) still works as one picture, same as before this
    node understood the JSON shape at all.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return [str(v) for v in parsed.values()]
    return [text]


def _fill_outfit_descriptions(items: list[str]) -> list[str]:
    """Clean each entry, substituting a fallback for a blank one instead of
    dropping it.

    A blank entry (H3LookSheetsDescribe couldn't get a usable caption for
    that specific picture, even after its own retries/fallback) must still
    occupy its slot in the list: `outfit_tags`/`outfit_ref` are sized off
    `len(outfit_items)`, and that count has to keep matching the outfit
    images actually wired into MiniMaxH3ReferenceToVideo — dropping an
    empty entry would silently shift every later <Picture N> tag out of
    sync with the images downstream.
    """
    cleaned = [t.strip().rstrip(".") or "the outfit shown" for t in items]
    return cleaned or ["the outfit shown"]


class H3LookSheetsPrompt:
    """Write H3's ref2va prompt for a 6-shot, all-neutral look turnaround.

    Two references go in — a person (<Picture 1>) and a look to dress them in
    (<Picture 2>, or <Picture 2>, <Picture 3>... when the outfit is shown
    across several photos). `person_description`/`outfit_description` take
    either plain text or H3LookSheetsDescribe's JSON output directly — a
    JSON object there is unwrapped into one description per picture
    automatically, so the number of <Picture N> tags for the outfit always
    matches how many entries were in it, no separate count input needed.
    The prompt asks H3 for a 6-shot turnaround that keeps only <Picture 1>'s
    identity (hair,
    body shape, skin, facial structure) and only the outfit pictures'
    clothing, discarding everything else either reference carries (the
    outfit pictures' own hair, body shape and proportions in particular):
    [Shot 1] full body -> [Shot 2] face close-up -> [Shot 3] left profile ->
    [Shot 4] right profile -> [Shot 5] back -> [Shot 6] a second, closer face
    shot. Every shot is neutral; for other expressions or a freely chosen
    shot list, use H3LookSheetsCustomPrompt instead.

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
        person_items = [t.strip().rstrip(".") for t in _parse_multi_description(person_description) if t.strip()]
        outfit_items = _fill_outfit_descriptions(_parse_multi_description(outfit_description))

        resolved_pronoun = (
            _detect_pronoun(person_items[0] if person_items else "")
            if picture_1_gender == _AUTO_PRONOUN else picture_1_gender
        )
        p = _PRONOUNS.get(resolved_pronoun, _NEUTRAL_PRONOUN)
        resolved_subject_type = (
            _detect_subject_type(" ".join(outfit_items))
            if picture_2_subject_type == _AUTO_PRONOUN else picture_2_subject_type
        )
        subject_noun = _SUBJECT_NOUNS.get(resolved_subject_type, _NEUTRAL_SUBJECT_NOUN)
        person = person_items[0] if person_items else "the same figure"
        set_dressing = backdrop.strip().rstrip(".") or "plain light neutral grey studio backdrop"
        outfit_tags = _picture_tags(2, len(outfit_items))
        outfit_ref = _join_tags_english(outfit_tags)
        # Each picture gets its own description + its own "from <Picture N>"
        # tag, rather than one shared outfit sentence — the outfit pictures
        # can be genuinely different garments, not just different angles of
        # the same one.
        outfit_fragment = ", ".join(
            f"{text} from {tag}" for text, tag in zip(outfit_items, outfit_tags)
        )

        total_shots = 6
        duration = float(video_duration_seconds)
        # Truncated to 1 decimal, not rounded: 0.833s -> 0.8s, never 0.83s
        # rounding up to overshoot the take's real length.
        step = max(0.1, math.floor((duration / total_shots) * 10) / 10)
        at = [_timecode(i * step) for i in range(total_shots)]
        shot_tags = ", ".join(f"[Shot {i + 1}]" for i in range(total_shots))

        header = (
            f"subject_definitions:\n<Subject 1> {person}, and the same figure "
            f"and proportions established in <Picture 1>. {outfit_fragment}, "
            f"the garment tailored to {p['poss']} own "
            "body shape and proportions from <Picture 1>, not the proportions "
            f"in {outfit_ref}. <Picture 1> hairstyle is transferred to <Subject "
            f"1> — none of the hair, hairstyle, or hair texture of "
            f"{subject_noun} shown in {outfit_ref} is carried over.\n"
            "<Picture 1> is the first frame of [Shot 1], showing <Subject 1> "
            f"full-body, facing the camera with a neutral expression against "
            f"{set_dressing}."
        )

        summary = (
            "summary:\n[reference generation] The target video presents "
            f"<Subject 1> in a {duration:g}-second sequence of "
            f"{total_shots} static shots, retaining only {p['poss']} hair, "
            "body shape, skin, and facial structure from <Picture 1>, while "
            f"preserving the outfit from {outfit_ref}. The background remains "
            f"{set_dressing}. All shots maintain identical framing, lighting, "
            "and pose, every expression neutral throughout."
        )

        outfit_retention = "\n".join(
            f"{tag} (appears in {shot_tags}): partially_preserved - "
            "retains only the outfit; no other visual elements or props "
            "preserved."
            for tag in outfit_tags
        )
        retention = (
            "retention_analysis:\n"
            f"<Subject 1> (appears in {shot_tags}): partially_preserved - "
            "retains only hair, hairstyle, body shape, skin, and facial "
            "structure; no clothing or movement preserved.\n"
            f"{outfit_retention}"
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
            "writing of any kind appears anywhere in the video. "
            f"<Subject 1> wears the outfit and accessories from {outfit_ref} "
            f"continuously from the very first frame through all {shots_word} "
            "shots, without any garment changing, shifting, or being "
            "removed. Between every shot, only the camera's position around "
            "<Subject 1> changes to "
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
        "body shape from the provided image, ignore outfit and clothes, and "
        "ignore the background or setting entirely. \n"
        "Very short description, only 1 sentence."
    ),
    "outfit": (
        "Analyzes the outfit, only describe the outfit, specifically "
        "detailing the clothing and accessories, skip hair description. "
        "Very short description, only 1 sentence"
    ),
}

#: Outfit conflict resolution: when two outfit photos both happen to show
#: e.g. footwear, the generic "describe the whole outfit" prompt makes each
#: image mention its own pair, and H3's prompt ends up listing two different
#: pairs of shoes for the same subject with no way to tell which to wear.
#: These per-category `..._image_front`/`..._image_back` combos (shown for
#: target=outfit) let the user assign which connected image is the source
#: for each garment category — and, since a garment like a dress can look
#: different front and back, separately for its front and back view.
#: Assigning any of them switches every connected image's system prompt to
#: an exclusion-aware one — the assigned image is told to describe *only*
#: its assigned categories (and from which side, if that matters), and
#: every other image is told to skip those same categories — so no two
#: images ever claim the same category. Leaving all of them at "auto" (the
#: default) keeps the original single generic "describe the whole outfit"
#: prompt per image, unchanged. This only feeds the description text itself
#: (`subject_definitions`) — which shots the final video actually needs a
#: back view for is a separate concern the prompt-building nodes handle on
#: their own, unrelated to this.
_OUTFIT_CATEGORIES = ["top", "bottom", "shoes", "accessories"]
_OUTFIT_VIEWS = ["front", "back"]
#: Only garments whose silhouette can meaningfully differ front vs back (a
#: top, a skirt/dress bottom) get separate front/back source fields — shoes
#: and accessories keep a single field each; a back-of-the-shoe or
#: back-of-the-bag detail isn't worth doubling their field count for.
_CATEGORIES_WITH_VIEW = {"top", "bottom"}


def _outfit_field_id(cat: str, view: str) -> str:
    """The combo's input id for this category/view — `{cat}_image_{view}`
    for a category that supports the front/back split, plain `{cat}_image`
    (view-less) otherwise."""
    return f"{cat}_image_{view}" if cat in _CATEGORIES_WITH_VIEW else f"{cat}_image"


_CATEGORY_PHRASES = {
    "top": "top / upper-body garment (shirt, blouse, jacket...)",
    "bottom": "bottom / lower-body garment (pants, skirt...)",
    "shoes": "footwear (shoes, boots)",
    "accessories": "accessories (bag, belt, jewelry, headwear, glasses...)",
}

#: Extra, category-specific instructions appended only when that category is
#: this image's own to describe — a "black boots" caption is ambiguous
#: between an ankle boot and a thigh-high one, and that ambiguity is exactly
#: what leads the final video generation to render ordinary shoes with
#: separate hosiery instead of one continuous over-the-knee boot (or vice
#: versa). Not every category needs this; only add an entry where the
#: category can vary in a way plain naming doesn't capture.
_CATEGORY_EXTRA_GUIDANCE = {
    "shoes": (
        "If it extends up the leg, state where it starts and how high it "
        "reaches (ankle, calf, knee, or mid-thigh)."
    ),
    "accessories": (
        "For a hoop, bangle, ring, or chain, describe the thickness of the "
        "material AND its size relative to a visible body part it's worn "
        "against or near (e.g. \"a hoop barely wider than the earlobe, "
        "sitting tight against it\" vs \"a hoop hanging well past the "
        "earlobe, several centimeters wide\"). For a headband, hairpin, "
        "barrette, or other hair accessory, state how far it extends "
        "across the head (e.g. \"a full headband spanning ear to ear\" vs "
        "\"a small clip on one side only, a few centimeters wide\") and "
        "where it sits (right at the hairline vs further back on the "
        "crown). A vague adjective like \"small\" or \"large\" alone, "
        "with nothing to compare it to, is not a usable size and leaves "
        "the actual extent guessed. For glasses/eyewear, describe the "
        "frame shape (round, square, cat-eye...), color/material, and "
        "how thick or thin the frame is. If more than one accessory item "
        "is visible (e.g. glasses worn together with a ring or a pin on "
        "the frame), describe every one of them, not just the most "
        "eye-catching one."
    ),
}


def _outfit_source_options(max_images: int) -> list[str]:
    """Same names as the `images` Autogrow's own sockets (`image_0`,
    `image_1`...) — picking a source this way is a direct match against the
    actual socket you connected, no separate 1-based/0-based numbering to
    keep straight."""
    return ["auto"] + [f"image_{i}" for i in range(max_images)]


def _outfit_wanted_phrase(categories_here: list[tuple[str, str]]) -> str:
    """English phrase for what this image should describe, e.g. "the top /
    upper-body garment (...)" (front, the common case), "the back of the
    footwear (...)" (back), or both joined when this image was assigned a
    mix — grouped by view since a single photo is realistically all-front
    or all-back (a camera can't shoot both sides of a person at once), even
    though different categories can each come from a different photo.
    """
    front_cats = [c for c, v in categories_here if v == "front"]
    back_cats = [c for c, v in categories_here if v == "back"]
    parts = []
    if front_cats:
        parts.append("the " + _join_tags_english([_CATEGORY_PHRASES[c] for c in front_cats]))
    if back_cats:
        parts.append("the back of the " + _join_tags_english([_CATEGORY_PHRASES[c] for c in back_cats]))
    return " and ".join(parts)


def _outfit_system_prompt(base_prompt: str, categories_here: list[tuple[str, str]],
                           categories_elsewhere: list[str]) -> str:
    """Build this image's outfit system prompt given the category split.

    `categories_here` is a list of `(category, view)` pairs assigned to
    this specific image; `categories_elsewhere` is just category names
    (view doesn't matter for exclusion — a category assigned to another
    image, front or back, is skipped here either way).

    No split at all (both empty) returns `base_prompt` verbatim — the
    original, single generic outfit description.
    """
    if categories_here:
        wanted = _outfit_wanted_phrase(categories_here)
        cats_only = {c for c, _ in categories_here}
        extra = " ".join(_CATEGORY_EXTRA_GUIDANCE[c] for c in cats_only if c in _CATEGORY_EXTRA_GUIDANCE)
        extra_clause = f" {extra}" if extra else ""
        return (
            f"Describe {wanted} shown in the image. Skip everything else "
            "the person is wearing, and skip hair. If it's ambiguous (e.g. a "
            "thigh-high boot that looks like hosiery), describe it anyway "
            f"rather than saying nothing is visible.{extra_clause} One short "
            "sentence only, no explanation."
        )
    if categories_elsewhere:
        skip = _join_tags_english([_CATEGORY_PHRASES[c] for c in categories_elsewhere])
        base = base_prompt.strip()
        if not base.endswith((".", "!", "?")):
            base += "."
        return (
            f"{base} Skip the {skip} — already described from a different "
            "reference photo; describe only what else this image shows of "
            "the outfit."
        )
    return base_prompt


def _outfit_fallback_prompt(categories_here: list[tuple[str, str]]) -> str:
    """A bare-bones, no-frills version of the scoped prompt — used only when
    the full scoped prompt (with its exclusion/ambiguity clauses) fails
    outright after retries. Deliberately minimal: the more clauses a prompt
    carries, the more this particular captioning model seems to misread one
    of them as a reason to refuse (e.g. "don't mention any other category"
    got read as "don't use category words at all", triggering a refusal on
    the word "skirt"). Still scoped to `categories_here` — never the fully
    generic whole-outfit prompt, which would throw away the category split
    entirely and reintroduce the original conflicting-items problem.
    """
    return f"Describe {_outfit_wanted_phrase(categories_here)} shown in the image, in one short sentence."


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


#: A genuine "very short, 1 sentence" description doesn't run this long or
#: span this many paragraphs. A small captioning VLM given a longer,
#: multi-clause system prompt (e.g. the category-scoped outfit prompts,
#: which carry conditional "if X, you MUST..." instructions) can start
#: reasoning out loud through the answer field instead of just answering —
#: narrating the instructions back, working through the image, and only
#: then giving a real sentence buried at the end. That text is neither
#: empty nor a refusal, so it would otherwise sail past `_is_bad_output`
#: and get used verbatim, garbled conclusion and all.
_MAX_REASONABLE_LENGTH = 320


def _is_bad_output(text: str) -> bool:
    text = text.strip()
    if not text or bool(_REFUSAL_PATTERN.search(text)):
        return True
    return len(text) > _MAX_REASONABLE_LENGTH or text.count("\n") >= 2


#: Small chat-tuned models occasionally echo the chat-template's own role
#: tag ("assistant") as literal generated text instead of stopping right
#: after it — a known quirk, not something specific to this node's prompt.
_LEADING_ROLE_TAG = re.compile(r"^\s*(assistant|system|user)\s*[:\n]+\s*", re.IGNORECASE)


def _strip_role_leak(text: str) -> str:
    return _LEADING_ROLE_TAG.sub("", text, count=1)


class H3LookSheetsDescribe(TextGenerate):
    """Describe a Look Sheet reference photo (or several) in one sentence each.

    Wraps core's `TextGenerate` (same tokenize/generate/decode call, reused
    via `super().execute()`) with the fixed system prompt H3LookSheetsPrompt
    expects for whichever reference this is — `target=person` for <Picture
    1>, `target=outfit` for <Picture 2> onward.

    `person` only ever needs `image_1` — a person reference is exactly one
    photo. `outfit` can use several sockets (`image_1`, `image_2`...); each
    is described separately, and `description` comes back as one JSON
    string keyed by `target`, `{"outfit_1": "...", "outfit_2": "...", ...}`
    (or `{"person_1": "..."}`) — never merged into one paragraph, so a set
    of genuinely different outfits (or different pieces of the same look)
    each keep their own description.

    Wire `description` straight into H3LookSheetsPrompt/CustomPrompt's
    `person_description`/`outfit_description` — those nodes unwrap this JSON
    themselves and write one `<Picture N>` fragment per entry automatically.
    To pull a single entry out for anything else, use core's "Extract Text
    from JSON" node (`json_string` = this output, `key` = `"outfit_1"`,
    `"outfit_2"`...).

    Connect the images in the exact order you'll wire them, unchanged, into
    MiniMaxH3ReferenceToVideo's ref_image sockets, right after the person's —
    H3 numbers <Picture i> by connection order on that node, not by anything
    this node can see.

    `debug` is the second output — for every connected image, the exact
    system prompt sent to the captioning VLM and its raw, unfiltered output
    for every attempt (including retries), so a refusal, an empty decode, or
    an unexpectedly-scoped category prompt can be diagnosed directly instead
    of guessed at from the final `description` alone.

    Choosing target=outfit reveals a source combo per garment category —
    `top` and `bottom` each get an `..._image_front` and an `..._image_back`
    (a top or a skirt/dress bottom can look different from behind), while
    `shoes` and `accessories` get one plain `..._image` each (not worth
    doubling for a back-of-the-shoe/bag detail). Each combo picks, by its
    exact socket name (`image_0`, `image_1`...), which connected image is
    the source. They default to "auto" and can be left alone entirely —
    nothing changes unless at least one is set. Once one is set, every
    connected image's prompt switches to an exclusion-aware one: an image
    assigned a category (front or back, for top/bottom) is told to describe
    *only* that category — and from the back specifically, when that's what
    was assigned — while every other image is told to skip it, so two
    outfit photos that both happen to show footwear never both end up
    describing a pair of shoes in the final prompt. Set `..._back` only
    when a distinct back-view photo exists and its design actually differs
    from the front (e.g. a dress with a different back) — the resulting
    text just becomes another entry in the JSON output, described from the
    back; nothing here decides which video shot gets to use it.
    """

    #: MiniMaxH3ReferenceToVideo caps ref_images at 9; one of those slots is
    #: always the person, leaving at most 8 for the outfit. Matching that cap
    #: here (rather than the tokenizer's own limit) keeps the two nodes'
    #: maximums aligned so a maxed-out outfit here never overflows the wiring
    #: downstream.
    MAX_IMAGES = 9

    @classmethod
    def define_schema(cls):
        parent = super().define_schema()
        inputs = []
        for inp in parent.inputs:
            if inp.id == "prompt":
                def _outfit_field_tooltip(cat: str, view: str) -> str:
                    source = (f"shows the {view} view of the {_CATEGORY_PHRASES[cat]}"
                              if cat in _CATEGORIES_WITH_VIEW
                              else f"is the source for the {_CATEGORY_PHRASES[cat]}")
                    back_note = (" Only set `..._back` when a back-view photo exists and its "
                                 "design differs from the front (e.g. a dress with a different "
                                 "back)." if cat in _CATEGORIES_WITH_VIEW else "")
                    return (
                        f"Which connected outfit image {source} — same socket name as in "
                        "`images` above (e.g. `image_0` is the first image socket). \"auto\" "
                        "leaves this unassigned — harmless with a single outfit image, but "
                        "with several, leaving every category at \"auto\" means each image "
                        "just describes the whole outfit on its own, which can produce two "
                        "conflicting items (e.g. two different pairs of shoes) if more than "
                        f"one image shows the same category.{back_note}"
                    )

                target_options = [
                    io.DynamicCombo.Option(key="person", inputs=[]),
                    io.DynamicCombo.Option(key="outfit", inputs=[
                        io.Combo.Input(
                            _outfit_field_id(cat, view),
                            options=_outfit_source_options(cls.MAX_IMAGES),
                            default="auto", optional=True,
                            tooltip=_outfit_field_tooltip(cat, view),
                        )
                        for cat in _OUTFIT_CATEGORIES
                        for view in (_OUTFIT_VIEWS if cat in _CATEGORIES_WITH_VIEW else ["front"])
                    ]),
                ]
                inputs.append(io.DynamicCombo.Input(
                    "target", options=target_options, display_name="target",
                    tooltip="Which reference these image(s) are: the person "
                    "(<Picture 1>, always image_1 only) or the outfit "
                    "(<Picture 2> onward — use image_2, image_3... for a "
                    "look shown across several photos). Choosing outfit "
                    "reveals per-category source fields below.",
                ))
            elif inp.id == "image":
                inputs.append(io.Autogrow.Input(
                    "images",
                    tooltip="One photo per socket, described separately and "
                    "joined in order. Connect these in the exact order "
                    "you'll wire them into MiniMaxH3ReferenceToVideo's "
                    "ref_image sockets.",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_", min=1, max=cls.MAX_IMAGES,
                    ),
                ))
            elif inp.id in ("video", "audio"):
                # This node only ever describes still reference photos.
                continue
            else:
                inputs.append(inp)
        return io.Schema(
            node_id="H3LookSheetsDescribe",
            display_name="Describe Reference (H3 Look Sheet)",
            category="H3LookSheets",
            inputs=inputs,
            outputs=[
                io.String.Output(display_name="description",
                    tooltip='JSON object, one key per connected image, keyed by target: '
                    '{"outfit_1": "...", "outfit_2": "...", ...}. Not merged — wire straight '
                    'into H3LookSheetsPrompt/CustomPrompt, or pull one entry out with core\'s '
                    '"Extract Text from JSON" node.'),
                io.String.Output(display_name="debug",
                    tooltip="Per connected image: the exact system prompt sent to the captioning "
                    "VLM and its raw, unfiltered output for every attempt (including retries) — "
                    "for diagnosing refusals, empty outputs, or unexpected category scoping."),
            ],
        )

    #: Retried on: an empty decode (first sampled token landed on
    #: end-of-sequence — bad luck of that seed draw) or a refusal (the vision
    #: model treating an ordinary photo as sensitive content). Neither is
    #: tied to a specific seed value; a different seed often just works.
    MAX_EMPTY_RETRIES = 3
    _RETRY_SEED_STRIDE = 104729  # an arbitrary large prime, just to jump seeds

    @classmethod
    def _describe_one(cls, clip, system_prompt, target, max_length, sampling_mode,
                       image, thinking, use_default_template) -> tuple[str, list[dict]]:
        """Returns `(cleaned_text, attempts)` — `attempts` logs, for every
        try (including retries), the seed used and the model's raw,
        unfiltered output (before `_strip_role_leak`), for the `debug`
        output."""
        mode = sampling_mode
        base_seed = mode.get("seed") if isinstance(mode, dict) else None
        attempts: list[dict] = []
        text = ""

        for attempt in range(cls.MAX_EMPTY_RETRIES + 1):
            out = super().execute(
                clip, system_prompt, max_length, mode, image=image,
                thinking=thinking, use_default_template=use_default_template,
            )
            raw = out.args[0] if out.args else ""
            attempts.append({"attempt": attempt, "seed": mode.get("seed") if isinstance(mode, dict) else None, "raw": raw})
            text = _strip_role_leak(raw)
            if not _is_bad_output(text):
                return text, attempts
            if base_seed is None or attempt == cls.MAX_EMPTY_RETRIES:
                # No seed to vary (sampling is off), or retries exhausted.
                break
            mode = dict(mode)
            mode["seed"] = (base_seed + (attempt + 1) * cls._RETRY_SEED_STRIDE) % 0xffffffffffffffff
            logging.warning(
                "[H3LookSheetsDescribe] %s for target=%s (seed %s), "
                "retrying with seed %s (attempt %d/%d)",
                "empty output" if not text.strip() else "refusal/too verbose",
                target, base_seed, mode["seed"], attempt + 1, cls.MAX_EMPTY_RETRIES,
            )

        # Exhausted retries on this one image — skip it rather than fail the
        # whole batch; the other images in the set still describe fine.
        logging.warning(
            "[H3LookSheetsDescribe] still bad output for target=%s after "
            "retries — skipping this image.", target,
        )
        return "", attempts

    @classmethod
    def execute(cls, clip, target, max_length, sampling_mode, images,
                thinking=False, use_default_template=True) -> io.NodeOutput:
        target_key = target.get("target") if isinstance(target, dict) else target
        base_prompt = _DESCRIBE_SYSTEM_PROMPTS[target_key]
        ordered = [(key, img) for key, img in (images or {}).items() if img is not None]
        if not ordered:
            raise ValueError("H3LookSheetsDescribe needs at least one image")

        # cat -> {"front": socket, "back": socket}. Keyed by the exact
        # `images` socket name ("image_0", "image_1"...) — same names shown
        # in the `..._image_front`/`..._image_back` combos, so a choice
        # there matches a connected socket directly, no separate numbering
        # to keep in sync.
        category_assignment: dict[str, dict[str, str]] = {}
        if target_key == "outfit" and isinstance(target, dict):
            for cat in _OUTFIT_CATEGORIES:
                for view in (_OUTFIT_VIEWS if cat in _CATEGORIES_WITH_VIEW else ["front"]):
                    choice = target.get(_outfit_field_id(cat, view), "auto")
                    if choice and choice != "auto":
                        category_assignment.setdefault(cat, {})[view] = choice
        # A category pointed at a socket that isn't currently connected (the
        # image was disconnected/bypassed after the combo was set, shifting
        # every later image_N down) is dropped rather than raised — it's a
        # routine thing to hit while iterating on a workflow, and erroring
        # the whole node over one stale combo is disproportionate. The
        # image just falls back to describing whatever's left unassigned.
        connected_keys = {key for key, _ in ordered}
        for cat in list(category_assignment.keys()):
            views = category_assignment[cat]
            for view in list(views.keys()):
                socket = views[view]
                if socket not in connected_keys:
                    logging.warning(
                        "[H3LookSheetsDescribe] %s is set to \"%s\", but that socket isn't "
                        "connected (connected: %s) — ignoring this assignment.",
                        _outfit_field_id(cat, view), socket, sorted(connected_keys),
                    )
                    del views[view]
            if not views:
                del category_assignment[cat]

        # Keyed by target + position among connected images ("outfit_0",
        # "outfit_1"...), not the raw "image_N" socket id — a socket
        # disconnected mid-list (e.g. image_2 unplugged while image_3 stays
        # connected) would otherwise leave a gap ("outfit_0", "outfit_1",
        # "outfit_3") instead of just shifting everything after it down by
        # one, which is what H3LookSheetsPrompt/CustomPrompt's sequential
        # <Picture N> numbering already assumes. Every connected socket
        # keeps its key in the output, even one that fails after retries
        # (empty string), so a downstream JSON lookup never has to guess
        # which keys made it through.
        descriptions = {}
        debug_blocks = []
        for position, (key, image) in enumerate(ordered):
            categories_here = [(cat, view) for cat, views in category_assignment.items()
                               for view, socket in views.items() if socket == key]
            categories_elsewhere = sorted({
                cat for cat, views in category_assignment.items()
                if views and not any(socket == key for socket in views.values())
            })
            system_prompt = _outfit_system_prompt(base_prompt, categories_here, categories_elsewhere) \
                if target_key == "outfit" else base_prompt
            out_key = f"{target_key}_{position}"
            text, attempts = cls._describe_one(
                clip, system_prompt, target_key, max_length, sampling_mode,
                image, thinking, use_default_template,
            )
            # The scoped prompt's own exclusion/ambiguity clauses can make
            # the captioning VLM refuse outright, even after seed retries
            # (one clause gets misread as a reason to refuse). Falling back
            # to a bare, clause-free version of the SAME scoped prompt
            # trades a bit of precision for actually getting a usable
            # description — never the fully generic whole-outfit prompt,
            # which would throw away the category split entirely and bring
            # back the original conflicting-items problem this is meant to
            # avoid.
            if not text and categories_here:
                logging.warning(
                    "[H3LookSheetsDescribe] category-scoped prompt for %s (%s) produced no "
                    "usable output after retries — retrying with a simpler scoped prompt "
                    "for this image.", out_key,
                    ", ".join(f"{cat}:{view}" for cat, view in categories_here),
                )
                text, fallback_attempts = cls._describe_one(
                    clip, _outfit_fallback_prompt(categories_here), target_key, max_length,
                    sampling_mode, image, thinking, use_default_template,
                )
                attempts += fallback_attempts
            descriptions[out_key] = text

            attempts_text = "\n".join(
                f"  attempt {a['attempt']} (seed {a['seed']}): {a['raw']!r}" for a in attempts
            )
            debug_blocks.append(
                f"=== {out_key} ({key}) ===\n"
                f"--- system prompt ---\n{system_prompt}\n"
                f"--- raw output(s) ---\n{attempts_text}"
            )

        debug = "\n\n".join(debug_blocks)

        if not any(descriptions.values()):
            logging.warning(
                "[H3LookSheetsDescribe] every image failed for target=%s — "
                "returning empty descriptions.", target_key,
            )
        return io.NodeOutput(json.dumps(descriptions, ensure_ascii=False), debug)


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

    Same <Picture 1>/outfit-pictures identity+outfit logic as
    H3LookSheetsPrompt, including automatic multi-image outfits from
    H3LookSheetsDescribe's JSON output — see that class's docstring — but
    the shot list itself is not fixed — each `shot_N` socket (fed by an
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
        person_items = [t.strip().rstrip(".") for t in _parse_multi_description(person_description) if t.strip()]
        outfit_items = _fill_outfit_descriptions(_parse_multi_description(outfit_description))

        resolved_pronoun = (
            _detect_pronoun(person_items[0] if person_items else "")
            if picture_1_gender == _AUTO_PRONOUN else picture_1_gender
        )
        p = _PRONOUNS.get(resolved_pronoun, _NEUTRAL_PRONOUN)
        resolved_subject_type = (
            _detect_subject_type(" ".join(outfit_items))
            if picture_2_subject_type == _AUTO_PRONOUN else picture_2_subject_type
        )
        subject_noun = _SUBJECT_NOUNS.get(resolved_subject_type, _NEUTRAL_SUBJECT_NOUN)
        person = person_items[0] if person_items else "the same figure"
        set_dressing = backdrop.strip().rstrip(".") or "plain light neutral grey studio backdrop"
        outfit_tags = _picture_tags(2, len(outfit_items))
        outfit_ref = _join_tags_english(outfit_tags)
        outfit_fragment = ", ".join(
            f"{text} from {tag}" for text, tag in zip(outfit_items, outfit_tags)
        )

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
            f"and proportions established in <Picture 1>. {outfit_fragment}, "
            f"the garment tailored to {p['poss']} own "
            "body shape and proportions from <Picture 1>, not the proportions "
            f"in {outfit_ref}. <Picture 1> hairstyle is transferred to <Subject "
            f"1> — none of the hair, hairstyle, or hair texture of "
            f"{subject_noun} shown in {outfit_ref} is carried over.\n"
            "<Picture 1> is the first frame of [Shot 1], showing <Subject 1> "
            f"in {anchor_framing}, {anchor_angle}, with a {anchor_expression} "
            f"expression, against {set_dressing}."
        )

        summary = (
            "summary:\n[reference generation] The target video presents "
            f"<Subject 1> in a {duration:g}-second sequence of "
            f"{total_shots} static shots, retaining only {p['poss']} hair, "
            "body shape, skin, and facial structure from <Picture 1>, while "
            f"preserving the outfit from {outfit_ref}. The background remains "
            f"{set_dressing}. Framing, angle and expression change shot to "
            "shot as described below; pose is otherwise held still within "
            "each shot."
        )

        outfit_retention = "\n".join(
            f"{tag} (appears in {shot_tags}): partially_preserved - "
            "retains only the outfit; no other visual elements or props "
            "preserved."
            for tag in outfit_tags
        )
        retention = (
            "retention_analysis:\n"
            f"<Subject 1> (appears in {shot_tags}): partially_preserved - "
            "retains only hair, hairstyle, body shape, skin, and facial "
            "structure; no clothing or movement preserved.\n"
            f"{outfit_retention}"
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
            "writing of any kind appears anywhere in the video. "
            f"<Subject 1> wears the outfit and accessories from {outfit_ref} "
            f"continuously from the very first frame through all {shots_word} "
            "shots, without any garment changing, shifting, or being "
            "removed. Between every shot, only the camera's position around "
            "<Subject 1> changes to "
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
                f"{opener}{framing_text} of <Subject 1>, wearing the outfit "
                f"from {outfit_ref}, {angle_text}, "
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
