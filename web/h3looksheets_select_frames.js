// Select Frames (H3 Look Sheet): show only the widgets the current scope
// reads. `selection_scope` picks which set of settings applies (`video_*` or
// `shot_*`); the other set is hidden.
//
// Widgets are hidden, never removed. A widget whose socket is connected stays
// visible so the link keeps a home.
//
// Saved workflows store widget values by position and links by input name, so
// `configure` rewrites older layouts into the current one before they are
// applied: the unprefixed widgets became `video_*`, `shot_*` is a copy of that
// set, and the subject description is one widget shared by both.
import { app } from "../../scripts/app.js";

const NODE_NAME = "H3LookSheetsSelectFrames";

const SCOPES = ["whole_video", "per_shot"];

// The settings each scope has its own copy of, in widget order.
const SETTINGS = [
  "saved_frame_count",
  "tier1_shots_clusters",
  "tier2_candidates",
  "tier2_prompt_how_to_select_frames",
  "tier2_free_vram_first",
  "tier2_max_token_length",
  "tier2_temperature",
  "tier2_thinking",
  "tier3_sharpness_diversity",
  "tier3_sharpness_weight",
];

// Shared by both scopes. Older layouts kept it inside each set, right after
// tier2_candidates, which made a set 11 values long.
const SUBJECT = "tier2_prompt_subject_description";
const OLD_SET_LENGTH = SETTINGS.length + 1;
const OLD_SUBJECT_INDEX = 3;
const SUBJECT_COPIES = [`video_${SUBJECT}`, `shot_${SUBJECT}`];

// Input names used by older versions of the node.
const RENAMES = {
  ...Object.fromEntries(SETTINGS.map((name) => [name, `video_${name}`])),
  frames_per_shot: "shot_saved_frame_count",
};

// shot_count / frames_per_shot for a layout that predates them.
const SHOT_COUNT_DEFAULT = 6;
const FRAMES_PER_SHOT_DEFAULT = 2;

// Layouts, as widget value arrays ("set" = one scope's 11 old settings):
//   original:           [...set]
//   scope at the end:   [...set, scope, shot_count, frames_per_shot]
//   scope first:        [scope, shot_count, frames_per_shot, ...set]
//   one set per scope:  [scope, shot_count, ...video set, ...shot set]
//   current:            [scope, shot_count, subject, ...video SETTINGS, ...shot SETTINGS]
// Each older layout is walked forward one step at a time to the current one.
function migrateWidgetValues(values) {
  const n = OLD_SET_LENGTH;
  let v = values;
  if (v.length === n) {
    v = [SCOPES[0], SHOT_COUNT_DEFAULT, FRAMES_PER_SHOT_DEFAULT, ...v];
  } else if (v.length === n + 3 && !SCOPES.includes(v[0])) {
    v = [...v.slice(n), ...v.slice(0, n)];
  }
  if (v.length === n + 3 && SCOPES.includes(v[0])) {
    const [scope, shotCount, framesPerShot, ...video] = v;
    // Both counts start equal, so tiers 2/3 never trigger inside a shot.
    const shot = [...video];
    shot[0] = framesPerShot;
    shot[1] = framesPerShot;
    v = [scope, shotCount, ...video, ...shot];
  }
  if (v.length !== 2 + 2 * n || !SCOPES.includes(v[0])) return values;
  const [scope, shotCount, ...sets] = v;
  const video = sets.slice(0, n);
  const shot = sets.slice(n);
  const [subject] = video.splice(OLD_SUBJECT_INDEX, 1);
  shot.splice(OLD_SUBJECT_INDEX, 1);
  return [scope, shotCount, subject, ...video, ...shot];
}

function migrateInputs(inputs) {
  // When each scope had its own subject description, the copy with a link
  // (else the video one) becomes the shared input; the other is left as is.
  const copies = inputs.filter((i) => SUBJECT_COPIES.includes(i?.name));
  const subject = inputs.some((i) => i?.name === SUBJECT)
    ? null
    : copies.find((i) => i.link != null) ?? copies.find((i) => i.name === SUBJECT_COPIES[0]);

  return inputs.map((input) => {
    const name = input === subject ? SUBJECT : RENAMES[input?.name];
    if (!name) return input;
    const renamed = { ...input, name };
    if (input.localized_name === input.name) renamed.localized_name = name;
    if (input.label === input.name) renamed.label = name;
    if (input.widget?.name === input.name) renamed.widget = { ...input.widget, name };
    return renamed;
  });
}

function hiddenWidgetNames(node) {
  const scope = node.widgets?.find((w) => w.name === "selection_scope")?.value;
  const perShot = scope === "per_shot";
  const inactive = perShot ? "video_" : "shot_";
  const hidden = new Set(SETTINGS.map((setting) => inactive + setting));
  if (!perShot) hidden.add("shot_count");
  return hidden;
}

function refresh(node, resize) {
  const hidden = hiddenWidgetNames(node);
  const wired = new Set((node.inputs ?? []).filter((i) => i.widget && i.link != null).map((i) => i.name));
  let changed = false;
  for (const widget of node.widgets ?? []) {
    const hide = hidden.has(widget.name) && !wired.has(widget.name);
    if (!!widget.hidden === hide) continue;
    // Same pair the frontend's own nodes set: `hidden` for the canvas layout,
    // `options.hidden` for DOM (multiline) widgets and the node panel.
    widget.hidden = hide;
    if (widget.options) widget.options.hidden = hide;
    changed = true;
  }
  if (!changed) return;
  if (resize) node.setSize([node.size[0], node.computeSize()[1]]);
  app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "H3LookSheets.SelectFrames",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const scope = this.widgets?.find((w) => w.name === "selection_scope");
      if (scope) {
        const callback = scope.callback;
        scope.callback = (...args) => {
          callback?.apply(scope, args);
          refresh(this, true);
        };
      }
      refresh(this, true);
    };

    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info, ...rest) {
      if (info) {
        info = { ...info };
        if (Array.isArray(info.widgets_values)) info.widgets_values = migrateWidgetValues(info.widgets_values);
        if (Array.isArray(info.inputs)) info.inputs = migrateInputs(info.inputs);
      }
      return configure.call(this, info, ...rest);
    };

    // After a workflow load the saved widget values and links are in place;
    // the saved node size is kept as it was.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      refresh(this, false);
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      onConnectionsChange?.apply(this, arguments);
      refresh(this, true);
    };
  },
});
