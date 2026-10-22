// Prompt View (H3 Look Sheet): a read-only, colorized mirror of an upstream
// prompt string, with a one-click copy button and an "Override prompt"
// toggle to edit a copy in place. Modelled on H3VideoRemake's prompt editor
// (web/h3remake_prompt.js) for the same tag-highlighting trick, trimmed down
// to just that one behavior — kept self-contained rather than importing from
// that other node pack.
import { app } from "../../scripts/app.js";

const NODE_NAME = "H3LookSheetsPromptView";
const WIDGET_TYPE = "H3LOOKSHEETS_PROMPT_VIEW";
const SECTIONS = ["subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"];

const CSS = `
.h3lv{display:flex;flex-direction:column;gap:6px;height:100%;min-height:0;box-sizing:border-box;padding:6px;
  font:12px/1.3 sans-serif;color:var(--fg-color,#ddd);background:rgba(0,0,0,.25);border-radius:6px;overflow:hidden}
.h3lv *{box-sizing:border-box}
.h3lv-bar{display:flex;align-items:center;gap:6px;flex:0 0 auto}
.h3lv-btn{border:1px solid #444;border-radius:5px;background:#222;color:#ccc;font-size:11px;padding:3px 10px;cursor:pointer}
.h3lv-btn:hover{border-color:#888}
.h3lv-btn.on{background:#3a2d52;color:#fff;border-color:#8a5cd0}
.h3lv-hint{color:#888;font-size:11px;margin-left:auto}
.h3lv-editor{position:relative;flex:1 1 auto;min-height:120px;background:#1b1b1b;border:1px solid #333;border-radius:6px;overflow:hidden}
.h3lv-editor.editing{border-color:#8a5cd0}
.h3lv-editor textarea,.h3lv-editor .back{position:absolute;inset:0;margin:0;border:0;padding:8px;font:12px/1.5 ui-monospace,Consolas,monospace;
  white-space:pre-wrap;overflow-wrap:break-word;overflow-y:auto;scrollbar-gutter:stable;tab-size:4}
.h3lv-editor textarea{resize:none;background:transparent;color:transparent;caret-color:#eee;outline:none}
.h3lv-editor textarea[readonly]{caret-color:transparent;cursor:default}
.h3lv-editor .back{color:#ccc;pointer-events:none;overflow-y:hidden}
.h3lv-editor mark{background:none;border-radius:3px}
.h3lv-editor mark.picture{color:#6ab0ff;background:rgba(106,176,255,.15)}
.h3lv-editor mark.subject{color:#e0a060;background:rgba(224,160,96,.15)}
.h3lv-editor mark.audio{color:#62d0a0;background:rgba(98,208,160,.15)}
.h3lv-editor mark.shot{color:#c99af0}
.h3lv-editor mark.header{color:#f0c85a}
`;

function injectStyle() {
  if (document.getElementById("h3looksheets-prompt-view-css")) return;
  const style = document.createElement("style");
  style.id = "h3looksheets-prompt-view-css";
  style.textContent = CSS;
  document.head.appendChild(style);
}

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined && child !== false) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const escapeHtml = (text) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const TOKEN = new RegExp(`&lt;(Picture|Subject|Audio) \\d+&gt;|\\[Shot \\d+\\]|^(?:${SECTIONS.join("|")})\\s*:`, "gm");

function highlight(text) {
  // Same characters as the textarea, only wrapped in marks, so the caret stays aligned. The trailing
  // space keeps a final empty line the same height in both layers.
  return escapeHtml(text).replace(TOKEN, (token, kind) => {
    const cls = kind ? kind.toLowerCase() : token.startsWith("[") ? "shot" : "header";
    return `<mark class="${cls}">${token}</mark>`;
  }) + " ";
}

class PromptView {
  constructor() {
    this.state = { override: false, text: "" };
    this.sourceText = "";

    this.copyBtn = el("button", { class: "h3lv-btn", onclick: () => this.copy() }, "Copy");
    this.overrideBtn = el("button", { class: "h3lv-btn", onclick: () => this.toggleOverride() }, "Override prompt");
    this.hint = el("span", { class: "h3lv-hint" }, "");
    const bar = el("div", { class: "h3lv-bar" }, [this.copyBtn, this.overrideBtn, this.hint]);

    this.back = el("div", { class: "back" });
    this.textarea = el("textarea", { spellcheck: "false", oninput: (e) => this.onInput(e) });
    this.textarea.addEventListener("scroll", () => { this.back.scrollTop = this.textarea.scrollTop; });
    this.editorWrap = el("div", { class: "h3lv-editor" }, [this.back, this.textarea]);

    this.root = el("div", { class: "h3lv" }, [bar, this.editorWrap]);
    // Keep typing and scrolling inside the editor instead of the canvas.
    this.root.addEventListener("keydown", (e) => e.stopPropagation());
    this.root.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });

    this.render();
  }

  get value() {
    return JSON.stringify(this.state);
  }

  load(value) {
    let parsed = {};
    try {
      parsed = value ? JSON.parse(value) : {};
    } catch {
      parsed = {};
    }
    this.state = { override: !!parsed.override, text: parsed.text ?? "" };
    this.render();
  }

  receivePrompt(text) {
    this.sourceText = text ?? "";
    if (!this.state.override) this.render();
  }

  displayedText() {
    return this.state.override ? this.state.text : this.sourceText;
  }

  onInput(e) {
    if (!this.state.override) return;
    this.state.text = e.target.value;
    this.back.innerHTML = highlight(this.state.text);
  }

  toggleOverride() {
    if (!this.state.override && !this.state.text) this.state.text = this.sourceText;
    this.state.override = !this.state.override;
    this.render();
  }

  async copy() {
    try {
      await navigator.clipboard.writeText(this.displayedText());
      const original = this.copyBtn.textContent;
      this.copyBtn.textContent = "Copied!";
      setTimeout(() => { this.copyBtn.textContent = original; }, 900);
    } catch (err) {
      console.error("[H3LookSheetsPromptView] copy failed:", err);
    }
  }

  render() {
    const text = this.displayedText();
    if (this.textarea.value !== text) this.textarea.value = text;
    this.textarea.readOnly = !this.state.override;
    this.back.innerHTML = highlight(text);
    this.editorWrap.classList.toggle("editing", this.state.override);
    this.overrideBtn.textContent = this.state.override ? "Revert to original" : "Override prompt";
    this.overrideBtn.classList.toggle("on", this.state.override);
    this.hint.textContent = this.state.override ? "editing a copy" : "read-only";
  }
}

app.registerExtension({
  name: "H3LookSheets.PromptView",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      this.setSize([420, 300]);
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      if (output?.h3looksheets_prompt?.[0] !== undefined) this.h3looksheetsPromptView?.receivePrompt(output.h3looksheets_prompt[0]);
    };
  },

  getCustomWidgets() {
    return {
      [WIDGET_TYPE](node, inputName) {
        injectStyle();
        const view = new PromptView();
        node.h3looksheetsPromptView = view;
        const widget = node.addDOMWidget(inputName, WIDGET_TYPE, view.root, {
          getValue: () => view.value,
          setValue: (value) => view.load(value),
          getMinHeight: () => 220,
          hideOnZoom: false,
          serialize: true,
        });
        widget.serializeValue = () => view.value;
        return { widget };
      },
    };
  },
});
