// Datasheet Images (H3 Look Sheet): a thumbnail grid of the images that go
// into the datasheet. Incoming frames (shown once the node has run) and
// images added with the "Add images" button can each be clicked on/off;
// added images also get a remove button.
//
// The widget value is {added: [{name, subfolder}], disabled: [key]}, where a
// key is "flow:<index>" or "input:<subfolder>/<name>" — the same keys the
// Python side checks.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "H3LookSheetsDatasheetImages";
const WIDGET_TYPE = "H3LOOKSHEETS_DATASHEET_IMAGES";
const UPLOAD_SUBFOLDER = "h3looksheets";

const CSS = `
.h3ds{display:flex;flex-direction:column;gap:6px;height:100%;min-height:0;box-sizing:border-box;padding:6px;
  font:12px/1.3 sans-serif;color:var(--fg-color,#ddd);background:rgba(0,0,0,.25);border-radius:6px;overflow:hidden}
.h3ds *{box-sizing:border-box}
.h3ds-bar{display:flex;align-items:center;gap:6px;flex:0 0 auto}
.h3ds-btn{border:1px solid #444;border-radius:5px;background:#222;color:#ccc;font-size:11px;padding:3px 10px;cursor:pointer}
.h3ds-btn:hover{border-color:#888}
.h3ds-hint{color:#888;font-size:11px;margin-left:auto}
.h3ds-grid{flex:1 1 auto;min-height:0;overflow-y:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));
  gap:6px;align-content:start}
.h3ds-empty{grid-column:1/-1;color:#777;font-size:11px;padding:8px;text-align:center}
.h3ds-tile{position:relative;border:2px solid #6ab0ff;border-radius:5px;overflow:hidden;cursor:pointer;background:#111;aspect-ratio:1}
.h3ds-tile img{width:100%;height:100%;object-fit:contain;display:block}
.h3ds-tile.off{border-color:#333}
.h3ds-tile.off img{opacity:.25}
.h3ds-tag{position:absolute;left:3px;top:3px;font-size:10px;padding:0 4px;border-radius:3px;background:rgba(0,0,0,.7);color:#ccc}
.h3ds-del{position:absolute;right:3px;top:3px;width:18px;height:18px;line-height:16px;text-align:center;border-radius:3px;
  background:rgba(0,0,0,.7);color:#ccc;font-size:12px}
.h3ds-del:hover{background:#a33;color:#fff}
`;

function injectStyle() {
  if (document.getElementById("h3looksheets-datasheet-images-css")) return;
  const style = document.createElement("style");
  style.id = "h3looksheets-datasheet-images-css";
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

const viewUrl = ({ filename, subfolder, type }) =>
  api.apiURL(`/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(subfolder ?? "")}&type=${type}`);

const addedKey = (item) => `input:${item.subfolder ?? ""}/${item.name}`;

class DatasheetImages {
  constructor(node) {
    this.node = node;
    this.state = { added: [], disabled: [] };
    this.flowThumbs = [];

    this.fileInput = el("input", { type: "file", accept: "image/*", multiple: true, style: "display:none", onchange: (e) => this.upload(e.target.files) });
    this.hint = el("span", { class: "h3ds-hint" });
    const addBtn = el("button", { class: "h3ds-btn", onclick: () => this.fileInput.click() }, "Add images");
    const allBtn = el("button", { class: "h3ds-btn", onclick: () => this.setAll(true) }, "All");
    const noneBtn = el("button", { class: "h3ds-btn", onclick: () => this.setAll(false) }, "None");
    this.grid = el("div", { class: "h3ds-grid" });
    this.root = el("div", { class: "h3ds" }, [el("div", { class: "h3ds-bar" }, [addBtn, allBtn, noneBtn, this.hint, this.fileInput]), this.grid]);
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
    this.state = { added: parsed.added ?? [], disabled: parsed.disabled ?? [] };
    this.render();
  }

  receiveFlow(thumbs) {
    this.flowThumbs = thumbs ?? [];
    this.render();
  }

  tiles() {
    const flow = this.flowThumbs.map((thumb, i) => ({ key: `flow:${i}`, src: viewUrl(thumb), tag: `#${i}` }));
    const added = this.state.added.map((item) => ({
      key: addedKey(item),
      src: viewUrl({ filename: item.name, subfolder: item.subfolder, type: "input" }),
      tag: "added",
      item,
    }));
    return [...flow, ...added];
  }

  toggle(key) {
    const disabled = new Set(this.state.disabled);
    if (disabled.has(key)) disabled.delete(key);
    else disabled.add(key);
    this.state.disabled = [...disabled];
    this.render();
  }

  setAll(on) {
    this.state.disabled = on ? [] : this.tiles().map((t) => t.key);
    this.render();
  }

  remove(item) {
    const key = addedKey(item);
    this.state.added = this.state.added.filter((a) => addedKey(a) !== key);
    this.state.disabled = this.state.disabled.filter((k) => k !== key);
    this.render();
  }

  async upload(files) {
    for (const file of files) {
      const body = new FormData();
      body.append("image", file);
      body.append("subfolder", UPLOAD_SUBFOLDER);
      body.append("type", "input");
      const resp = await api.fetchApi("/upload/image", { method: "POST", body });
      if (resp.status !== 200) {
        console.error("[H3LookSheetsDatasheetImages] upload failed:", resp.status, resp.statusText);
        continue;
      }
      const { name, subfolder } = await resp.json();
      const item = { name, subfolder: subfolder ?? "" };
      if (!this.state.added.some((a) => addedKey(a) === addedKey(item))) this.state.added.push(item);
    }
    this.fileInput.value = "";
    this.render();
  }

  render() {
    const tiles = this.tiles();
    const disabled = new Set(this.state.disabled);
    this.grid.replaceChildren(
      ...tiles.map((tile) =>
        el("div", { class: disabled.has(tile.key) ? "h3ds-tile off" : "h3ds-tile", title: tile.key, onclick: () => this.toggle(tile.key) }, [
          el("img", { src: tile.src, loading: "lazy" }),
          el("span", { class: "h3ds-tag" }, tile.tag),
          tile.item && el("span", { class: "h3ds-del", title: "Remove", onclick: (e) => { e.stopPropagation(); this.remove(tile.item); } }, "×"),
        ]),
      ),
    );
    if (!tiles.length) {
      const wired = this.node.inputs?.some((i) => i.name === "images" && i.link != null);
      this.grid.append(el("div", { class: "h3ds-empty" }, wired ? "Run the workflow once to see the incoming images." : "Add images or connect an image input."));
    }
    const on = tiles.filter((t) => !disabled.has(t.key)).length;
    this.hint.textContent = `${on} / ${tiles.length} selected`;
  }
}

app.registerExtension({
  name: "H3LookSheets.DatasheetImages",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      this.setSize([420, 360]);
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      if (output?.h3looksheets_flow_images) this.h3looksheetsDatasheetImages?.receiveFlow(output.h3looksheets_flow_images);
    };
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      onConnectionsChange?.apply(this, arguments);
      this.h3looksheetsDatasheetImages?.render();
    };
  },

  getCustomWidgets() {
    return {
      [WIDGET_TYPE](node, inputName) {
        injectStyle();
        const view = new DatasheetImages(node);
        node.h3looksheetsDatasheetImages = view;
        const widget = node.addDOMWidget(inputName, WIDGET_TYPE, view.root, {
          getValue: () => view.value,
          setValue: (value) => view.load(value),
          getMinHeight: () => 240,
          hideOnZoom: false,
          serialize: true,
        });
        widget.serializeValue = () => view.value;
        return { widget };
      },
    };
  },
});
