"""Generate a drag-to-position editor for Fig 47.2 (two-panel side view).

LEFT panel:  motion parallel to camera plane (Ż = 0)
RIGHT panel: motion along the optical axis (Ẋ = Ẏ = 0)

Each panel has its own coordinate system. The editor shows both panels side by
side with the book image as background and lets you drag every structural
element and label.
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_02.png"
out_html = HERE / "editor-output" / "fig47_02.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.2 - Drag-to-position editor</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui;margin:0;padding:16px;background:#f5f5f5;color:#222}
  h1{margin:0 0 8px;font-size:18px}
  .top{display:flex;gap:16px;align-items:flex-start;margin-bottom:12px;flex-wrap:wrap}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button{padding:6px 12px;background:#2563eb;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:13px}
  button.secondary{background:#6b7280}
  input[type=range]{width:160px;vertical-align:middle}
  label{font-size:13px;color:#444}
  .editor{display:flex;gap:16px}
  .canvas-wrap{position:relative;width:1300px;height:420px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
  .canvas-wrap img{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  .canvas-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%}
  .panel-info{flex:1;min-width:340px;background:#fff;padding:12px;border:1px solid #ccc;border-radius:4px}
  .panel-info h3{margin:8px 0 4px;font-size:13px;color:#555;border-bottom:1px solid #eee;padding-bottom:4px}
  .panel-info pre{margin:0;font-family:'Cascadia Code',Consolas,monospace;font-size:11px;background:#f9fafb;border:1px solid #eee;padding:8px;border-radius:4px;white-space:pre-wrap}
  .handle{cursor:move;stroke-width:1.5;transition:fill-opacity 0.1s}
  .handle:hover{fill-opacity:.8}
  .handle.point  {fill:#16a34a;fill-opacity:.4;stroke:#16a34a}
  .handle.vel    {fill:#ef4444;fill-opacity:.4;stroke:#ef4444}
  .handle.axis   {fill:#111;fill-opacity:.3;stroke:#111}
  .handle.plane  {fill:#3b82f6;fill-opacity:.4;stroke:#3b82f6}
  .handle.label  {fill:#a855f7;fill-opacity:.4;stroke:#a855f7}
  .label-text{font-size:14px;fill:#111;font-weight:700;pointer-events:none;font-family:'Times New Roman',serif}
  .label-text.title{font-size:13px;font-weight:600}
  .axis-grid{stroke:#e5e7eb;stroke-width:1}
  .panel-divider{stroke:#aaa;stroke-width:1;stroke-dasharray:6 4}
  .legend{font-size:12px;color:#555;margin-top:8px;line-height:1.7}
  .swatch{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin-right:4px}
</style></head>
<body>
<h1>Figure 47.2 - Drag elements to match the book (LEFT panel: lateral; RIGHT panel: axial)</h1>
<div class="top">
  <div class="controls">
    <label>Book opacity <input type="range" id="opacity" min="0" max="100" value="45"/></label>
    <label><input type="checkbox" id="showGrid" checked/> Grid</label>
    <label><input type="checkbox" id="snap"/> Snap to 0.1</label>
    <button onclick="resetAll()" class="secondary">Reset</button>
    <button onclick="exportPython()">Export Python</button>
  </div>
</div>
<div class="legend">
  <span class="swatch" style="background:#16a34a"></span>green = world points (3D)
  &nbsp;&nbsp;<span class="swatch" style="background:#ef4444"></span>red = velocity arrow endpoints
  &nbsp;&nbsp;<span class="swatch" style="background:#111"></span>black = axis tips, camera origin
  &nbsp;&nbsp;<span class="swatch" style="background:#3b82f6"></span>blue = image-plane endpoints
  &nbsp;&nbsp;<span class="swatch" style="background:#a855f7"></span>purple = label positions
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 1300 420" preserveAspectRatio="xMidYMid meet" id="canvas">
      <defs>
        <marker id="redArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#ef4444"/>
        </marker>
        <marker id="blackArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#111"/>
        </marker>
      </defs>
    </svg>
  </div>
  <div class="panel-info">
    <h3>Coordinates</h3>
    <pre id="coords"></pre>
    <h3>Python - paste into the notebook cell</h3>
    <pre id="export"></pre>
  </div>
</div>
<script>
// Two panels on a single canvas. Each panel has its OWN coordinate system.
// LEFT panel screen X:   0 - 650
// RIGHT panel screen X: 650 - 1300
// Both panels share screen Y: 0 - 420
// Each panel's world coords match the matplotlib code: x in [-0.7, 5.5], y in [-1.3, 3.0]
const CANVAS_W = 1300, CANVAS_H = 420;
const PANEL_W = 650;
const AXIS = { xmin: -0.7, xmax: 5.5, ymin: -1.3, ymax: 3.0 };

const DEFAULTS = {
  // LEFT PANEL (lateral motion) ---
  L_O:         {pos: [ 0.00,  0.00], kind: 'axis',  panel: 'L'},
  L_X_tip:     {pos: [ 0.00,  2.50], kind: 'axis',  panel: 'L'},
  L_Z_tip:     {pos: [ 5.10,  0.00], kind: 'axis',  panel: 'L'},
  L_plane_top: {pos: [ 0.80,  1.80], kind: 'plane', panel: 'L'},
  L_plane_bot: {pos: [ 0.80, -0.80], kind: 'plane', panel: 'L'},
  L_W1:        {pos: [ 3.60,  1.60], kind: 'point', panel: 'L'},
  L_W2:        {pos: [ 3.60, -0.80], kind: 'point', panel: 'L'},
  L_V1_end:    {pos: [ 3.60,  2.50], kind: 'vel',   panel: 'L'},
  L_V2_end:    {pos: [ 3.60,  0.10], kind: 'vel',   panel: 'L'},
  L_label_O:        {pos: [-0.05, -0.45], kind: 'label', panel: 'L', text: 'Camera origin'},
  L_label_X:        {pos: [ 0.10,  2.55], kind: 'label', panel: 'L', text: 'X'},
  L_label_Z:        {pos: [ 5.15, -0.20], kind: 'label', panel: 'L', text: 'Z'},
  L_label_plane:    {pos: [ 0.55, -1.10], kind: 'label', panel: 'L', text: 'Image plane'},
  L_label_title:    {pos: [ 0.30,  2.90], kind: 'label', panel: 'L', text: 'Motion parallel to camera plane (Ż=0)'},

  // RIGHT PANEL (axial motion) ---
  R_O:         {pos: [ 0.00,  0.00], kind: 'axis',  panel: 'R'},
  R_X_tip:     {pos: [ 0.00,  2.50], kind: 'axis',  panel: 'R'},
  R_Z_tip:     {pos: [ 5.10,  0.00], kind: 'axis',  panel: 'R'},
  R_plane_top: {pos: [ 0.80,  1.80], kind: 'plane', panel: 'R'},
  R_plane_bot: {pos: [ 0.80, -0.80], kind: 'plane', panel: 'R'},
  R_W1:        {pos: [ 3.60,  1.60], kind: 'point', panel: 'R'},
  R_W2:        {pos: [ 3.60, -0.80], kind: 'point', panel: 'R'},
  R_V1_end:    {pos: [ 4.50,  1.60], kind: 'vel',   panel: 'R'},   // axial = horizontal velocity
  R_V2_end:    {pos: [ 4.50, -0.80], kind: 'vel',   panel: 'R'},
  R_label_O:        {pos: [-0.05, -0.45], kind: 'label', panel: 'R', text: 'Camera origin'},
  R_label_X:        {pos: [ 0.10,  2.55], kind: 'label', panel: 'R', text: 'X'},
  R_label_Z:        {pos: [ 5.15, -0.20], kind: 'label', panel: 'R', text: 'Z'},
  R_label_plane:    {pos: [ 0.55, -1.10], kind: 'label', panel: 'R', text: 'Image plane'},
  R_label_title:    {pos: [ 0.30,  2.90], kind: 'label', panel: 'R', text: 'Motion along optical axis (Ẋ=Ẏ=0)'},
};
let elements = JSON.parse(JSON.stringify(DEFAULTS));

function worldToScreen(panel, x, y) {
  const offsetX = (panel === 'L' ? 0 : PANEL_W);
  return [
    offsetX + (x - AXIS.xmin) / (AXIS.xmax - AXIS.xmin) * PANEL_W,
    CANVAS_H - (y - AXIS.ymin) / (AXIS.ymax - AXIS.ymin) * CANVAS_H
  ];
}
function screenToWorld(sx, sy, panel) {
  const offsetX = (panel === 'L' ? 0 : PANEL_W);
  const x = AXIS.xmin + ((sx - offsetX) / PANEL_W) * (AXIS.xmax - AXIS.xmin);
  const y = AXIS.ymin + (1 - sy / CANVAS_H) * (AXIS.ymax - AXIS.ymin);
  return [x, y];
}
function maybeSnap(v){return document.getElementById('snap').checked ? Math.round(v*10)/10 : v;}
function svgEl(tag, attrs){
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function arrow(svg, p1, p2, color, lw){
  svg.appendChild(svgEl('line', {
    x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1],
    stroke: color, 'stroke-width': lw, 'stroke-linecap': 'round',
    'marker-end': color === '#ef4444' ? 'url(#redArrow)' : 'url(#blackArrow)',
  }));
}

function lineIntersect(p1, p2, p3, p4){
  const [x1,y1]=p1,[x2,y2]=p2,[x3,y3]=p3,[x4,y4]=p4;
  const denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
  const t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom;
  return [x1 + t*(x2-x1), y1 + t*(y2-y1)];
}

function renderPanel(svg, panel){
  const E = (key) => elements[panel + '_' + key].pos;

  // Grid
  if (document.getElementById('showGrid').checked){
    for (let x = Math.ceil(AXIS.xmin); x <= Math.floor(AXIS.xmax); x++){
      const sx = worldToScreen(panel, x, 0)[0];
      svg.appendChild(svgEl('line', {x1: sx, y1: 0, x2: sx, y2: CANVAS_H, class: 'axis-grid'}));
    }
    for (let y = Math.ceil(AXIS.ymin); y <= Math.floor(AXIS.ymax); y++){
      const sy = worldToScreen(panel, 0, y)[1];
      svg.appendChild(svgEl('line', {x1: panel === 'L' ? 0 : PANEL_W, y1: sy, x2: panel === 'L' ? PANEL_W : CANVAS_W, y2: sy, class: 'axis-grid'}));
    }
  }

  // Axes
  const O = E('O'); const Xtip = E('X_tip'); const Ztip = E('Z_tip');
  const O_s = worldToScreen(panel, ...O);
  const Xt_s = worldToScreen(panel, ...Xtip);
  const Zt_s = worldToScreen(panel, ...Ztip);
  arrow(svg, O_s, Xt_s, '#111', 1.5);
  arrow(svg, O_s, Zt_s, '#111', 1.5);

  // Image plane (vertical line between top and bottom endpoints)
  const pt_s = worldToScreen(panel, ...E('plane_top'));
  const pb_s = worldToScreen(panel, ...E('plane_bot'));
  svg.appendChild(svgEl('line', {x1: pt_s[0], y1: pt_s[1], x2: pb_s[0], y2: pb_s[1], stroke: '#111', 'stroke-width': 1.6}));

  // Two world points
  for (const idx of ['1','2']){
    const W = E('W'+idx);
    const Vend = E('V'+idx+'_end');
    const W_s = worldToScreen(panel, ...W);
    const Vend_s = worldToScreen(panel, ...Vend);

    // Compute image-plane projections (intersection of O→W and O→Vend with the image plane line)
    const plane_top = E('plane_top'); const plane_bot = E('plane_bot');
    const proj_world = lineIntersect(O, W, plane_top, plane_bot);
    const proj_s = worldToScreen(panel, ...proj_world);
    const Vend_proj_world = lineIntersect(O, Vend, plane_top, plane_bot);
    const Vend_proj_s = worldToScreen(panel, ...Vend_proj_world);

    // SOLID green ray from O all the way to W (the original projection ray through the image plane)
    svg.appendChild(svgEl('line', {x1: O_s[0], y1: O_s[1], x2: W_s[0], y2: W_s[1], stroke: '#22c55e', 'stroke-width': 2.5}));
    // DASHED green ray from O all the way to Vend (the moved projection ray)
    svg.appendChild(svgEl('line', {x1: O_s[0], y1: O_s[1], x2: Vend_s[0], y2: Vend_s[1], stroke: '#22c55e', 'stroke-width': 2.0, 'stroke-dasharray': '6 4'}));

    // Cyan dot at image-plane projection of W
    svg.appendChild(svgEl('circle', {cx: proj_s[0], cy: proj_s[1], r: 6, fill: '#22d3ee', stroke: '#111', 'stroke-width': 0.8}));
    // Red arrow on image plane from proj_W to proj_Vend (the ṗ vector)
    arrow(svg, proj_s, Vend_proj_s, '#ef4444', 2.5);
    // Green dot at world point
    svg.appendChild(svgEl('circle', {cx: W_s[0], cy: W_s[1], r: 7, fill: '#16a34a'}));
    // Red velocity arrow on the world (the Ẋ or Ż vector)
    arrow(svg, W_s, Vend_s, '#ef4444', 2.5);
  }

  // Camera origin dot
  svg.appendChild(svgEl('circle', {cx: O_s[0], cy: O_s[1], r: 7, fill: '#111'}));
}

function renderLabels(svg, panel){
  for (const key of ['label_O','label_X','label_Z','label_plane','label_title']){
    const el = elements[panel + '_' + key];
    const [sx, sy] = worldToScreen(panel, ...el.pos);
    const cls = key === 'label_title' ? 'label-text title' : 'label-text';
    const t = svgEl('text', {x: sx, y: sy, class: cls});
    t.textContent = el.text;
    svg.appendChild(t);
  }
}

function renderHandles(svg){
  for (const key of Object.keys(elements)){
    const el = elements[key];
    const [sx, sy] = worldToScreen(el.panel, ...el.pos);
    const attrs = {'data-key': key};
    if (el.kind === 'label'){
      attrs.x = sx - 12; attrs.y = sy - 14; attrs.width = 24; attrs.height = 16;
      attrs.class = 'handle label';
      svg.appendChild(svgEl('rect', attrs));
    } else {
      attrs.cx = sx; attrs.cy = sy; attrs.r = 11;
      attrs.class = 'handle ' + el.kind;
      svg.appendChild(svgEl('circle', attrs));
    }
  }
  document.querySelectorAll('.handle').forEach(h => h.addEventListener('mousedown', startDrag));
}

function render(){
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

  // Panel divider
  svg.appendChild(svgEl('line', {x1: PANEL_W, y1: 0, x2: PANEL_W, y2: CANVAS_H, class: 'panel-divider'}));

  renderPanel(svg, 'L');
  renderPanel(svg, 'R');
  renderLabels(svg, 'L');
  renderLabels(svg, 'R');
  renderHandles(svg);
  updatePanel();
}

let dragging = null;
function startDrag(e){
  e.preventDefault();
  dragging = e.target.dataset.key;
  document.addEventListener('mousemove', doDrag);
  document.addEventListener('mouseup', endDrag);
}
function doDrag(e){
  if (!dragging) return;
  const svg = document.getElementById('canvas');
  const rect = svg.getBoundingClientRect();
  const sx = (e.clientX - rect.left) / rect.width * CANVAS_W;
  const sy = (e.clientY - rect.top) / rect.height * CANVAS_H;
  const wp = screenToWorld(sx, sy, elements[dragging].panel);
  elements[dragging].pos = [maybeSnap(wp[0]), maybeSnap(wp[1])];
  render();
}
function endDrag(){
  dragging = null;
  document.removeEventListener('mousemove', doDrag);
  document.removeEventListener('mouseup', endDrag);
}

function updatePanel(){
  const fmt = (xy) => '(' + xy[0].toFixed(2).padStart(5) + ', ' + xy[1].toFixed(2).padStart(5) + ')';
  const lines = [];
  lines.push('# LEFT panel (lateral, Ż=0)');
  for (const k of ['L_O','L_X_tip','L_Z_tip','L_plane_top','L_plane_bot','L_W1','L_W2','L_V1_end','L_V2_end']){
    lines.push(k.padEnd(13) + ' = ' + fmt(elements[k].pos));
  }
  lines.push('');
  lines.push('# RIGHT panel (axial, Ẋ=Ẏ=0)');
  for (const k of ['R_O','R_X_tip','R_Z_tip','R_plane_top','R_plane_bot','R_W1','R_W2','R_V1_end','R_V2_end']){
    lines.push(k.padEnd(13) + ' = ' + fmt(elements[k].pos));
  }
  document.getElementById('coords').textContent = lines.join('\n');
  exportPython();
}

function exportPython(){
  const f = (xy) => 'torch.tensor([' + xy[0].toFixed(3) + ', ' + xy[1].toFixed(3) + '])';
  const keys = ['O','X_tip','Z_tip','plane_top','plane_bot','W1','W2','V1_end','V2_end'];
  const labelKeys = ['label_O','label_X','label_Z','label_plane','label_title'];

  let code = '# Auto-generated by output/figure-editor/fig47_02.html\n';
  for (const panel of ['L','R']){
    const tag = panel === 'L' ? 'lateral' : 'axial';
    code += `\n# --- ${tag.toUpperCase()} panel ---\n`;
    for (const k of keys){
      code += `${tag}_${k.padEnd(11)} = ${f(elements[panel + '_' + k].pos)}\n`;
    }
    code += `${tag}_labels = {\n`;
    for (const lk of labelKeys){
      const el = elements[panel + '_' + lk];
      const name = lk.replace('label_', '');
      code += `    ${JSON.stringify(name)}: ${f(el.pos)},\n`;
    }
    code += '}\n';
  }
  document.getElementById('export').textContent = code;
}

function resetAll(){ elements = JSON.parse(JSON.stringify(DEFAULTS)); render(); }

document.getElementById('opacity').addEventListener('input', e => {
  document.getElementById('bookImg').style.opacity = (e.target.value / 100).toString();
});
document.getElementById('bookImg').style.opacity = 0.45;
document.getElementById('showGrid').addEventListener('change', render);
document.getElementById('snap').addEventListener('change', render);

render();
</script>
</body></html>
"""

html = HTML.replace("__BASE64__", book_b64)
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html} ({len(html)/1024:.0f} KB)")
