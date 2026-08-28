"""Drag-to-position editor for Fig 47.5 (scene + motion field).

The book shows two panels:
  LEFT  — illustrated scene (sky, cloud, 2 trees, ground) + 'Camera motion' arrow
  RIGHT — same scene faded + motion field overlay

This editor uses the LEFT panel of the book as the drag target. The notebook cell
draws BOTH panels using the same positions (right panel = faded copy + quiver).

Configurable elements:
  * 2 tree positions (foreground big tree, background small tree) + size sliders
  * 1 cloud position + size slider
  * Horizon y (slider)
  * 'Camera motion' arrow (2 dots: start + end)
  * 'Camera motion' label position
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_05_left.png"
out_html = HERE / "editor-output" / "fig47_05.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.5 - Drag-to-position editor</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui;margin:0;padding:16px;background:#f5f5f5;color:#222}
  h1{margin:0 0 8px;font-size:18px}
  .top{display:flex;gap:16px;align-items:flex-start;margin-bottom:12px;flex-wrap:wrap}
  .controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  button{padding:6px 12px;background:#2563eb;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:13px}
  button.secondary{background:#6b7280}
  input[type=range]{width:120px;vertical-align:middle}
  label{font-size:12px;color:#444}
  .editor{display:flex;gap:16px}
  .canvas-wrap{position:relative;width:700px;height:520px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
  .canvas-wrap img{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  .canvas-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%}
  .panel-info{flex:1;min-width:340px;background:#fff;padding:12px;border:1px solid #ccc;border-radius:4px}
  .panel-info h3{margin:8px 0 4px;font-size:13px;color:#555;border-bottom:1px solid #eee;padding-bottom:4px}
  .panel-info pre{margin:0;font-family:'Cascadia Code',Consolas,monospace;font-size:11px;background:#f9fafb;border:1px solid #eee;padding:8px;border-radius:4px;white-space:pre-wrap}
  .handle{cursor:move;stroke-width:1.5;transition:fill-opacity 0.1s}
  .handle:hover{fill-opacity:.8}
  .handle.tree {fill:#16a34a;fill-opacity:.4;stroke:#16a34a}
  .handle.cloud{fill:#bfdbfe;fill-opacity:.55;stroke:#1e40af}
  .handle.arrow{fill:#ef4444;fill-opacity:.4;stroke:#ef4444}
  .handle.label{fill:#a855f7;fill-opacity:.4;stroke:#a855f7}
  .label-text{font-size:13px;fill:#111;font-weight:600;pointer-events:none;font-family:system-ui}
</style></head>
<body>
<h1>Figure 47.5 - drag scene elements; sliders control sizes</h1>
<div class="top">
  <div class="controls">
    <label>Book opacity <input type="range" id="opacity" min="0" max="100" value="40"/></label>
    <label>FG tree size <input type="range" id="fgSize" min="20" max="80" value="42"/> <span id="fgSizeVal">0.42</span></label>
    <label>BG tree size <input type="range" id="bgSize" min="10" max="50" value="22"/> <span id="bgSizeVal">0.22</span></label>
    <label>Cloud size <input type="range" id="cloudSize" min="15" max="60" value="28"/> <span id="cloudSizeVal">0.28</span></label>
    <label>Horizon y <input type="range" id="horY" min="-50" max="40" value="-5"/> <span id="horYVal">-0.05</span></label>
    <label><input type="checkbox" id="snap"/> Snap to 0.05</label>
    <button onclick="resetAll()" class="secondary">Reset</button>
    <button onclick="exportPython()">Export Python</button>
  </div>
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 700 520" preserveAspectRatio="xMidYMid meet" id="canvas">
      <defs>
        <marker id="redArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#ef4444"/>
        </marker>
      </defs>
    </svg>
  </div>
  <div class="panel-info">
    <h3>Python - paste into the notebook cell</h3>
    <pre id="export"></pre>
  </div>
</div>
<script>
const CANVAS_W = 700, CANVAS_H = 520;
const AXIS = { xmin: -1.4, xmax: 1.4, ymin: -1.0, ymax: 1.05 };

let FG_SIZE = 0.42, BG_SIZE = 0.22, CLOUD_SIZE = 0.28, HORIZON_Y = -0.05;

const DEFAULTS = {
  fg_tree:  {pos: [-0.78, -0.40], kind: 'tree'},
  bg_tree:  {pos: [ 0.58, -0.30], kind: 'tree'},
  cloud:    {pos: [ 0.30,  0.60], kind: 'cloud'},
  cam_start:{pos: [-0.55,  0.85], kind: 'arrow'},
  cam_end:  {pos: [-0.15,  0.85], kind: 'arrow'},
  label_cm: {pos: [-0.50,  0.95], kind: 'label', text: 'Camera motion'},
};
let elements = JSON.parse(JSON.stringify(DEFAULTS));

function worldToScreen(x, y){
  return [
    (x - AXIS.xmin) / (AXIS.xmax - AXIS.xmin) * CANVAS_W,
    CANVAS_H - (y - AXIS.ymin) / (AXIS.ymax - AXIS.ymin) * CANVAS_H
  ];
}
function screenToWorld(sx, sy){
  const x = AXIS.xmin + (sx / CANVAS_W) * (AXIS.xmax - AXIS.xmin);
  const y = AXIS.ymin + (1 - sy / CANVAS_H) * (AXIS.ymax - AXIS.ymin);
  return [x, y];
}
function maybeSnap(v){return document.getElementById('snap').checked ? Math.round(v*20)/20 : v;}
function svgEl(tag, attrs){
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function drawTree(svg, cx, cy, scale){
  // Tree = brown trunk rectangle + green circular canopy
  const w_world = AXIS.xmax - AXIS.xmin;
  const h_world = AXIS.ymax - AXIS.ymin;
  const px_per_x = CANVAS_W / w_world;
  const px_per_y = CANVAS_H / h_world;

  const trunkW = 0.05 * scale;
  const trunkH = 0.55 * scale;
  const canopyR = 0.30 * scale;
  // Trunk: from (cx - trunkW/2, cy - trunkH) to (cx + trunkW/2, cy)
  const t_tl = worldToScreen(cx - trunkW/2, cy);
  const t_br = worldToScreen(cx + trunkW/2, cy - trunkH);
  svg.appendChild(svgEl('rect', {
    x: t_tl[0], y: t_tl[1], width: t_br[0] - t_tl[0], height: t_br[1] - t_tl[1],
    fill: '#7a4f1a'
  }));
  // Canopy circle centred at (cx, cy + canopyR * 0.6)
  const c_centre = worldToScreen(cx, cy + canopyR * 0.6);
  const c_rx = canopyR * px_per_x;
  const c_ry = canopyR * px_per_y;
  svg.appendChild(svgEl('ellipse', {cx: c_centre[0], cy: c_centre[1], rx: c_rx, ry: c_ry,
                                     fill: '#2b8a3e'}));
}

function drawCloud(svg, cx, cy, scale){
  // Cloud = several overlapping circles
  for (const [dx, dy, r] of [[-0.15, 0.0, 0.10], [0.0, 0.06, 0.13], [0.10, 0.0, 0.10], [0.18, -0.02, 0.08]]){
    const c = worldToScreen(cx + dx * (scale / 0.28), cy + dy * (scale / 0.28));
    const w_world = AXIS.xmax - AXIS.xmin;
    const h_world = AXIS.ymax - AXIS.ymin;
    const rx = r * (scale / 0.28) * (CANVAS_W / w_world);
    const ry = r * (scale / 0.28) * (CANVAS_H / h_world);
    svg.appendChild(svgEl('ellipse', {cx: c[0], cy: c[1], rx: rx, ry: ry, fill: '#ffffff'}));
  }
}

function render(){
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

  // Sky (top half)
  const sky_top = worldToScreen(AXIS.xmin, AXIS.ymax);
  const sky_horizon = worldToScreen(AXIS.xmax, HORIZON_Y);
  svg.appendChild(svgEl('rect', {x: 0, y: sky_top[1], width: CANVAS_W,
                                  height: sky_horizon[1] - sky_top[1],
                                  fill: '#bfe3ff', 'fill-opacity': 0.7}));
  // Ground (below horizon)
  const ground_bot = worldToScreen(AXIS.xmax, AXIS.ymin);
  svg.appendChild(svgEl('rect', {x: 0, y: sky_horizon[1], width: CANVAS_W,
                                  height: ground_bot[1] - sky_horizon[1],
                                  fill: '#a3d4a0', 'fill-opacity': 0.7}));

  // Cloud
  drawCloud(svg, elements.cloud.pos[0], elements.cloud.pos[1], CLOUD_SIZE);

  // Trees (background first so foreground sits on top)
  drawTree(svg, elements.bg_tree.pos[0], elements.bg_tree.pos[1], BG_SIZE);
  drawTree(svg, elements.fg_tree.pos[0], elements.fg_tree.pos[1], FG_SIZE);

  // Camera motion arrow
  const a0 = worldToScreen(...elements.cam_start.pos);
  const a1 = worldToScreen(...elements.cam_end.pos);
  svg.appendChild(svgEl('line', {x1: a0[0], y1: a0[1], x2: a1[0], y2: a1[1],
                                  stroke: '#ef4444', 'stroke-width': 3, 'stroke-linecap': 'round',
                                  'marker-end': 'url(#redArrow)'}));

  // Label
  const lbl_s = worldToScreen(...elements.label_cm.pos);
  const t = svgEl('text', {x: lbl_s[0], y: lbl_s[1], class: 'label-text'});
  t.textContent = elements.label_cm.text;
  svg.appendChild(t);

  // Handles (drawn last so they sit on top)
  for (const key of Object.keys(elements)){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const attrs = {'data-key': key};
    if (el.kind === 'label'){
      attrs.x = sx - 30; attrs.y = sy - 14; attrs.width = 70; attrs.height = 18;
      attrs.class = 'handle label';
      svg.appendChild(svgEl('rect', attrs));
    } else {
      attrs.cx = sx; attrs.cy = sy; attrs.r = 12;
      attrs.class = 'handle ' + el.kind;
      svg.appendChild(svgEl('circle', attrs));
    }
  }
  document.querySelectorAll('.handle').forEach(h => h.addEventListener('mousedown', startDrag));
  exportPython();
}

let dragging = null;
function startDrag(e){ e.preventDefault(); dragging = e.target.dataset.key; document.addEventListener('mousemove', doDrag); document.addEventListener('mouseup', endDrag); }
function doDrag(e){
  if (!dragging) return;
  const svg = document.getElementById('canvas');
  const rect = svg.getBoundingClientRect();
  const sx = (e.clientX - rect.left) / rect.width * CANVAS_W;
  const sy = (e.clientY - rect.top) / rect.height * CANVAS_H;
  const wp = screenToWorld(sx, sy);
  elements[dragging].pos = [maybeSnap(wp[0]), maybeSnap(wp[1])];
  render();
}
function endDrag(){ dragging = null; document.removeEventListener('mousemove', doDrag); document.removeEventListener('mouseup', endDrag); }

function exportPython(){
  const f = (xy) => 'torch.tensor([' + xy[0].toFixed(3) + ', ' + xy[1].toFixed(3) + '])';
  const code =
    '# Auto-generated by output/figure-editor/fig47_05.html\n' +
    'fg_tree_pos = ' + f(elements.fg_tree.pos) + '\n' +
    'bg_tree_pos = ' + f(elements.bg_tree.pos) + '\n' +
    'cloud_pos   = ' + f(elements.cloud.pos)   + '\n' +
    'cam_start   = ' + f(elements.cam_start.pos) + '\n' +
    'cam_end     = ' + f(elements.cam_end.pos)   + '\n' +
    'label_cm    = ' + f(elements.label_cm.pos)  + '\n' +
    '\n' +
    'FG_SIZE    = ' + FG_SIZE.toFixed(3)    + '\n' +
    'BG_SIZE    = ' + BG_SIZE.toFixed(3)    + '\n' +
    'CLOUD_SIZE = ' + CLOUD_SIZE.toFixed(3) + '\n' +
    'HORIZON_Y  = ' + HORIZON_Y.toFixed(3)  + '\n';
  document.getElementById('export').textContent = code;
}

function resetAll(){
  elements = JSON.parse(JSON.stringify(DEFAULTS));
  FG_SIZE = 0.42; BG_SIZE = 0.22; CLOUD_SIZE = 0.28; HORIZON_Y = -0.05;
  document.getElementById('fgSize').value = 42; document.getElementById('fgSizeVal').textContent = '0.42';
  document.getElementById('bgSize').value = 22; document.getElementById('bgSizeVal').textContent = '0.22';
  document.getElementById('cloudSize').value = 28; document.getElementById('cloudSizeVal').textContent = '0.28';
  document.getElementById('horY').value = -5; document.getElementById('horYVal').textContent = '-0.05';
  render();
}

document.getElementById('opacity').addEventListener('input', e => {
  document.getElementById('bookImg').style.opacity = (e.target.value / 100).toString();
});
document.getElementById('bookImg').style.opacity = 0.40;
document.getElementById('fgSize').addEventListener('input', e => {
  FG_SIZE = e.target.value / 100; document.getElementById('fgSizeVal').textContent = FG_SIZE.toFixed(2); render();
});
document.getElementById('bgSize').addEventListener('input', e => {
  BG_SIZE = e.target.value / 100; document.getElementById('bgSizeVal').textContent = BG_SIZE.toFixed(2); render();
});
document.getElementById('cloudSize').addEventListener('input', e => {
  CLOUD_SIZE = e.target.value / 100; document.getElementById('cloudSizeVal').textContent = CLOUD_SIZE.toFixed(2); render();
});
document.getElementById('horY').addEventListener('input', e => {
  HORIZON_Y = e.target.value / 100; document.getElementById('horYVal').textContent = HORIZON_Y.toFixed(2); render();
});
document.getElementById('snap').addEventListener('change', render);

render();
</script>
</body></html>
"""

html = HTML.replace("__BASE64__", book_b64)
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html} ({len(html)/1024:.0f} KB)")
