"""Generate a drag-to-position editor for Fig 47.3 (Gibson's bird) — V2.

Re-built based on a close reading of the book image:
  * 7 birds in TOP trajectory (sizes hardcoded to match book proportions)
  * 2 birds in BOTTOM trajectory
  * 4 plane corners
  * Optical centre O at bottom-left of plane
  * Vanishing point INSIDE the image plane
  * Mini bird silhouettes inside the plane (auto-projected from world birds via O)
  * Red velocity arrows between consecutive world birds
  * 4 labels

All structural elements are individually draggable.
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_03.png"
out_html = HERE / "editor-output" / "fig47_03.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.3 - Drag-to-position editor (v2)</title>
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
  .canvas-wrap{position:relative;width:1300px;height:620px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
  .canvas-wrap img{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  .canvas-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%}
  .panel-info{flex:1;min-width:340px;background:#fff;padding:12px;border:1px solid #ccc;border-radius:4px}
  .panel-info h3{margin:8px 0 4px;font-size:13px;color:#555;border-bottom:1px solid #eee;padding-bottom:4px}
  .panel-info pre{margin:0;font-family:'Cascadia Code',Consolas,monospace;font-size:11px;background:#f9fafb;border:1px solid #eee;padding:8px;border-radius:4px;white-space:pre-wrap}
  .handle{cursor:move;stroke-width:1.5;transition:fill-opacity 0.1s}
  .handle:hover{fill-opacity:.8}
  .handle.O      {fill:#111;fill-opacity:.4;stroke:#111}
  .handle.corner {fill:#3b82f6;fill-opacity:.4;stroke:#3b82f6}
  .handle.bird1  {fill:#16a34a;fill-opacity:.4;stroke:#16a34a}
  .handle.bird2  {fill:#f59e0b;fill-opacity:.4;stroke:#f59e0b}
  .handle.vp     {fill:#ef4444;fill-opacity:.4;stroke:#ef4444}
  .handle.mini   {fill:#14b8a6;fill-opacity:.4;stroke:#14b8a6}
  .handle.label  {fill:#a855f7;fill-opacity:.4;stroke:#a855f7}
  .label-text{font-size:14px;fill:#111;font-weight:700;pointer-events:none;font-family:'Times New Roman',serif}
  .axis-grid{stroke:#e5e7eb;stroke-width:1}
  .legend{font-size:12px;color:#555;margin-top:8px;line-height:1.7}
  .swatch{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin-right:4px}
</style></head>
<body>
<h1>Figure 47.3 - drag each bird, plane corner, and label independently</h1>
<div class="top">
  <div class="controls">
    <label>Book opacity <input type="range" id="opacity" min="0" max="100" value="40"/></label>
    <label><input type="checkbox" id="showGrid" checked/> Grid</label>
    <label><input type="checkbox" id="snap"/> Snap to 0.1</label>
    <button onclick="resetAll()" class="secondary">Reset</button>
    <button onclick="exportPython()">Export Python</button>
  </div>
</div>
<div class="legend">
  <span class="swatch" style="background:#111"></span>O
  &nbsp;<span class="swatch" style="background:#3b82f6"></span>plane corners
  &nbsp;<span class="swatch" style="background:#16a34a"></span>top trajectory (7 birds)
  &nbsp;<span class="swatch" style="background:#f59e0b"></span>bottom trajectory (2 birds)
  &nbsp;<span class="swatch" style="background:#ef4444"></span>vanishing point
  &nbsp;<span class="swatch" style="background:#14b8a6"></span>mini-bird line (drag start &amp; end)
  &nbsp;<span class="swatch" style="background:#a855f7"></span>labels
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 1300 620" preserveAspectRatio="xMidYMid meet" id="canvas">
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
const CANVAS_W = 1300, CANVAS_H = 620;
const AXIS = { xmin: 0.0, xmax: 11.6, ymin: -0.5, ymax: 5.0 };

// Sizes hardcoded to match the book's perspective proportions.
// Top trajectory: 7 birds. First bird is small (just entering), 2nd is largest, then progressively smaller.
const T1_SIZES = [0.40, 0.85, 0.70, 0.55, 0.45, 0.38, 0.32];
// Bottom trajectory: 2 birds — one inside/near the plane (small), one to the right (medium).
const T2_SIZES = [0.30, 0.65];

const DEFAULTS = {
  // Optical centre
  O:         {pos: [ 2.281, 0.834], kind: 'O'},

  // Image plane corners (from your last export)
  plane_bl:  {pos: [ 2.361,  1.074], kind: 'corner'},
  plane_br:  {pos: [ 3.906, -0.482], kind: 'corner'},
  plane_tr:  {pos: [ 3.915,  1.599], kind: 'corner'},
  plane_tl:  {pos: [ 2.370,  3.128], kind: 'corner'},

  // Top trajectory: 7 birds
  T1_b0:     {pos: [ 2.727,  3.377], kind: 'bird1'},
  T1_b1:     {pos: [ 4.282,  3.714], kind: 'bird1'},
  T1_b2:     {pos: [ 5.818,  3.910], kind: 'bird1'},
  T1_b3:     {pos: [ 7.417,  4.203], kind: 'bird1'},
  T1_b4:     {pos: [ 8.873,  4.390], kind: 'bird1'},
  T1_b5:     {pos: [10.391,  4.621], kind: 'bird1'},
  T1_b6:     {pos: [11.508,  4.763], kind: 'bird1'},

  // Bottom trajectory: 2 birds
  T2_b0:     {pos: [ 3.731,  1.047], kind: 'bird2'},
  T2_b1:     {pos: [ 6.649,  1.510], kind: 'bird2'},

  // Mini birds line inside the image plane — drag start (closest to viewer, biggest)
  // and end (closest to vanishing point, smallest). Auto-interpolates 5 birds total.
  mini_start: {pos: [ 2.50,  0.95], kind: 'mini'},
  mini_end:   {pos: [ 2.95,  1.10], kind: 'mini'},

  // Vanishing point
  VP:        {pos: [ 2.209,  0.594], kind: 'vp'},

  // Labels (each floats independently and is draggable)
  label_O:   {pos: [ 0.400,  0.050], kind: 'label', text: 'O'},
  label_V1:  {pos: [ 3.200,  4.850], kind: 'label', text: 'V'},
  label_V2:  {pos: [ 5.693,  1.474], kind: 'label', text: 'V'},
  label_VP:  {pos: [ 1.800,  0.400], kind: 'label', text: 'Vanishing point'},
};
let elements = JSON.parse(JSON.stringify(DEFAULTS));

function worldToScreen(x, y) {
  return [
    (x - AXIS.xmin) / (AXIS.xmax - AXIS.xmin) * CANVAS_W,
    CANVAS_H - (y - AXIS.ymin) / (AXIS.ymax - AXIS.ymin) * CANVAS_H
  ];
}
function screenToWorld(sx, sy) {
  const x = AXIS.xmin + (sx / CANVAS_W) * (AXIS.xmax - AXIS.xmin);
  const y = AXIS.ymin + (1 - sy / CANVAS_H) * (AXIS.ymax - AXIS.ymin);
  return [x, y];
}
function maybeSnap(v){return document.getElementById('snap').checked ? Math.round(v*10)/10 : v;}
function svgEl(tag, attrs){
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function lineIntersect(p1, p2, p3, p4){
  const [x1,y1]=p1,[x2,y2]=p2,[x3,y3]=p3,[x4,y4]=p4;
  const denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
  const t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom;
  return [x1 + t*(x2-x1), y1 + t*(y2-y1)];
}

// Draw a bird at screen position (sx, sy). halfW = each wing's horizontal reach in pixels.
// Uses two quadratic Bézier arcs (each wing tip dips down at the outside, arcs up to a peak,
// then comes back down to the body in the middle).
function birdPath(sx, sy, halfW, archH){
  const dip = halfW * 0.15;                          // outer wing tip dips slightly below body
  const peakX_l = sx - halfW * 0.55;
  const peakX_r = sx + halfW * 0.55;
  return 'M ' + (sx - halfW) + ',' + (sy + dip)
       + ' Q ' + peakX_l + ',' + (sy - archH) + ' ' + sx + ',' + sy
       + ' Q ' + peakX_r + ',' + (sy - archH) + ' ' + (sx + halfW) + ',' + (sy + dip);
}

function renderTrajectory(svg, prefix, sizes){
  const n = sizes.length;
  const positions = [];
  for (let i = 0; i < n; i++) positions.push(elements[prefix + '_b' + i].pos);

  const O_s = worldToScreen(...elements.O.pos);
  const planeA = elements.plane_tl.pos;
  const planeB = elements.plane_br.pos;

  // Green projection rays from O to each bird
  for (const pos of positions){
    const ps = worldToScreen(...pos);
    svg.appendChild(svgEl('line', {
      x1: O_s[0], y1: O_s[1], x2: ps[0], y2: ps[1],
      stroke: '#22c55e', 'stroke-width': 1.0, opacity: 0.75
    }));
  }

  // (Mini birds + their velocity arrow are drawn separately as a draggable line — see renderMiniLine)

  // Red velocity arrows between consecutive birds
  for (let i = 0; i < n - 1; i++){
    const p0 = positions[i], p1 = positions[i+1];
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) continue;
    const dirX = dx/len, dirY = dy/len;
    const s_w = [p0[0] + dirX * sizes[i] * 0.65, p0[1] + dirY * sizes[i] * 0.65];
    const e_w = [p1[0] - dirX * sizes[i+1] * 0.65, p1[1] - dirY * sizes[i+1] * 0.65];
    const s_s = worldToScreen(...s_w);
    const e_s = worldToScreen(...e_w);
    svg.appendChild(svgEl('line', {
      x1: s_s[0], y1: s_s[1], x2: e_s[0], y2: e_s[1],
      stroke: '#ef4444', 'stroke-width': 3.0, 'stroke-linecap': 'round',
      'marker-end': 'url(#redArrow)'
    }));
  }

  // Big birds at each position
  for (let i = 0; i < n; i++){
    const ps = worldToScreen(...positions[i]);
    const half = sizes[i] * (CANVAS_W / (AXIS.xmax - AXIS.xmin));
    svg.appendChild(svgEl('path', {
      d: birdPath(ps[0], ps[1], half, half * 0.45),
      stroke: '#111', 'stroke-width': 2.2, fill: 'none',
      'stroke-linecap': 'round', 'stroke-linejoin': 'round'
    }));
  }
}

function renderMiniLine(svg){
  // 5 mini birds along the line from mini_start to mini_end. Sizes shrink from 0.40 (closest)
  // to 0.18 (smallest, closest to the vanishing point in the book).
  const N_MINI = 5;
  const MINI_SIZES = [0.40, 0.32, 0.26, 0.22, 0.18];
  const start = elements.mini_start.pos;
  const end   = elements.mini_end.pos;
  const positions = [];
  for (let i = 0; i < N_MINI; i++){
    const t = i / (N_MINI - 1);
    positions.push([start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])]);
  }

  // Mini birds
  for (let i = 0; i < N_MINI; i++){
    const ps = worldToScreen(...positions[i]);
    let half = MINI_SIZES[i] * 0.40 * (CANVAS_W / (AXIS.xmax - AXIS.xmin));
    half = Math.max(half, 8);
    svg.appendChild(svgEl('path', {
      d: birdPath(ps[0], ps[1], half, half * 0.55),
      stroke: '#111', 'stroke-width': 1.6, fill: 'none',
      'stroke-linecap': 'round', 'stroke-linejoin': 'round'
    }));
  }

  // Red velocity arrow between the first and last mini bird (one overall arrow showing ṗ direction)
  const dx = end[0] - start[0], dy = end[1] - start[1];
  const len = Math.hypot(dx, dy);
  if (len > 1e-6){
    const dirX = dx / len, dirY = dy / len;
    const shrink_w = 0.12;
    const s_w = [start[0] + dirX * shrink_w, start[1] + dirY * shrink_w];
    const e_w = [end[0]   - dirX * shrink_w, end[1]   - dirY * shrink_w];
    const s_s = worldToScreen(...s_w);
    const e_s = worldToScreen(...e_w);
    svg.appendChild(svgEl('line', {
      x1: s_s[0], y1: s_s[1], x2: e_s[0], y2: e_s[1],
      stroke: '#ef4444', 'stroke-width': 2.4, 'stroke-linecap': 'round',
      'marker-end': 'url(#redArrow)'
    }));
  }
}

function render(){
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

  // Grid
  if (document.getElementById('showGrid').checked){
    for (let x = Math.ceil(AXIS.xmin); x <= Math.floor(AXIS.xmax); x++){
      const sx = worldToScreen(x, 0)[0];
      svg.appendChild(svgEl('line', {x1: sx, y1: 0, x2: sx, y2: CANVAS_H, class: 'axis-grid'}));
    }
    for (let y = Math.ceil(AXIS.ymin); y <= Math.floor(AXIS.ymax); y++){
      const sy = worldToScreen(0, y)[1];
      svg.appendChild(svgEl('line', {x1: 0, y1: sy, x2: CANVAS_W, y2: sy, class: 'axis-grid'}));
    }
  }

  // Image plane
  const corners = ['plane_bl','plane_br','plane_tr','plane_tl'].map(k => worldToScreen(...elements[k].pos));
  svg.appendChild(svgEl('path', {
    d: 'M ' + corners[0][0] + ',' + corners[0][1] +
       ' L ' + corners[1][0] + ',' + corners[1][1] +
       ' L ' + corners[2][0] + ',' + corners[2][1] +
       ' L ' + corners[3][0] + ',' + corners[3][1] + ' Z',
    stroke: '#111', 'stroke-width': 2.4, fill: 'none', 'stroke-linejoin': 'round'
  }));

  // Trajectories
  renderTrajectory(svg, 'T1', T1_SIZES);
  renderTrajectory(svg, 'T2', T2_SIZES);

  // Mini-bird line inside the image plane (independent of trajectories — drag both endpoints)
  renderMiniLine(svg);

  // Vanishing point (small filled dot)
  const VP_s = worldToScreen(...elements.VP.pos);
  svg.appendChild(svgEl('circle', {cx: VP_s[0], cy: VP_s[1], r: 5, fill: '#111'}));

  // Optical centre O (big black dot at the dragged position)
  const O_s = worldToScreen(...elements.O.pos);
  svg.appendChild(svgEl('circle', {cx: O_s[0], cy: O_s[1], r: 10, fill: '#111'}));

  // Labels (each at its own draggable position)
  for (const key of ['label_O','label_V1','label_V2','label_VP']){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const t = svgEl('text', {x: sx, y: sy, class: 'label-text'});
    t.textContent = el.text;
    svg.appendChild(t);
  }

  // Handles
  for (const key of Object.keys(elements)){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const attrs = {'data-key': key};
    if (el.kind === 'label'){
      attrs.x = sx - 14; attrs.y = sy - 14; attrs.width = 28; attrs.height = 16;
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
    '# Auto-generated by output/figure-editor/fig47_03.html\n' +
    '\n' +
    'O           = ' + f(elements.O.pos) + '\n' +
    'plane = torch.tensor([\n' +
    '    [' + elements.plane_bl.pos[0].toFixed(3) + ', ' + elements.plane_bl.pos[1].toFixed(3) + '],   # bottom-LEFT\n' +
    '    [' + elements.plane_br.pos[0].toFixed(3) + ', ' + elements.plane_br.pos[1].toFixed(3) + '],   # bottom-RIGHT\n' +
    '    [' + elements.plane_tr.pos[0].toFixed(3) + ', ' + elements.plane_tr.pos[1].toFixed(3) + '],   # top-RIGHT\n' +
    '    [' + elements.plane_tl.pos[0].toFixed(3) + ', ' + elements.plane_tl.pos[1].toFixed(3) + '],   # top-LEFT\n' +
    '    [' + elements.plane_bl.pos[0].toFixed(3) + ', ' + elements.plane_bl.pos[1].toFixed(3) + '],\n' +
    '])\n' +
    '\n' +
    '# Top trajectory (7 birds), sizes match the book\n' +
    'T1_birds = [\n' +
    [0,1,2,3,4,5,6].map(i => '    ' + f(elements['T1_b'+i].pos)).join(',\n') + ',\n' +
    ']\n' +
    'T1_sizes = [0.40, 0.85, 0.70, 0.55, 0.45, 0.38, 0.32]\n' +
    '\n' +
    '# Bottom trajectory (2 birds)\n' +
    'T2_birds = [\n' +
    [0,1].map(i => '    ' + f(elements['T2_b'+i].pos)).join(',\n') + ',\n' +
    ']\n' +
    'T2_sizes = [0.30, 0.65]\n' +
    '\n' +
    'VP          = ' + f(elements.VP.pos) + '\n' +
    '\n' +
    '# Mini-bird line inside the image plane (5 birds linearly interpolated, sizes [0.40,0.32,0.26,0.22,0.18])\n' +
    'mini_start  = ' + f(elements.mini_start.pos) + '\n' +
    'mini_end    = ' + f(elements.mini_end.pos)   + '\n' +
    '\n' +
    'labels = {\n' +
    '    "O":  ' + f(elements.label_O.pos)  + ',\n' +
    '    "V1": ' + f(elements.label_V1.pos) + ',\n' +
    '    "V2": ' + f(elements.label_V2.pos) + ',\n' +
    '    "VP": ' + f(elements.label_VP.pos) + ',\n' +
    '}\n';
  document.getElementById('export').textContent = code;
}

function resetAll(){ elements = JSON.parse(JSON.stringify(DEFAULTS)); render(); }

document.getElementById('opacity').addEventListener('input', e => {
  document.getElementById('bookImg').style.opacity = (e.target.value / 100).toString();
});
document.getElementById('bookImg').style.opacity = 0.40;
document.getElementById('showGrid').addEventListener('change', render);
document.getElementById('snap').addEventListener('change', render);

render();
</script>
</body></html>
"""

html = HTML.replace("__BASE64__", book_b64)
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html} ({len(html)/1024:.0f} KB)")
