"""Generate a standalone HTML drag-to-position editor for Fig 47.1.

Configurable elements:
  * Plane corners (4) - reshape the perspective parallelogram
  * O, P, Pdot_end       - the 3D point and the end of velocity vector
  * p, p_dash            - independently draggable image-plane points (ṗ arrow endpoints)
  * Five label positions - each math symbol can be placed anywhere

A "Re-auto-snap p, p'" button recomputes p / p_dash from the current O, P, Pdot_end, plane.
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_01.png"
out_html = HERE / "editor-output" / "fig47_01.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Fig 47.1 - Drag-to-position editor</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui;margin:0;padding:16px;background:#f5f5f5;color:#222}
  h1{margin:0 0 8px;font-size:18px}
  .top{display:flex;gap:16px;align-items:flex-start;margin-bottom:12px;flex-wrap:wrap}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button{padding:6px 12px;background:#2563eb;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:13px}
  button.secondary{background:#6b7280}
  button.warn{background:#a855f7}
  input[type=range]{width:160px;vertical-align:middle}
  label{font-size:13px;color:#444}
  .editor{display:flex;gap:16px}
  .canvas-wrap{position:relative;width:900px;height:600px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
  .canvas-wrap img{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  .canvas-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%}
  .panel{flex:1;min-width:340px;background:#fff;padding:12px;border:1px solid #ccc;border-radius:4px}
  .panel h3{margin:8px 0 4px;font-size:13px;color:#555;border-bottom:1px solid #eee;padding-bottom:4px}
  .panel pre{margin:0;font-family:'Cascadia Code',Consolas,monospace;font-size:12px;background:#f9fafb;border:1px solid #eee;padding:8px;border-radius:4px;white-space:pre-wrap;overflow-x:auto}
  .handle{cursor:move;stroke-width:1.5;transition:fill-opacity 0.1s}
  .handle:hover{fill-opacity:.8}
  .handle.point  {fill:#ef4444;fill-opacity:.35;stroke:#ef4444}
  .handle.corner {fill:#3b82f6;fill-opacity:.35;stroke:#3b82f6}
  .handle.label  {fill:#10b981;fill-opacity:.40;stroke:#10b981}
  .label-text{font-size:18px;fill:#111;font-weight:700;pointer-events:none;font-family:'Times New Roman',serif;font-style:italic}
  .axis-grid{stroke:#e5e7eb;stroke-width:1}
  .axis-zero{stroke:#9ca3af;stroke-width:1.2;stroke-dasharray:4 3}
  .legend{font-size:12px;color:#555;margin-top:8px;margin-bottom:8px;line-height:1.6}
  .swatch{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin-right:4px}
  .swatch.sq{border-radius:2px}
</style>
</head>
<body>
<h1>Figure 47.1 - Drag elements to visually match the book</h1>
<div class="top">
  <div class="controls">
    <label>Book opacity <input type="range" id="opacity" min="0" max="100" value="45"/></label>
    <label><input type="checkbox" id="showGrid" checked/> Grid</label>
    <label><input type="checkbox" id="snap"/> Snap to 0.1</label>
    <button onclick="autoSnapProjections()" class="warn">Re-auto-snap p, p'</button>
    <button onclick="resetAll()" class="secondary">Reset</button>
    <button onclick="exportPython()">Export Python</button>
  </div>
</div>
<div class="legend">
  <span class="swatch" style="background:#3b82f6"></span>blue = plane corners
  &nbsp;&nbsp;
  <span class="swatch" style="background:#ef4444"></span>red = point handles (O, P, P+Ṗ, p, ṗ endpoint)
  &nbsp;&nbsp;
  <span class="swatch sq" style="background:#10b981"></span>green = label handles (drag each math symbol independently)
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 900 600" preserveAspectRatio="xMidYMid meet" id="canvas">
      <defs>
        <marker id="redArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#ef4444"/>
        </marker>
      </defs>
    </svg>
  </div>
  <div class="panel">
    <h3>Coordinates (matplotlib world)</h3>
    <pre id="coords"></pre>
    <h3>Python - paste into the notebook cell</h3>
    <pre id="export"></pre>
  </div>
</div>
<script>
const AXIS = { xmin: -0.75, xmax: 8.1, ymin: -0.4, ymax: 4.8 };
const CANVAS_W = 900, CANVAS_H = 600;

const DEFAULTS = {
  O:         {pos: [-0.30, 0.45], kind: 'point'},
  P:         {pos: [ 7.00, 2.40], kind: 'point'},
  Pdot_end:  {pos: [ 5.30, 3.60], kind: 'point'},
  p:         {pos: [ 1.99, 1.06], kind: 'point'},
  p_dash:    {pos: [ 1.73, 1.59], kind: 'point'},
  plane_bl:  {pos: [ 0.40, 0.65], kind: 'corner'},
  plane_br:  {pos: [ 2.45, 0.15], kind: 'corner'},
  plane_tr:  {pos: [ 2.45, 3.75], kind: 'corner'},
  plane_tl:  {pos: [ 0.40, 4.25], kind: 'corner'},
  label_O:    {pos: [-0.35, -0.05], kind: 'label', text: 'O'},
  label_P:    {pos: [ 7.22,  2.35], kind: 'label', text: 'P'},
  label_p:    {pos: [ 2.11,  0.74], kind: 'label', text: 'p'},
  label_pdot: {pos: [ 1.31,  1.79], kind: 'label', text: 'ṗ'},
  label_Pdot: {pos: [ 5.72,  3.72], kind: 'label', text: 'Ṗ'},
};
let elements = deepClone(DEFAULTS);

function deepClone(o){ return JSON.parse(JSON.stringify(o)); }

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
function lineIntersect(a, b, c, d) {
  const [x1,y1]=a,[x2,y2]=b,[x3,y3]=c,[x4,y4]=d;
  const denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
  const t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom;
  return [x1 + t*(x2-x1), y1 + t*(y2-y1)];
}
function maybeSnap(v){
  if (document.getElementById('snap').checked){
    return Math.round(v*10)/10;
  }
  return v;
}
function svgEl(tag, attrs){
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function render() {
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

  if (document.getElementById('showGrid').checked) {
    for (let x = Math.ceil(AXIS.xmin); x <= Math.floor(AXIS.xmax); x++){
      const sx = worldToScreen(x, 0)[0];
      svg.appendChild(svgEl('line', {x1: sx, y1: 0, x2: sx, y2: CANVAS_H, class: (x===0?'axis-zero':'axis-grid')}));
    }
    for (let y = Math.ceil(AXIS.ymin); y <= Math.floor(AXIS.ymax); y++){
      const sy = worldToScreen(0, y)[1];
      svg.appendChild(svgEl('line', {x1: 0, y1: sy, x2: CANVAS_W, y2: sy, class: (y===0?'axis-zero':'axis-grid')}));
    }
  }

  // Plane
  const corners = ['plane_bl','plane_br','plane_tr','plane_tl'].map(k => worldToScreen(...elements[k].pos));
  svg.appendChild(svgEl('path', {
    d: 'M ' + corners[0][0] + ',' + corners[0][1] +
       ' L ' + corners[1][0] + ',' + corners[1][1] +
       ' L ' + corners[2][0] + ',' + corners[2][1] +
       ' L ' + corners[3][0] + ',' + corners[3][1] + ' Z',
    stroke: '#111', 'stroke-width': 2.5, fill: 'none', 'stroke-linejoin': 'round'
  }));

  // Screen coords of every point
  const O_s    = worldToScreen(...elements.O.pos);
  const P_s    = worldToScreen(...elements.P.pos);
  const Pde_s  = worldToScreen(...elements.Pdot_end.pos);
  const p_s    = worldToScreen(...elements.p.pos);
  const pd_s   = worldToScreen(...elements.p_dash.pos);

  // Solid green O→P  (extended slightly past P to mimic the book)
  svg.appendChild(svgEl('line', {x1: O_s[0], y1: O_s[1], x2: P_s[0], y2: P_s[1], stroke: '#22c55e', 'stroke-width': 3, 'stroke-linecap': 'round'}));
  // Dashed green O→Pdot_end
  svg.appendChild(svgEl('line', {x1: O_s[0], y1: O_s[1], x2: Pde_s[0], y2: Pde_s[1], stroke: '#22c55e', 'stroke-width': 2.5, 'stroke-dasharray': '6 4', 'stroke-linecap': 'round'}));
  // Red Ṗ arrow
  svg.appendChild(svgEl('line', {x1: P_s[0], y1: P_s[1], x2: Pde_s[0], y2: Pde_s[1], stroke: '#ef4444', 'stroke-width': 5, 'marker-end': 'url(#redArrow)', 'stroke-linecap': 'round'}));
  // Red ṗ arrow (now explicit from elements.p to elements.p_dash)
  svg.appendChild(svgEl('line', {x1: p_s[0], y1: p_s[1], x2: pd_s[0], y2: pd_s[1], stroke: '#ef4444', 'stroke-width': 4, 'marker-end': 'url(#redArrow)', 'stroke-linecap': 'round'}));

  // Solid black dots
  svg.appendChild(svgEl('circle', {cx: O_s[0], cy: O_s[1], r: 9, fill: '#111'}));
  svg.appendChild(svgEl('circle', {cx: P_s[0], cy: P_s[1], r: 12, fill: '#111'}));
  svg.appendChild(svgEl('circle', {cx: p_s[0], cy: p_s[1], r: 6, fill: '#111'}));

  // Labels — each rendered at its own independent position
  for (const key of ['label_O','label_P','label_p','label_pdot','label_Pdot']){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const t = svgEl('text', {x: sx, y: sy, class: 'label-text'});
    t.textContent = el.text;
    svg.appendChild(t);
  }

  // Handles drawn last so they sit on top
  for (const key of Object.keys(elements)){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const shapeAttrs = {'data-key': key};
    if (el.kind === 'label'){
      // Small square handle so labels are visually distinct
      shapeAttrs.x = sx - 8; shapeAttrs.y = sy - 14; shapeAttrs.width = 16; shapeAttrs.height = 16;
      shapeAttrs.class = 'handle label';
      svg.appendChild(svgEl('rect', shapeAttrs));
    } else if (el.kind === 'corner'){
      shapeAttrs.cx = sx; shapeAttrs.cy = sy; shapeAttrs.r = 13;
      shapeAttrs.class = 'handle corner';
      svg.appendChild(svgEl('circle', shapeAttrs));
    } else {
      shapeAttrs.cx = sx; shapeAttrs.cy = sy; shapeAttrs.r = 13;
      shapeAttrs.class = 'handle point';
      svg.appendChild(svgEl('circle', shapeAttrs));
    }
  }
  document.querySelectorAll('.handle').forEach(h => {
    h.addEventListener('mousedown', startDrag);
  });

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
  const wp = screenToWorld(sx, sy);
  elements[dragging].pos = [maybeSnap(wp[0]), maybeSnap(wp[1])];
  render();
}
function endDrag(){
  dragging = null;
  document.removeEventListener('mousemove', doDrag);
  document.removeEventListener('mouseup', endDrag);
}

function autoSnapProjections(){
  // Recompute p and p_dash as intersections of (O→P) and (O→Pdot_end) with the plane diagonal top-left ↔ bottom-right
  const p_world      = lineIntersect(elements.O.pos, elements.P.pos,        elements.plane_tl.pos, elements.plane_br.pos);
  const pdash_world  = lineIntersect(elements.O.pos, elements.Pdot_end.pos, elements.plane_tl.pos, elements.plane_br.pos);
  elements.p.pos      = [maybeSnap(p_world[0]),      maybeSnap(p_world[1])];
  elements.p_dash.pos = [maybeSnap(pdash_world[0]),  maybeSnap(pdash_world[1])];
  render();
}

function updatePanel(){
  const fmt = (xy) => '(' + xy[0].toFixed(2).padStart(5) + ', ' + xy[1].toFixed(2).padStart(5) + ')';
  const lines = [];
  lines.push('# structural points');
  lines.push('O          = ' + fmt(elements.O.pos));
  lines.push('P          = ' + fmt(elements.P.pos));
  lines.push('Pdot_end   = ' + fmt(elements.Pdot_end.pos));
  lines.push('p          = ' + fmt(elements.p.pos));
  lines.push('p_dash     = ' + fmt(elements.p_dash.pos));
  lines.push('');
  lines.push('# plane corners (BL, BR, TR, TL)');
  lines.push('plane_bl   = ' + fmt(elements.plane_bl.pos));
  lines.push('plane_br   = ' + fmt(elements.plane_br.pos));
  lines.push('plane_tr   = ' + fmt(elements.plane_tr.pos));
  lines.push('plane_tl   = ' + fmt(elements.plane_tl.pos));
  lines.push('');
  lines.push('# label positions');
  lines.push('label_O    = ' + fmt(elements.label_O.pos));
  lines.push('label_P    = ' + fmt(elements.label_P.pos));
  lines.push('label_p    = ' + fmt(elements.label_p.pos));
  lines.push('label_pdot = ' + fmt(elements.label_pdot.pos));
  lines.push('label_Pdot = ' + fmt(elements.label_Pdot.pos));
  document.getElementById('coords').textContent = lines.join('\n');
  exportPython();
}

function exportPython(){
  const f = (xy) => 'torch.tensor([' + xy[0].toFixed(3) + ', ' + xy[1].toFixed(3) + '])';
  const code =
    '# Auto-generated by the drag-and-drop editor (output/figure-editor/fig47_01.html)\n' +
    '# --- structural points ---\n' +
    'O        = ' + f(elements.O.pos)        + '\n' +
    'P        = ' + f(elements.P.pos)        + '\n' +
    'Pdot_end = ' + f(elements.Pdot_end.pos) + '\n' +
    'p        = ' + f(elements.p.pos)        + '\n' +
    'p_dash   = ' + f(elements.p_dash.pos)   + '\n' +
    '\n' +
    '# --- plane corners (closed loop: BL -> BR -> TR -> TL -> BL) ---\n' +
    'plane = torch.tensor([\n' +
    '    [' + elements.plane_bl.pos[0].toFixed(3) + ', ' + elements.plane_bl.pos[1].toFixed(3) + '],   # bottom-LEFT  (BACK)\n' +
    '    [' + elements.plane_br.pos[0].toFixed(3) + ', ' + elements.plane_br.pos[1].toFixed(3) + '],   # bottom-RIGHT (FRONT)\n' +
    '    [' + elements.plane_tr.pos[0].toFixed(3) + ', ' + elements.plane_tr.pos[1].toFixed(3) + '],   # top-RIGHT    (FRONT)\n' +
    '    [' + elements.plane_tl.pos[0].toFixed(3) + ', ' + elements.plane_tl.pos[1].toFixed(3) + '],   # top-LEFT     (BACK)\n' +
    '    [' + elements.plane_bl.pos[0].toFixed(3) + ', ' + elements.plane_bl.pos[1].toFixed(3) + '],\n' +
    '])\n' +
    '\n' +
    '# --- label positions (each math symbol is placed manually) ---\n' +
    'label_positions = {\n' +
    '    "O":    ' + f(elements.label_O.pos)    + ',\n' +
    '    "P":    ' + f(elements.label_P.pos)    + ',\n' +
    '    "p":    ' + f(elements.label_p.pos)    + ',\n' +
    '    "pdot": ' + f(elements.label_pdot.pos) + ',\n' +
    '    "Pdot": ' + f(elements.label_Pdot.pos) + ',\n' +
    '}\n';
  document.getElementById('export').textContent = code;
}

function resetAll(){
  elements = deepClone(DEFAULTS);
  render();
}

document.getElementById('opacity').addEventListener('input', function(e){
  document.getElementById('bookImg').style.opacity = (e.target.value / 100).toString();
});
document.getElementById('bookImg').style.opacity = 0.45;
document.getElementById('showGrid').addEventListener('change', render);
document.getElementById('snap').addEventListener('change', render);

render();
</script>
</body>
</html>
"""

html = HTML_TEMPLATE.replace("__BASE64__", book_b64)
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html} ({len(html)/1024:.0f} KB)")
