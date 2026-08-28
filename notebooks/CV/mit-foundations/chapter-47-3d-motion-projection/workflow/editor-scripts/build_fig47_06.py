"""Drag-to-position editor for Fig 47.6 (forward camera motion).

Same 5-dot pyramid structure as Fig 47.4, but rotated:
  * Apex (dot) on the LEFT of each camera
  * Image plane (quadrilateral) on the RIGHT of each apex
  * 4 edges from corners converging at the apex

5 yellow dots define the pyramid shape (apex + 4 corners of camera C0).
4 blue dots position cameras C1-C4 (each shares C0's shape).
Red arrows run between consecutive apex dots, starting AFTER each image plane's
right edge and ending just before the next apex.
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_06.png"
out_html = HERE / "editor-output" / "fig47_06.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.6 - Drag-to-position editor</title>
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
  .canvas-wrap{position:relative;width:1300px;height:280px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
  .canvas-wrap img{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  .canvas-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%}
  .panel-info{flex:1;min-width:340px;background:#fff;padding:12px;border:1px solid #ccc;border-radius:4px}
  .panel-info h3{margin:8px 0 4px;font-size:13px;color:#555;border-bottom:1px solid #eee;padding-bottom:4px}
  .panel-info pre{margin:0;font-family:'Cascadia Code',Consolas,monospace;font-size:11px;background:#f9fafb;border:1px solid #eee;padding:8px;border-radius:4px;white-space:pre-wrap}
  .handle{cursor:move;stroke-width:1.5;transition:fill-opacity 0.1s}
  .handle:hover{fill-opacity:.8}
  .handle.shape  {fill:#eab308;fill-opacity:.5;stroke:#a16207}
  .handle.pos    {fill:#3b82f6;fill-opacity:.4;stroke:#3b82f6}
  .axis-grid{stroke:#e5e7eb;stroke-width:1}
  .legend{font-size:12px;color:#555;margin-top:8px;line-height:1.7}
  .swatch{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin-right:4px}
</style></head>
<body>
<h1>Figure 47.6 - drag the 5 yellow dots (apex + 4 corners). Apex on LEFT, image plane on RIGHT.</h1>
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
  <span class="swatch" style="background:#eab308"></span>yellow = pyramid shape (apex + 4 image-plane corners of camera 1)
  &nbsp;&nbsp;<span class="swatch" style="background:#3b82f6"></span>blue = apex positions of cameras 2-5
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 1300 280" preserveAspectRatio="xMidYMid meet" id="canvas">
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
const CANVAS_W = 1300, CANVAS_H = 280;
const AXIS = { xmin: -0.5, xmax: 12.5, ymin: -1.0, ymax: 1.0 };

// 5 shape dots define the pyramid for camera 0 (apex on LEFT, image plane on RIGHT).
const DEFAULTS = {
  C0_apex: {pos: [ 1.50,  0.00], kind: 'shape'},
  C0_TL:   {pos: [ 2.00,  0.50], kind: 'shape'},
  C0_TR:   {pos: [ 2.50,  0.50], kind: 'shape'},
  C0_BR:   {pos: [ 2.50, -0.50], kind: 'shape'},
  C0_BL:   {pos: [ 2.00, -0.50], kind: 'shape'},
  C1: {pos: [ 3.70, 0.00], kind: 'pos'},
  C2: {pos: [ 5.90, 0.00], kind: 'pos'},
  C3: {pos: [ 8.10, 0.00], kind: 'pos'},
  C4: {pos: [10.30, 0.00], kind: 'pos'},
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
function maybeSnap(v){return document.getElementById('snap').checked ? Math.round(v*10)/10 : v;}
function svgEl(tag, attrs){
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function getShape(){
  const apex = elements.C0_apex.pos;
  return {
    TL: [elements.C0_TL.pos[0] - apex[0], elements.C0_TL.pos[1] - apex[1]],
    TR: [elements.C0_TR.pos[0] - apex[0], elements.C0_TR.pos[1] - apex[1]],
    BR: [elements.C0_BR.pos[0] - apex[0], elements.C0_BR.pos[1] - apex[1]],
    BL: [elements.C0_BL.pos[0] - apex[0], elements.C0_BL.pos[1] - apex[1]],
  };
}
// Largest +x offset of any corner — used to position arrow tails just past the image plane
function shapeMaxX(shape){
  return Math.max(shape.TL[0], shape.TR[0], shape.BR[0], shape.BL[0]);
}

function drawCamera(svg, apex, shape){
  const TL = worldToScreen(apex[0] + shape.TL[0], apex[1] + shape.TL[1]);
  const TR = worldToScreen(apex[0] + shape.TR[0], apex[1] + shape.TR[1]);
  const BR = worldToScreen(apex[0] + shape.BR[0], apex[1] + shape.BR[1]);
  const BL = worldToScreen(apex[0] + shape.BL[0], apex[1] + shape.BL[1]);
  const apexS = worldToScreen(...apex);

  svg.appendChild(svgEl('path', {
    d: 'M ' + TL[0] + ',' + TL[1] + ' L ' + TR[0] + ',' + TR[1]
     + ' L ' + BR[0] + ',' + BR[1] + ' L ' + BL[0] + ',' + BL[1] + ' Z',
    stroke: '#111', 'stroke-width': 1.5, fill: 'none', 'stroke-linejoin': 'round'
  }));
  for (const c of [TL, TR, BL, BR]){
    svg.appendChild(svgEl('line', {x1: c[0], y1: c[1], x2: apexS[0], y2: apexS[1],
                                    stroke: '#111', 'stroke-width': 1.2}));
  }
  svg.appendChild(svgEl('circle', {cx: apexS[0], cy: apexS[1], r: 5, fill: '#111'}));
}

function render(){
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

  if (document.getElementById('showGrid').checked){
    for (let x = Math.ceil(AXIS.xmin); x <= Math.floor(AXIS.xmax); x++){
      const sx = worldToScreen(x, 0)[0];
      svg.appendChild(svgEl('line', {x1: sx, y1: 0, x2: sx, y2: CANVAS_H, class: 'axis-grid'}));
    }
  }

  const shape = getShape();
  const maxX = shapeMaxX(shape);
  const apexes = [elements.C0_apex.pos, elements.C1.pos, elements.C2.pos, elements.C3.pos, elements.C4.pos];

  // Red arrows between consecutive cameras:
  //   tail just past previous camera's image plane (apex.x + maxX + 0.1)
  //   head just before next apex (next.x - 0.05)
  for (let i = 0; i < apexes.length - 1; i++){
    const p0 = apexes[i], p1 = apexes[i+1];
    const ay = (p0[1] + p1[1]) / 2;
    const s_w = [p0[0] + maxX + 0.10, ay];
    const e_w = [p1[0] - 0.05, ay];
    const s_s = worldToScreen(...s_w);
    const e_s = worldToScreen(...e_w);
    svg.appendChild(svgEl('line', {x1: s_s[0], y1: s_s[1], x2: e_s[0], y2: e_s[1],
                                    stroke: '#ef4444', 'stroke-width': 4.5, 'stroke-linecap': 'round',
                                    'marker-end': 'url(#redArrow)'}));
  }
  // Leading arrow (BEFORE camera 1, points INTO C0's apex)
  const first = apexes[0];
  const lead_s = worldToScreen(first[0] - 1.4, first[1]);
  const lead_e = worldToScreen(first[0] - 0.05, first[1]);
  svg.appendChild(svgEl('line', {x1: lead_s[0], y1: lead_s[1], x2: lead_e[0], y2: lead_e[1],
                                  stroke: '#ef4444', 'stroke-width': 4.5, 'stroke-linecap': 'round',
                                  'marker-end': 'url(#redArrow)'}));
  // Trailing arrow (AFTER camera 5)
  const last = apexes[apexes.length-1];
  const tail_s = worldToScreen(last[0] + maxX + 0.10, last[1]);
  const tail_e = worldToScreen(last[0] + maxX + 1.5, last[1]);
  svg.appendChild(svgEl('line', {x1: tail_s[0], y1: tail_s[1], x2: tail_e[0], y2: tail_e[1],
                                  stroke: '#ef4444', 'stroke-width': 4.5, 'stroke-linecap': 'round',
                                  'marker-end': 'url(#redArrow)'}));

  // Draw all 5 cameras
  for (const apex of apexes) drawCamera(svg, apex, shape);

  // Handles
  for (const key of Object.keys(elements)){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    svg.appendChild(svgEl('circle', {cx: sx, cy: sy, r: 13, class: 'handle ' + el.kind, 'data-key': key}));
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
  const shape = getShape();
  const code =
    '# Auto-generated by output/figure-editor/fig47_06.html\n' +
    '# Each camera is a pyramid: apex (dot) on the LEFT, image plane on the RIGHT.\n' +
    '# All 5 cameras share the same shape; only the apex position differs per camera.\n' +
    '\n' +
    '# --- shape: offsets from each apex to its 4 image-plane corners ---\n' +
    'corner_offsets = {\n' +
    '    "TL": ' + f(shape.TL) + ',\n' +
    '    "TR": ' + f(shape.TR) + ',\n' +
    '    "BR": ' + f(shape.BR) + ',\n' +
    '    "BL": ' + f(shape.BL) + ',\n' +
    '}\n' +
    '\n' +
    '# --- apex positions ---\n' +
    'apexes = [\n' +
    '    ' + f(elements.C0_apex.pos) + ',\n' +
    '    ' + f(elements.C1.pos)      + ',\n' +
    '    ' + f(elements.C2.pos)      + ',\n' +
    '    ' + f(elements.C3.pos)      + ',\n' +
    '    ' + f(elements.C4.pos)      + ',\n' +
    ']\n';
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
