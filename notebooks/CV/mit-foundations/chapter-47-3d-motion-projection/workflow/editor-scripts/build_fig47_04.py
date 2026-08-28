"""Drag-to-position editor for Fig 47.4 (lateral camera translation).

Each camera is a pyramid: 4 image-plane corners + 1 apex (dot at the bottom).
The SHAPE is defined by 5 draggable dots on camera C0 (apex + 4 corners).
Cameras C1-C4 share that same shape but each has its own apex position.

Total handles:
  * 5 yellow dots — shape definition (apex_0, TL, TR, BR, BL of the prototype)
  * 4 blue dots   — apex positions of cameras 1-4 (their image planes inherit C0's shape)
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_04.png"
out_html = HERE / "editor-output" / "fig47_04.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.4 - Drag-to-position editor</title>
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
  .canvas-wrap{position:relative;width:1300px;height:250px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
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
<h1>Figure 47.4 - drag the 5 yellow dots to shape the pyramid; drag the 4 blue dots to place cameras 2-5</h1>
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
  <span class="swatch" style="background:#eab308"></span>yellow = shape dots (apex + 4 image-plane corners of camera 1)
  &nbsp;&nbsp;<span class="swatch" style="background:#3b82f6"></span>blue = apex positions of cameras 2-5
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 1300 250" preserveAspectRatio="xMidYMid meet" id="canvas">
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
const CANVAS_W = 1300, CANVAS_H = 250;
const AXIS = { xmin: -0.5, xmax: 12.5, ymin: -0.5, ymax: 1.0 };

// 5 shape dots (defining camera 0: apex + 4 corners) + 4 apex positions for the other cameras
const DEFAULTS = {
  C0_apex: {pos: [ 1.50, 0.00], kind: 'shape'},
  C0_TL:   {pos: [ 0.95, 0.50], kind: 'shape'},
  C0_TR:   {pos: [ 2.05, 0.50], kind: 'shape'},
  C0_BR:   {pos: [ 1.90, 0.20], kind: 'shape'},
  C0_BL:   {pos: [ 1.10, 0.20], kind: 'shape'},
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

// Return SHAPE = offsets of TL/TR/BR/BL from C0_apex
function getShape(){
  const apex = elements.C0_apex.pos;
  return {
    TL: [elements.C0_TL.pos[0] - apex[0], elements.C0_TL.pos[1] - apex[1]],
    TR: [elements.C0_TR.pos[0] - apex[0], elements.C0_TR.pos[1] - apex[1]],
    BR: [elements.C0_BR.pos[0] - apex[0], elements.C0_BR.pos[1] - apex[1]],
    BL: [elements.C0_BL.pos[0] - apex[0], elements.C0_BL.pos[1] - apex[1]],
  };
}

// Draw a pyramid camera at apex (cx, cy) using the given shape offsets
function drawCamera(svg, apex, shape){
  const TL = worldToScreen(apex[0] + shape.TL[0], apex[1] + shape.TL[1]);
  const TR = worldToScreen(apex[0] + shape.TR[0], apex[1] + shape.TR[1]);
  const BR = worldToScreen(apex[0] + shape.BR[0], apex[1] + shape.BR[1]);
  const BL = worldToScreen(apex[0] + shape.BL[0], apex[1] + shape.BL[1]);
  const apexS = worldToScreen(...apex);

  // Image plane (quadrilateral outline)
  svg.appendChild(svgEl('path', {
    d: 'M ' + TL[0] + ',' + TL[1] + ' L ' + TR[0] + ',' + TR[1]
     + ' L ' + BR[0] + ',' + BR[1] + ' L ' + BL[0] + ',' + BL[1] + ' Z',
    stroke: '#111', 'stroke-width': 1.5, fill: 'none', 'stroke-linejoin': 'round'
  }));

  // 4 pyramid edges from corners to apex
  for (const c of [TL, TR, BL, BR]){
    svg.appendChild(svgEl('line', {x1: c[0], y1: c[1], x2: apexS[0], y2: apexS[1],
                                    stroke: '#111', 'stroke-width': 1.2}));
  }
  // Apex dot
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
  const apexes = [elements.C0_apex.pos, elements.C1.pos, elements.C2.pos, elements.C3.pos, elements.C4.pos];

  // Red arrows between consecutive apex dots
  for (let i = 0; i < apexes.length - 1; i++){
    const p0 = apexes[i], p1 = apexes[i+1];
    const s_w = [p0[0] + 0.10, (p0[1] + p1[1])/2];
    const e_w = [p1[0] - 0.10, (p0[1] + p1[1])/2];
    const s_s = worldToScreen(...s_w);
    const e_s = worldToScreen(...e_w);
    svg.appendChild(svgEl('line', {x1: s_s[0], y1: s_s[1], x2: e_s[0], y2: e_s[1],
                                    stroke: '#ef4444', 'stroke-width': 4.5, 'stroke-linecap': 'round',
                                    'marker-end': 'url(#redArrow)'}));
  }
  // Leading arrow
  const first = apexes[0];
  const lead_s = worldToScreen(first[0] - 1.4, first[1]);
  const lead_e = worldToScreen(first[0] - 0.10, first[1]);
  svg.appendChild(svgEl('line', {x1: lead_s[0], y1: lead_s[1], x2: lead_e[0], y2: lead_e[1],
                                  stroke: '#ef4444', 'stroke-width': 4.5, 'stroke-linecap': 'round',
                                  'marker-end': 'url(#redArrow)'}));
  // Trailing arrow
  const last = apexes[apexes.length-1];
  const tail_s = worldToScreen(last[0] + 0.10, last[1]);
  const tail_e = worldToScreen(last[0] + 1.4, last[1]);
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
  const fOffset = (xy) => 'torch.tensor([' + xy[0].toFixed(3) + ', ' + xy[1].toFixed(3) + '])';
  const code =
    '# Auto-generated by output/figure-editor/fig47_04.html\n' +
    '# Pyramid shape is defined by the 5 yellow dots on camera C0.\n' +
    '# All cameras share this shape; each has its own apex position.\n' +
    '\n' +
    '# --- shape (offsets from each camera apex to its 4 image-plane corners) ---\n' +
    'corner_offsets = {\n' +
    '    "TL": ' + fOffset(shape.TL) + ',\n' +
    '    "TR": ' + fOffset(shape.TR) + ',\n' +
    '    "BR": ' + fOffset(shape.BR) + ',\n' +
    '    "BL": ' + fOffset(shape.BL) + ',\n' +
    '}\n' +
    '\n' +
    '# --- apex positions (one per camera) ---\n' +
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
