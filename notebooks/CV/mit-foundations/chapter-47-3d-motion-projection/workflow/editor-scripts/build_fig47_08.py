"""Drag-to-position editor for Fig 47.8 (Euler angles diagram).

Elements: origin, 3 axis tips (X/Y/Z), 4 image-plane corners, inner-plane axes,
points P and p, 3 rotation arrows (yaw/pitch/roll), and all labels.

The 3 rotation arrows are drawn as oriented 3/4 ellipses around each axis:
  * Yaw — horizontal ellipse around the Y (vertical) axis
  * Pitch — tilted ellipse around the X (diagonal) axis
  * Roll — vertical ellipse around the Z (horizontal) axis
Drag the centre of each rotation arrow to position it; the orientation is fixed.
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_08.png"
out_html = HERE / "editor-output" / "fig47_08.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.8 - Drag-to-position editor</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui;margin:0;padding:16px;background:#f5f5f5;color:#222}
  h1{margin:0 0 8px;font-size:18px}
  .top{display:flex;gap:16px;align-items:flex-start;margin-bottom:12px;flex-wrap:wrap}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  button{padding:6px 12px;background:#2563eb;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:13px}
  button.secondary{background:#6b7280}
  input[type=range]{width:140px;vertical-align:middle}
  label{font-size:13px;color:#444}
  .editor{display:flex;gap:16px}
  .canvas-wrap{position:relative;width:900px;height:720px;background:#fff;border:1px solid #ccc;border-radius:4px;flex-shrink:0;user-select:none}
  .canvas-wrap img{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  .canvas-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%}
  .panel-info{flex:1;min-width:340px;background:#fff;padding:12px;border:1px solid #ccc;border-radius:4px}
  .panel-info h3{margin:8px 0 4px;font-size:13px;color:#555;border-bottom:1px solid #eee;padding-bottom:4px}
  .panel-info pre{margin:0;font-family:'Cascadia Code',Consolas,monospace;font-size:11px;background:#f9fafb;border:1px solid #eee;padding:8px;border-radius:4px;white-space:pre-wrap;max-height:600px;overflow:auto}
  .handle{cursor:move;stroke-width:1.5;transition:fill-opacity 0.1s}
  .handle:hover{fill-opacity:.8}
  .handle.axis    {fill:#111;fill-opacity:.35;stroke:#111}
  .handle.corner  {fill:#3b82f6;fill-opacity:.4;stroke:#3b82f6}
  .handle.inner   {fill:#16a34a;fill-opacity:.4;stroke:#16a34a}
  .handle.point   {fill:#22d3ee;fill-opacity:.5;stroke:#0891b2}
  .handle.rot     {fill:#ef4444;fill-opacity:.4;stroke:#ef4444}
  .handle.label   {fill:#a855f7;fill-opacity:.4;stroke:#a855f7}
  .label-text{font-size:14px;fill:#111;font-weight:700;pointer-events:none;font-family:'Times New Roman',serif}
  .axis-grid{stroke:#e5e7eb;stroke-width:1}
  .legend{font-size:11px;color:#555;margin-top:8px;line-height:1.7}
  .swatch{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin-right:4px}
</style></head>
<body>
<h1>Figure 47.8 - drag axes, plane corners, points, and rotation-arrow centres</h1>
<div class="top">
  <div class="controls">
    <label>Book opacity <input type="range" id="opacity" min="0" max="100" value="40"/></label>
    <label>Rot. arrow size <input type="range" id="rotSize" min="10" max="50" value="20"/> <span id="rotSizeVal">20</span></label>
    <label><input type="checkbox" id="showGrid" checked/> Grid</label>
    <label><input type="checkbox" id="snap"/> Snap to 0.1</label>
    <button onclick="resetAll()" class="secondary">Reset</button>
    <button onclick="exportPython()">Export Python</button>
  </div>
</div>
<div class="legend">
  <span class="swatch" style="background:#111"></span>black = origin &amp; axis tips
  &nbsp;<span class="swatch" style="background:#3b82f6"></span>blue = image-plane corners
  &nbsp;<span class="swatch" style="background:#16a34a"></span>green = inner x/y axes
  &nbsp;<span class="swatch" style="background:#22d3ee"></span>cyan = P and p
  &nbsp;<span class="swatch" style="background:#ef4444"></span>red = rotation arrow centres
  &nbsp;<span class="swatch" style="background:#a855f7"></span>purple = labels
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 900 720" preserveAspectRatio="xMidYMid meet" id="canvas">
      <defs>
        <marker id="blackArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#111"/>
        </marker>
        <marker id="greenArrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#16a34a"/>
        </marker>
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
const CANVAS_W = 900, CANVAS_H = 720;
const AXIS = { xmin: -1.0, xmax: 8.0, ymin: -0.5, ymax: 6.0 };
let ROT_SIZE = 20;   // rotation arrow ellipse radius in pixels (controlled by slider)

const DEFAULTS = {
  // 3D axes
  origin: {pos: [ 0.5, 0.5], kind: 'axis'},
  Y_tip:  {pos: [ 0.5, 5.0], kind: 'axis'},
  X_tip:  {pos: [ 3.5, 3.0], kind: 'axis'},
  Z_tip:  {pos: [ 7.0, 0.5], kind: 'axis'},

  // Image plane corners
  plane_bl: {pos: [ 1.7, 0.3], kind: 'corner'},
  plane_br: {pos: [ 3.0, 0.6], kind: 'corner'},
  plane_tr: {pos: [ 3.0, 2.2], kind: 'corner'},
  plane_tl: {pos: [ 1.7, 1.9], kind: 'corner'},

  // Inner plane axes — origin + 2 tips
  inner_O:    {pos: [ 1.8, 0.4], kind: 'inner'},
  inner_x_tip:{pos: [ 2.6, 0.6], kind: 'inner'},
  inner_y_tip:{pos: [ 1.8, 1.8], kind: 'inner'},

  // Cyan points
  p: {pos: [ 2.3, 1.1], kind: 'point'},
  P: {pos: [ 6.0, 3.8], kind: 'point'},

  // Rotation arrow centres
  yaw_c:   {pos: [ 0.3, 4.3], kind: 'rot'},
  pitch_c: {pos: [ 2.3, 2.5], kind: 'rot'},
  roll_c:  {pos: [ 6.0, 0.8], kind: 'rot'},

  // Labels
  label_origin: {pos: [ 0.3, 0.0], kind: 'label', text: '(0,0,0)'},
  label_Y:      {pos: [ 0.6, 5.2], kind: 'label', text: 'Y'},
  label_X:      {pos: [ 3.6, 3.0], kind: 'label', text: 'X'},
  label_Z:      {pos: [ 7.1, 0.4], kind: 'label', text: 'Z'},
  label_yaw:    {pos: [-0.7, 4.5], kind: 'label', text: 'Yaw θ_Y'},
  label_pitch:  {pos: [ 1.8, 3.0], kind: 'label', text: 'Pitch θ_X'},
  label_roll:   {pos: [ 6.0, 0.3], kind: 'label', text: 'Roll θ_Z'},
  label_p:      {pos: [ 2.4, 1.4], kind: 'label', text: 'p'},
  label_P:      {pos: [ 6.2, 4.0], kind: 'label', text: 'P'},
  // Small green inner-plane axis labels
  label_inner_x:{pos: [ 2.7, 0.7], kind: 'label', text: 'x'},
  label_inner_y:{pos: [ 1.9, 1.6], kind: 'label', text: 'y'},
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

// Draw a rotation arrow centred at (cx, cy) screen pixels.
// kind: 'yaw' (horizontal ellipse around vertical Y axis)
//       'pitch' (tilted ~45° ellipse around diagonal X axis)
//       'roll'  (vertical ellipse around horizontal Z axis)
function drawRotationArrow(svg, cx, cy, kind){
  // Each ellipse is defined by (rx, ry) and a rotation angle.
  let rx, ry, rotDeg;
  if (kind === 'yaw')   { rx = ROT_SIZE * 1.3; ry = ROT_SIZE * 0.45; rotDeg =   0; }
  if (kind === 'pitch') { rx = ROT_SIZE * 1.1; ry = ROT_SIZE * 0.45; rotDeg = -30; }
  if (kind === 'roll')  { rx = ROT_SIZE * 0.50; ry = ROT_SIZE * 1.2; rotDeg =   0; }

  // 3/4 arc from theta=0 to theta=1.5π (in the ellipse's own frame), then rotated by rotDeg
  // Compute the ellipse's start and end points in screen coords.
  const rad = (deg) => deg * Math.PI / 180;
  const rotRad = rad(rotDeg);
  const cosR = Math.cos(rotRad), sinR = Math.sin(rotRad);
  const startT = 0.1 * Math.PI;
  const endT   = 1.7 * Math.PI;
  // Local (ellipse-frame) points
  const lsx = rx * Math.cos(startT), lsy = ry * Math.sin(startT);
  const lex = rx * Math.cos(endT),   ley = ry * Math.sin(endT);
  // Rotate to global frame
  const sx = cx + (lsx * cosR - lsy * sinR);
  const sy = cy + (lsx * sinR + lsy * cosR);
  const ex = cx + (lex * cosR - ley * sinR);
  const ey = cy + (lex * sinR + ley * cosR);
  // SVG arc command: A rx,ry rotation large-arc-flag sweep-flag x,y
  // sweepFlag = 1 for clockwise in screen coords
  svg.appendChild(svgEl('path', {
    d: `M ${sx},${sy} A ${rx},${ry} ${rotDeg} 1 1 ${ex},${ey}`,
    stroke: '#ef4444', 'stroke-width': 2.5, fill: 'none', 'stroke-linecap': 'round',
    'marker-end': 'url(#redArrow)'
  }));
}

function render(){
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

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

  const O_s = worldToScreen(...elements.origin.pos);

  // 3D axes (origin to tips, with arrowheads)
  for (const tipKey of ['Y_tip','X_tip','Z_tip']){
    const t_s = worldToScreen(...elements[tipKey].pos);
    svg.appendChild(svgEl('line', {x1: O_s[0], y1: O_s[1], x2: t_s[0], y2: t_s[1],
                                    stroke: '#111', 'stroke-width': 2.2,
                                    'marker-end': 'url(#blackArrow)'}));
  }

  // Image plane (light grey filled quadrilateral)
  const corners = ['plane_bl','plane_br','plane_tr','plane_tl'].map(k => worldToScreen(...elements[k].pos));
  svg.appendChild(svgEl('path', {
    d: `M ${corners[0][0]},${corners[0][1]} L ${corners[1][0]},${corners[1][1]} L ${corners[2][0]},${corners[2][1]} L ${corners[3][0]},${corners[3][1]} Z`,
    fill: '#e7eef0', 'fill-opacity': 0.55,
    stroke: '#111', 'stroke-width': 1.3
  }));

  // Inner x and y axes on the image plane (green)
  const iO_s = worldToScreen(...elements.inner_O.pos);
  const ix_s = worldToScreen(...elements.inner_x_tip.pos);
  const iy_s = worldToScreen(...elements.inner_y_tip.pos);
  svg.appendChild(svgEl('line', {x1: iO_s[0], y1: iO_s[1], x2: ix_s[0], y2: ix_s[1],
                                  stroke: '#16a34a', 'stroke-width': 1.8,
                                  'marker-end': 'url(#greenArrow)'}));
  svg.appendChild(svgEl('line', {x1: iO_s[0], y1: iO_s[1], x2: iy_s[0], y2: iy_s[1],
                                  stroke: '#16a34a', 'stroke-width': 1.8,
                                  'marker-end': 'url(#greenArrow)'}));

  // Dashed line from origin (through p) to P
  const p_s = worldToScreen(...elements.p.pos);
  const P_s = worldToScreen(...elements.P.pos);
  svg.appendChild(svgEl('line', {x1: O_s[0], y1: O_s[1], x2: P_s[0], y2: P_s[1],
                                  stroke: '#888', 'stroke-width': 1.0, 'stroke-dasharray': '4 3'}));

  // Cyan points
  svg.appendChild(svgEl('circle', {cx: p_s[0], cy: p_s[1], r: 7, fill: '#22d3ee', stroke: '#0891b2', 'stroke-width': 1.0}));
  svg.appendChild(svgEl('circle', {cx: P_s[0], cy: P_s[1], r: 9, fill: '#22d3ee', stroke: '#0891b2', 'stroke-width': 1.0}));

  // Origin black dot
  svg.appendChild(svgEl('circle', {cx: O_s[0], cy: O_s[1], r: 7, fill: '#111'}));

  // Rotation arrows
  const yaw_s   = worldToScreen(...elements.yaw_c.pos);
  const pitch_s = worldToScreen(...elements.pitch_c.pos);
  const roll_s  = worldToScreen(...elements.roll_c.pos);
  drawRotationArrow(svg, yaw_s[0], yaw_s[1], 'yaw');
  drawRotationArrow(svg, pitch_s[0], pitch_s[1], 'pitch');
  drawRotationArrow(svg, roll_s[0], roll_s[1], 'roll');

  // Labels
  for (const key of Object.keys(elements)){
    if (!key.startsWith('label_')) continue;
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const attrs = {x: sx, y: sy, class: 'label-text'};
    if (key === 'label_inner_x' || key === 'label_inner_y'){
      attrs.fill = '#16a34a';   // green to match the inner axis arrows
      attrs['font-size'] = 12;
    }
    const t = svgEl('text', attrs);
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
      attrs.cx = sx; attrs.cy = sy; attrs.r = 11;
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
  const labels = {};
  for (const k of Object.keys(elements)){
    if (k.startsWith('label_')) labels[k.replace('label_','')] = elements[k].pos;
  }
  let code = '# Auto-generated by output/figure-editor/fig47_08.html\n';
  code += 'origin   = ' + f(elements.origin.pos) + '\n';
  code += 'Y_tip    = ' + f(elements.Y_tip.pos)  + '\n';
  code += 'X_tip    = ' + f(elements.X_tip.pos)  + '\n';
  code += 'Z_tip    = ' + f(elements.Z_tip.pos)  + '\n';
  code += '\nplane = torch.tensor([\n';
  for (const k of ['plane_bl','plane_br','plane_tr','plane_tl','plane_bl']){
    code += '    [' + elements[k.replace('_bl_extra','_bl')].pos[0].toFixed(3) + ', ' + elements[k.replace('_bl_extra','_bl')].pos[1].toFixed(3) + '],\n';
  }
  code += '])\n\n';
  code += 'inner_O     = ' + f(elements.inner_O.pos)     + '\n';
  code += 'inner_x_tip = ' + f(elements.inner_x_tip.pos) + '\n';
  code += 'inner_y_tip = ' + f(elements.inner_y_tip.pos) + '\n';
  code += '\np = ' + f(elements.p.pos) + '\n';
  code += 'P = ' + f(elements.P.pos) + '\n';
  code += '\nyaw_centre   = ' + f(elements.yaw_c.pos)   + '\n';
  code += 'pitch_centre = ' + f(elements.pitch_c.pos) + '\n';
  code += 'roll_centre  = ' + f(elements.roll_c.pos)  + '\n';
  code += 'ROT_SIZE     = ' + ROT_SIZE + '   # arrow size (pixels in editor; convert to world units in matplotlib)\n';
  code += '\nlabels = {\n';
  for (const [k, v] of Object.entries(labels)){
    code += '    "' + k + '": ' + f(v) + ',\n';
  }
  code += '}\n';
  document.getElementById('export').textContent = code;
}

function resetAll(){
  elements = JSON.parse(JSON.stringify(DEFAULTS));
  ROT_SIZE = 20; document.getElementById('rotSize').value = 20; document.getElementById('rotSizeVal').textContent = '20';
  render();
}

document.getElementById('opacity').addEventListener('input', e => {
  document.getElementById('bookImg').style.opacity = (e.target.value / 100).toString();
});
document.getElementById('bookImg').style.opacity = 0.40;
document.getElementById('showGrid').addEventListener('change', render);
document.getElementById('snap').addEventListener('change', render);
document.getElementById('rotSize').addEventListener('input', e => {
  ROT_SIZE = parseInt(e.target.value);
  document.getElementById('rotSizeVal').textContent = ROT_SIZE;
  render();
});

render();
</script>
</body></html>
"""

html = HTML.replace("__BASE64__", book_b64)
out_html.write_text(html, encoding="utf-8")
print(f"Wrote {out_html} ({len(html)/1024:.0f} KB)")
