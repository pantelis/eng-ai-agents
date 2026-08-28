"""Drag-to-position editor for Fig 47.7 (forward motion toward a brick wall).

LEFT panel: brick wall + "Camera motion" red triangle arrow at bottom.
RIGHT panel: same scene faded + radial motion-field quiver centred on the wall (FoE).

Configurable:
  * Wall centre position (drag handle)
  * Wall width/height (sliders)
  * Horizon y — where the gray ground begins (slider)
  * Camera motion red triangle position (drag)
  * 'Camera motion' label position (drag)
  * FoE position — focus of expansion for the right panel (drag)
"""
import base64
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "editor-output").mkdir(exist_ok=True)

book_png = HERE / "book-figures" / "fig47_07_left.png"
out_html = HERE / "editor-output" / "fig47_07.html"

book_b64 = base64.b64encode(book_png.read_bytes()).decode()

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/>
<title>Fig 47.7 - Drag-to-position editor</title>
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
  .handle.wall {fill:#7a3b18;fill-opacity:.45;stroke:#3d1d0a}
  .handle.foe  {fill:#3b82f6;fill-opacity:.4;stroke:#3b82f6}
  .handle.arrow{fill:#ef4444;fill-opacity:.4;stroke:#ef4444}
  .handle.label{fill:#a855f7;fill-opacity:.4;stroke:#a855f7}
  .label-text{font-size:13px;fill:#111;font-weight:600;pointer-events:none;font-family:system-ui}
</style></head>
<body>
<h1>Figure 47.7 - drag wall centre, FoE, camera arrow, label; sliders control wall size and horizon</h1>
<div class="top">
  <div class="controls">
    <label>Book opacity <input type="range" id="opacity" min="0" max="100" value="40"/></label>
    <label>Wall width <input type="range" id="ww" min="50" max="200" value="170"/> <span id="wwVal">1.70</span></label>
    <label>Wall height <input type="range" id="wh" min="40" max="160" value="115"/> <span id="whVal">1.15</span></label>
    <label>Horizon y <input type="range" id="horY" min="-60" max="20" value="-35"/> <span id="horYVal">-0.35</span></label>
    <label><input type="checkbox" id="snap"/> Snap to 0.05</label>
    <button onclick="resetAll()" class="secondary">Reset</button>
    <button onclick="exportPython()">Export Python</button>
  </div>
</div>
<div class="editor">
  <div class="canvas-wrap" id="wrap">
    <img id="bookImg" src="data:image/png;base64,__BASE64__"/>
    <svg viewBox="0 0 700 520" preserveAspectRatio="xMidYMid meet" id="canvas">
      <defs></defs>
    </svg>
  </div>
  <div class="panel-info">
    <h3>Python - paste into the notebook cell</h3>
    <pre id="export"></pre>
  </div>
</div>
<script>
const CANVAS_W = 700, CANVAS_H = 520;
const AXIS = { xmin: -1.4, xmax: 1.4, ymin: -1.05, ymax: 1.05 };

let WALL_W = 1.70, WALL_H = 1.15, HORIZON_Y = -0.35;

const DEFAULTS = {
  wall_centre: {pos: [ 0.00, 0.275], kind: 'wall'},
  foe:         {pos: [ 0.00, 0.275], kind: 'foe'},
  cam_arrow:   {pos: [ 0.00, -0.80], kind: 'arrow'},
  label_cm:    {pos: [-0.18, -0.95], kind: 'label', text: 'Camera motion'},
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

function drawBrickWall(svg, alpha){
  const cx = elements.wall_centre.pos[0];
  const cy = elements.wall_centre.pos[1];
  const x_left = cx - WALL_W / 2;
  const x_right = cx + WALL_W / 2;
  const y_bot = cy - WALL_H / 2;
  const y_top = cy + WALL_H / 2;

  const rows = 16;
  const brick_h = WALL_H / rows;
  const brick_w_target = 0.17;
  for (let row = 0; row < rows; row++){
    const y0 = y_bot + row * brick_h;
    const offset = (row % 2 === 0) ? 0 : brick_w_target / 2;
    let x = x_left + offset;
    while (x < x_right){
      const x_end = Math.min(x + brick_w_target, x_right);
      const tl = worldToScreen(x, y0 + brick_h);
      const br = worldToScreen(x_end, y0);
      svg.appendChild(svgEl('rect', {
        x: tl[0], y: tl[1], width: br[0] - tl[0], height: br[1] - tl[1],
        fill: '#7a3b18', 'fill-opacity': alpha, stroke: '#2c160a', 'stroke-width': 0.6
      }));
      x = x_end;
    }
  }
}

function render(){
  const svg = document.getElementById('canvas');
  [...svg.childNodes].forEach(n => { if (n.nodeName !== 'defs') svg.removeChild(n); });

  // Ground (below horizon) — gray
  const horizon_s = worldToScreen(AXIS.xmin, HORIZON_Y);
  const bottom_s = worldToScreen(AXIS.xmax, AXIS.ymin);
  svg.appendChild(svgEl('rect', {x: 0, y: horizon_s[1], width: CANVAS_W,
                                  height: bottom_s[1] - horizon_s[1],
                                  fill: '#a0a0a0', 'fill-opacity': 0.7}));
  // Background above horizon — light gray
  const top_s = worldToScreen(AXIS.xmin, AXIS.ymax);
  svg.appendChild(svgEl('rect', {x: 0, y: top_s[1], width: CANVAS_W,
                                  height: horizon_s[1] - top_s[1],
                                  fill: '#ededed', 'fill-opacity': 0.7}));

  // Brick wall
  drawBrickWall(svg, 1.0);

  // Camera motion red triangle (apex pointing UP toward wall)
  const tri_centre = elements.cam_arrow.pos;
  const tri_pts = [
    worldToScreen(tri_centre[0] - 0.07, tri_centre[1]),
    worldToScreen(tri_centre[0] + 0.07, tri_centre[1]),
    worldToScreen(tri_centre[0],         tri_centre[1] + 0.13),
  ];
  svg.appendChild(svgEl('polygon', {
    points: tri_pts.map(p => p.join(',')).join(' '),
    fill: '#ef4444', stroke: '#ef4444', 'stroke-width': 1
  }));

  // FoE marker (small cross — only visible as a handle, no visual rendering needed)

  // Label
  const lbl_s = worldToScreen(...elements.label_cm.pos);
  const t = svgEl('text', {x: lbl_s[0], y: lbl_s[1], class: 'label-text', 'text-anchor': 'middle'});
  t.textContent = elements.label_cm.text;
  svg.appendChild(t);

  // Handles
  for (const key of Object.keys(elements)){
    const el = elements[key];
    const [sx, sy] = worldToScreen(...el.pos);
    const attrs = {'data-key': key};
    if (el.kind === 'label'){
      attrs.x = sx - 40; attrs.y = sy - 14; attrs.width = 90; attrs.height = 18;
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
    '# Auto-generated by output/figure-editor/fig47_07.html\n' +
    'wall_centre = ' + f(elements.wall_centre.pos) + '\n' +
    'foe         = ' + f(elements.foe.pos)         + '\n' +
    'cam_arrow   = ' + f(elements.cam_arrow.pos)   + '\n' +
    'label_cm    = ' + f(elements.label_cm.pos)    + '\n' +
    '\n' +
    'WALL_W    = ' + WALL_W.toFixed(3)    + '\n' +
    'WALL_H    = ' + WALL_H.toFixed(3)    + '\n' +
    'HORIZON_Y = ' + HORIZON_Y.toFixed(3) + '\n';
  document.getElementById('export').textContent = code;
}

function resetAll(){
  elements = JSON.parse(JSON.stringify(DEFAULTS));
  WALL_W = 1.70; WALL_H = 1.15; HORIZON_Y = -0.35;
  document.getElementById('ww').value = 170; document.getElementById('wwVal').textContent = '1.70';
  document.getElementById('wh').value = 115; document.getElementById('whVal').textContent = '1.15';
  document.getElementById('horY').value = -35; document.getElementById('horYVal').textContent = '-0.35';
  render();
}

document.getElementById('opacity').addEventListener('input', e => {
  document.getElementById('bookImg').style.opacity = (e.target.value / 100).toString();
});
document.getElementById('bookImg').style.opacity = 0.40;
document.getElementById('ww').addEventListener('input', e => {
  WALL_W = e.target.value / 100; document.getElementById('wwVal').textContent = WALL_W.toFixed(2); render();
});
document.getElementById('wh').addEventListener('input', e => {
  WALL_H = e.target.value / 100; document.getElementById('whVal').textContent = WALL_H.toFixed(2); render();
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
