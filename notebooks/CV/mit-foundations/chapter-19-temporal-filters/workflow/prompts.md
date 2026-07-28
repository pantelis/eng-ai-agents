# Prompts — Ch 19 Temporal Filters

## 1. Digest the chapter
> Extract the math in full. For each figure state what it shows. Give exact
> equations for: temporal derivatives; the spatiotemporal Gaussian and its
> derivatives; motion as orientation in x-t; the Fourier constraint plane;
> velocity-tuned and velocity-nulling filters; IIR/causality. List every example
> image/video and whether colour or grayscale.

Correction learned: the WebFetch summary claimed "all grayscale", but the actual
figure images are the **colour** pedestrian video — always open the real image.

## 2. Use a real pedestrian clip
> The book's exact video isn't distributable. Use OpenCV's vtest.avi (static
> camera, people walking), pre-extract frames to a compact npz asset so no video
> decoder is needed at run time. Pick blur/null velocities from the walkers'
> actual speeds. Make the x-t slice show static as vertical streaks and movers
> as diagonal streaks.

## 3. Space-time filtering
> Implement 3D (t,y,x) kernels: separable spatiotemporal Gaussian, skewable by a
> velocity shear x - vx t; temporal/spatial Gaussian derivatives. Filter the
> volume with F.conv3d. Show: velocity-matched blur (object at that velocity
> stays sharp), and the nulling filter h = gt + vx gx + vy gy (erases the object
> at that velocity; nulling v=0 removes the static background).

## 4. Fourier signature
> For a 1D pulse moving at v = 0, -0.5, -1, show the x-t band and the 2D |DFT| as
> a 3D surface — the sinc ridge lies on the line wt + v wx = 0, tilting with speed.

## 5. Validate in the container
> Execute via docker compose torch.dev.cpu; recopy the extracted figures to
> images/fig19_*; add the notebook-database.yml entry.

## Lessons carried forward
- **Check the real figure images**, not the fetch summary (colour vs grayscale).
- **Signed-around-gray** display for derivative/nulling outputs (`show_signed`).
- A number beside every figure: kernels sum to 1; g_t nulls static bg to ~2e-9.
