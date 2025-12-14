A 2D heat diffusion simulation in C++ controlled by Python configuration.

## What This Demonstrates

- Real use case: Python as configuration for scientific computing
- Edit `config.py`, run again—no recompilation
- C++ handles computation, Python handles parameters

## The Physics

The 2D heat equation with finite differences:
```
∂T/∂t = α∇²T

T_new[y][x] = T[y][x] + α * (T[y+1][x] + T[y-1][x] + T[y][x+1] + T[y][x-1] - 4*T[y][x])
```

Where α is the diffusion rate.

## Build and Run

```bash
make run
```

## Try It

1. Change `heat_source_temp` in `config.py` → see different heat spread
2. Change `diffusion_rate` → see faster/slower diffusion
3. Change `grid_width/height` → see larger/smaller simulation