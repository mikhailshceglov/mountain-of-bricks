
# 2D Static Friction Contact Solver

## Overview
Computes contact forces (normal and friction) for a static pile of 2D rectangular bricks using Coulomb friction model with ε-regularization (Baraff 1991, §8.1).

## Features
- QP-based contact force solver
- ε-approximation for static friction (smooth interpolation)
- Contact classification: sticking/sliding/near-cone
- Equilibrium validation (∑F ≈ 0, ∑M ≈ 0)
- Friction cone constraint validation (|τ| ≤ μf)
- Visual output with force vectors

## Installation
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install numpy cvxpy matplotlib
```

## Usage

### Demo Scene
```bash
python main.py --mode demo --num-bricks 25 --output results/
```

### Custom JSON Scene
```bash
python main.py --mode json --input scene.json --output results/
```

### JSON Format
```json
{
  "gravity": 9.81,
  "mu": 0.5,
  "epsilon": 1e-3,
  "bodies": [
    {"x": 0.0, "y": 0.5, "width": 1.0, "height": 0.5, "angle": 0.0, "mass": 1.0}
  ]
}
```

## Output
- `contact_forces.csv`: Contact-level force data
- `body_forces.csv`: Per-body force summaries
- `validation.txt`: Equilibrium and constraint checks
- `visualization.png`: Force diagram

## Interpretation

### Contact Classifications
- **sticking**: |v_tangent| < ε, |τ| < μf (static friction active)
- **sliding**: |v_tangent| ≥ ε (dynamic friction, saturated)
- **near-cone**: |τ| ≈ μf (at friction limit)

### Validation Metrics
- `force_residual`: ‖∑F‖ per body (should be < tol_F)
- `moment_residual`: ‖∑M‖ per body (should be < tol_M)
- `friction_cone_violation`: max(|τ| - μf, 0) per contact

## Parameters
- `mu`: Coulomb friction coefficient (default 0.5)
- `epsilon`: Regularization parameter (default 1e-3 m/s)
- `gravity`: Gravitational acceleration (default 9.81 m/s²)
- `tol_F`: Force equilibrium tolerance (default 1e-4 N)
- `tol_M`: Moment equilibrium tolerance (default 1e-4 N·m)

## Method
Formulates static equilibrium as convex QP:
```
minimize   (1/2) f^T Q f + c^T f
subject to A_eq f = b_eq  (equilibrium)
           f_normal ≥ 0   (non-penetration)
           |f_tangent| ≤ μ f_normal (friction cone, via ε-approx)
```

The ε-regularization replaces discontinuous static friction with:
```
f_tangent = -μ f_normal * v_tangent / max(|v_tangent|, ε)
```

## References
- Baraff, D. (1991). Coping with Friction for Non-penetrating Rigid Body Simulation. SIGGRAPH '91.
