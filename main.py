#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for 2D static friction contact solver.
Handles CLI, scene generation, solving, and output.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from geometry import RigidBody, create_demo_scene
from contact_detection import detect_contacts
from solver import ContactForceSolver  # адаптер, который вызывает QP внутри
from visualization import visualize_scene
from validation import validate_and_write_report


def load_scene_from_json(filepath: str):
    """Load scene configuration from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    bodies = []
    for b in data["bodies"]:
        body = RigidBody(
            x=b["x"],
            y=b["y"],
            width=b["width"],
            height=b["height"],
            angle=b.get("angle", 0.0),
            mass=b["mass"],
        )
        bodies.append(body)

    params = {
        "gravity": data.get("gravity", 9.81),
        "mu": data.get("mu", 0.5),
        "epsilon": data.get("epsilon", 1e-3),
    }
    return bodies, params


def main():
    parser = argparse.ArgumentParser(description="2D Static Friction Contact Force Solver")
    parser.add_argument("--mode", choices=["demo", "json"], default="demo")
    parser.add_argument("--num-bricks", type=int, default=25)
    parser.add_argument("--input", type=str, default="scene.json")
    parser.add_argument("--output", type=str, default="results/")

    # физпараметры
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--gravity", type=float, default=9.81)

    # настройки решателя
    parser.add_argument("--solver", type=str, default="OSQP", choices=["OSQP", "ECOS", "SCS"])
    parser.add_argument("--lambda-reg", type=float, default=1e-6)

    # допуски для валидации
    parser.add_argument("--tol-force", type=float, default=1e-6)
    parser.add_argument("--tol-moment", type=float, default=1e-6)
    parser.add_argument("--tol-friction", type=float, default=1e-4)

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 2D Static Friction Contact Solver ===\n")

    # Load or generate scene
    if args.mode == "demo":
        print(f"Generating demo scene with {args.num_bricks} bricks...")
        bodies = create_demo_scene(args.num_bricks)
        params = {"gravity": args.gravity, "mu": args.mu, "epsilon": args.epsilon}
    else:
        print(f"Loading scene from {args.input}...")
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found!")
            return 1
        bodies, params = load_scene_from_json(args.input)
        # Override from CLI
        params["mu"] = args.mu if args.mu is not None else params["mu"]
        params["epsilon"] = args.epsilon if args.epsilon is not None else params["epsilon"]
        params["gravity"] = args.gravity if args.gravity is not None else params["gravity"]

    print(f"  Bodies: {len(bodies)}")
    print(f"  Gravity: {params['gravity']} m/s²")
    print(f"  Friction coefficient μ: {params['mu']}")
    print(f"  Regularization ε: {params['epsilon']}\n")

    # Detect contacts
    print("Detecting contacts...")
    contacts = detect_contacts(bodies)
    print(f"  Found {len(contacts)} contacts\n")

    if len(contacts) == 0:
        print("Warning: No contacts detected. Check scene configuration.")
        return 1

    # Solve for contact forces
    print("Solving for contact forces (QP formulation)...")
    solver = ContactForceSolver(
        bodies=bodies,
        contacts=contacts,
        mu=params["mu"],
        epsilon=params["epsilon"],
        gravity=params["gravity"],
        solver_name=getattr(args, "solver", "OSQP"),
        lambda_reg=getattr(args, "lambda_reg", 1e-6),
    )

    try:
        solver.solve()
        print("  Solver converged successfully\n")
    except Exception as e:
        print(f"  Solver failed: {e}\n")
        return 1

    # ---------- Save results (CSV) ----------
    print("Saving results...")

    # CONTACTS -> contact_forces.csv
    rows = []
    for i, c in enumerate(contacts):
        b1_id = bodies.index(c.body1) if getattr(c, "body1", None) is not None else -1
        b2_id = bodies.index(c.body2) if getattr(c, "body2", None) is not None else -1

        px, py = c.point
        nx, ny = c.normal
        tx, ty = c.tangent

        fN_mag = float(getattr(c, "f_normal", 0.0))
        fT_vec = np.asarray(getattr(c, "f_tangent", (0.0, 0.0)), dtype=float)
        fT_mag = float(np.linalg.norm(fT_vec))
        fN_x, fN_y = fN_mag * nx, fN_mag * ny
        fT_x, fT_y = float(fT_vec[0]), float(fT_vec[1])

        row = {
            "contact_id": i,
            "body1_id": b1_id, "body2_id": b2_id,
            "px": float(px), "py": float(py),
            "nx": float(nx), "ny": float(ny),
            "tx": float(tx), "ty": float(ty),
            "f_normal_x": fN_x, "f_normal_y": fN_y, "f_normal_mag": fN_mag,
            "f_tangent_x": fT_x, "f_tangent_y": fT_y, "f_tangent_mag": fT_mag,
            "mu": float(params["mu"]),
            "v_tangent": float(getattr(c, "v_tangent", 0.0)),
            "classification": getattr(c, "classification", "unknown"),
            "cone_status": getattr(c, "cone_status", "unknown"),
        }
        # slack (если решатель его записал в контакт)
        if hasattr(c, "s") and c.s is not None:
            row["s"] = float(c.s)

        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "contact_forces.csv", index=False)
    print(f"  Contact forces: {output_dir / 'contact_forces.csv'}")

    # BODIES -> body_forces.csv (базовые поля; суммы пересчитает валидация)
    body_rows = []
    for i, body in enumerate(bodies):
        body_rows.append({
            "body_id": i,
            "mass": float(body.mass),
            "x": float(body.x),
            "y": float(body.y),
            "total_fx": 0.0,
            "total_fy": 0.0,
            "total_moment": 0.0,
            "force_residual": 0.0,
            "moment_residual": 0.0,
        })
    pd.DataFrame(body_rows).to_csv(output_dir / "body_forces.csv", index=False)
    print(f"  Body summaries: {output_dir / 'body_forces.csv'}")

    # ---------- Validation ----------
    print("\nValidating solution & writing report...")
    acceptance = validate_and_write_report(
        out_dir=output_dir,
        tol_force=args.tol_force,
        tol_moment=args.tol_moment,
        tol_fric=args.tol_friction,
        g=args.gravity,
    )
    print(f"  Validation report: {output_dir / 'validation.txt'}")

    # Visualization
    print("\nGenerating visualization...")
    viz_file = output_dir / "visualization.png"
    visualize_scene(bodies, contacts, str(viz_file))
    print(f"  Visualization: {viz_file}")

    # Summary
    print("\n=== SUMMARY ===")
    all_pass = all(v == "PASS" for v in acceptance.values())
    if all_pass:
        print("✓ All acceptance criteria PASSED")
    else:
        print("✗ Some acceptance criteria FAILED (see validation.txt)")

    print(f"\nResults saved to: {output_dir}/")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
