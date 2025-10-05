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
from solver import ContactForceSolver  # ваш существующий решатель
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
    parser.add_argument("--mode", choices=["demo", "json"], default="demo",
                        help="Execution mode: demo (auto-generated) or json (from file)")
    parser.add_argument("--num-bricks", type=int, default=25,
                        help="Number of bricks for demo mode (default: 25)")
    parser.add_argument("--input", type=str, default="scene.json",
                        help="Input JSON file for json mode")
    parser.add_argument("--output", type=str, default="results/",
                        help="Output directory for results")

    # физпараметры
    parser.add_argument("--mu", type=float, default=0.5, help="Friction coefficient (default: 0.5)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Regularization ε (default: 1e-3)")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration (default: 9.81)")

    # настройки решателя (если ваш ContactForceSolver их поддерживает — отлично; если нет, игнорируются)
    parser.add_argument("--solver", type=str, default="OSQP", choices=["OSQP", "ECOS", "SCS"],
                        help="Underlying convex solver (default: OSQP)")
    parser.add_argument("--lambda-reg", type=float, default=1e-6,
                        help="Tikhonov regularization λ for forces (default: 1e-6)")

    # допуски для валидации
    parser.add_argument("--tol-force", type=float, default=1e-6, help="Force residual tolerance (default: 1e-6)")
    parser.add_argument("--tol-moment", type=float, default=1e-6, help="Moment residual tolerance (default: 1e-6)")
    parser.add_argument("--tol-friction", type=float, default=1e-4, help="Friction cone tolerance (default: 1e-4)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate scene
    print(f"=== 2D Static Friction Contact Solver ===\n")

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
        # Override with CLI params if provided
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
    # ожидаемые валидатором колонки:
    # contact_id, body1_id, body2_id, px, py, nx, ny, tx, ty,
    # f_normal_x, f_normal_y, f_normal_mag,
    # f_tangent_x, f_tangent_y, f_tangent_mag,
    # mu, [s], v_tangent, classification, cone_status
    contact_rows = []
    for i, c in enumerate(contacts):
        b1_id = bodies.index(c.body1) if getattr(c, "body1", None) is not None else -1
        b2_id = bodies.index(c.body2) if getattr(c, "body2", None) is not None else -1

        px, py = c.point  # точка контакта
        nx, ny = c.normal
        tx, ty = c.tangent

        # силы: у вас в объекте контакта были: c.f_normal (скаляр) и c.f_tangent (вектор 2D)
        # восстановим компонентную и модульную форму
        fN_mag = float(getattr(c, "f_normal", 0.0))
        fT_vec = np.asarray(getattr(c, "f_tangent", (0.0, 0.0)), dtype=float)
        fT_mag = float(np.linalg.norm(fT_vec))

        fN_x = fN_mag * nx
        fN_y = fN_mag * ny
        fT_x = float(fT_vec[0])
        fT_y = float(fT_vec[1])

        # slack (если в контакте есть поле s/slack — сохраним; иначе 0.0)
        s_val = None
        if hasattr(c, "s"):
            s_val = float(c.s) if c.s is not None else 0.0
        elif hasattr(c, "slack"):
            s_val = float(c.slack) if c.slack is not None else 0.0

        row = {
            "contact_id": i,
            "body1_id": b1_id,
            "body2_id": b2_id,
            "px": float(px),
            "py": float(py),
            "nx": float(nx),
            "ny": float(ny),
            "tx": float(tx),
            "ty": float(ty),
            "f_normal_x": fN_x,
            "f_normal_y": fN_y,
            "f_normal_mag": fN_mag,
            "f_tangent_x": fT_x,
            "f_tangent_y": fT_y,
            "f_tangent_mag": fT_mag,
            "mu": float(params["mu"]),
            "v_tangent": float(getattr(c, "v_tangent", 0.0)),
            "classification": getattr(c, "classification", "unknown"),
            "cone_status": getattr(c, "cone_status", "unknown"),
        }
        if s_val is not None:
            row["s"] = s_val

        contact_rows.append(row)

    contacts_df = pd.DataFrame(contact_rows)
    contact_file = output_dir / "contact_forces.csv"
    contacts_df.to_csv(contact_file, index=False)
    print(f"  Contact forces: {contact_file}")

    # BODIES -> body_forces.csv
    # Эти поля валидатор позднее перезапишет корректными суммами,
    # но базовую информацию (mass, x, y) важно сохранить.
    body_rows = []
    for i, body in enumerate(bodies):
        # Если ваш старый validate_solution уже посчитал эти поля — используем;
        # иначе — нули, их пересчитает validate_and_write_report.
        total_force = getattr(body, "total_force", (0.0, 0.0))
        total_moment = getattr(body, "total_moment", 0.0)
        force_residual = getattr(body, "force_residual", 0.0)
        moment_residual = getattr(body, "moment_residual", 0.0)

        body_rows.append({
            "body_id": i,
            "mass": float(body.mass),
            "x": float(body.x),
            "y": float(body.y),
            "total_fx": float(total_force[0]) if isinstance(total_force, (list, tuple, np.ndarray)) else float(total_force),
            "total_fy": float(total_force[1]) if isinstance(total_force, (list, tuple, np.ndarray)) else 0.0,
            "total_moment": float(total_moment),
            "force_residual": float(force_residual),
            "moment_residual": float(moment_residual),
        })

    bodies_df = pd.DataFrame(body_rows)
    body_file = output_dir / "body_forces.csv"
    bodies_df.to_csv(body_file, index=False)
    print(f"  Body summaries: {body_file}")

    # ---------- Validation (пересчитает суммы и сформирует отчёт) ----------
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
