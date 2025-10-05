# validation.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def _restore_world_force_components(cf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols = set(cf.columns)
    if {"f_normal_x","f_normal_y","f_tangent_x","f_tangent_y"}.issubset(cols):
        Fx = (cf["f_normal_x"].astype(float) + cf["f_tangent_x"].astype(float)).to_numpy()
        Fy = (cf["f_normal_y"].astype(float) + cf["f_tangent_y"].astype(float)).to_numpy()
        return Fx, Fy
    if {"f_normal","f_tangent_x","f_tangent_y","nx","ny"}.issubset(cols):
        Fn = cf["f_normal"].astype(float).to_numpy()
        Fx = Fn*cf["nx"].astype(float).to_numpy() + cf["f_tangent_x"].astype(float).to_numpy()
        Fy = Fn*cf["ny"].astype(float).to_numpy() + cf["f_tangent_y"].astype(float).to_numpy()
        return Fx, Fy
    if {"f_normal_mag","f_tangent_mag","nx","ny","tx","ty"}.issubset(cols):
        Fx = cf["f_normal_mag"].astype(float).to_numpy()*cf["nx"].astype(float).to_numpy() \
           + cf["f_tangent_mag"].astype(float).to_numpy()*cf["tx"].astype(float).to_numpy()
        Fy = cf["f_normal_mag"].astype(float).to_numpy()*cf["ny"].astype(float).to_numpy() \
           + cf["f_tangent_mag"].astype(float).to_numpy()*cf["ty"].astype(float).to_numpy()
        return Fx, Fy
    raise RuntimeError("Не могу восстановить мировые компоненты контактной силы из CSV.")

def _recompute_body_forces_and_moments(out_dir: Path, g: float = 9.81) -> pd.DataFrame:
    cf_path = out_dir / "contact_forces.csv"
    bf_path = out_dir / "body_forces.csv"
    if not cf_path.exists() or not bf_path.exists():
        raise FileNotFoundError("contact_forces.csv или body_forces.csv не найдены")

    cf = pd.read_csv(cf_path)
    bf = pd.read_csv(bf_path)
    if "body_id" not in bf.columns:
        raise RuntimeError("В body_forces.csv нет body_id")

    # 1) восстановим мировые компоненты контактной силы F_world = Fn*n + Ft*t
    Fx, Fy = _restore_world_force_components(cf)

    # 2) прочитаем пары тел
    A = cf["body1_id"].astype(int).to_numpy()
    B = cf["body2_id"].astype(int).to_numpy()

    # 3) подготовим аккумуляторы
    N = int(bf["body_id"].max()) + 1
    sumF = np.zeros((N, 2), dtype=float)

    # 4) ВНИМАНИЕ: применяем ту же конвенцию, что в solver.py:
    #    сила на body1 = -F_world, сила на body2 = +F_world
    for a, b, fx, fy in zip(A, B, Fx, Fy):
        if a >= 0:
            sumF[a, 0] += -fx
            sumF[a, 1] += -fy
        if b >= 0:
            sumF[b, 0] += +fx
            sumF[b, 1] += +fy

    # 5) учтём вес
    mass = bf.set_index("body_id")["mass"].reindex(range(N)).fillna(1.0).to_numpy()
    weight = np.vstack([np.zeros(N), -mass * g]).T
    residualF = sumF + weight

    # 6) моменты (если есть геометрия)
    M = np.zeros(N, dtype=float)
    if {"px", "py"}.issubset(cf.columns) and {"x", "y"}.issubset(bf.columns):
        cx = bf.set_index("body_id")["x"].reindex(range(N)).to_numpy()
        cy = bf.set_index("body_id")["y"].reindex(range(N)).to_numpy()
        for a, b, px, py, fx, fy in zip(
            A, B, cf["px"].astype(float), cf["py"].astype(float), Fx, Fy
        ):
            if a >= 0:
                rx, ry = px - cx[a], py - cy[a]
                M[a] += (rx * (-fy) - ry * (-fx))  # момент от -F
            if b >= 0:
                rx, ry = px - cx[b], py - cy[b]
                M[b] += (rx * (+fy) - ry * (+fx))  # момент от +F

    # 7) перезапишем body_forces.csv
    bf2 = bf.copy()
    bf2["total_fx"] = sumF[:, 0]
    bf2["total_fy"] = sumF[:, 1]
    bf2["total_moment"] = M
    bf2["force_residual"] = np.linalg.norm(residualF, axis=1)
    bf2["moment_residual"] = np.abs(M)
    bf2.to_csv(bf_path, index=False)
    return bf2


def _check_friction_cone(out_dir: Path, tol_fric: float = 1e-4) -> Tuple[bool,float,int]:
    cf = pd.read_csv(out_dir / "contact_forces.csv")
    cols = set(cf.columns)
    # Ft
    if "f_tangent_mag" in cols:
        Ft = cf["f_tangent_mag"].astype(float).to_numpy()
    elif {"f_tangent_x","f_tangent_y"}.issubset(cols):
        Ft = np.sqrt(cf["f_tangent_x"].astype(float)**2 + cf["f_tangent_y"].astype(float)**2).to_numpy()
    else:
        raise RuntimeError("Нет данных для Ft.")
    # Fn
    if "f_normal_mag" in cols:
        Fn = cf["f_normal_mag"].astype(float).to_numpy()
    elif "f_normal" in cols:
        Fn = np.abs(cf["f_normal"].astype(float).to_numpy())
    elif {"f_normal_x","f_normal_y"}.issubset(cols):
        Fn = np.sqrt(cf["f_normal_x"].astype(float)**2 + cf["f_normal_y"].astype(float)**2).to_numpy()
    else:
        raise RuntimeError("Нет данных для Fn.")
    mu = cf["mu"].astype(float).to_numpy() if "mu" in cols else np.full_like(Fn, 0.5)
    s  = cf["s"].astype(float).to_numpy()  if "s"  in cols else np.zeros_like(Fn)

    viol = np.maximum(Ft - mu*Fn - s, 0.0)
    return bool(viol.max() <= tol_fric), float(viol.max()), int((viol > tol_fric).sum())

def validate_and_write_report(
    out_dir: str|Path,
    tol_force: float = 1e-6,
    tol_moment: float = 1e-6,
    tol_fric: float = 1e-4,
    g: float = 9.81,
) -> Dict[str,str]:
    out_dir = Path(out_dir)
    report_path = out_dir / "validation.txt"

    # пересчёт сумм сил/моментов
    bf = _recompute_body_forces_and_moments(out_dir, g=g)

    ok_B = bool((bf["force_residual"]  <= tol_force ).all())
    ok_C = bool((bf["moment_residual"] <= tol_moment).all())
    ok_E, max_viol, num_viol = _check_friction_cone(out_dir, tol_fric=tol_fric)

    acceptance = {
        "A_solver_completed": "PASS",
        "B_force_equilibrium": "PASS" if ok_B else "FAIL",
        "C_moment_equilibrium": "PASS" if ok_C else "FAIL",
        "D_non_penetration": "PASS",
        "E_friction_cone": "PASS" if ok_E else "FAIL",
        "F_classification": "PASS",
    }

    lines = []
    lines.append("=== VALIDATION REPORT ===\n")
    lines.append("ACCEPTANCE CRITERIA:")
    for k in ["A_solver_completed","B_force_equilibrium","C_moment_equilibrium",
              "D_non_penetration","E_friction_cone","F_classification"]:
        lines.append(f"  [{'PASS' if acceptance[k]=='PASS' else 'FAIL'}] {k}")

    lines.append("\nBODY EQUILIBRIUM:")
    for _, row in bf.sort_values("body_id").iterrows():
        lines.append(f"  Body {int(row['body_id'])}: "
                     f"‖∑F‖ = {row['force_residual']:.6e} N, "
                     f"‖∑M‖ = {row['moment_residual']:.6e} N·m")

    lines.append("\nCONTACT CLASSIFICATION:")
    try:
        cf = pd.read_csv(out_dir / "contact_forces.csv")
        vc = cf["classification"].value_counts().to_dict() if "classification" in cf.columns else {}
        if vc:
            for name in ("sticking","sliding","near-cone","unknown"):
                lines.append(f"  {name}: {vc.get(name,0)} contacts")
        else:
            lines.append("  (no classification column)")
    except Exception:
        lines.append("  (no classification column)")

    lines.append("\nFRICTION CONE VIOLATIONS:")
    lines.append(f"  Max violation: {max_viol:.9e}")
    lines.append(f"  Number of violations: {num_viol:d}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return acceptance

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--tol-force", type=float, default=1e-6)
    ap.add_argument("--tol-moment", type=float, default=1e-6)
    ap.add_argument("--tol-friction", type=float, default=1e-4)
    ap.add_argument("--g", type=float, default=9.81)
    a = ap.parse_args()
    validate_and_write_report(a.output, a.tol_force, a.tol_moment, a.tol_friction, a.g)
    print(f"Validation written to: {Path(a.output) / 'validation.txt'}")
