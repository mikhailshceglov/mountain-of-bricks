# solver.py
# -*- coding: utf-8 -*-
"""
QP-based static contact force solver with slack variables for friction cone (ε-regularized).

Идея:
- Переменные: f (все компоненты контактных сил, как правило разложенные на нормали и тангенсы),
  и s >= 0 — slack по контакту (нарушение конуса трения).
- Ограничения:
    A_eq @ f = b_eq                         (равновесие по телам)
    f_n >= 0                                (нормальные силы неотрицательны)
    |f_t| <= mu * f_n + s                   (в 2D линейно через две неравенства)
    s >= 0
- Цель:
    0.5 * || W f ||_2^2 + (1/(2ε)) * || s ||_2^2
  где W = λ I — маленькая Tikhonov-регуляризация сил, ε ~ 1e-3…1e-2.

Результат:
- Возвращает решение f и s; сохраняет s в contacts_df для последующей валидации.
"""

from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import cvxpy as cp


class QPContactSolver:
    """
    Контейнер, который принимает на вход уже собранные матрицы равновесия и
    индексы переменных, соответствующих f_n и f_t, а также dataframe контактов.
    """

    def __init__(
        self,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        contacts_df: pd.DataFrame,
        idx_n: np.ndarray,
        idx_t: np.ndarray,
        mu_per_contact: np.ndarray,
        epsilon: float = 1e-3,
        lambda_reg: float = 1e-6,
        solver_name: str = "OSQP",
    ):
        """
        A_eq @ f = b_eq. Вектор f — это все компоненты сил, порядок должен
        соответствовать idx_n/idx_t.

        idx_n, idx_t — массивы индексов (длины = число контактов).
        mu_per_contact — массив μ_i по контактам.

        contacts_df — датасет контактов, в который мы добавим столбец 's'.
        """
        self.A_eq = np.asarray(A_eq, dtype=float)
        self.b_eq = np.asarray(b_eq, dtype=float)
        self.contacts_df = contacts_df.copy()
        self.idx_n = np.asarray(idx_n, dtype=int)
        self.idx_t = np.asarray(idx_t, dtype=int)
        self.mu = np.asarray(mu_per_contact, dtype=float)
        self.eps = float(epsilon)
        self.lmbd = float(lambda_reg)
        self.solver_name = solver_name

        assert self.A_eq.shape[0] == self.b_eq.shape[0], "A_eq, b_eq shape mismatch"
        assert self.idx_n.shape == self.idx_t.shape, "idx_n and idx_t must match"
        assert len(self.mu) == len(self.idx_n), "len(mu) must equal #contacts"

        # Размерность f (число компонент сил)
        self.n_vars = self.A_eq.shape[1]
        self.n_contacts = len(self.idx_n)

    def solve(self, verbose: bool = False):
        """
        Assemble already done in __init__/external; solve QP with:
        A_eq @ f + r = b_eq,  (r = residuals, heavily penalized)
        f_n >= 0,
        |f_t| <= mu f_n + s,
        s >= 0.
        Objective: 0.5*lambda*||f||^2 + (1/(2*eps))*||s||^2 + (alpha/2)*||r||^2
        """
        n, m = self.n_vars, self.n_contacts
        n_eq = self.A_eq.shape[0]

        f = cp.Variable(n)                   # все компоненты сил
        s = cp.Variable(m, nonneg=True)      # slack по конусу трения
        r = cp.Variable(n_eq)                # резерв к равенствам (может быть полож/отриц), штрафуем сильно

        # равенства с резервом
        constraints = [ self.A_eq @ f + r == self.b_eq ]

        # извлекаем f_n и f_t по индексам
        f_n = cp.hstack([f[i] for i in self.idx_n])
        f_t = cp.hstack([f[i] for i in self.idx_t])

        # 2D трение: |f_t| <= mu f_n + s
        constraints += [
            f_n >= 0,
            f_t <= self.mu * f_n + s,
            -f_t <= self.mu * f_n + s,
        ]

        # цель: тикхоновская рег-ция по силам + штраф slack + ОЧЕНЬ большой штраф на r
        alpha = 1e8  # можно ослабить до 1e6, если будет слишком жёстко
        obj = 0.0
        if self.lmbd > 0:
            obj += 0.5 * self.lmbd * cp.sum_squares(f)
        obj += (1.0 / (2.0 * self.eps)) * cp.sum_squares(s)
        obj += 0.5 * alpha * cp.sum_squares(r)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # попробуем более «терпеливый» солвер; OSQP ок, но ECOS/SCS чаще стабильно решают такое
        try_order = [self.solver_name.upper()] if self.solver_name else []
        try_order += [sn for sn in ["ECOS", "OSQP", "SCS"] if sn not in try_order]

        last_err = None
        for sn in try_order:
            try:
                prob.solve(solver=getattr(cp, sn), verbose=verbose)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception as e:
                last_err = e
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP not solved: status={prob.status}; last_err={last_err}")

        f_val = f.value.astype(float).ravel()
        s_val = s.value.astype(float).ravel()
        # r можно вернуть при желании: r_val = r.value.astype(float).ravel()
        return f_val, s_val


    def write_slack_to_contacts(self, s_val: np.ndarray) -> pd.DataFrame:
        """Сохраняет s в contacts_df (колонка 's') и возвращает копию."""
        df = self.contacts_df.copy()
        if "s" in df.columns:
            df = df.drop(columns=["s"])
        df.insert(len(df.columns), "s", s_val)
        return df


# ---------- Утилита верхнего уровня (drop-in) ----------

def solve_contacts_qp_with_slack(
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    contacts_df: pd.DataFrame,
    idx_n: Iterable[int],
    idx_t: Iterable[int],
    mu_per_contact: Iterable[float],
    epsilon: float = 1e-3,
    lambda_reg: float = 1e-6,
    solver_name: str = "OSQP",
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Удобная обёртка: решает QP и возвращает словарь:
      {
        "f": np.ndarray (решение по всем компонентам сил),
        "s": np.ndarray (slack по контактам),
        "contacts_df": pd.DataFrame (contacts_df + колонка 's')
      }
    """
    solver = QPContactSolver(
        A_eq=A_eq,
        b_eq=b_eq,
        contacts_df=contacts_df,
        idx_n=np.asarray(list(idx_n), dtype=int),
        idx_t=np.asarray(list(idx_t), dtype=int),
        mu_per_contact=np.asarray(list(mu_per_contact), dtype=float),
        epsilon=epsilon,
        lambda_reg=lambda_reg,
        solver_name=solver_name,
    )
    f_val, s_val = solver.solve(verbose=verbose)
    contacts_with_s = solver.write_slack_to_contacts(s_val)
    return {"f": f_val, "s": s_val, "contacts_df": contacts_with_s}

# ====== Adapter to keep old API: ContactForceSolver ======

class ContactForceSolver:
    """
    Backward-compatible adapter used by main.py.
    Builds equilibrium equations from (bodies, contacts), then solves QP with slacks.
    After solve(), writes results back into each contact:
        c.f_normal: float
        c.f_tangent: np.ndarray shape (2,)
        c.s: float
        c.classification: str in {"sticking", "near-cone", "sliding"}
        c.cone_status: str in {"within-cone", "near-cone", "outside-cone"}
        c.v_tangent: float  (0.0 for static model)
    """
    def __init__(
        self,
        bodies,
        contacts,
        mu: float = 0.5,
        epsilon: float = 1e-3,
        gravity: float = 9.81,
        solver_name: str = "OSQP",
        lambda_reg: float = 1e-6,
    ):
        self.bodies = bodies
        self.contacts = contacts
        self.mu = float(mu)
        self.eps = float(epsilon)
        self.g = float(gravity)
        self.solver_name = solver_name
        self.lambda_reg = float(lambda_reg)

        # built during assembly
        self.idx_n = None
        self.idx_t = None
        self.mu_vec = None
        self.A_eq = None
        self.b_eq = None

    def _assemble(self):
        """
        Build A_eq f = b_eq.
        Variables: per-contact (fn, ft) => 2*m variables.
        For each body k: force-x, force-y, moment about body center.
        """
        import numpy as np

        m = len(self.contacts)
        n_bodies = len(self.bodies)

        # variable order: [fn_0, ft_0, fn_1, ft_1, ...]
        # indices for convenience:
        idx_n = np.arange(0, 2*m, 2, dtype=int)
        idx_t = np.arange(1, 2*m, 2, dtype=int)
        self.idx_n, self.idx_t = idx_n, idx_t
        self.mu_vec = np.full(m, self.mu, dtype=float)

        # Rows: for each body: Fx=0, Fy=0, M=0  => 3*n_bodies equations
        rows = 3 * n_bodies
        cols = 2 * m
        A = np.zeros((rows, cols), dtype=float)
        b = np.zeros(rows, dtype=float)

        # external forces: gravity on each body
        # Fx_ext = 0; Fy_ext = -m*g
        for k, body in enumerate(self.bodies):
            row_fx = 3*k + 0
            row_fy = 3*k + 1
            row_m  = 3*k + 2
            b[row_fx] = 0.0
            b[row_fy] = + body.mass * self.g  # move to RHS: sum(contacts) + (0, -mg) = 0  -> b_y = +mg
            b[row_m]  = 0.0

        # fill contact contributions for both bodies (body1 ~ A, body2 ~ B)
        # Force on body1: +fn*n + ft*t
        # Force on body2: -(fn*n + ft*t)
        # Moment: r x F, where r = (p - c_body), F as above; z-moment scalar = r_x*F_y - r_y*F_x
        for i, c in enumerate(self.contacts):
            # индексы переменных в векторе f
            fn_col = 2 * i
            ft_col = 2 * i + 1

            # геометрия контакта
            px, py = c.point
            nx, ny = c.normal
            tx, ty = c.tangent

            # --- НОРМАЛИЗАЦИЯ И ОРТОНОРМАЛИЗАЦИЯ (на всякий случай) ---
            # нормаль
            n_norm = (nx*nx + ny*ny) ** 0.5
            if n_norm > 0:
                nx, ny = nx / n_norm, ny / n_norm
            else:
                # fallback: если вдруг норма нулевая — делаем вертикальную нормаль
                nx, ny = 0.0, 1.0

            # тангенс
            t_norm = (tx*tx + ty*ty) ** 0.5
            if t_norm > 0:
                tx, ty = tx / t_norm, ty / t_norm
            else:
                # если тангенс не задан, делаем перпендикуляр к n
                tx, ty = -ny, nx

            # делаем t строго перпендикулярным n (Gram-Schmidt 1 шаг)
            dot_nt = nx*tx + ny*ty
            if abs(dot_nt) > 1e-8:
                tx, ty = tx - dot_nt*nx, ty - dot_nt*ny
                t_norm = (tx*tx + ty*ty) ** 0.5
                if t_norm > 0:
                    tx, ty = tx / t_norm, ty / t_norm
                else:
                    tx, ty = -ny, nx
            # --- END НОРМАЛИЗАЦИИ ---

            # --- FIX: авто-swap n↔t, если похоже, что они перепутаны ---
            # земля: одна сторона контакта отсутствует (None) -> ожидаем вертикальную нормаль (|ny| >= |nx|)
            body1_is_ground = (getattr(c, "body1", None) is None)
            body2_is_ground = (getattr(c, "body2", None) is None)

            need_swap = False
            if body1_is_ground or body2_is_ground:
                # для опоры "пол": нормаль должна быть ближе к вертикали, чем тангенс
                # сравниваем "вертикальность" n и t по компоненте |y|
                if abs(ny) < abs(ty):
                    need_swap = True
            else:
                # для кирпич-кирпич: нормаль должна быть "более нормальна", чем t
                # критерий: модуль проекции на нормальное направление больше для n, чем для t
                score_n = abs(ny) + 0.1*abs(nx)   # чуть «нагружаем» вертикаль
                score_t = abs(ty) + 0.1*abs(tx)
                if score_n < score_t:
                    need_swap = True

            if need_swap:
                nx, ny, tx, ty = tx, ty, nx, ny
            # --- END FIX ---

            # >>> ВСТАВИТЬ: обновить контакт исправленными единичными векторами <<<
            c.normal  = (float(nx), float(ny))
            c.tangent = (float(tx), float(ty))

            # helper: вклад контакта в строки тела (Fx,Fy,M) с указанным знаком
            def add_body_contrib(body, sgn: float):
                k = self.bodies.index(body)
                row_fx = 3*k + 0
                row_fy = 3*k + 1
                row_m  = 3*k + 2

                # вклад по силам
                A[row_fx, fn_col] += sgn * nx
                A[row_fx, ft_col] += sgn * tx
                A[row_fy, fn_col] += sgn * ny
                A[row_fy, ft_col] += sgn * ty

                # вклад по моменту: r x F, r = p - c
                cx, cy = body.x, body.y
                rx, ry = (px - cx), (py - cy)
                # момент от единичного fn: r x n; от единичного ft: r x t
                A[row_m, fn_col] += sgn * (rx*ny - ry*nx)
                A[row_m, ft_col] += sgn * (rx*ty - ry*tx)

            # вклад в оба тела (землю/пол пропускаем: body=None)
            if getattr(c, "body1", None) is not None:
                add_body_contrib(c.body1, +1.0)
            if getattr(c, "body2", None) is not None:
                add_body_contrib(c.body2, -1.0)

        self.A_eq, self.b_eq = A, b



    def solve(self, verbose: bool = False):
        """
        Assemble and solve. Writes back results into contact objects.
        """
        import numpy as np

        self._assemble()

        # Build a contacts_df skeleton to carry slack 's' back
        rows = []
        for i, c in enumerate(self.contacts):
            rows.append({
                "contact_id": i,
                "body1_id": self.bodies.index(c.body1) if getattr(c, "body1", None) is not None else -1,
                "body2_id": self.bodies.index(c.body2) if getattr(c, "body2", None) is not None else -1,
                "px": float(c.point[0]),
                "py": float(c.point[1]),
                "nx": float(c.normal[0]),
                "ny": float(c.normal[1]),
                "tx": float(c.tangent[0]),
                "ty": float(c.tangent[1]),
            })
        contacts_df = pd.DataFrame(rows)

        # Solve QP with slacks
        res = solve_contacts_qp_with_slack(
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            contacts_df=contacts_df,
            idx_n=self.idx_n,
            idx_t=self.idx_t,
            mu_per_contact=self.mu_vec,
            epsilon=self.eps,
            lambda_reg=self.lambda_reg,
            solver_name=self.solver_name,
            verbose=verbose,
        )
        f = res["f"]
        s = res["s"]

        # write back into contacts
        for i, c in enumerate(self.contacts):
            fn = float(f[2*i + 0])
            ft = float(f[2*i + 1])

            # store as world components consistent with main.py expectations:
            nx, ny = c.normal
            tx, ty = c.tangent
            c.f_normal = fn  # scalar (magnitude along normal)
            c.f_tangent = np.array([ft*tx, ft*ty], dtype=float)  # world components
            c.v_tangent = 0.0  # static model

            # slack
            c.s = float(s[i])

            # classification / cone status
            mu = self.mu
            within = abs(ft) <= mu*max(fn, 0.0) + 1e-9
            near   = abs(ft) <= mu*max(fn, 0.0) + 5e-4  # small band
            if within:
                c.classification = "sticking"
                c.cone_status = "within-cone"
            elif near:
                c.classification = "sticking"
                c.cone_status = "near-cone"
            else:
                c.classification = "sticking" if c.s <= 1e-6 else "sliding"
                c.cone_status = "outside-cone"
