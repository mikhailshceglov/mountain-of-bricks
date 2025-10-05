#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver.py — QP-решатель контактных сил для 2D статической задачи с трением.

Содержит:
- QPContactSolver: чистый QP (f, s, r)
- solve_contacts_qp_with_slack: обёртка
- ContactForceSolver: адаптер под проект (сборка A_eq, b_eq из bodies/contacts)

Конвенции:
- На контакт i заводим 2 переменные: fn_i (вдоль нормали n_i, неотриц.) и ft_i (вдоль касательной t_i, со знаком).
- Вклад в равновесие:
    для body1:  +fn*n + ft*t
    для body2:  -(fn*n + ft*t)
- Внутри сборки мы ортонормируем (n, t) и при необходимости меняем их местами (auto-swap),
  после чего **записываем исправленные векторы обратно в контакт** (важно для согласованности с CSV/валидацией).
"""

from __future__ import annotations

from typing import Iterable, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import cvxpy as cp


# =============================== QP core ===============================

class QPContactSolver:
    """
    QP-решатель с slack-переменными для конуса трения и резервом r для равенств.

    Переменные:
      f ∈ R^n    — все контактные компоненты (обычно [fn_0, ft_0, fn_1, ft_1, ...])
      s ∈ R^m    — slack по контакту (Ft <= mu*Fn + s), s >= 0
      r ∈ R^ne   — резерв к равенствам A_eq @ f + r = b_eq (штрафуем сильно)

    Ограничения:
      A_eq @ f + r = b_eq
      f_n >= 0
      -mu*f_n - s <= f_t <= mu*f_n + s
      s >= 0

    Цель (выпуклая):
      0.5*lambda*||f||^2  +  (1/(2*eps))*||s||^2  +  (alpha/2)*||r||^2
    """

    def __init__(
        self,
        A_eq: np.ndarray,
        b_eq: np.ndarray,
        contacts_df: pd.DataFrame,
        idx_n: Iterable[int],
        idx_t: Iterable[int],
        mu_per_contact: Iterable[float],
        epsilon: float = 1e-3,
        lambda_reg: float = 1e-6,
        solver_name: str = "OSQP",
    ):
        self.A_eq = np.asarray(A_eq, dtype=float)
        self.b_eq = np.asarray(b_eq, dtype=float).ravel()
        self.contacts_df = contacts_df.copy()
        self.idx_n = np.asarray(list(idx_n), dtype=int)
        self.idx_t = np.asarray(list(idx_t), dtype=int)
        self.mu = np.asarray(list(mu_per_contact), dtype=float).ravel()
        self.eps = float(epsilon)
        self.lmbd = float(lambda_reg)
        self.solver_name = str(solver_name)

        assert self.A_eq.shape[0] == self.b_eq.shape[0], "A_eq и b_eq несовместимы по строкам"
        assert self.idx_n.shape == self.idx_t.shape, "idx_n и idx_t должны иметь одинаковую длину"
        assert len(self.mu) == len(self.idx_n), "Длина mu должна равняться числу контактов"

        self.n_contacts = len(self.idx_n)
        self.n_vars = self.A_eq.shape[1]

    def solve(self, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Решить QP и вернуть (f, s)."""
        n = self.n_vars
        m = self.n_contacts
        n_eq = self.A_eq.shape[0]

        f = cp.Variable(n)                # все компоненты сил
        s = cp.Variable(m, nonneg=True)   # slack для конуса трения
        r = cp.Variable(n_eq)             # резерв к равенствам (может быть +/-)

        # Равенства с резервом
        constraints = [self.A_eq @ f + r == self.b_eq]

        # Извлекаем f_n и f_t по индексам
        f_n = cp.hstack([f[i] for i in self.idx_n])
        f_t = cp.hstack([f[i] for i in self.idx_t])

        # Конические ограничения (2D): |f_t| <= mu * f_n + s
        constraints += [
            f_n >= 0,
            f_t <= self.mu * f_n + s,
            -f_t <= self.mu * f_n + s,
        ]

        # Цель: рег. сил + штраф slack + большой штраф на r
        alpha = 1e8
        obj = 0.0
        if self.lmbd > 0:
            obj += 0.5 * self.lmbd * cp.sum_squares(f)
        obj += (1.0 / (2.0 * self.eps)) * cp.sum_squares(s)
        obj += 0.5 * alpha * cp.sum_squares(r)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # Порядок пробуемых солверов
        order = [self.solver_name.upper()] if self.solver_name else []
        for sn in ("ECOS", "OSQP", "SCS"):
            if sn not in order:
                order.append(sn)

        last_err = None
        for sn in order:
            try:
                prob.solve(solver=getattr(cp, sn), verbose=verbose)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception as e:
                last_err = e

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"QP not solved: status={prob.status}; last_err={last_err}")

        f_val = np.asarray(f.value, dtype=float).ravel()
        s_val = np.asarray(s.value, dtype=float).ravel()
        return f_val, s_val

    def write_slack_to_contacts(self, s_val: np.ndarray) -> pd.DataFrame:
        """Вернуть contacts_df с добавленной колонкой 's' (slack per contact)."""
        df = self.contacts_df.copy()
        if "s" in df.columns:
            df = df.drop(columns=["s"])
        df.insert(len(df.columns), "s", np.asarray(s_val, dtype=float))
        return df


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
    Удобная обёртка вокруг QPContactSolver: возвращает f, s и contacts_df с колонкой 's'.
    """
    solver = QPContactSolver(
        A_eq=A_eq,
        b_eq=b_eq,
        contacts_df=contacts_df,
        idx_n=idx_n,
        idx_t=idx_t,
        mu_per_contact=mu_per_contact,
        epsilon=epsilon,
        lambda_reg=lambda_reg,
        solver_name=solver_name,
    )
    f_val, s_val = solver.solve(verbose=verbose)
    contacts_with_s = solver.write_slack_to_contacts(s_val)
    return {"f": f_val, "s": s_val, "contacts_df": contacts_with_s}


# =============================== Adapter for project ===============================

class ContactForceSolver:
    """
    Backward-compatible адаптер, используемый main.py.

    Делаает:
      - собирает систему равновесия A_eq f = b_eq (3 уравнения на тело: Fx, Fy, M);
      - нормализует (n,t), чинит перепутанные пары (auto-swap), делает t ⟂ n;
      - решает QP со slack’ами (через QPContactSolver);
      - записывает силы обратно в объекты контактов:
            c.f_normal: float (модуль по нормали)
            c.f_tangent: np.ndarray(2,) мировые компоненты
            c.s: float (slack)
            c.v_tangent: 0.0  (статическая постановка)
            c.classification: sticking/near-cone/sliding
            c.cone_status: within-cone/near-cone/outside-cone
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

        # заполнятся в _assemble
        self.idx_n: Optional[np.ndarray] = None
        self.idx_t: Optional[np.ndarray] = None
        self.mu_vec: Optional[np.ndarray] = None
        self.A_eq: Optional[np.ndarray] = None
        self.b_eq: Optional[np.ndarray] = None

    # ---------- helpers ----------

    @staticmethod
    def _orthonormalize_and_fix(nt_pair: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Ортонормируем n=(nx,ny), t=(tx,ty) и делаем t ⟂ n. Возвращаем (nx,ny,tx,ty)."""
        nx, ny, tx, ty = nt_pair
        # нормализуем n
        nn = (nx * nx + ny * ny) ** 0.5
        if nn > 0:
            nx, ny = nx / nn, ny / nn
        else:
            nx, ny = 0.0, 1.0
        # нормализуем t
        tn = (tx * tx + ty * ty) ** 0.5
        if tn > 0:
            tx, ty = tx / tn, ty / tn
        else:
            tx, ty = -ny, nx
        # Gram–Schmidt для t перпендикулярно n
        dot_nt = nx * tx + ny * ty
        if abs(dot_nt) > 1e-8:
            tx, ty = tx - dot_nt * nx, ty - dot_nt * ny
            tn = (tx * tx + ty * ty) ** 0.5
            if tn > 0:
                tx, ty = tx / tn, ty / tn
            else:
                tx, ty = -ny, nx
        return nx, ny, tx, ty

    # ---------- assemble equilibrium ----------

    def _assemble(self):
        """
        Build A_eq f = b_eq.
        Переменные: per-contact (fn, ft) => 2*m variables.
        Для каждого тела k: строки [Fx=0, Fy=0, M=0].
        Конвенция знаков (должна совпадать с валидацией!):
            на body1 действует -F, на body2 действует +F,
            где F_world = fn * n + ft * t.
        """
        bodies = self.bodies
        contacts = self.contacts
        m = len(contacts)
        nb = len(bodies)

        # индексы переменных: [fn0, ft0, fn1, ft1, ...]
        idx_n = np.arange(0, 2 * m, 2, dtype=int)
        idx_t = np.arange(1, 2 * m, 2, dtype=int)
        self.idx_n, self.idx_t = idx_n, idx_t
        self.mu_vec = np.full(m, self.mu, dtype=float)

        # матрица/вектор равновесия
        rows = 3 * nb
        cols = 2 * m
        A = np.zeros((rows, cols), dtype=float)
        b = np.zeros(rows, dtype=float)

        # Правая часть: переносим вес на RHS:
        #   sum(F_contact) + (0, -mg) = 0   ⇒   b_y = +mg
        for k, body in enumerate(bodies):
            b[3 * k + 0] = 0.0
            b[3 * k + 1] = + body.mass * self.g
            b[3 * k + 2] = 0.0

        # Заполняем по контактам
        for i, c in enumerate(contacts):
            fn_col = 2 * i
            ft_col = 2 * i + 1

            px, py = float(c.point[0]), float(c.point[1])
            nx, ny = float(c.normal[0]), float(c.normal[1])
            tx, ty = float(c.tangent[0]), float(c.tangent[1])

            # 1) Ортонормируем (сделаем t ⟂ n и обе единичными)
            nx, ny, tx, ty = self._orthonormalize_and_fix((nx, ny, tx, ty))

            # 2) Геометрически корректируем (n, t)
            body1 = getattr(c, "body1", None)
            body2 = getattr(c, "body2", None)

            if body1 is None or body2 is None:
                # Контакт с опорой (земля). Хотим нормаль строго "вверх".
                # Если смотрит вниз — инвертируем обе (n,t) → (−n,−t).
                if ny < 0:
                    nx, ny, tx, ty = -nx, -ny, -tx, -ty
                # Если вдруг вертикаль ушла в t — свапнем их и снова обеспечим ny ≥ 0.
                if abs(ny) < abs(ty):
                    nx, ny, tx, ty = tx, ty, nx, ny
                    if ny < 0:
                        nx, ny, tx, ty = -nx, -ny, -tx, -ty
            else:
                # Кирпич–кирпич: нормаль должна быть сонаправлена направлению от body1 к body2.
                dx = float(body2.x) - float(body1.x)
                dy = float(body2.y) - float(body1.y)
                dn = (dx * dx + dy * dy) ** 0.5
                if dn > 0:
                    dx, dy = dx / dn, dy / dn
                else:
                    # Если центры совпали — возьмём произвольный ориентир
                    dx, dy = 1.0, 0.0

                # Если по направлению между центрами ближе оказывается t, значит n/t перепутаны — свапаем.
                if abs(nx * dx + ny * dy) < abs(tx * dx + ty * dy):
                    nx, ny, tx, ty = tx, ty, nx, ny

                # Добиваемся, чтобы n смотрела из body1 в body2 (n·d >= 0)
                if (nx * dx + ny * dy) < 0:
                    nx, ny, tx, ty = -nx, -ny, -tx, -ty

            # 3) Записываем исправленные единичные векторы обратно в контакт (для CSV/визуализации)
            c.normal = (float(nx), float(ny))
            c.tangent = (float(tx), float(ty))

            # Локальный помощник: добавить вклад в строки тела k со знаком sgn
            def add_body_contrib(body, sgn: float):
                k = bodies.index(body)
                row_fx = 3 * k + 0
                row_fy = 3 * k + 1
                row_m  = 3 * k + 2

                # Сила
                A[row_fx, fn_col] += sgn * nx
                A[row_fx, ft_col] += sgn * tx
                A[row_fy, fn_col] += sgn * ny
                A[row_fy, ft_col] += sgn * ty

                # Момент вокруг центра тела: r × F, где r = p - c
                cx, cy = float(body.x), float(body.y)
                rx, ry = (px - cx), (py - cy)
                A[row_m, fn_col] += sgn * (rx * ny - ry * nx)  # r × n
                A[row_m, ft_col] += sgn * (rx * ty - ry * tx)  # r × t

            # Конвенция знаков: на body1 действует -F, на body2 — +F
            if body1 is not None:
                add_body_contrib(body1, -1.0)
            if body2 is not None:
                add_body_contrib(body2, +1.0)


        self.A_eq, self.b_eq = A, b

    # ---------- solve and write back ----------

    def solve(self, verbose: bool = False):
        """Собрать систему, решить QP, записать силы и метаданные обратно в контакты."""
        self._assemble()

        # skeleton DF для возврата 's' (на всякий случай)
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

        # решаем QP
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

        # записать результаты обратно в объекты контактов
        for i, c in enumerate(self.contacts):
            fn = float(f[2 * i + 0])   # скаляр вдоль n
            ft = float(f[2 * i + 1])   # скаляр вдоль t (со знаком)

            nx, ny = c.normal
            tx, ty = c.tangent

            # сохранить в формате, который ждёт main.py:
            c.f_normal = fn
            c.f_tangent = np.array([ft * tx, ft * ty], dtype=float)  # мировые компоненты
            c.v_tangent = 0.0  # статическая постановка
            c.s = float(s[i])

            # классификация и статус конуса (для CSV/визуализации)
            mu = self.mu
            within = abs(ft) <= mu * max(fn, 0.0) + 1e-9
            near = abs(ft) <= mu * max(fn, 0.0) + 5e-4
            if within:
                c.classification = "sticking"
                c.cone_status = "within-cone"
            elif near:
                c.classification = "sticking"
                c.cone_status = "near-cone"
            else:
                c.classification = "sliding"
                c.cone_status = "outside-cone"
