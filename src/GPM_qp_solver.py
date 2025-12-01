
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SoftQPSolverConfig:
    """Настройки итерационного метода решения мягкой QP-задачи равновесия.

    Параметры:
        max_iters (int): Максимальное количество итераций.
        step_size (float): Шаг градиентного спуска (alpha).
        epsilon_reg (float): Регуляризация ε в целевой функции (штраф на ||λ||^2).
        tol_grad (float): Порог по норме градиента для критерия сходимости.
        tol_eq (float): Порог по норме остатка равновесия ||J^T λ + Q|| для критерия сходимости.
        verbose (bool): Если True, печатать прогресс (каждые 1000 итераций).
    """
    max_iters: int = 10000
    step_size: float = 1e-3
    epsilon_reg: float = 1e-4
    tol_grad: float = 1e-6
    tol_eq: float = 1e-6
    verbose: bool = False

def project_to_friction_cone(lambda_vec: np.ndarray, mu: float) -> np.ndarray:
    """
    Проецирует вектор контактных сил λ на декартово произведение конусов трения K.

    Структура λ: длина 2 * N_contacts, пары (λ_N, λ_T) по контактам:
        lambda_vec[2*k]   = λ_N_k
        lambda_vec[2*k+1] = λ_T_k

    Для каждого контакта конус задан условиями:
        λ_N >= 0
        |λ_T| <= mu * λ_N

    Проекция работает по каждому контакту независимо:
      1) если λ_N <= 0, то ставим λ_N = 0 и λ_T = 0 (контакт не несет нагрузки);
      2) если λ_N > 0 и |λ_T| <= mu * λ_N, оставляем (λ_N, λ_T) как есть;
      3) если λ_N > 0 и |λ_T| > mu * λ_N, то проецируем на границу:
           λ_T = sign(λ_T) * mu * λ_N.

    Важно: функция возвращает НОВЫЙ массив (копию), не модифицируя входной вектор in-place.
    """
    lam = np.array(lambda_vec, dtype=float, copy=True).reshape(-1)
    N = lam.shape[0]
    if N == 0:
        return lam

    # Число контактов
    N_contacts = N // 2
    for k in range(N_contacts):
        idxN = 2 * k
        idxT = 2 * k + 1
        lamN = lam[idxN]
        lamT = lam[idxT]
        if lamN <= 0.0:
            # Контакт не нагружен
            lam[idxN] = 0.0
            lam[idxT] = 0.0
        else:
            # Фрикционный конус
            if abs(lamT) > mu * lamN:
                # Проекция на грань конуса трения
                lam[idxT] = np.sign(lamT) * mu * lamN
            # Если |λ_T| <= mu * λ_N, то (λ_N, λ_T) уже внутри конуса – оставляем как есть
    return lam

def solve_soft_qp_equilibrium(
    J_T: np.ndarray,
    Q: np.ndarray,
    mu: float,
    config: SoftQPSolverConfig,
    lambda_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, str, float, float]:
    """
    Решает задачу мягкого равновесия:

        min_{λ ∈ K}  1/2 * || J^T λ + Q ||^2 + (ε / 2) * ||λ||^2,

    где K — конус трения по каждому контакту:
        λ_N >= 0, |λ_T| <= mu * λ_N.

    Входные аргументы:
      - J_T: матрица размера (M, N), где M = 3 * N_bodies, N = 2 * N_contacts.
             Это транспонированная матрица контактов J^T.
      - Q:   вектор размера (M,) или (M, 1) — внешний обобщенный груз (силы и моменты).
      - mu:  коэффициент трения μ (одинаковый для всех контактов).
      - config: экземпляр SoftQPSolverConfig с настройками итерационного метода.
      - lambda_init: начальное приближение λ формы (N,), если None — использовать нулевой вектор.

    Возвращаемое значение — кортеж:
      (lambda_opt, status, objective_value, equilibrium_error)

      - lambda_opt: np.ndarray формы (N,) — найденное значение вектора λ.
      - status: str — строка-статус:
          * "optimal"    — если достигнута сходимость по norm(grad) или norm(eq_residual);
          * "max_iters"  — если достигнут лимит итераций max_iters;
          * "no_contacts" — если N == 0;
          * "bad_input"  — при несогласованных размерностях (например, shape J_T и Q не стыкуются).
      - objective_value: float — значение цели в найденной точке:
            1/2 * || J^T λ + Q ||^2 + (ε / 2) * ||λ||^2.
      - equilibrium_error: float — норма остатка равновесия:
            || J^T λ + Q ||_2.
    """
    # Приведение Q к одномерному массиву нужной длины
    Q_vec = np.array(Q, dtype=float).reshape(-1)
    if not isinstance(J_T, np.ndarray):
        J_T = np.array(J_T, dtype=float)
    if J_T.ndim != 2:
        return (np.array([], dtype=float), "bad_input", np.nan, np.nan)
    M, N = J_T.shape
    if Q_vec.shape[0] != M:
        return (np.array([], dtype=float), "bad_input", np.nan, np.nan)
    if N == 0:
        # Нет контактов: возвращаем пустое решение, остаток равновесия просто ||Q||
        equilibrium_error = np.linalg.norm(Q_vec)
        return (np.array([], dtype=float), "no_contacts", 0.0, equilibrium_error)
    if N % 2 != 0:
        return (np.array([], dtype=float), "bad_input", np.nan, np.nan)
    if lambda_init is not None:
        lambda_vec = np.array(lambda_init, dtype=float).reshape(-1)
        if lambda_vec.shape[0] != N:
            return (np.array([], dtype=float), "bad_input", np.nan, np.nan)
    else:
        lambda_vec = np.zeros(N, dtype=float)

    J = J_T.T

    status = "max_iters"
    # Итерационный процесс проекционного градиентного спуска
    for iter_num in range(1, config.max_iters + 1):
        eq_residual = J_T @ lambda_vec + Q_vec  
        grad = J @ eq_residual + config.epsilon_reg * lambda_vec
        # Проверяем критерии сходимости
        norm_grad = np.linalg.norm(grad)
        norm_eq = np.linalg.norm(eq_residual)
        if norm_grad <= config.tol_grad or norm_eq <= config.tol_eq:
            status = "optimal"
            break
            # Шаг градиента с проекцией на конус трения
        lambda_half = lambda_vec - config.step_size * grad
        lambda_vec = project_to_friction_cone(lambda_half, mu)
        if config.verbose and iter_num % 1000 == 0:
            obj_value = 0.5 * (norm_eq ** 2) + 0.5 * config.epsilon_reg * (np.linalg.norm(lambda_vec) ** 2)
            print(f"Iteration {iter_num}: norm_grad={norm_grad:.3e}, norm_eq={norm_eq:.3e}, objective={obj_value:.3e}")

    eq_residual = J_T @ lambda_vec + Q_vec
    equilibrium_error = np.linalg.norm(eq_residual)
    objective_value = 0.5 * (equilibrium_error ** 2) + 0.5 * config.epsilon_reg * (np.linalg.norm(lambda_vec) ** 2)
    return (lambda_vec, status, objective_value, equilibrium_error)
