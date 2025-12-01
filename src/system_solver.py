import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Если True — используем самописный солвер мягкого равновесия (GPM_qp_solver).
# Если False — используем готовый QP-решатель cvxopt.solvers.qp.
USE_MANUAL_SOLVER: bool = True


# Отдельные настройки точности для самописного и готового солверов
MANUAL_TOL_GRAD: float = 1e-4
MANUAL_TOL_EQ: float = 1e-6
MANUAL_MAX_ITERS: int = 5000
MANUAL_EPSILON_REG: float = 1e-4

# Порог, при котором мы считаем, что решение «примерно годное», даже если статус max_iters
MANUAL_APPROX_VIS_TOL: float = 1e1  

# Самописный солвер (проекционный градиент)
from GPM_qp_solver import SoftQPSolverConfig, solve_soft_qp_equilibrium

# Готовый QP-солвер cvxopt
if not USE_MANUAL_SOLVER:
    import cvxopt.solvers
    import cvxopt.base as cvx

    # Тут своя точность для солвера
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-7
    cvxopt.solvers.options['reltol'] = 1e-6
    cvxopt.solvers.options['feastol'] = 1e-7

from load_config import BrickConfig
from contact_finder import Contact, ContactPoint


@dataclass
class QPSolution:
    lambda_values: np.ndarray
    status: str
    objective_value: float
    equilibrium_error: float = np.nan


def get_tangent(n: Tuple[float, float]) -> Tuple[float, float]:
    # Находит единичный вектор тангенса t (перпендикулярно нормали n в 2D): t=(-ny, nx)
    return (-n[1], n[0])


def calculate_jacobian_full(config: BrickConfig, contacts: List[ContactPoint]) -> np.ndarray:
    """
    Рассчитывает ТРАНСПОНИРОВАННУЮ матрицу Якоби J^T

    Возвращаемая матрица J_T имеет размер: (3 * N_bricks) x (2 * N_contacts)

    Строки соответствуют обобщённым силам/моментам по кирпичам: на каждый кирпич 3 строки Fx, Fy, Mz

    Столбцы соответствуют компонентам контактных сил: по 2 столбца на контакт λ_N, λ_T
    """
    
    N_c = len(contacts)
    N_b = config.N_bricks

    # J имеет размер (2 * N_c) x (3 * N_b), затем мы вернём J.T размером (3 * N_b) x (2 * N_c)
    J = np.zeros((2 * N_c, 3 * N_b))

    for k, contact in enumerate(contacts):
        n_global = np.array(contact.n_global, dtype=float)
        t_global = np.array(get_tangent(contact.n_global), dtype=float)
        contact_point = np.array(contact.point, dtype=float)

        # Кирпич 1 (brick1) вклад с положительным знаком
        i = contact.brick1_id
        R_i = np.array(config.R_list[i][:2], dtype=float)   # (x,y) центра масс
        r_i = contact_point - R_i                           # плечо силы

        # Нормальная сила
        J[2 * k, 3 * i]     = n_global[0]  # Fx
        J[2 * k, 3 * i + 1] = n_global[1]  # Fy
        J[2 * k, 3 * i + 2] = r_i[0] * n_global[1] - r_i[1] * n_global[0]  # Mz = rn

        # Тангенциальная сила (строка 2k + 1)
        J[2 * k + 1, 3 * i]     = t_global[0]  # Fx
        J[2 * k + 1, 3 * i + 1] = t_global[1]  # Fy
        J[2 * k + 1, 3 * i + 2] = r_i[0] * t_global[1] - r_i[1] * t_global[0]  # Mz = rt

        # Кирпич 2 (brick2) вклад с отрицательным знаком, если это не земля
        j = contact.brick2_id
        if j != -1:
            R_j = np.array(config.R_list[j][:2], dtype=float)
            r_j = contact_point - R_j

            # Нормальная сила, действующая на brick2, противоположна n_global
            J[2 * k, 3 * j]     = -n_global[0]
            J[2 * k, 3 * j + 1] = -n_global[1]
            J[2 * k, 3 * j + 2] = -(r_j[0] * n_global[1] - r_j[1] * n_global[0])

            # Тангенциальная сила, действующая на brick2, противоположна t_global
            J[2 * k + 1, 3 * j]     = -t_global[0]
            J[2 * k + 1, 3 * j + 1] = -t_global[1]
            J[2 * k + 1, 3 * j + 2] = -(r_j[0] * t_global[1] - r_j[1] * t_global[0])

    return J.T


def setup_system_matrices(config: BrickConfig) -> np.ndarray:
    """
    Рассчитывает вектор внешних сил Q (только гравитация, квазистатика).

    Возвращает одномерный вектор Q формы (3 * N_bricks,) где для каждого кирпича:
        Q[3*i]   = 0     (Fx)
        Q[3*i+1] = -m*g  (Fy)
        Q[3*i+2] = 0     (Mz)
    """
    
    N_b = config.N_bricks
    Q = np.zeros(3 * N_b, dtype=float)

    for i in range(N_b):
        Q[3 * i + 1] = -config.mass * config.g

    return Q


# =======================
#      РЕШАТЕЛЬ QP
# =======================

def solve_qp_equilibrium(config: BrickConfig, contacts: List[ContactPoint]) -> QPSolution:
    """
    Формулирует и решает задачу для контактных сил.

    Если USE_MANUAL_SOLVER = True: использует самописный солвер мягкого равновесия

    Если USE_MANUAL_SOLVER = False: использует классический QP-решатель cvxopt.solvers.qp
    """
    
    N_c = len(contacts)
    if N_c == 0:
        # Нет контактов нет сил, система либо висит, либо падает
        return QPSolution(
            lambda_values=np.array([], dtype=float),
            status='no_contacts',
            objective_value=0.0,
            equilibrium_error=0.0,
        )

    # Общие матрицы для обоих солверов
    J_T = calculate_jacobian_full(config, contacts)
    Q = setup_system_matrices(config)

    # ===========================
    #   Метод Проекции Градиента
    # ===========================
    if USE_MANUAL_SOLVER:
        solver_config = SoftQPSolverConfig(
            max_iters=MANUAL_MAX_ITERS,
            #step_size=1e-3,
            step_size=5e-3,
            epsilon_reg=MANUAL_EPSILON_REG,
            tol_grad=MANUAL_TOL_GRAD,
            tol_eq=MANUAL_TOL_EQ,
            verbose=False,  # True для отладки
        )

        lambda_values, status, objective_value, equilibrium_error = solve_soft_qp_equilibrium(
            J_T=J_T,
            Q=Q,
            mu=config.mu,
            config=solver_config,
            lambda_init=None,
        )

        # Пересчитаем остаток равновесия на всякий случай
        if lambda_values.size > 0:
            eq_residual = J_T @ lambda_values + Q
            equilibrium_error = float(np.linalg.norm(eq_residual))
        else:
            equilibrium_error = float(np.linalg.norm(Q))

        return QPSolution(
            lambda_values=lambda_values,
            status=status,
            objective_value=float(objective_value),
            equilibrium_error=float(equilibrium_error),
        )
        
    # ===========
    #   CVXOPT
    # ===========

    import cvxopt.base as cvx
    import cvxopt.solvers

    N_variables = 2 * N_c

    # Целевая функция: P = I, q = 0
    P = cvx.matrix(np.identity(N_variables))
    q = cvx.matrix(np.zeros((N_variables, 1)))

    # Ограничения равенства: J^T * lambda = -Q
    A_eq = cvx.matrix(J_T)
    b_eq = cvx.matrix(-Q.reshape((-1, 1)))

    # Ограничения неравенства: G * lambda <= h
    N_ineq = N_c * 3
    G = np.zeros((N_ineq, N_variables))
    h = np.zeros((N_ineq, 1))

    mu = config.mu
    for k in range(N_c):
        idx_N = 2 * k
        idx_T = 2 * k + 1

        # No Tension: -lambda_N <= 0
        row_no_tension = 3 * k
        G[row_no_tension, idx_N] = -1.0

        # Friction: lambda_T - mu * lambda_N <= 0
        row_friction_plus = 3 * k + 1
        G[row_friction_plus, idx_N] = -mu
        G[row_friction_plus, idx_T] = 1.0

        # Friction: -lambda_T - mu * lambda_N <= 0
        row_friction_minus = 3 * k + 2
        G[row_friction_minus, idx_N] = -mu
        G[row_friction_minus, idx_T] = -1.0

    G_cvx = cvx.matrix(G)
    h_cvx = cvx.matrix(h)

    # Можно отдельно настроить точность cvxopt именно здесь, если нужно отличать от MANUAL_* параметров
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = 1e-7
    cvxopt.solvers.options['reltol'] = 1e-6
    cvxopt.solvers.options['feastol'] = 1e-7

    try:
        solution = cvxopt.solvers.qp(P, q, G_cvx, h_cvx, A_eq, b_eq)
    except ValueError:
        return QPSolution(
            lambda_values=np.zeros(N_variables, dtype=float),
            status='solver_error',
            objective_value=np.nan,
            equilibrium_error=np.nan,
        )

    lambda_values = np.array(solution['x']).flatten()
    raw_status = solution['status']

    # Нормализуем статус немного для анализатора
    if raw_status == 'optimal':
        status = 'optimal'
    elif 'infeasible' in raw_status:
        status = 'infeasible'
    else:
        status = raw_status

    objective_value = float(solution.get('primal objective', np.nan))

    eq_residual = J_T @ lambda_values + Q
    equilibrium_error = float(np.linalg.norm(eq_residual))

    return QPSolution(
        lambda_values=lambda_values,
        status=status,
        objective_value=objective_value,
        equilibrium_error=equilibrium_error,
    )


# резы
def analyze_equilibrium_stability(
    config: BrickConfig,
    contacts: List[ContactPoint],
    qp_solution: QPSolution,
) -> Dict:
    """
    Анализ устойчивости системы на основе результата QP-решения.

    ВАЖНО: даже если статус не 'optimal', но есть lambda_values,
    мы всё равно считаем и показываем контактные силы — чтобы видеть,
    что делает солвер на сложных сценах.
    """
    analysis: Dict = {
        'stability': 'UNSTABLE',
        'friction_ratios': [],
        'no_tension_violations': 0,
        'sliding_risk': 'NONE',
        'equilibrium_error': qp_solution.equilibrium_error,
        'contact_forces': [],
    }

    status = qp_solution.status

    # Классифицируем устойчивость грубо по статусу + ошибке равновесия
    if status == 'optimal':
        analysis['stability'] = 'STABLE'
    elif status == 'max_iters':
        # Если ошибка маленькая численно ок, но итераций не хватило
        if qp_solution.equilibrium_error <= MANUAL_APPROX_VIS_TOL:
            analysis['stability'] = 'APPROX_STABLE_MAX_ITERS'
        else:
            analysis['stability'] = 'UNSTABLE_MAX_ITERS'
    elif status == 'infeasible':
        analysis['stability'] = 'UNSTABLE_INFEASIBLE'
    else:
        analysis['stability'] = status

    lambda_values = qp_solution.lambda_values
    if lambda_values.size == 0 or len(contacts) == 0:
        return analysis

    mu = config.mu
    sliding_risk_max = 0.0

    for k, contact in enumerate(contacts):
        idx_N = 2 * k
        idx_T = 2 * k + 1

        if idx_T >= len(lambda_values):
            break

        lambda_N = float(lambda_values[idx_N])
        lambda_T = float(lambda_values[idx_T])
        abs_lambda_T = abs(lambda_T)

        # Проверка на потерю контакта/растяжение
        if lambda_N < -config.epsilon:
            analysis['no_tension_violations'] += 1

        # Отношение трения (Friction Ratio)
        if lambda_N > config.epsilon:
            friction_ratio = abs_lambda_T / (mu * lambda_N)
        else:
            friction_ratio = np.inf if abs_lambda_T > config.epsilon else 0.0

        sliding_risk_max = max(sliding_risk_max, friction_ratio)
        analysis['friction_ratios'].append(
            (contact.brick1_id, contact.brick2_id, friction_ratio)
        )

        analysis['contact_forces'].append({
            'contact_id': k,
            'brick1': contact.brick1_id,
            'brick2': contact.brick2_id,
            'point': contact.point,
            'lambda_N': lambda_N,
            'lambda_T': lambda_T,
            'ratio': friction_ratio,
        })

    # Общее заключение о риске скольжения
    if sliding_risk_max > 1.0 + config.epsilon:
        analysis['sliding_risk'] = 'VIOLATED'
        if 'STABLE' in analysis['stability']:
            analysis['stability'] = 'UNSTABLE_FRICTION'
    elif sliding_risk_max > 0.95:
        analysis['sliding_risk'] = 'HIGH'

    if analysis['no_tension_violations'] > 0:
        analysis['stability'] = 'UNSTABLE_TENSION'

    return analysis


def print_equilibrium_analysis(analysis: Dict):
    """Выводит результаты анализа равновесия в консоль."""
    print("\n\n=== АНАЛИЗ СТАТИЧЕСКОГО РАВНОВЕСИЯ (QP) ===")

    status = analysis['stability']
    if 'STABLE' in status:
        print(f"---> Общий статус: **{status}**")
    else:
        print(f"---> Общий статус: **{status}**")

    print(f"\nТочность равновесия (L2-норма ошибки): {analysis['equilibrium_error']:.2e}")

    print(f"\nРиск проскальзывания: **{analysis['sliding_risk']}**")
    if analysis['no_tension_violations'] > 0:
        print(f"---> Нарушения No Tension (растяжение): {analysis['no_tension_violations']}")

    print("\n--- Распределение Контактных Сил ---")

    if analysis['contact_forces']:
        header = (
            f"{'ID':<4} {'B1':<4} {'B2':<4} "
            f"{'Lambda_N (N)':>15} {'Lambda_T (N)':>15} {'Ratio (|Ft/mu*Fn|)':>24}"
        )
        print(header)
        print("-" * len(header))

        for force in analysis['contact_forces']:
            ratio = force['ratio']
            if np.isinf(ratio):
                ratio_str = "∞"
            else:
                ratio_str = f"{ratio:.3f}"
                if ratio > 1.0:
                    ratio_str = f"- {ratio_str}"
                elif ratio > 0.95:
                    ratio_str = f"🟡 {ratio_str}"

            print(
                f"{force['contact_id']:<4} {force['brick1']:<4} {force['brick2']:<4} "
                f"{force['lambda_N']:>15.4f} {force['lambda_T']:>15.4f} "
                f"{ratio_str:>24}"
            )
    else:
        print("Нет контактных сил для анализа.")


def solve_system_equilibrium(config: BrickConfig, contact_analysis: Contact) -> Dict:
    """
    Основная функция, объединяющая:
      1) проверку геометрии
      2) решение QP
      3) анализ устойчивости
      4) вывод результатов
    """

    if contact_analysis.overlaps or contact_analysis.underground_bricks:
        print("---> ФАТАЛЬНАЯ ОШИБКА: Обнаружены перекрытия или проникновение в землю. Решение QP невозможно.")
        return {'stability': 'FATAL_OVERLAP_OR_PENETRATION'}

    if contact_analysis.floating_bricks:
        print(
            f"---> ПРЕДУПРЕЖДЕНИЕ: Кирпичи {contact_analysis.floating_bricks} висят в воздухе. "
            f"Система будет неустойчивой."
        )

    if not contact_analysis.contacts:
        print("---> Система не содержит контактов. Если нет гравитации, она стабильна; иначе — неустойчива.")
        if config.g != 0:
            return {'stability': 'UNSTABLE_FLOATING_SYSTEM'}
        return {'stability': 'STABLE_NO_FORCES'}

    # Решение QP
    qp_solution = solve_qp_equilibrium(config, contact_analysis.contacts)

    # Анализ результата
    analysis = analyze_equilibrium_stability(config, contact_analysis.contacts, qp_solution)

    # Вывод
    print_equilibrium_analysis(analysis)

    return analysis