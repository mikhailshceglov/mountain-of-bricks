import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import cvxopt.solvers
import cvxopt.base as cvx

from load_config import BrickConfig
from contact_finder import Contact, ContactPoint

# Если rotation_matrix, get_tangent и другие хелперы вынесены в отдельный файл,
# их нужно импортировать. Здесь оставляем get_tangent, а rotation_matrix удаляем, 
# так как она не используется в финальном коде (оставлена только как пример).

@dataclass
class QPSolution:
    """Контейнер для результата решения QP"""
    lambda_values: np.ndarray
    status: str
    objective_value: float
    equilibrium_error: float = np.nan

# Устанавливаем решатель
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-7
cvxopt.solvers.options['reltol'] = 1e-6
cvxopt.solvers.options['feastol'] = 1e-7

# --- ГЛОБАЛЬНЫЕ МАТЕМАТИЧЕСКИЕ ФУНКЦИИ ---

def get_tangent(n: Tuple[float, float]) -> Tuple[float, float]:
    """Находит единичный вектор тангенса t (перпендикулярно нормали n, в 2D)"""
    # В 2D тангенс - это просто поворот нормали на 90 градусов: t = (-ny, nx)
    return (-n[1], n[0])

def calculate_jacobian_full(config: BrickConfig, contacts: List[ContactPoint]) -> np.ndarray:
    """
    Рассчитывает транспонированную матрицу Якоби (J.T).
    J.T[i, j] - это вклад j-й контактной силы (lambda) в i-ю обобщенную силу (F_x, F_y, M_z).
    Размерность: (3 * N_bricks) x (2 * N_contacts)
    """
    N_c = len(contacts)
    N_b = config.N_bricks
    
    # Размерность J: (2 * N_c) строк x (3 * N_b) столбцов
    # Мы строим J, а возвращаем J.T
    J = np.zeros((2 * N_c, 3 * N_b))

    for k, contact in enumerate(contacts):
        n_global = np.array(contact.n_global)
        t_global = np.array(get_tangent(contact.n_global))
        contact_point = np.array(contact.point)

        # 1. Обработка кирпича 1 (Brick 1) - Вклад с положительным знаком
        i = contact.brick1_id
        R_i = np.array(config.R_list[i][:2])  # (x, y) центра масс
        r_i = contact_point - R_i             # Плечо силы
        
        # Индексы: 2*k для lambda_N, 2*k + 1 для lambda_T
        
        # Вклад нормальной силы (Строка 2k)
        J[2 * k, 3 * i]     = n_global[0]  # Fx
        J[2 * k, 3 * i + 1] = n_global[1]  # Fy
        J[2 * k, 3 * i + 2] = r_i[0] * n_global[1] - r_i[1] * n_global[0] # Mz (r x n)

        # Вклад тангенциальной силы (Строка 2k + 1)
        J[2 * k + 1, 3 * i]     = t_global[0]  # Fx
        J[2 * k + 1, 3 * i + 1] = t_global[1]  # Fy
        J[2 * k + 1, 3 * i + 2] = r_i[0] * t_global[1] - r_i[1] * t_global[0] # Mz (r x t)

        # 2. Обработка кирпича 2 (Brick 2) - Вклад с отрицательным знаком
        j = contact.brick2_id
        if j != -1:  # Если это не земля
            R_j = np.array(config.R_list[j][:2])
            r_j = contact_point - R_j
            
            # Нормальная сила, действующая на Brick j, противоположна n_global
            J[2 * k, 3 * j]     = -n_global[0]
            J[2 * k, 3 * j + 1] = -n_global[1]
            J[2 * k, 3 * j + 2] = -(r_j[0] * n_global[1] - r_j[1] * n_global[0]) # Mz: r x (-n) = -(r x n)

            # Тангенциальная сила, действующая на Brick j, противоположна t_global
            J[2 * k + 1, 3 * j]     = -t_global[0]
            J[2 * k + 1, 3 * j + 1] = -t_global[1]
            J[2 * k + 1, 3 * j + 2] = -(r_j[0] * t_global[1] - r_j[1] * t_global[0]) # Mz: r x (-t) = -(r x t)

    return J.T # Возвращаем транспонированную матрицу J.T

def setup_system_matrices(config: BrickConfig) -> np.ndarray:
    """
    Рассчитывает вектор внешних сил Q (только гравитация в статике).
    Размерность Q: (3 * N_bricks) x 1
    """
    N_b = config.N_bricks
    Q = np.zeros((3 * N_b, 1))
    
    # Внешние силы (гравитация)
    for i in range(N_b):
        # F_y (индекс 3*i + 1)
        Q[3 * i + 1, 0] = -config.mass * config.g
        
    return Q

# --- РЕШАТЕЛЬ QP ---

def solve_qp_equilibrium(config: BrickConfig, contacts: List[ContactPoint]) -> QPSolution:
    """
    Формулирует и решает задачу QP: min (1/2 * lambda.T * I * lambda),
    при ограничениях равновесия и физических законов.
    """
    N_c = len(contacts)
    N_variables = 2 * N_c  # lambda_N и lambda_T для каждого контакта
    
    if N_c == 0:
        return QPSolution(np.array([]), 'no_contacts', 0.0)

    # 1. Целевая функция: P = I, q = 0
    P = cvx.matrix(np.identity(N_variables))
    q = cvx.matrix(np.zeros((N_variables, 1)))

    # 2. Ограничения равенства (Равновесие): J.T * lambda = -Q
    J_T = calculate_jacobian_full(config, contacts) 
    Q = setup_system_matrices(config)              
    
    A_eq = cvx.matrix(J_T)
    b_eq = cvx.matrix(-Q)

    # 3. Ограничения неравенства: G * lambda <= h (24 ограничения для 8 контактов)
    N_ineq = N_c * 3  # N_c (No Tension) + N_c*2 (Friction Cone)
    G = np.zeros((N_ineq, N_variables))
    h = np.zeros((N_ineq, 1))
    
    mu = config.mu
    
    for k in range(N_c):
        idx_N = 2 * k   # Индекс lambda_N
        idx_T = 2 * k + 1 # Индекс lambda_T
        
        # A. Непроникновение: -lambda_N <= 0 (Строка 3*k)
        row_no_tension = 3 * k
        G[row_no_tension, idx_N] = -1.0
        
        # B. Конус трения: lambda_T - mu*lambda_N <= 0 (Строка 3*k + 1)
        row_friction_plus = 3 * k + 1
        G[row_friction_plus, idx_N] = -mu
        G[row_friction_plus, idx_T] = 1.0
        
        # C. Конус трения: -lambda_T - mu*lambda_N <= 0 (Строка 3*k + 2)
        row_friction_minus = 3 * k + 2
        G[row_friction_minus, idx_N] = -mu
        G[row_friction_minus, idx_T] = -1.0
        
    G_cvx = cvx.matrix(G)
    h_cvx = cvx.matrix(h) # h остается нулевым вектором

    # 4. Решение
    try:
        solution = cvxopt.solvers.qp(P, q, G_cvx, h_cvx, A_eq, b_eq)
    except ValueError as e:
        return QPSolution(np.zeros(N_variables), 'solver_error', np.nan)

    # 5. Обработка результата
    lambda_values = np.array(solution['x']).flatten()
    status = solution['status']
    objective_value = solution['primal objective'] if 'primal objective' in solution else np.nan

    if status == 'optimal':
        # Проверка ошибки равновесия (насколько точно выполнено J.T * lambda = -Q)
        equilibrium_check = J_T @ lambda_values + Q.flatten()
        equilibrium_error = np.linalg.norm(equilibrium_check)
    else:
        equilibrium_error = np.nan

    return QPSolution(lambda_values, status, objective_value, equilibrium_error)


# --- АНАЛИЗ РЕЗУЛЬТАТОВ ---

def analyze_equilibrium_stability(config: BrickConfig, contacts: List[ContactPoint], qp_solution: QPSolution) -> Dict:
    """Интерпретирует решение QP для пользователя."""
    analysis = {
        'stability': 'UNSTABLE' if qp_solution.status != 'optimal' else 'STABLE',
        'friction_ratios': [],
        'no_tension_violations': 0,
        'sliding_risk': 'NONE',
        'equilibrium_error': qp_solution.equilibrium_error,
        'contact_forces': []
    }
    
    if qp_solution.status != 'optimal':
        analysis['stability'] = 'UNSTABLE_INFEASIBLE' if qp_solution.status == 'infeasible' else qp_solution.status
        return analysis

    lambda_values = qp_solution.lambda_values
    mu = config.mu
    sliding_risk_max = 0.0
    
    for k, contact in enumerate(contacts):
        idx_N = 2 * k
        idx_T = 2 * k + 1
        
        lambda_N = lambda_values[idx_N]
        lambda_T = lambda_values[idx_T]
        abs_lambda_T = abs(lambda_T)
        
        # A. Проверка на потерю контакта/растяжение
        if lambda_N < -config.epsilon: 
            analysis['no_tension_violations'] += 1
            
        # B. Отношение трения (Friction Ratio)
        if lambda_N > config.epsilon:
            friction_ratio = abs_lambda_T / (mu * lambda_N)
        else:
            # Если контактная сила ~0, но тангенциальная есть, то это проскальзывание/неустойчивость
            friction_ratio = np.inf if abs_lambda_T > config.epsilon else 0.0

        sliding_risk_max = max(sliding_risk_max, friction_ratio)
        analysis['friction_ratios'].append((contact.brick1_id, contact.brick2_id, friction_ratio))
        
        # Сохранение найденных сил
        analysis['contact_forces'].append({
            'contact_id': k,
            'brick1': contact.brick1_id,
            'brick2': contact.brick2_id,
            'point': contact.point,
            'lambda_N': lambda_N,
            'lambda_T': lambda_T,
            'ratio': friction_ratio
        })

    # Общее заключение о риске скольжения
    if sliding_risk_max > 1.0 + config.epsilon:
        analysis['sliding_risk'] = 'VIOLATED'
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
        print(f"✅ Общий статус: **{status}**")
    else:
        print(f"❌ Общий статус: **{status}**")
        
    print(f"\nТочность равновесия (L2-норма ошибки): {analysis['equilibrium_error']:.2e}")
    
    # 2. Анализ рисков
    print(f"\nРиск проскальзывания: **{analysis['sliding_risk']}**")
    if analysis['no_tension_violations'] > 0:
        print(f"🚨 Нарушения No Tension (растяжение): {analysis['no_tension_violations']}")
    
    # 3. Детальный анализ сил
    print("\n--- Распределение Контактных Сил ---")
    
    if analysis['contact_forces']:
        header = f"{'ID':<4} {'B1':<4} {'B2':<4} {'Lambda_N (N)':>15} {'Lambda_T (N)':>15} {'Ratio (|Ft/mu*Fn|)':>24}"
        print(header)
        print("-" * len(header))
        
        for force in analysis['contact_forces']:
            ratio_str = f"{force['ratio']:.3f}"
            if force['ratio'] > 1.0:
                 ratio_str = f"🛑 {ratio_str}"
            elif force['ratio'] > 0.95:
                 ratio_str = f"🟡 {ratio_str}"
            
            print(f"{force['contact_id']:<4} {force['brick1']:<4} {force['brick2']:<4} "
                  f"{force['lambda_N']:>15.4f} {force['lambda_T']:>15.4f} "
                  f"{ratio_str:>24}")
    else:
        print("Нет контактных сил для анализа.")

# --- ОСНОВНАЯ ФУНКЦИЯ РЕШЕНИЯ ---

def solve_system_equilibrium(config: BrickConfig, contact_analysis: Contact) -> Dict:
    """
    Основная функция, объединяющая расчеты и анализ.
    """
    
    # 1. Проверка на фатальные ошибки
    if contact_analysis.overlaps or contact_analysis.underground_bricks:
        print("❌ ФАТАЛЬНАЯ ОШИБКА: Обнаружены перекрытия или проникновение в землю. Решение QP невозможно.")
        return {'stability': 'FATAL_OVERLAP_OR_PENETRATION'}

    if contact_analysis.floating_bricks:
         print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Кирпичи {contact_analysis.floating_bricks} висят в воздухе. Система будет неустойчивой.")
         
    if not contact_analysis.contacts:
        print("✅ Система не содержит контактов. Если нет гравитации, она стабильна; иначе - неустойчива.")
        if config.g != 0:
             return {'stability': 'UNSTABLE_FLOATING_SYSTEM'}
        return {'stability': 'STABLE_NO_FORCES'}
    
    # 2. Решение QP
    qp_solution = solve_qp_equilibrium(config, contact_analysis.contacts)
    
    # 3. Анализ результата
    analysis = analyze_equilibrium_stability(config, contact_analysis.contacts, qp_solution)
    
    # 4. Вывод
    print_equilibrium_analysis(analysis)
    
    return analysis

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ (Для отладки) ---
# ... (Этот блок нужно удалить или изменить для рабочего проекта)