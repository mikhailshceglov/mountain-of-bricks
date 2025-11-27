# system_solver.py
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
from load_config import BrickConfig
from contact_finder import Contact

@dataclass
class QPSolution:
    """Результат решения QP задачи"""
    lambda_values: np.ndarray  # Вектор контактных сил [λ_N1, λ_T1, λ_N2, λ_T2, ...]
    status: str  # 'optimal', 'infeasible', 'unbounded', 'max_iterations'
    objective_value: float
    equilibrium_error: float
    iterations: int
    solve_time: float

def calculate_jacobian_full(config: BrickConfig, contacts: Contact) -> np.ndarray:
    """
    Расчет полной матрицы Якоби J системы
    """
    N_bricks = len(config.R_list)
    N_contacts = len(contacts.contacts)
    
    # Матрица Якоби: 2 строки на контакт (нормальная и тангенциальная составляющие)
    # 3 столбца на кирпич (x, y, theta)
    J = np.zeros((2 * N_contacts, 3 * N_bricks))
    
    for contact_idx, contact_point in enumerate(contacts.contacts):
        # Индексы для этого контакта в матрице J
        row_idx_normal = 2 * contact_idx
        row_idx_tangential = 2 * contact_idx + 1
        
        # Точка контакта
        x_contact, y_contact = contact_point.point
        
        # Для контактов между кирпичами
        if contact_point.brick2_id != -1:  # Контакт кирпич-кирпич
            brick1_id = contact_point.brick1_id
            brick2_id = contact_point.brick2_id
            
            # Координаты центров кирпичей
            x1, y1, theta1 = config.R_list[brick1_id]
            x2, y2, theta2 = config.R_list[brick2_id]
            
            # Нормаль и тангенс для контакта
            # Для угол-угол контактов используем направление между центрами
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 1e-10:
                normal_x = dx / distance
                normal_y = dy / distance
            else:
                normal_x = 1.0
                normal_y = 0.0
            
            # Тангенциальное направление (перпендикулярно нормали)
            tangential_x = -normal_y
            tangential_y = normal_x
            
            # Для кирпича 1
            col_idx1_x = 3 * brick1_id
            col_idx1_y = 3 * brick1_id + 1
            col_idx1_theta = 3 * brick1_id + 2
            
            # Нормальная составляющая для кирпича 1
            J[row_idx_normal, col_idx1_x] = normal_x
            J[row_idx_normal, col_idx1_y] = normal_y
            J[row_idx_normal, col_idx1_theta] = -normal_y * (x_contact - x1) + normal_x * (y_contact - y1)
            
            # Тангенциальная составляющая для кирпича 1
            J[row_idx_tangential, col_idx1_x] = tangential_x
            J[row_idx_tangential, col_idx1_y] = tangential_y
            J[row_idx_tangential, col_idx1_theta] = -tangential_y * (x_contact - x1) + tangential_x * (y_contact - y1)
            
            # Для кирпича 2
            col_idx2_x = 3 * brick2_id
            col_idx2_y = 3 * brick2_id + 1
            col_idx2_theta = 3 * brick2_id + 2
            
            # Нормальная составляющая для кирпича 2
            J[row_idx_normal, col_idx2_x] = -normal_x
            J[row_idx_normal, col_idx2_y] = -normal_y
            J[row_idx_normal, col_idx2_theta] = normal_y * (x_contact - x2) - normal_x * (y_contact - y2)
            
            # Тангенциальная составляющая для кирпича 2
            J[row_idx_tangential, col_idx2_x] = -tangential_x
            J[row_idx_tangential, col_idx2_y] = -tangential_y
            J[row_idx_tangential, col_idx2_theta] = tangential_y * (x_contact - x2) - tangential_x * (y_contact - y2)
            
        else:  # Контакт с землей
            brick_id = contact_point.brick1_id
            x_center, y_center, theta = config.R_list[brick_id]
            
            # Для контакта с землей нормаль направлена вверх
            normal_x = 0.0
            normal_y = 1.0
            tangential_x = 1.0
            tangential_y = 0.0
            
            col_idx_x = 3 * brick_id
            col_idx_y = 3 * brick_id + 1
            col_idx_theta = 3 * brick_id + 2
            
            # Нормальная составляющая
            J[row_idx_normal, col_idx_x] = normal_x
            J[row_idx_normal, col_idx_y] = normal_y
            J[row_idx_normal, col_idx_theta] = -normal_y * (x_contact - x_center) + normal_x * (y_contact - y_center)
            
            # Тангенциальная составляющая
            J[row_idx_tangential, col_idx_x] = tangential_x
            J[row_idx_tangential, col_idx_y] = tangential_y
            J[row_idx_tangential, col_idx_theta] = -tangential_y * (x_contact - x_center) + tangential_x * (y_contact - y_center)
    
    return J

def setup_system_matrices(config: BrickConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Настройка матриц системы: матрицы масс и вектора внешних сил
    """
    N_bricks = len(config.R_list)
    
    # Расчет момента инерции I для прямоугольника
    W = config.width
    H = config.height
    I = (1/12) * config.mass * (W**2 + H**2)
    
    # Матрица масс (диагональная)
    M = np.zeros((3 * N_bricks, 3 * N_bricks))
    
    # Вектор внешних сил (гравитация)
    Q = np.zeros(3 * N_bricks)
    
    for i in range(N_bricks):
        # Диагональные элементы матрицы масс
        M[3*i, 3*i] = config.mass          # m * x''
        M[3*i+1, 3*i+1] = config.mass      # m * y''
        M[3*i+2, 3*i+2] = I                # I * theta''
        
        # Вектор гравитации (только по Y)
        Q[3*i + 1] = -config.mass * config.g
    
    return M, Q

class SimpleQPSolver:
    """
    Простой QP решатель для задачи статического равновесия
    """
    
    def __init__(self, max_iterations=1000, tolerance=1e-8, verbose=False):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
    def _print(self, message):
        if self.verbose:
            print(message)
    
    def solve(self, J: np.ndarray, Q: np.ndarray, mu: float) -> QPSolution:
        """
        Решение QP задачи статического равновесия
        """
        start_time = time.time()
        
        self._print("🔧 Запуск QP решателя...")
        self._print(f"   Размер J: {J.shape}")
        self._print(f"   Размер Q: {Q.shape}")
        self._print(f"   Коэф. трения μ: {mu}")
        
        N_contacts = J.shape[0] // 2
        N_variables = 2 * N_contacts
        
        if N_contacts == 0:
            return QPSolution(
                lambda_values=np.array([]),
                status='no_contacts',
                objective_value=0.0,
                equilibrium_error=np.inf,
                iterations=0,
                solve_time=0.0
            )
        
        # 1. Инициализация
        lambda_vec = np.ones(N_variables) * 0.1
        
        # 2. Матрица Гессе (единичная для нашей целевой функции)
        H = np.eye(N_variables)
        
        # 3. Ограничения равенства: J^T * λ = -Q
        A_eq = J.T
        b_eq = -Q
        
        # 4. Решаем задачу с ограничениями равенства методом наименьших квадратов
        try:
            # Решаем систему методом наименьших квадратов
            lambda_vec, residuals, rank, s = np.linalg.lstsq(A_eq, b_eq, rcond=None)
            
            # Если решение не найдено, используем псевдо-обратную матрицу
            if len(lambda_vec) == 0:
                lambda_vec = np.linalg.pinv(A_eq) @ b_eq
            
            self._print("   ✅ Система решена методом наименьших квадратов")
            
        except np.linalg.LinAlgError:
            self._print("   ❌ Ошибка решения системы, используем псевдо-обратную матрицу")
            lambda_vec = np.linalg.pinv(A_eq) @ b_eq
        
        # 5. Применяем ограничения неравенства
        lambda_vec = self._apply_inequality_constraints(lambda_vec, mu, N_contacts)
        
        # 6. Проверяем сходимость и вычисляем ошибки
        objective_value = 0.5 * lambda_vec.T @ H @ lambda_vec
        equilibrium_error = np.linalg.norm(A_eq @ lambda_vec - b_eq)
        
        solve_time = time.time() - start_time
        
        # Проверяем выполнение ограничений
        feasible = self._check_feasibility(lambda_vec, mu, N_contacts, equilibrium_error)
        
        status = 'optimal' if feasible else 'infeasible'
        
        self._print(f"   ✅ QP решение завершено")
        self._print(f"   📊 Статус: {status}")
        self._print(f"   🔢 Итераций: 1")
        self._print(f"   📏 Норма сил: {np.linalg.norm(lambda_vec):.6f}")
        self._print(f"   🎯 Ошибка равновесия: {equilibrium_error:.6e}")
        self._print(f"   ⏱️ Время решения: {solve_time:.4f} сек")
        
        return QPSolution(
            lambda_values=lambda_vec,
            status=status,
            objective_value=objective_value,
            equilibrium_error=equilibrium_error,
            iterations=1,
            solve_time=solve_time
        )
    
    def _apply_inequality_constraints(self, lambda_vec: np.ndarray, mu: float, N_contacts: int) -> np.ndarray:
        """
        Применение ограничений неравенства методом проекции
        """
        for i in range(N_contacts):
            idx_N = 2 * i      # Нормальная сила
            idx_T = 2 * i + 1  # Тангенциальная сила
            
            lambda_N = lambda_vec[idx_N]
            lambda_T = lambda_vec[idx_T]
            
            # Ограничение 1: λ_N >= 0
            if lambda_N < 0:
                lambda_vec[idx_N] = 0.0
                lambda_N = 0.0
            
            # Ограничения 2-3: |λ_T| <= μ * λ_N
            max_friction = mu * lambda_N
            if abs(lambda_T) > max_friction:
                lambda_vec[idx_T] = np.sign(lambda_T) * max_friction
        
        return lambda_vec
    
    def _check_feasibility(self, lambda_vec: np.ndarray, mu: float, N_contacts: int, 
                          equilibrium_error: float) -> bool:
        """
        Проверка выполнения всех ограничений
        """
        # Проверка ограничений равенства
        if equilibrium_error > self.tolerance:
            return False
        
        # Проверка ограничений неравенства
        for i in range(N_contacts):
            idx_N = 2 * i
            idx_T = 2 * i + 1
            
            lambda_N = lambda_vec[idx_N]
            lambda_T = lambda_vec[idx_T]
            
            # λ_N >= 0
            if lambda_N < -self.tolerance:
                return False
            
            # |λ_T| <= μ * λ_N
            if abs(lambda_T) > mu * lambda_N + self.tolerance:
                return False
        
        return True

def solve_qp_equilibrium(config: BrickConfig, contacts: Contact, verbose: bool = True) -> QPSolution:
    """
    Основная функция решения QP задачи статического равновесия
    """
    print("🔧 Начинаем решение QP задачи...")
    
    try:
        # 1. Расчет матрицы Якоби
        print("    📐 Расчет матрицы Якоби J...")
        J = calculate_jacobian_full(config, contacts)
        print(f"    ✅ Размерность J: {J.shape}")
        
        # 2. Настройка вектора внешних сил
        print("    ⚖️  Настройка вектора внешних сил Q...")
        M, Q = setup_system_matrices(config)
        print(f"    ✅ Размерность Q: {Q.shape}")
        
        # 3. Решение QP задачи
        solver = SimpleQPSolver(verbose=verbose)
        solution = solver.solve(J, Q, config.mu)
        
        return solution
        
    except Exception as e:
        print(f"    💥 Ошибка при решении QP: {e}")
        import traceback
        traceback.print_exc()
        return QPSolution(
            lambda_values=np.array([]),
            status='error',
            objective_value=0.0,
            equilibrium_error=np.inf,
            iterations=0,
            solve_time=0.0
        )

def analyze_equilibrium_stability(config: BrickConfig, contacts: Contact, 
                                qp_solution: QPSolution) -> Dict:
    """
    Анализ устойчивости системы на основе QP решения
    """
    analysis = {
        'is_stable': False,
        'status': qp_solution.status,
        'total_contacts': len(contacts.contacts),
        'contact_forces': {},
        'warnings': [],
        'recommendations': []
    }
    
    if qp_solution.status == 'optimal':
        analysis['is_stable'] = True
        analysis['equilibrium_error'] = qp_solution.equilibrium_error
        
        # Анализ распределения сил по контактам
        for i, contact_point in enumerate(contacts.contacts):
            if i * 2 + 1 >= len(qp_solution.lambda_values):
                continue
                
            idx_N = 2 * i
            idx_T = 2 * i + 1
            
            lambda_N = qp_solution.lambda_values[idx_N]
            lambda_T = qp_solution.lambda_values[idx_T]
            
            # Создаем ключ для контакта
            if contact_point.brick2_id == -1:
                contact_key = f"B{contact_point.brick1_id}-BGround"
            else:
                contact_key = f"B{contact_point.brick1_id}-B{contact_point.brick2_id}"
            
            analysis['contact_forces'][contact_key] = {
                'normal_force': lambda_N,
                'tangential_force': lambda_T,
                'friction_ratio': abs(lambda_T) / (config.mu * lambda_N) if lambda_N > 1e-10 else np.inf,
            }
            
            # Проверка на граничные условия трения
            if lambda_N > 1e-10 and abs(lambda_T) / (config.mu * lambda_N) > 0.95:
                analysis['warnings'].append(
                    f"Контакт {contact_key} близок к пределу трения (отношение: {abs(lambda_T)/(config.mu * lambda_N):.3f})"
                )
        
        # Проверка минимальных нормальных сил
        if analysis['contact_forces']:
            min_normal_force = min(
                analysis['contact_forces'][key]['normal_force'] 
                for key in analysis['contact_forces']
            )
            
            if min_normal_force < 1e-6:
                analysis['warnings'].append(
                    f"Обнаружены очень малые нормальные силы (min: {min_normal_force:.2e})"
                )
        
        analysis['recommendations'].append("✅ Система находится в статическом равновесии")
        
    else:
        analysis['is_stable'] = False
        
        if qp_solution.status == 'infeasible':
            analysis['warnings'].append("❌ Задача несовместна: невозможно удовлетворить всем ограничениям")
            analysis['recommendations'].append("Проверьте перекрытия кирпичей и контакты с землей")
            
        elif qp_solution.status == 'unbounded':
            analysis['warnings'].append("❌ Задача неограничена: возможно, недостаточно ограничений")
            analysis['recommendations'].append("Проверьте матрицу Якоби и ограничения трения")
            
        elif qp_solution.status == 'no_contacts':
            analysis['warnings'].append("❌ Нет контактов между кирпичами")
            analysis['recommendations'].append("Проверьте расположение кирпичей")
            
        else:
            analysis['warnings'].append(f"❌ Ошибка решения: {qp_solution.status}")
            analysis['recommendations'].append("Проверьте входные данные и параметры решателя")
    
    return analysis

def print_equilibrium_analysis(analysis: Dict):
    """
    Красивый вывод анализа устойчивости
    """
    print("\n" + "="*60)
    print("📊 АНАЛИЗ СТАТИЧЕСКОГО РАВНОВЕСИЯ")
    print("="*60)
    
    print(f"📈 Статус устойчивости: {'✅ СТАБИЛЬНА' if analysis['is_stable'] else '❌ НЕСТАБИЛЬНА'}")
    print(f"🎯 Статус QP: {analysis['status']}")
    print(f"🔗 Всего контактов: {analysis['total_contacts']}")
    
    if analysis['is_stable']:
        print(f"📏 Ошибка равновесия: {analysis.get('equilibrium_error', 0):.2e}")
        
        print(f"\n📋 Распределение сил по контактам:")
        for contact_key, forces in analysis['contact_forces'].items():
            print(f"    {contact_key}:")
            print(f"      ┣ Нормальная сила: {forces['normal_force']:8.4f}")
            print(f"      ┣ Тангенциальная сила: {forces['tangential_force']:8.4f}")
            if forces['normal_force'] > 1e-10:
                print(f"      ┗ Отношение трения: {forces['friction_ratio']:8.4f}")
            else:
                print(f"      ┗ Отношение трения: {'∞':>8}")
    
    if analysis['warnings']:
        print(f"\n⚠️  Предупреждения:")
        for warning in analysis['warnings']:
            print(f"    • {warning}")
    
    if analysis['recommendations']:
        print(f"\n💡 Рекомендации:")
        for recommendation in analysis['recommendations']:
            print(f"    • {recommendation}")
    
    print("="*60)