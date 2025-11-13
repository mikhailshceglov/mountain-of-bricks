import numpy as np
import cvxopt
from cvxopt import matrix

class ContactSolver:
    def __init__(self, bricks, contacts, ground_contacts=None):
        self.bricks = bricks
        self.contacts = contacts
        self.ground_contacts = ground_contacts or []
        self.all_contacts = contacts + self.ground_contacts
        
    def solve(self):
        """Решает задачу для определения контактных сил"""
        if not self.all_contacts:
            print("Нет контактов для решения!")
            return False
            
        force_variables = []
        for contact in self.all_contacts:
            force_variables.extend(contact.force_variables)
        
        print(f"Всего переменных сил: {len(force_variables)}")
        
        # Строим систему уравнений
        A_eq_list = []
        b_eq_list = []
        
        for brick in self.bricks:
            A_brick, b_brick = brick.get_equilibrium_equations(force_variables, self.all_contacts)
            if A_brick is not None and b_brick is not None:
                A_eq_list.append(A_brick)
                b_eq_list.append(b_brick)
        
        if not A_eq_list:
            print("Нет уравнений равновесия!")
            return False
            
        A_eq = np.vstack(A_eq_list)
        b_eq = np.hstack(b_eq_list)
        
        print(f"Уравнений: {A_eq.shape[0]}, Переменных: {A_eq.shape[1]}")
        
        # Пробуем разные методы решения
        if A_eq.shape[0] > A_eq.shape[1]:
            print("Система переопределена, используем МНК...")
            return self.solve_least_squares(A_eq, b_eq, force_variables)
        else:
            # Пробуем QP, если не получится - МНК
            try:
                return self.solve_qp_method(A_eq, b_eq, force_variables)
            except:
                print("QP не удался, используем МНК...")
                return self.solve_least_squares(A_eq, b_eq, force_variables)
    
    def solve_qp_method(self, A_eq, b_eq, force_variables):
        """Решает QP задачу"""
        P = np.eye(len(force_variables)) * 0.001
        q = np.zeros(len(force_variables))
        
        # Ограничения трения
        G_list = []
        h_list = []
        
        for contact in self.all_contacts:
            mu = min(contact.brick_a.friction_coef, getattr(contact.brick_b, 'friction_coef', 1.0))
            G_contact, h_contact = contact.get_friction_constraints(force_variables, mu)
            G_list.append(G_contact)
            h_list.append(h_contact)
        
        if G_list:
            G_ineq = np.vstack(G_list)
            h_ineq = np.hstack(h_list)
        else:
            G_ineq = np.zeros((0, len(force_variables)))
            h_ineq = np.array([])
        
        # Решаем QP
        P_mat = matrix(P.astype(float))
        q_mat = matrix(q.astype(float))
        A_eq_mat = matrix(A_eq.astype(float))
        b_eq_mat = matrix(b_eq.astype(float))
        
        if G_ineq.size > 0:
            G_ineq_mat = matrix(G_ineq.astype(float))
            h_ineq_mat = matrix(h_ineq.astype(float))
        else:
            G_ineq_mat = None
            h_ineq_mat = None
        
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P_mat, q_mat, G_ineq_mat, h_ineq_mat, A_eq_mat, b_eq_mat)
        
        if solution['status'] != 'optimal':
            raise Exception(f"QP не оптимально: {solution['status']}")
        
        result = np.array(solution['x']).flatten()
        self.distribute_forces(result, force_variables)
        print("✅ QP решение найдено!")
        return True
    
    def solve_least_squares(self, A_eq, b_eq, force_variables):
        """Решает методом наименьших квадратов"""
        try:
            solution = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
            self.distribute_forces(solution, force_variables)
            print("✅ МНК решение найдено!")
            return True
        except:
            print("❌ МНК не удался, используем упрощённое решение")
            self.debug_simple_solution()
            return False
    
    def distribute_forces(self, solution, force_variables):
        """Распределяет найденные силы"""
        for brick in self.bricks:
            brick.solved_forces = []
            brick.contact_forces = []
        
        print("\nРАСПРЕДЕЛЕНИЕ СИЛ:")
        for contact in self.all_contacts:
            fn_idx = force_variables.index(f"f_n_{contact.id}")
            ft_idx = force_variables.index(f"f_t_{contact.id}")
            
            contact.normal_force = max(0, solution[fn_idx])
            contact.tangent_force = solution[ft_idx]
            
            total_force = contact.total_force
            contact.brick_a.solved_forces.append((contact, total_force))
            contact.brick_a.contact_forces.append(total_force)
            
            if hasattr(contact.brick_b, 'solved_forces') and contact.brick_b.id != -1:
                contact.brick_b.solved_forces.append((contact, -total_force))
                contact.brick_b.contact_forces.append(-total_force)
    
    def debug_simple_solution(self):
        """Упрощённое решение"""
        for brick in self.bricks:
            brick.solved_forces = []
            brick.contact_forces = []
        
        for contact in self.all_contacts:
            contact.normal_force = contact.brick_a.mass * 4.9
            contact.tangent_force = 0.0
            
            total_force = contact.total_force
            contact.brick_a.solved_forces.append((contact, total_force))
            contact.brick_a.contact_forces.append(total_force)
            
            if hasattr(contact.brick_b, 'solved_forces') and contact.brick_b.id != -1:
                contact.brick_b.solved_forces.append((contact, -total_force))
                contact.brick_b.contact_forces.append(-total_force)
    
    def analyze_stability(self):
        """Анализирует устойчивость"""
        critical = []
        safe = []
        
        for contact in self.all_contacts:
            try:
                safety = contact.get_safety_factor()
                if safety < 1.5:
                    critical.append((contact, safety))
                else:
                    safe.append((contact, safety))
            except:
                safe.append((contact, float('inf')))
        
        return critical, safe