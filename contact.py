import numpy as np

class Contact:
    next_id = 0
    
    def __init__(self, brick_a, brick_b, point, normal):
        self.id = Contact.next_id
        Contact.next_id += 1
        
        self.brick_a = brick_a
        self.brick_b = brick_b
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)
        
        # Тангенциальное направление (перпендикулярно нормали)
        self.tangent = np.array([-self.normal[1], self.normal[0]])
        
        # Переменные для QP решателя
        self.force_variables = [f"f_n_{self.id}", f"f_t_{self.id}"]
        
        # Решённые силы
        self.normal_force = 0.0
        self.tangent_force = 0.0
    
    def get_friction_constraints(self, force_variables, mu):
        """
        Возвращает ограничения трения для этого контакта
        |f_t| <= μ * f_n
        f_n >= 0
        """
        fn_idx = force_variables.index(f"f_n_{self.id}")
        ft_idx = force_variables.index(f"f_t_{self.id}")
        
        # Матрица неравенств: G * f <= h
        G = np.zeros((3, len(force_variables)))
        h = np.zeros(3)
        
        # f_n >= 0
        G[0, fn_idx] = -1
        h[0] = 0
        
        # f_t <= μ * f_n
        G[1, fn_idx] = -mu
        G[1, ft_idx] = 1
        h[1] = 0
        
        # -f_t <= μ * f_n
        G[2, fn_idx] = -mu
        G[2, ft_idx] = -1
        h[2] = 0
        
        return G, h
    
    @property
    def total_force(self):
        """Суммарная сила в контакте"""
        return self.normal * self.normal_force + self.tangent * self.tangent_force
    
    def get_safety_factor(self):
        """Коэффициент запаса по трению"""
        if abs(self.normal_force) < 1e-10:
            return float('inf')
        friction_ratio = abs(self.tangent_force) / self.normal_force
        if friction_ratio < 1e-10:
            return float('inf')
        return min(self.brick_a.friction_coef, getattr(self.brick_b, 'friction_coef', 1.0)) / friction_ratio
    
    def is_slipping(self):
        """Проверяет, проскальзывает ли контакт"""
        try:
            return self.get_safety_factor() <= 1.0
        except:
            return False