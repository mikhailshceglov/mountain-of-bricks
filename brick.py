import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Brick:
    next_id = 0
    
    def __init__(self, x, y, width, height, mass=1.0, friction_coef=0.5):
        self.id = Brick.next_id
        Brick.next_id += 1
        
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.mass = mass
        self.friction_coef = friction_coef
        
        self.velocity = np.array([0.0, 0.0])
        self.angular_velocity = 0.0
        self.angle = 0.0
        
        # Физические свойства
        self.density = mass / (width * height)
        self.inertia = mass * (width**2 + height**2) / 12
        
        # Силы будут вычисляться решателем
        self.contact_forces = []
        self.weight_force = np.array([0, -mass * 9.81])
        
    @property
    def center(self):
        return np.array([self.x + self.width/2, self.y + self.height/2])
    
    @property
    def vertices(self):
        """Возвращает вершины кирпича"""
        return [
            np.array([self.x, self.y]),
            np.array([self.x + self.width, self.y]),
            np.array([self.x + self.width, self.y + self.height]),
            np.array([self.x, self.y + self.height])
        ]
    
    @property
    def edges(self):
        """Возвращает рёбра кирпича"""
        verts = self.vertices
        return [
            (verts[0], verts[1]),
            (verts[1], verts[2]),
            (verts[2], verts[3]),
            (verts[3], verts[0])
        ]
    
    def contains_point(self, point):
        """Проверяет, находится ли точка внутри кирпича"""
        px, py = point
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)
    
    def get_equilibrium_equations(self, force_variables, contacts):
        """
        Возвращает матрицы для уравнений равновесия этого кирпича
        """
        # 3 уравнения: sum(Fx)=0, sum(Fy)=0, sum(M)=0
        A_eq = np.zeros((3, len(force_variables)))
        b_eq = np.zeros(3)
        
        # Вес кирпича (сила тяжести)
        b_eq[1] = -self.weight_force[1]  # Fy = -mg
        
        relevant_contacts = [c for c in contacts if c.brick_a == self or c.brick_b == self]
        
        if not relevant_contacts:
            return None, None
        
        for contact in relevant_contacts:
            # Найти индексы переменных сил для этого контакта
            try:
                fn_idx = force_variables.index(f"f_n_{contact.id}")
                ft_idx = force_variables.index(f"f_t_{contact.id}")
                
                # Определить направление силы (кирпич A или B)
                sign = 1 if contact.brick_a == self else -1
                
                # Точка приложения силы относительно центра
                r = contact.point - self.center
                
                # Уравнения сил
                # Fx: f_t * t_x + f_n * n_x
                A_eq[0, fn_idx] += sign * contact.normal[0]
                A_eq[0, ft_idx] += sign * contact.tangent[0]
                
                # Fy: f_t * t_y + f_n * n_y  
                A_eq[1, fn_idx] += sign * contact.normal[1]
                A_eq[1, ft_idx] += sign * contact.tangent[1]
                
                # Момент: r x F = r_x*F_y - r_y*F_x
                A_eq[2, fn_idx] += sign * (r[0]*contact.normal[1] - r[1]*contact.normal[0])
                A_eq[2, ft_idx] += sign * (r[0]*contact.tangent[1] - r[1]*contact.tangent[0])
            except ValueError:
                continue
        
        return A_eq, b_eq
    
    def draw(self, ax, force_scale=0.01):
        """Отрисовка кирпича и действующих сил"""
        # Рисуем кирпич
        rect = Rectangle((self.x, self.y), self.width, self.height, 
                        linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(rect)
        
        # Рисуем центр масс
        ax.plot(self.center[0], self.center[1], 'ro', markersize=3)
        
        # Рисуем силы, если они есть
        if hasattr(self, 'solved_forces'):
            self.draw_forces(ax, force_scale)
    
    def draw_forces(self, ax, force_scale=0.01):
        """Отрисовка сил, действующих на кирпич"""
        center = self.center
        
        # Сила тяжести (вниз)
        weight_end = center + self.weight_force * force_scale
        ax.arrow(center[0], center[1], 
                self.weight_force[0] * force_scale, self.weight_force[1] * force_scale,
                head_width=0.1, head_length=0.1, fc='red', ec='red', label='Weight')
        
        # Контактные силы
        for contact, force in self.solved_forces:
            if np.linalg.norm(force) > 1e-5:
                force_scaled = force * force_scale
                ax.arrow(contact.point[0], contact.point[1],
                        force_scaled[0], force_scaled[1],
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue')