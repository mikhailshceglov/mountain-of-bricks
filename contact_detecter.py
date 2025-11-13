import numpy as np

class ContactDetector:
    def __init__(self, tolerance=1e-5):
        self.tolerance = tolerance
    
    def detect_contacts(self, bricks, ground_y=0):
        """Обнаруживает все контакты между кирпичами и с землёй"""
        contacts = []
        ground_contacts = []
        
        # Контакты между кирпичами
        for i in range(len(bricks)):
            for j in range(i + 1, len(bricks)):
                brick_contacts = self.detect_brick_contact(bricks[i], bricks[j])
                contacts.extend(brick_contacts)
        
        # Контакты с землёй
        for brick in bricks:
            ground_contacts.extend(self.detect_ground_contact(brick, ground_y))
        
        return contacts, ground_contacts
    
    def detect_brick_contact(self, brick_a, brick_b):
        """Обнаруживает контакты между двумя кирпичами"""
        contacts = []
        
        # Проверяем пересечение
        if not self.bricks_intersect(brick_a, brick_b):
            return contacts
        
        # Проверяем контакты по рёбрам
        for edge_a in brick_a.edges:
            for edge_b in brick_b.edges:
                contact_point, normal = self.edge_contact(edge_a, edge_b, brick_a, brick_b)
                if contact_point is not None:
                    contacts.append(Contact(brick_a, brick_b, contact_point, normal))
        
        return contacts
    
    def bricks_intersect(self, brick_a, brick_b):
        """Проверяет пересечение двух кирпичей"""
        return not (brick_a.x + brick_a.width <= brick_b.x or
                   brick_b.x + brick_b.width <= brick_a.x or
                   brick_a.y + brick_a.height <= brick_b.y or
                   brick_b.y + brick_b.height <= brick_a.y)
    
    def edge_contact(self, edge_a, edge_b, brick_a, brick_b):
        """Определяет контакт между двумя рёбрами"""
        a1, a2 = edge_a
        b1, b2 = edge_b
        
        # Проверяем параллельность (горизонтальные или вертикальные рёбра)
        vec_a = a2 - a1
        vec_b = b2 - b1
        
        # Горизонтальное ребро
        if abs(vec_a[1]) < self.tolerance and abs(vec_b[0]) < self.tolerance:
            return self.horizontal_vertical_contact(edge_a, edge_b, brick_a, brick_b)
        # Вертикальное ребро  
        elif abs(vec_a[0]) < self.tolerance and abs(vec_b[1]) < self.tolerance:
            return self.vertical_horizontal_contact(edge_a, edge_b, brick_a, brick_b)
        
        return None, None
    
    def horizontal_vertical_contact(self, horizontal_edge, vertical_edge, brick_a, brick_b):
        """Контакт горизонтального и вертикального рёбер"""
        h1, h2 = horizontal_edge
        v1, v2 = vertical_edge
        
        # Проверяем пересечение
        if (min(h1[0], h2[0]) <= v1[0] <= max(h1[0], h2[0]) and
            min(v1[1], v2[1]) <= h1[1] <= max(v1[1], v2[1])):
            
            contact_point = np.array([v1[0], h1[1]])
            
            # Нормаль направлена от brick_a к brick_b
            center_a = brick_a.center
            center_b = brick_b.center
            direction = center_b - center_a
            
            # Определяем нормаль на основе относительного положения
            if direction[0] > 0:
                normal = np.array([-1, 0])  # Контакт справа от brick_a
            else:
                normal = np.array([1, 0])   # Контакт слева от brick_a
            
            return contact_point, normal
        
        return None, None
    
    def vertical_horizontal_contact(self, vertical_edge, horizontal_edge, brick_a, brick_b):
        """Контакт вертикального и горизонтального рёбер"""
        # Аналогично предыдущему, но роли поменялись
        return self.horizontal_vertical_contact(horizontal_edge, vertical_edge, brick_b, brick_a)
    
    def detect_ground_contact(self, brick, ground_y):
        """Обнаруживает контакты кирпича с землёй"""
        contacts = []
        
        if abs(brick.y - ground_y) < self.tolerance:
            # Контакт по нижнему ребру
            bottom_edge = (brick.vertices[0], brick.vertices[1])
            
            # Создаём фиктивный "земляной" кирпич
            class Ground:
                def __init__(self):
                    self.friction_coef = 1.0  # Высокое трение о землю
            
            ground = Ground()
            
            # Создаём контакты по углам нижнего ребра
            for point in [brick.vertices[0], brick.vertices[1]]:
                contacts.append(Contact(brick, ground, point, np.array([0, 1])))
        
        return contacts