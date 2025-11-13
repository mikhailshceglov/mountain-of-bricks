import numpy as np

class ContactDetector:
    def __init__(self, tolerance=1e-5):
        self.tolerance = tolerance
    
    def detect_contacts(self, bricks, ground_y=0):
        """Обнаруживает все контакты между кирпичами и с землёй"""
        contacts = []
        ground_contacts = []
        
        # Сначала импортируем Contact здесь, чтобы избежать циклических импортов
        from contact import Contact
        
        # Контакты между кирпичами
        for i in range(len(bricks)):
            for j in range(i + 1, len(bricks)):
                brick_contacts = self.detect_brick_contact(bricks[i], bricks[j])
                contacts.extend(brick_contacts)
        
        # Контакты с землёй
        for brick in bricks:
            ground_contacts.extend(self.detect_ground_contact(brick, ground_y))
        
        print(f"Найдено {len(contacts)} межкирпичных контактов")
        print(f"Найдено {len(ground_contacts)} контактов с землёй")
        
        if len(contacts) == 0:
            print("ПРЕДУПРЕЖДЕНИЕ: Не найдено контактов между кирпичами!")
        
        return contacts, ground_contacts
    
    def detect_brick_contact(self, brick_a, brick_b):
        """Обнаруживает контакты между двумя кирпичами"""
        from contact import Contact
        contacts = []
        
        # brick_a над brick_b
        if abs((brick_a.y + brick_a.height) - brick_b.y) < self.tolerance:
            overlap = self.get_horizontal_overlap(brick_a, brick_b)
            if overlap > 0:
                overlap_center = (max(brick_a.x, brick_b.x) + min(brick_a.x + brick_a.width, brick_b.x + brick_b.width)) / 2
                contact_point = np.array([overlap_center, brick_b.y])
                normal = np.array([0, 1])
                contacts.append(Contact(brick_a, brick_b, contact_point, normal))
                print(f"  Контакт: К{brick_a.id} над К{brick_b.id}")
        
        # brick_b над brick_a
        elif abs((brick_b.y + brick_b.height) - brick_a.y) < self.tolerance:
            overlap = self.get_horizontal_overlap(brick_a, brick_b)
            if overlap > 0:
                overlap_center = (max(brick_a.x, brick_b.x) + min(brick_a.x + brick_a.width, brick_b.x + brick_b.width)) / 2
                contact_point = np.array([overlap_center, brick_a.y])
                normal = np.array([0, 1])
                contacts.append(Contact(brick_b, brick_a, contact_point, normal))
                print(f"  Контакт: К{brick_b.id} над К{brick_a.id}")
        
        return contacts
    
    def get_horizontal_overlap(self, brick_a, brick_b):
        """Вычисляет горизонтальное перекрытие двух кирпичей"""
        overlap = min(brick_a.x + brick_a.width, brick_b.x + brick_b.width) - max(brick_a.x, brick_b.x)
        return max(0, overlap)
    
    def detect_ground_contact(self, brick, ground_y):
        """Обнаруживает контакты кирпича с землёй"""
        from contact import Contact
        contacts = []
        
        if abs(brick.y - ground_y) < self.tolerance:
            # Создаём фиктивный "земляной" кирпич
            class Ground:
                def __init__(self):
                    self.friction_coef = 1.0
                    self.id = -1
                    self.mass = float('inf')
            
            ground = Ground()
            
            # Создаём контакты по углам
            for x in [brick.x, brick.x + brick.width]:
                contact_point = np.array([x, ground_y])
                normal = np.array([0, 1])
                contacts.append(Contact(brick, ground, contact_point, normal))
        
        return contacts