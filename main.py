import numpy as np
import matplotlib.pyplot as plt
from brick import Brick
from contact_detector import ContactDetector
from contact_solver import ContactSolver

def create_brick_wall():
    """Создаёт тестовую стену из кирпичей"""
    bricks = []
    
    # Основание
    bricks.append(Brick(0, 0, 2, 0.5, mass=5))
    bricks.append(Brick(2, 0, 2, 0.5, mass=5))
    
    # Второй ряд
    bricks.append(Brick(1, 0.5, 2, 0.5, mass=4))
    
    # Третий ряд  
    bricks.append(Brick(0.5, 1.0, 2, 0.5, mass=4))
    
    # Четвёртый ряд (нестабильный)
    bricks.append(Brick(1.5, 1.5, 2, 0.5, mass=3))
    
    return bricks

def create_brick_arch():
    """Создаёт арку из кирпичей"""
    bricks = []
    
    # Основание
    bricks.append(Brick(0, 0, 1, 0.3, mass=3))
    bricks.append(Brick(3, 0, 1, 0.3, mass=3))
    
    # Опоры
    bricks.append(Brick(0.5, 0.3, 0.8, 0.3, mass=2))
    bricks.append(Brick(2.7, 0.3, 0.8, 0.3, mass=2))
    
    # Замковый камень
    bricks.append(Brick(1.4, 0.6, 1.2, 0.3, mass=4))
    
    return bricks

def main():
    # Создаём кирпичи
    bricks = create_brick_wall()
    # bricks = create_brick_arch()  # Раскомментируйте для теста арки
    
    # Обнаруживаем контакты
    detector = ContactDetector()
    contacts, ground_contacts = detector.detect_contacts(bricks)
    
    print(f"Обнаружено контактов: {len(contacts)}")
    print(f"Контактов с землёй: {len(ground_contacts)}")
    
    # Решаем контактные силы
    solver = ContactSolver(bricks, contacts, ground_contacts)
    success = solver.solve()
    
    if success:
        print("Силы успешно вычислены!")
        
        # Анализируем устойчивость
        critical, safe = solver.analyze_stability()
        print(f"Критических контактов: {len(critical)}")
        print(f"Безопасных контактов: {len(safe)}")
        
        for contact, safety in critical:
            print(f"Контакт {contact.id}: запас прочности {safety:.2f}")
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Левая панель - кирпичи и силы
    ax1.set_title('Кирпичи и силы')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Рисуем землю
    ax1.axhline(y=0, color='brown', linewidth=2, label='Земля')
    
    # Рисуем кирпичи и силы
    force_scale = 0.01  # Масштаб для отображения сил
    for brick in bricks:
        brick.draw(ax1, force_scale)
    
    # Рисуем контакты
    for contact in contacts + ground_contacts:
        color = 'red' if contact.is_slipping() else 'green'
        ax1.plot(contact.point[0], contact.point[1], 'o', color=color, markersize=6)
        
        # Подпись контакта
        ax1.text(contact.point[0], contact.point[1] + 0.1, 
                f'{contact.id}', fontsize=8, ha='center')
    
    ax1.legend()
    
    # Правая панель - информация о силах
    ax2.set_title('Информация о силах')
    ax2.axis('off')
    
    # Выводим информацию о силах
    info_text = "СИЛЫ В КОНТАКТАХ:\n\n"
    
    for i, contact in enumerate(contacts + ground_contacts[:10]):  # Показываем первые 10
        brick_b_name = "Земля" if not hasattr(contact.brick_b, 'id') else f"Кирпич {contact.brick_b.id}"
        info_text += f"Контакт {contact.id}: {contact.brick_a.id} -> {brick_b_name}\n"
        info_text += f"  Нормальная: {contact.normal_force:.2f} N\n"
        info_text += f"  Тангенциальная: {contact.tangent_force:.2f} N\n"
        info_text += f"  Запас прочности: {contact.get_safety_factor():.2f}\n"
        info_text += f"  Проскальзывает: {'ДА' if contact.is_slipping() else 'нет'}\n\n"
    
    if len(contacts + ground_contacts) > 10:
        info_text += f"... и ещё {len(contacts + ground_contacts) - 10} контактов"
    
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=9, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
