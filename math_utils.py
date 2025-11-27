# math_utils.py

import numpy as np
from typing import Tuple

def get_tangent(n: Tuple[float, float]) -> Tuple[float, float]:
    """
    Рассчитывает единичный вектор тангенса (t), который перпендикулярен
    единичному вектору нормали (n) в 2D.
    
    Вектор тангенса рассчитывается поворотом нормали на 90 градусов
    против часовой стрелки: t = (-ny, nx).
    
    Параметры:
        n (Tuple[float, float]): Вектор нормали (nx, ny).
        
    Возвращает:
        Tuple[float, float]: Вектор тангенса (tx, ty).
    """
    nx, ny = n
    # Проверка нормализации не требуется, так как в calculate_jacobian_full
    # нормализуется в contact_analyzer.
    return (-ny, nx)

def rotation_matrix(angle: float) -> np.ndarray:
    """
    Рассчитывает 2x2 матрицу поворота для преобразования координат.
    
    Параметры:
        angle (float): Угол поворота в радианах.
        
    Возвращает:
        np.ndarray: Матрица поворота 2x2.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([
        [c, -s], 
        [s, c]
    ])
    return R

def normalize_vector(v: Tuple[float, float], tolerance: float = 1e-10) -> Tuple[float, float]:
    """
    Нормализует 2D вектор.
    
    Параметры:
        v (Tuple[float, float]): Вектор (vx, vy).
        
    Возвращает:
        Tuple[float, float]: Нормализованный вектор.
    """
    vx, vy = v
    norm = np.sqrt(vx**2 + vy**2)
    
    if norm < tolerance:
        # Если вектор близок к нулевому, возвращаем нулевой вектор
        return (0.0, 0.0)
    
    return (vx / norm, vy / norm)

# Примечание: Функция для расчета расстояния (например, point_to_line_distance)
# должна остаться в 'contact_analyzer.py', так как она специфична для логики контактов.