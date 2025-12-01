import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

def calculate_inertia(mass: float, width: float, height: float) -> float:
    # вычисляем момент инерции
    # I = (1/12) * M * (w^2 + h^2)
    if mass <= 0 or width <= 0 or height <= 0:
        return 0.0
    return (1.0 / 12.0) * mass * (width**2 + height**2)

@dataclass
class BrickConfig:
    # конфигурация кирпичей
    id: int
    description: str
    N_bricks: int
    mu: float
    epsilon: float
    g: float
    width: float
    height: float
    mass: float
    I: float = field(init=False) 
    R_list: List[Tuple[float, float, float]]
    
    def __post_init__(self):
        #добавляем в конфигурацию момент инерции
        self.I = calculate_inertia(self.mass, self.width, self.height)
    
    @classmethod
    def from_dict(cls, config_dict):
        # конфигурация из словаря
        return cls(
            id=config_dict.get('id', 0),
            description=config_dict.get('description', 'Unknown'),
            N_bricks=config_dict.get('N_bricks', len(config_dict.get('R_list', []))), 
            mu=config_dict.get('mu', 0.5),
            epsilon=config_dict.get('epsilon', 0.0001),
            g=config_dict.get('g', 9.81),
            width=config_dict.get('width', 1.0),
            height=config_dict.get('height', 0.4),
            mass=config_dict.get('mass', 1.0),
            R_list=[tuple(R) for R in config_dict.get('R_list', [])]
        )

def load_config_from_file(filename: str) -> BrickConfig:
    #загрузка из файла
    if not os.path.exists(filename):
        raise FileNotFoundError(f"файл '{filename}' не найден")
    
    with open(filename, 'r') as f:
        try:
            config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"ошибка в файле '{filename}': {e}")
    
    # проверка обязательных полей
    required_fields = ['R_list', 'width', 'height', 'mass']
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"отсутствует обязательное поле: {field}")
    
    return BrickConfig.from_dict(config_dict)

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        # загрузка конфигурации
        config = load_config_from_file(filename)
        
        print(f"данные загружены из '{filename}'")
        
    except FileNotFoundError as e:
        print(f"ошибка: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ошибка в конфигурации: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()