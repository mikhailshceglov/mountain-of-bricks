import json
import numpy as np

# Общие физические константы
G = 9.81
EPSILON = 1e-4

# Параметры кирпича (одинаковые для всех)
WIDTH = 1.0
HEIGHT = 0.4
MASS = 1.0
I = MASS * (WIDTH**2 + HEIGHT**2) / 12

def create_config(config_id, filename):
    """Создает и сохраняет конфигурацию в JSON файл."""
    
    config = {
        "id": config_id,
        "description": filename.replace(".json", ""),
        "N_bricks": 0,
        "mu": 0.5,
        "epsilon": EPSILON,
        "g": G,
        "width": WIDTH,
        "height": HEIGHT,
        "mass": MASS,
        "I": I,
        "R_list": [] # Список [X, Y, Theta] для каждого кирпича
    }
    
    # --- Конфигурация 1: Арка (Arch) ---
    if config_id == 1:
        config["N_bricks"] = 2
        THETA = 20.0 * np.pi / 180
        R1 = [
            -WIDTH/2 * np.cos(THETA) + HEIGHT/2 * np.sin(THETA),
            HEIGHT/2 * np.cos(THETA) + WIDTH/2 * np.sin(THETA),
            THETA
        ]
        R2 = [
            WIDTH/2 * np.cos(THETA) - HEIGHT/2 * np.sin(THETA),
            HEIGHT/2 * np.cos(THETA) + WIDTH/2 * np.sin(THETA),
            -THETA
        ]
        config["R_list"] = [R1, R2]

    # --- Конфигурация 2: Стопка (Stack) ---
    elif config_id == 2:
        config["N_bricks"] = 2
        R1 = [0.0, HEIGHT/2, 0.0]
        R2 = [0.0, HEIGHT/2 + HEIGHT, 0.0]
        config["R_list"] = [R1, R2]
        
    # --- Конфигурация 3: Столб на двух опорах (Bridge) ---
    elif config_id == 3:
        config["N_bricks"] = 3
        # B1 и B2 на полу с зазором
        GAP = 0.1
        R1 = [-(WIDTH + GAP) / 2, HEIGHT/2, 0.0]
        R2 = [ (WIDTH + GAP) / 2, HEIGHT/2, 0.0]
        # B3 лежит сверху
        R3 = [0.0, HEIGHT + HEIGHT/2, 0.0] 
        config["R_list"] = [R1, R2, R3]

    # --- Конфигурация 4: Склоненный к стене (Leaning against wall) ---
    elif config_id == 4:
        config["N_bricks"] = 2
        
        THETA1_DEG = 45.0 # Новый угол
        THETA1 = THETA1_DEG * np.pi / 180
        THETA2 = 0.0 # Горизонтально
        
        W = config['width']
        H = config['height']
        
        # --- 1. Позиция B1 (Наклонный 45°) ---
        
        cos_t = np.cos(THETA1)
        sin_t = np.sin(THETA1)
        
        # R1_y (Высота центра масс над полом, чтобы нижний угол был на Y=0)
        R1_y = W/2 * sin_t + H/2 * cos_t 
        # R1_x (Позиция по X для касания в X=0)
        R1_x = W/2 * cos_t - H/2 * sin_t 
        
        R1 = [R1_x, R1_y, THETA1]
        
        # --- 2. Позиция B2 (Горизонтальный 0°) ---
        
        # 2a. Точка контакта P_c (Правый верхний угол B1)
        # B1: r_ur = (W/2, H/2)
        Pc_x = R1[0] + W/2 * cos_t - H/2 * sin_t
        Pc_y = R1[1] + W/2 * sin_t + H/2 * cos_t
        
        # 2b. Расчет центра B2
        
        # B2: R2_y = H/2 (стоит на полу).
        R2_y = H / 2
        
        # R2_x: X2 = Pc_x + W/2 (чтобы левый верхний угол B2 касался Pc)
        R2_x = Pc_x + W/2 
        
        # Примечание: Мы уверены, что B2 стоит на полу (Y=H/2).
        # Если Pc_y > H, то B1 поднимает B2, что и создает опору.
        
        R2 = [R2_x, R2_y, THETA2]
        
        config["R_list"] = [R1, R2]

    # Сохранение в файл
    with open(filename, 'w') as f:
        # Преобразование numpy массивов в списки для JSON
        config["R_list"] = [r.tolist() if isinstance(r, np.ndarray) else r for r in config["R_list"]]
        json.dump(config, f, indent=4)
        
    print(f"Конфигурация сохранена: {filename}")

# --- Создание всех конфигураций ---
#create_config(1, "config_arch.json")
#create_config(2, "stack.json")
#create_config(3, "bridge.json")
create_config(4, "leaning_wall.json")
#create_config(5, "overhang.json")