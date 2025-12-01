import argparse
import sys
import os

# === ДОБАВЛЯЕМ src В PYTHONPATH ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from load_config import load_config_from_file
from contact_finder import analyze_contacts, Contact
from visualization import BrickVisualizer
from system_solver import (
    solve_qp_equilibrium,
    analyze_equilibrium_stability,
    print_equilibrium_analysis,
)

def main():
    parser = argparse.ArgumentParser(
        description='визуализация конфигураций кирпичей из json файлов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''примеры использования: python main.py name_config.json'''
    )
    
    parser.add_argument('filename', help='Путь к JSON файлу с конфигурацией кирпичей')

    args = parser.parse_args()
    
    try:
        #загружаем конфигурацию из JSON файла
        print(f"загрузка конфигурации из файла: {args.filename}")
        config = load_config_from_file(args.filename)
        print("конфигурация загружена")
        print()

        #ищем контакты кирпичей
        print("поиск контактов между кирпичами")
        contact = analyze_contacts(config)
        print("контакты найдены")
        print()

        R_list = config.R_list
        if (len(contact.contacts)<3*len(R_list)/2):
            print("задача не разрешима с помощью солвера")
            visualizer = BrickVisualizer()
            visualizer.visualize_system(config, contact)
            exit(1)

        #ищем силы через qp-солвер
        print(f"решаем qp для конфигурации: {config.description}")
        contact_list = contact.contacts
        qp_solution = solve_qp_equilibrium(config, contact_list) 
        print()
        
        #проверка устойчивости системы
        analysis = analyze_equilibrium_stability(config, contact_list, qp_solution)
        print_equilibrium_analysis(analysis)
        
        #=визуализируем конфигурацию
        visualizer = BrickVisualizer()

        visualizer.visualize_system(config, contact, analysis)
        
        print("визуализация завершена успешно")
        
    except FileNotFoundError as e:
        print(f"ошибка: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ошибка в конфигурации: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
