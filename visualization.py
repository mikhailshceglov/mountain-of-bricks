import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import numpy as np
import argparse
import sys
import os

from typing import Dict, List, Tuple 

# Убедитесь, что эти модули импортированы и доступны
from load_config import BrickConfig
from contact_finder import Contact
from math_utils import get_tangent 

class BrickVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
      
    def create_brick_patch(self, x, y, width, height, angle, color='lightblue', edgecolor='black'):
        """Создание патча для кирпича с поворотом"""
        brick = Rectangle((x - width/2, y - height/2), width, height,
                          linewidth=2, edgecolor=edgecolor, facecolor=color,
                          alpha=0.8)
      
        # Применяем поворот
        transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
        brick.set_transform(transform)
      
        return brick
  
    def draw_ground(self, x_min, x_max, ground_level=0.0):
        """Рисование земли на уровне y=0"""
        ground_width = x_max - x_min
        ground = Rectangle((x_min, ground_level - 0.1), ground_width, 0.1,
                          linewidth=1, edgecolor='saddlebrown',
                          facecolor='peru', alpha=0.9, zorder=0)
        self.ax.add_patch(ground)
      
        self.ax.axhline(y=ground_level, color='black', linewidth=1, alpha=0.5, zorder=2)
  
    def draw_contacts(self, contact: Contact):
        """Отрисовка контактов между кирпичами"""
        for contact in contact.contacts:
            x, y = contact.point
          
            if contact.type == 'corner-corner':
                color = 'red'
            elif contact.type == 'corner-edge':
                color = 'blue'
            else:
                color = 'green'
          
            # Рисуем точку контакта с zorder=10
            contact_point = Circle((x, y), radius=0.02, color=color,
                                   alpha=0.8, zorder=10)
            self.ax.add_patch(contact_point)

    def draw_contact_forces(self, contact_analysis: Contact, qp_analysis: Dict, max_force_length: float = 0.2):
        """
        Отрисовка векторов контактных сил Fn и Ft с динамической нормализацией.
        """
        
        # ⚠️ ИСПРАВЛЕНИЕ 1: Обработка объекта QPSolution или Dict
        if isinstance(qp_analysis, dict):
            forces = qp_analysis.get('contact_forces', []) 
        elif hasattr(qp_analysis, 'contact_forces'):
            forces = qp_analysis.contact_forces
        else:
            print("Ошибка: qp_analysis не содержит данных о силах (нужен Dict или объект с атрибутом contact_forces).")
            return
            
        if not forces:
            return

        # 1. Найти максимальную силу (величину)
        max_lambda = 0.0
        for force_data in forces:
            # Используем .get() для безопасности, если структура force_data — словарь
            lambda_N = force_data.get('lambda_N', 0.0)
            lambda_T = force_data.get('lambda_T', 0.0)
            total_force_magnitude = np.sqrt(lambda_N**2 + lambda_T**2)
            max_lambda = max(max_lambda, total_force_magnitude)

        # 2. Рассчитать коэффициент масштабирования
        if max_lambda < 1e-6:
            scale = 0.0
        else:
            scale = max_force_length / max_lambda 
            
        print(f"Динамический масштаб сил: {scale:.4f} (Max Force: {max_lambda:.2f} N)")

        # ⚠️ ИСПРАВЛЕНИЕ 2: Фиксированный, гарантированно видимый размер головы стрелки
        ABSOLUTE_HEAD_WIDTH = 0.04
        ABSOLUTE_HEAD_LENGTH = 0.08

        # 3. Отрисовка с рассчитанным масштабом
        for k, force_data in enumerate(forces):
            if k >= len(contact_analysis.contacts):
                continue
                
            contact = contact_analysis.contacts[k]
            
            x, y = contact.point
            lambda_N = force_data.get('lambda_N', 0.0)
            lambda_T = force_data.get('lambda_T', 0.0)
            
            # Пропускаем нулевые контакты
            if abs(lambda_N) < 1e-6 and abs(lambda_T) < 1e-6:
                continue 

            n_global = np.array(contact.n_global)
            t_global = np.array(get_tangent(contact.n_global))
            
            # --- Нормальная сила (Fn) ---
            Fn_x, Fn_y = lambda_N * n_global * scale
            self.ax.arrow(x, y, Fn_x, Fn_y, 
                          head_width=ABSOLUTE_HEAD_WIDTH, head_length=ABSOLUTE_HEAD_LENGTH, 
                          fc='orange', ec='red', linewidth=1.5, zorder=15, 
                          label='Fn' if k == 0 else None)
            
            # --- Тангенциальная сила (Ft) ---
            Ft_x, Ft_y = lambda_T * t_global * scale
            self.ax.arrow(x, y, Ft_x, Ft_y, 
                          head_width=ABSOLUTE_HEAD_WIDTH, head_length=ABSOLUTE_HEAD_LENGTH, 
                          fc='darkgreen', ec='green', linewidth=1.5, linestyle='--', zorder=15, 
                          label='Ft' if k == 0 else None)

            # Добавляем подпись к точке контакта
            self.ax.text(x + 0.1 * max_force_length, y + 0.1 * max_force_length, 
                         f"N={lambda_N:.2f}\nT={lambda_T:.2f}",
                         fontsize=7, ha='left', va='bottom', color='black',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none', pad=1), 
                         zorder=15)


    def draw_underground_bricks(self, contact: Contact, config: BrickConfig):
        """Выделение кирпичей, которые входят в землю"""
        for brick_id in contact.underground_bricks:
            x, y, angle = config.R_list[brick_id]
            underground_frame = Rectangle((x - config.width/2 - 0.02, y - config.height/2 - 0.02),
                                          config.width + 0.04, config.height + 0.04,
                                          linewidth=3, edgecolor='brown',
                                          facecolor='none', alpha=0.7, zorder=5)
          
            transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
            underground_frame.set_transform(transform)
            self.ax.add_patch(underground_frame)
  
    def draw_overlaps(self, contact: Contact, config: BrickConfig):
        """Выделение кирпичей с перекрытиями"""
        overlapped_bricks = set()
        for brick1, brick2, _ in contact.overlaps:
            overlapped_bricks.add(brick1)
            overlapped_bricks.add(brick2)
      
        for brick_id in overlapped_bricks:
            x, y, angle = config.R_list[brick_id]
            overlap_frame = Rectangle((x - config.width/2 - 0.02, y - config.height/2 - 0.02),
                                      config.width + 0.04, config.height + 0.04,
                                      linewidth=3, edgecolor='red',
                                      facecolor='none', alpha=0.7, zorder=5)
          
            transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
            overlap_frame.set_transform(transform)
            self.ax.add_patch(overlap_frame)
  
    def draw_floating_bricks(self, contact: Contact, config: BrickConfig):
        """Выделение кирпичей, висящих в воздухе"""
        for brick_id in contact.floating_bricks:
            x, y, angle = config.R_list[brick_id]
            floating_frame = Rectangle((x - config.width/2 - 0.02, y - config.height/2 - 0.02),
                                      config.width + 0.04, config.height + 0.04,
                                      linewidth=3, edgecolor='yellow',
                                      facecolor='none', alpha=0.7, zorder=5)
          
            transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
            floating_frame.set_transform(transform)
            self.ax.add_patch(floating_frame)
  
    def calculate_display_bounds(self, R_list, brick_width, brick_height):
        """Вычисление границ отображения с учетом земли и отступов"""
        if not R_list:
            return -1, 1, -1, 1
      
        x_vals = [R[0] for R in R_list]
        y_vals = [R[1] for R in R_list]
      
        x_min = min(x_vals)
        x_max = max(x_vals)
        y_max = max(y_vals)
      
        max_brick_dimension = max(brick_width, brick_height) * 1.2
      
        x_margin = max_brick_dimension * 1.5
        y_margin = max_brick_dimension * 1.5
      
        display_x_min = x_min - x_margin
        display_x_max = x_max + x_margin
      
        ground_level = 0.0
        display_y_max = y_max + y_margin
      
        display_y_min = ground_level - 0.5 
      
        return display_x_min, display_x_max, display_y_min, display_y_max
  
    def visualize_config(self, config: BrickConfig, contact: Contact = None,
                         qp_analysis: Dict = None, 
                         save_path=None, show_grid=True, show_info=True,
                         show_ground=True, title=None, 
                         max_force_length: float = 0.2): 
        """Визуализация конфигурации кирпичей с контактами"""
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
      
        width = config.width
        height = config.height
        R_list = config.R_list
        description = config.description
        config_id = config.id
      
        display_title = title if title else description
      
        x_min, x_max, y_min, y_max = self.calculate_display_bounds(R_list, width, height)
      
        if show_ground:
            self.draw_ground(x_min, x_max)
      
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon',
                  'lightseagreen', 'plum', 'wheat', 'lightsteelblue']
      
        for i in range(len(R_list)):
            x, y, angle = R_list[i]
            color = colors[i % len(colors)]
          
            brick = self.create_brick_patch(x, y, width, height, angle, color)
            self.ax.add_patch(brick)
          
            self.ax.text(x, y, str(i), ha='center', va='center', 
                         fontsize=10, fontweight='bold', color='darkblue', zorder=12) 
      
        if contact:
            self.draw_contacts(contact)
            self.draw_overlaps(contact, config)
            self.draw_floating_bricks(contact, config)
            self.draw_underground_bricks(contact, config)

        if qp_analysis:
            self.draw_contact_forces(contact, qp_analysis, max_force_length)
      
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X координата', fontsize=12)
        self.ax.set_ylabel('Y координата', fontsize=12)
      
        title_text = f'Конфигурация кирпичей: {display_title} (ID: {config_id})'
        if contact and (contact.overlaps or contact.floating_bricks):
            title_text += " ⚠"
        self.ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
      
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
      
        if show_info:
            info_text = f"Количество кирпичей: {len(R_list)}\n"
            info_text += f"Ширина кирпича: {width}\n"
            info_text += f"Высота кирпича: {height}\n"
            info_text += f"Масса: {config.mass}\n"
            info_text += f"Коэф. трения (μ): {config.mu}\n"
            info_text += f"Момент инерции: {config.I:.6f}"
          
            if contact:
                info_text += f"\n\nКонтакты: {len(contact.contacts)}"
                info_text += f"\nПерекрытия: {len(contact.overlaps)}"
                info_text += f"\nВ воздухе: {len(contact.floating_bricks)}"
                info_text += f"\nВ земле: {len(contact.underground_bricks)}"
              
                if contact.warnings:
                    info_text += f"\n\nПредупреждения:"
                    for warning in contact.warnings[:3]: 
                        info_text += f"\n• {warning}"
                    if len(contact.warnings) > 3:
                        info_text += f"\n• ... и ещё {len(contact.warnings) - 3}"
          
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round',
                         facecolor='wheat', alpha=0.8), fontsize=9, zorder=20)
      
        if contact and show_info:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                           markersize=8, label='Угол-угол'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                           markersize=6, label='Угол-ребро'),
                plt.Line2D([0], [0], color='red', marker='>', linestyle='-', linewidth=2, markerfacecolor='orange', label='Нормальная сила (Fn)'),
                plt.Line2D([0], [0], color='green', marker='>', linestyle='--', linewidth=2, markerfacecolor='darkgreen', label='Тангенциальная сила (Ft)'),
                plt.Line2D([0], [0], color='red', linewidth=3, label='Перекрытие'),
                plt.Line2D([0], [0], color='yellow', linewidth=3, label='В воздухе'),
                plt.Line2D([0], [0], color='brown', linewidth=3, label='В земле')
            ]
            
            # ⚠️ ИСПРАВЛЕНИЕ 3: Удаление 'zorder', вызывающего ошибку в старых версиях Matplotlib
            self.ax.legend(handles=legend_elements, loc='upper right',
                           bbox_to_anchor=(0.98, 0.98), fontsize=8) 
      
        if show_grid:
            self.ax.grid(True, alpha=0.3)
      
        plt.tight_layout()
      
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Визуализация сохранена как: {save_path}")
      
        plt.show()