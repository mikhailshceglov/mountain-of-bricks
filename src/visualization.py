import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D
import numpy as np
import argparse
import sys
import os

from typing import Dict, List, Tuple 

from load_config import BrickConfig
from contact_finder import Contact, ContactPoint 

# поворот на 90 градусов
def get_tangent(n: Tuple[float, float]) -> Tuple[float, float]:
    return (-n[1], n[0])

class BrickVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
      
    def create_brick_patch(self, x, y, width, height, angle, color='lightblue', edgecolor='black'):
        # поворот кирпича
        brick = Rectangle((x - width/2, y - height/2), width, height, linewidth=2, edgecolor=edgecolor, facecolor=color, alpha=0.8)
      
        # применяем поворот
        transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
        brick.set_transform(transform)
      
        return brick
  
    def draw_ground(self, x_min, x_max, ground_level=0.0):
        # рисование земли 
        ground_width = x_max - x_min
        ground = Rectangle((x_min, ground_level - 0.1), ground_width, 0.1, linewidth=1, edgecolor='saddlebrown', facecolor='peru', alpha=0.9, zorder=0)
        self.ax.add_patch(ground)
        
        self.ax.fill_between([x_min, x_max], ground_level, ground_level - 0.5,  color='peru', alpha=0.5, zorder=0)
        
        self.ax.axhline(y=ground_level, color='black', linewidth=1, alpha=0.5, zorder=2)
  
    def draw_contacts(self, contact: Contact):
        # отрисовка контактов между кирпичами
        for contact_point in contact.contacts:
            x, y = contact_point.point
          
            if contact_point.type == 'corner-corner':
                color = 'red'
                radius = 0.015
            elif contact_point.type == 'corner-edge':
                color = 'blue'
                radius = 0.015
            else: # контакт с землей
                color = 'green'
                radius = 0.015
          
            # рисуем точку контакта 
            contact_circle = Circle((x, y), radius=radius, color=color, alpha=0.8, zorder=10)
            self.ax.add_patch(contact_circle)

    def draw_gravity_forces(self, config: 'BrickConfig', qp_analysis: Dict, max_force_length: float = 0.2):
        # отрисовка силы тяжести для кирпича
        if not config.R_list:
            return

        # находим максимальную силу для масштабирования
        if isinstance(qp_analysis, dict):
            forces = qp_analysis.get('contact_forces', []) 
        elif hasattr(qp_analysis, 'contact_forces'):
            forces = qp_analysis.contact_forces
        else:
            return
              
        if not forces:
            return

        # находим максимальную силу
        max_lambda = 0.0
        for force_data in forces:
            lambda_N = force_data.get('lambda_N', 0.0)
            lambda_T = force_data.get('lambda_T', 0.0)
            total_force_magnitude = np.sqrt(lambda_N**2 + lambda_T**2)
            max_lambda = max(max_lambda, total_force_magnitude)

        # расчет коэффициента масштабирования
        if max_lambda < 1e-6:
            scale = 0.0
        else:
            scale = max_force_length / max_lambda 

        # используем общий масштаб
        #scale = max_force_length / max_weight
        
        ABSOLUTE_HEAD_WIDTH = 0.04
        ABSOLUTE_HEAD_LENGTH = 0.08

        for i, r in enumerate(config.R_list):
            x, y = r[0], r[1] 

            g_force = -config.mass * config.g
            
            # длина вектора в масштабе
            dy = g_force * scale 

            self.ax.arrow(x, y, 0, dy, head_width=ABSOLUTE_HEAD_WIDTH, head_length=ABSOLUTE_HEAD_LENGTH, fc='purple', ec='darkviolet', linewidth=3, zorder=10, label='Сила тяжести (G)' if i == 0 else None)


    def draw_contact_forces(self, contact_analysis: Contact, qp_analysis: Dict, max_force_length: float = 0.2):
        # отрисовка контактных сил
        if isinstance(qp_analysis, dict):
            forces = qp_analysis.get('contact_forces', []) 
        elif hasattr(qp_analysis, 'contact_forces'):
            forces = qp_analysis.contact_forces
        else:
            print("ошибка: qp_analysis не содержит данных о силах.")
            return
              
        if not forces:
            return

        # находим максимальную силу
        max_lambda = 0.0
        for force_data in forces:
            lambda_N = force_data.get('lambda_N', 0.0)
            lambda_T = force_data.get('lambda_T', 0.0)
            total_force_magnitude = np.sqrt(lambda_N**2 + lambda_T**2)
            max_lambda = max(max_lambda, total_force_magnitude)

        # расчет коэффициента масштабирования
        if max_lambda < 1e-6:
            scale = 0.0
        else:
            scale = max_force_length / max_lambda 
              
        print(f"масштаб сил: {scale:.4f} (максимальная сила: {max_lambda:.2f} N)")

        ABSOLUTE_HEAD_WIDTH = 0.04
        ABSOLUTE_HEAD_LENGTH = 0.08

        # отрисовка с рассчитанным масштабом
        for k, force_data in enumerate(forces):
            if k >= len(contact_analysis.contacts):
                continue
                  
            contact = contact_analysis.contacts[k]
              
            x, y = contact.point
            lambda_N = force_data.get('lambda_N', 0.0)
            lambda_T = force_data.get('lambda_T', 0.0)
              
            # пропускаем нулевые контакты
            if abs(lambda_N) < 1e-6 and abs(lambda_T) < 1e-6:
                continue 

            n_global = np.array(contact.n_global)
            t_global = np.array(get_tangent(contact.n_global))
              
            
            Fn_x, Fn_y = lambda_N * n_global * scale
            Ft_x, Ft_y = lambda_T * t_global * scale
              
            # сила реакции опоры
            self.ax.arrow(x, y, Fn_x, Fn_y, head_width=ABSOLUTE_HEAD_WIDTH, head_length=ABSOLUTE_HEAD_LENGTH, fc='orange', ec='red', linewidth=1.5, zorder=15, label='Fn (на Br1)' if k == 0 else None)
              
            # сила трения
            self.ax.arrow(x, y, Ft_x, Ft_y, head_width=ABSOLUTE_HEAD_WIDTH, head_length=ABSOLUTE_HEAD_LENGTH, fc='darkgreen', ec='green', linewidth=1.5, linestyle='--', zorder=15, label='Ft (на Br1)' if k == 0 else None)

    def draw_underground_bricks(self, contact: Contact, config: BrickConfig):
        # выделение кирпичей, которые входят в землю
        for brick_id in contact.underground_bricks:
            x, y, angle = config.R_list[brick_id]
            underground_frame = Rectangle((x - config.width/2 - 0.02, y - config.height/2 - 0.02), config.width + 0.04, config.height + 0.04, linewidth=3, edgecolor='brown', facecolor='none', alpha=0.7, zorder=5)
          
            transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
            underground_frame.set_transform(transform)
            self.ax.add_patch(underground_frame)
  
    def draw_overlaps(self, contact: Contact, config: BrickConfig):
        # выделение кирпичей с перекрытиями
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
  
    def draw_flying_bricks(self, contact: Contact, config: BrickConfig):
        # выделение кирпичей, висящих в воздухе
        for brick_id in contact.flying_bricks:
            x, y, angle = config.R_list[brick_id]
            flying_frame = Rectangle((x - config.width/2 - 0.02, y - config.height/2 - 0.02), config.width + 0.04, config.height + 0.04, linewidth=3, edgecolor='yellow', facecolor='none', alpha=0.7, zorder=5)
          
            transform = Affine2D().rotate_around(x, y, angle) + self.ax.transData
            flying_frame.set_transform(transform)
            self.ax.add_patch(flying_frame)
  
    def calculate_display_bounds(self, R_list, brick_width, brick_height):
        # вычисление границ отображения 
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
  
    def visualize_system(self, config: BrickConfig, contact: Contact = None,
                         qp_analysis: Dict = None, 
                         save_path=None, show_grid=True, show_info=True,
                         show_ground=True, title=None, 
                         max_force_length: float = 0.2): 
        # визуализация конфигурации кирпичей
          
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
      
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightseagreen', 'plum', 'wheat', 'lightsteelblue']
      
        for i in range(len(R_list)):
            x, y, angle = R_list[i]
            color = colors[i % len(colors)]
          
            brick = self.create_brick_patch(x, y, width, height, angle, color)
            self.ax.add_patch(brick)
          
            self.ax.text(x, y, str(i), ha='center', va='center', 
                          fontsize=10, fontweight='bold', color='darkblue', zorder=12) 
        
        if config.g != 0:
            self.draw_gravity_forces(config, qp_analysis, max_force_length)
      
        if contact:
            self.draw_contacts(contact)
            self.draw_overlaps(contact, config)
            self.draw_flying_bricks(contact, config)
            self.draw_underground_bricks(contact, config)

        if qp_analysis:
            self.draw_contact_forces(contact, qp_analysis, max_force_length)
      
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x координата', fontsize=12)
        self.ax.set_ylabel('y координата', fontsize=12)
      
        title_text = f'конфигурация кирпичей: {display_title} (ID: {config_id})'
        if contact and (contact.overlaps or contact.flying_bricks):
            title_text += " !есть варнинги!"
        self.ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
      
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
      
        if show_info:
            info_text = f"количество кирпичей: {len(R_list)}\n"
            info_text += f"ширина кирпича: {width}\n"
            info_text += f"высота кирпича: {height}\n"
            info_text += f"масса: {config.mass}\n"
            info_text += f"коэф. трения (μ): {config.mu}\n"
          
            if contact:
                info_text += f"\n\nконтакты: {len(contact.contacts)}"
                info_text += f"\nперекрытия: {len(contact.overlaps)}"
                info_text += f"\nв воздухе: {len(contact.flying_bricks)}"
                info_text += f"\nв земле: {len(contact.underground_bricks)}"
              
                if contact.warnings:
                    info_text += f"\n\nварнинги:"
                    for warning in contact.warnings[:3]: 
                        info_text += f"\n {warning}"
              
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9, zorder=20)
      
        if contact and show_info:
            legend_elements = [
                # типы контактов
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='угол-угол'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label='угол-ребро'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=6, label='ребро-ребро/земля'),
                           
                # cилы из солвера
                plt.Line2D([0], [0], color='red', marker='>', linestyle='-', linewidth=2, markerfacecolor='orange', label='Fn'),
                plt.Line2D([0], [0], color='green', marker='>', linestyle='--', linewidth=2, markerfacecolor='darkgreen', label='Ft'),
                
                # cила тяжести
                plt.Line2D([0], [0], color='darkviolet', marker='>', linestyle='--', linewidth=2, markerfacecolor='violet', alpha=0.5, label='сила тяжести'),
                
                # статусы 
                plt.Line2D([0], [0], color='red', linewidth=3, label='перекрытие'),
                plt.Line2D([0], [0], color='yellow', linewidth=3, label='в воздухе'),
                plt.Line2D([0], [0], color='brown', linewidth=3, label='в земле')
            ]
          
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8) 
      
        if show_grid:
            self.ax.grid(True, alpha=0.3)
      
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
      
        plt.show()