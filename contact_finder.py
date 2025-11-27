import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from load_config import BrickConfig

@dataclass
class ContactPoint:
    #точка контакта
    brick1_id: int
    brick2_id: int  # -1 для земли
    point: Tuple[float, float]  #(x, y)
    n_global: Tuple[float, float] #(normal_x, normal_y)
    type: str  #corner-corner/corner-edge/edge-edge/ground/floating
    distance: float

@dataclass
class Contact:
    #класс контактов
    contacts: List[ContactPoint]
    overlaps: List[Tuple[int, int, float]]  #brick1_id, brick2_id, overlap_area
    floating_bricks: List[int]  #id кирпичей без контактов
    underground_bricks: List[int]  #id кирпичей, которые входят в землю
    warnings: List[str]

class ContactAnalyzer:
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance

    def point_on_segment(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        #проверка, что точка лежит на отрезке
        #расстояние от точки до прямой
        line_dist = abs((b[1]-a[1])*c[0] - (b[0]-a[0])*c[1] + b[0]*a[1] - b[1]*a[0])
        #норма отрезка
        norm_sq = (b[1]-a[1])**2 + (b[0]-a[0])**2
        if norm_sq > 1e-12:
            line_dist /= np.sqrt(norm_sq)
        else:
            #если отрезок - это точка, считаем dist до нее
            line_dist = np.sqrt((c[0]-a[0])**2 + (c[1]-a[1])**2) # Если отрезок - это точка, считаем dist до нее
        
        #проверка что проекция точки находится между концами отрезка
        dot1 = (c[0]-a[0])*(b[0]-a[0]) + (c[1]-a[1])*(b[1]-a[1])
        dot2 = (c[0]-b[0])*(a[0]-b[0]) + (c[1]-b[1])*(a[1]-b[1])
        
        return (line_dist <= self.tolerance and dot1 >= -self.tolerance and dot2 >= -self.tolerance)
    
    def check_overlap(self, corners1: List[Tuple[float, float]], corners2: List[Tuple[float, float]]) -> float:
        #проверка перекрытия кирпичей        
        def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
            #проверка что точка внутри многоугольника=
            x, y = point
            inside = False
            n = len(polygon)
            
            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]
                
                #проверяем точки на границе
                if self.point_on_segment((x1, y1), (x2, y2), (x, y)):
                    return False
                
                #метод лучей
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
                    
            return inside
            
        #проверяем перекрытие
        for point in corners1:
            if point_in_polygon(point, corners2):
                return 1.0  #есть перекрытие
        
        for point in corners2:
            if point_in_polygon(point, corners1):
                return 1.0  #есть перекрытие
        
        return 0.0  #нет перекрытия
        
    def get_brick_corners(self, x: float, y: float, width: float, height: float, angle: float) -> List[Tuple[float, float]]:
        #углы кирпича с учетом поворота
        half_w = width / 2
        half_h = height / 2
        corners_local = [
            (-half_w, -half_h),  #нижний левый
            ( half_w, -half_h),  #нижний правый
            ( half_w,  half_h),  #верхний правый
            (-half_w,  half_h)   #верхний левый
        ]
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        corners_global = []
        for cx, cy in corners_local:
            #поворот
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            #смещение
            corners_global.append((x + rx, y + ry))
        
        return corners_global
    
    def get_brick_edges(self, corners: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        #ребра кирпичей
        edges = []
        for i in range(len(corners)):
            edges.append((corners[i], corners[(i + 1) % len(corners)]))
        return edges
    
    def point_to_line_distance(self, point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        #расстояние от точки до отрезка
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        length_sq = dx * dx + dy * dy
        
        #точка вместо отрезка
        if length_sq == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        #часть отрезка отначала до точки прекции
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        
        #координаты проекции
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        #расстояние до отрезка
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def check_underground_bricks(self, brick_corners: List[List[Tuple[float, float]]]) -> List[int]:
        #проверка, что кирпичи выше земли
        underground_bricks = []
        ground_level = 0.0
        
        for i, corners in enumerate(brick_corners):
            min_y = min(corner[1] for corner in corners)
            
            if min_y < ground_level - self.tolerance:
                underground_bricks.append(i)
        
        return underground_bricks
    
    def find_ground_contacts(self, brick_corners: List[List[Tuple[float, float]]], brick_edges: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]]) -> List[ContactPoint]:
        #кирпичи на земле
        ground_contacts = []
        ground_level = 0.0
        
        #нормаль для земли всегда
        N_GLOBAL_GROUND = (0.0, 1.0) 

        for i, corners in enumerate(brick_corners):
            for corner in corners:
                distance = abs(corner[1] - ground_level)
                if distance <= self.tolerance:
                    contact = ContactPoint(
                        brick1_id=i,
                        brick2_id=-1,
                        point=corner,
                        n_global=N_GLOBAL_GROUND,
                        type='ground',
                        distance=distance
                    )
                    ground_contacts.append(contact)
        
        return ground_contacts
    
    def find_contacts(self, config: BrickConfig) -> Contact:
        #поиск контактов
        contacts = []
        overlaps = []
        floating_bricks = set(range(len(config.R_list)))
        warnings = []
        
        brick_corners = []
        brick_edges = []
        
        #вычисляем углы и ребра
        for i, (x, y, angle) in enumerate(config.R_list):
            corners = self.get_brick_corners(x, y, config.width, config.height, angle)
            brick_corners.append(corners)
            edges = self.get_brick_edges(corners)
            brick_edges.append(edges)
        
        #проверяем, входит ли кирпич в землю
        underground_bricks = self.check_underground_bricks(brick_corners)
        for brick_id in underground_bricks:
            warnings.append(f"кирпич{brick_id} входит в землю")
        
        #находим контакты с землей
        ground_contacts = self.find_ground_contacts(brick_corners, brick_edges)
        contacts.extend(ground_contacts)
        
        #обновляем множество кирпичей в воздухе
        for contact in ground_contacts:
            floating_bricks.discard(contact.brick1_id)
        
        #проверяем, летит ли кирпич в воздухе
        for i in range(len(config.R_list)):
            for j in range(i + 1, len(config.R_list)):
                brick1_corners = brick_corners[i]
                brick2_corners = brick_corners[j]
                brick1_edges = brick_edges[i]
                brick2_edges = brick_edges[j]
                
                #дефолтная нормаль для угол-угол
                N_GLOBAL_DEFAULT = (0.0, 1.0) 

                #проверяем контакты угол-угол
                for corner1 in brick1_corners:
                    for corner2 in brick2_corners:
                        distance = np.sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)
                        if distance <= self.tolerance:
                            contact = ContactPoint(
                                brick1_id=i,
                                brick2_id=j,
                                point=((corner1[0] + corner2[0]) / 2, 
                                       (corner1[1] + corner2[1]) / 2),
                                n_global=N_GLOBAL_DEFAULT,
                                type='corner-corner',
                                distance=distance
                            )
                            contacts.append(contact)
                            floating_bricks.discard(i)
                            floating_bricks.discard(j)
                            
                #проверяем контакты угол-ребро
                #угол кирпича i на ребре кирпича j (нормаль от b2 к b1)
                for corner in brick1_corners:
                    for edge in brick2_edges:
                        distance = self.point_to_line_distance(corner, edge[0], edge[1])
                        if distance <= self.tolerance:
                            
                            #нормаль, перпендикулярная ребру b2
                            e_x = edge[1][0] - edge[0][0]
                            e_y = edge[1][1] - edge[0][1]
                            
                            n_x, n_y = e_y, -e_x
                            
                            #нормализация
                            norm = np.sqrt(n_x**2 + n_y**2)
                            if norm > 1e-10:
                                n_x /= norm
                                n_y /= norm
                            
                            n_global_ij = (n_x, n_y)

                            contact = ContactPoint(
                                brick1_id=i,
                                brick2_id=j,
                                point=corner,
                                n_global=n_global_ij, 
                                type='corner-edge',
                                distance=distance
                            )
                            contacts.append(contact)
                            floating_bricks.discard(i)
                            floating_bricks.discard(j)
                            
                #угол кирпича j на ребре кирпича i (нормаль от b1 к b2)
                for corner in brick2_corners:
                    for edge in brick1_edges:
                        distance = self.point_to_line_distance(corner, edge[0], edge[1])
                        if distance <= self.tolerance:
                            
                            #расчет нормали, перпендикулярной ребру b1
                            e_x = edge[1][0] - edge[0][0]
                            e_y = edge[1][1] - edge[0][1]
                            
                            n_x, n_y = e_y, -e_x
                            
                            #нормализация
                            norm = np.sqrt(n_x**2 + n_y**2)
                            if norm > 1e-10:
                                n_x /= norm
                                n_y /= norm
                            
                            n_global_ji = (n_x, n_y)

                            contact = ContactPoint(
                                brick1_id=i,
                                brick2_id=j,
                                point=corner,
                                n_global=n_global_ji,
                                type='corner-edge',
                                distance=distance
                            )
                            contacts.append(contact)
                            floating_bricks.discard(i)
                            floating_bricks.discard(j)
                            
                #проверяем перекрытия
                overlap_detected = self.check_overlap(brick1_corners, brick2_corners)
                if overlap_detected > 0:
                    overlaps.append((i, j, overlap_detected))
                    warnings.append(f"Перекрытие между кирпичами {i} и {j}")
        
        #проверяем кирпичи, которые висят в воздухе
        floating_list = list(floating_bricks)
        for brick_id in floating_list:
            warnings.append(f"Кирпич {brick_id} висит в воздухе")
        
        return Contact(
            contacts=contacts,
            overlaps=overlaps,
            floating_bricks=floating_list,
            underground_bricks=underground_bricks,
            warnings=warnings
        )
    
    def print_analysis(self, analysis: Contact):
        #вывод контактов        
        ground_contacts = [c for c in analysis.contacts if c.brick2_id == -1]
        brick_contacts = [c for c in analysis.contacts if c.brick2_id != -1]
        
        print(f"\nконтакты с землей: {len(ground_contacts)}")
        for contact in ground_contacts:
            print(f" кирпич {contact.brick1_id} - земля: {contact.type} в ({contact.point[0]:.3f}, {contact.point[1]:.3f}) [N={contact.n_global}]")
        
        print(f"\nконтакты между кирпичами: {len(brick_contacts)}")
        for contact in brick_contacts:
            print(f"  кирпич {contact.brick1_id} - кирпич {contact.brick2_id}: "
                  f"{contact.type} в ({contact.point[0]:.3f}, {contact.point[1]:.3f}) [N={contact.n_global}]")
        
        print(f"\nперекрытия: {len(analysis.overlaps)}")
        for brick1, brick2, area in analysis.overlaps:
            print(f" кирпич{brick1} - кирпич{brick2}: площадь~{area:.6f}")
        
        print(f"\nrирпичи в воздухе: {len(analysis.floating_bricks)}")
        if analysis.floating_bricks:
            print(f" id:{analysis.floating_bricks}")
        
        print(f"\nrирпичи в земле: {len(analysis.underground_bricks)}")
        if analysis.underground_bricks:
            print(f" id:{analysis.underground_bricks}")
        
        print(f"\nпредупреждения: {len(analysis.warnings)}")
        for warning in analysis.warnings:
            print(f" {warning}")

def analyze_contacts(config: BrickConfig, tolerance=1e-6) -> Contact:
    analyzer = ContactAnalyzer(tolerance)
    return analyzer.find_contacts(config)