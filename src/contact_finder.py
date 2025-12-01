import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from load_config import BrickConfig 

@dataclass
class ContactPoint:
    # точка контакта
    brick1_id: int
    brick2_id: int  # -1 для земли
    point: Tuple[float, float]  # (x, y)
    n_global: Tuple[float, float]  # вектор силы реакции опоры
    t_global: Tuple[float, float]  # верктор силы трения
    type: str  # тип контакта: corner-corner/corner-edge/edge-edge/ground/flying
    distance: float

@dataclass
class Contact:
    # класс контактов
    contacts: List[ContactPoint]
    overlaps: List[Tuple[int, int, float]]  # brick1_id, brick2_id, площадб пересечения
    flying_bricks: List[int]  # id кирпичей без контактов ("летающих")
    underground_bricks: List[int]  # id кирпичей, которые входят в землю
    warnings: List[str]

class ContactAnalyzer:
    def __init__(self, config: BrickConfig, tolerance=1e-6):
        self.tolerance = tolerance
        self.config = config
        self.brick_centers = [(r[0], r[1]) for r in config.R_list]
        self.ground_level = 0.0

    def _compute_normal_and_tangent(self, edge_start: Tuple[float, float], edge_end: Tuple[float, float], point_on_brick1: Tuple[float, float], brick1_id: int, brick2_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # вычисление сил в контактах
        # вектор ребра
        e_x = edge_end[0] - edge_start[0]
        e_y = edge_end[1] - edge_start[1]
        
        # нормали (перпендикулярно ребру)
        n_a = (e_y, -e_x)
        n_b = (-e_y, e_x)
        
        norm = np.sqrt(e_x**2 + e_y**2)
        
        if norm < 1e-10:
            # ребро - это точка
            return (0.0, 0.0), (0.0, 0.0)

        # нормализация
        n_a = (n_a[0] / norm, n_a[1] / norm)
        n_b = (n_b[0] / norm, n_b[1] / norm)
        
        # центр масс brick1
        if brick1_id != -1:
            cm1 = self.brick_centers[brick1_id]
        else:
            # кирпич - это земля 
            return (0.0, 0.0), (0.0, 0.0) 

        # нормаль, направленную в сторону ыерхнего кирпича
        vec_to_cm1 = (cm1[0] - point_on_brick1[0], cm1[1] - point_on_brick1[1])
        
        # выбираем нормаль, направленную к верхнему кирпичу
        dot_a = vec_to_cm1[0] * n_a[0] + vec_to_cm1[1] * n_a[1]
        dot_b = vec_to_cm1[0] * n_b[0] + vec_to_cm1[1] * n_b[1]

        if dot_a > dot_b:
            n_target = n_a
        else:
            n_target = n_b
            
        n_global = n_target
        
        # сила трения (поворот на 90 градусов)
        t_global = (n_global[1], -n_global[0]) 

        return n_global, t_global


    def point_on_segment(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        #проверка, что точка лежит на отрезке
        #расстояние от точки до прямой
        line_dist = abs((b[1]-a[1])*c[0] - (b[0]-a[0])*c[1] + b[0]*a[1] - b[1]*a[0])
        # длина отрезка
        norm_sq = (b[1]-a[1])**2 + (b[0]-a[0])**2
        if norm_sq > 1e-12:
            line_dist /= np.sqrt(norm_sq)
        else:
            # если отрезок - это точка, считаем dist до нее
            line_dist = np.sqrt((c[0]-a[0])**2 + (c[1]-a[1])**2)

        # проверка что проекция точки находится между концами отрезка
        dot1 = (c[0]-a[0])*(b[0]-a[0]) + (c[1]-a[1])*(b[1]-a[1])
        dot2 = (c[0]-b[0])*(a[0]-b[0]) + (c[1]-b[1])*(a[1]-b[1])
        
        return (line_dist <= self.tolerance and dot1 >= -self.tolerance and dot2 >= -self.tolerance)
    
    def check_overlap(self, corners1: List[Tuple[float, float]], corners2: List[Tuple[float, float]]) -> float:
        # проверка перекрытия кирпичей    
        def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
            # проверка что точка внутри многоугольника=
            x, y = point
            inside = False
            n = len(polygon)
            
            #mпроверяем точки на границе
            for i in range(n):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % n]
                
                if self.point_on_segment((x1, y1), (x2, y2), (x, y)):
                    return False
                # метод лучей
                if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                    inside = not inside
                    
            return inside
        
        # проверка что один из углов внутри  
        for point in corners1:
            if point_in_polygon(point, corners2):
                return 1.0
        
        for point in corners2:
            if point_in_polygon(point, corners1):
                return 1.0
        
        return 0.0
        
    def get_brick_corners(self, x: float, y: float, width: float, height: float, angle: float) -> List[Tuple[float, float]]:
        # углы кирпича с учетом поворота
        half_w = width / 2
        half_h = height / 2
        corners_local = [
            (-half_w, -half_h),
            ( half_w, -half_h),
            ( half_w,  half_h),
            (-half_w,  half_h)
        ]
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        corners_global = []
        for cx, cy in corners_local:
            # поворот
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            # смещение
            corners_global.append((x + rx, y + ry))
        
        return corners_global
    
    def get_brick_edges(self, corners: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        # ребра кирпичей
        edges = []
        for i in range(len(corners)):
            edges.append((corners[i], corners[(i + 1) % len(corners)]))
        return edges
    
    def point_to_line_distance(self, point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        # расстояние от точки до отрезка
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        dx = x2 - x1
        dy = y2 - y1
        
        length_sq = dx * dx + dy * dy
        
        # если точка вместо отрезка
        if length_sq == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        # часть отрезка отначала до точки прекции 
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        
        # координаты проекции
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # расстояние до отрезка
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def check_underground_bricks(self, brick_corners: List[List[Tuple[float, float]]]) -> List[int]:
        # проверка, что кирпичи выше земли
        underground_bricks = []
        
        for i, corners in enumerate(brick_corners):
            min_y = min(corner[1] for corner in corners)
            
            if min_y < self.ground_level - self.tolerance:
                underground_bricks.append(i)
        
        return underground_bricks
    
    def find_ground_contacts(self, brick_corners: List[List[Tuple[float, float]]], brick_edges: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]]) -> List[ContactPoint]:
        # контакты углов с землей 
        ground_contacts = []
        
        N_GLOBAL_GROUND = (0.0, 1.0)  # реакция от опоры от земли
        T_GLOBAL_GROUND = (-1.0, 0.0) # сила трения с землей

        for i, corners in enumerate(brick_corners):
            for corner in corners:
                distance = corner[1] - self.ground_level
                
                #  точка находится на уровне земли (или отличается на епсилон)
                if abs(distance) <= self.tolerance and distance <= self.tolerance:
                    contact = ContactPoint(
                        brick1_id=i,
                        brick2_id=-1,
                        point=corner,
                        n_global=N_GLOBAL_GROUND,
                        t_global=T_GLOBAL_GROUND, # Добавляем тангенс
                        type='ground',
                        distance=abs(distance)
                    )
                    ground_contacts.append(contact)
        
        return ground_contacts
    
    def find_contacts(self, config: BrickConfig) -> Contact:
        # поиск контактов непосредственно
        contacts = []
        overlaps = []
        flying_bricks = set(range(len(config.R_list)))
        warnings = []
        
        brick_corners = []
        brick_edges = []
        
        # находим углы и ребра
        for i, (x, y, angle) in enumerate(config.R_list):
            corners = self.get_brick_corners(x, y, config.width, config.height, angle)
            brick_corners.append(corners)
            edges = self.get_brick_edges(corners)
            brick_edges.append(edges)
        
        # центры масс
        self.brick_centers = [(r[0], r[1]) for r in config.R_list]
        
        # входит ли кирпич в землю
        underground_bricks = self.check_underground_bricks(brick_corners)
        for brick_id in underground_bricks:
            warnings.append(f"Кирпич {brick_id} входит в землю")
        
        # находим контакты с землей
        ground_contacts = self.find_ground_contacts(brick_corners, brick_edges)
        contacts.extend(ground_contacts)
        
        # обновляем множество кирпичей в воздухе
        for contact in ground_contacts:
            flying_bricks.discard(contact.brick1_id)
        
        # проверяем контакты между кирпичами
        for j in range(len(config.R_list)):
            for i in range(j + 1, len(config.R_list)):
                brick1_corners = brick_corners[i]
                brick2_corners = brick_corners[j]
                brick1_edges = brick_edges[i]
                brick2_edges = brick_edges[j]
                r1_x = self.brick_centers[i][0]
                r2_x = self.brick_centers[j][0]
                
                # угол-угол
                for corner1 in brick1_corners:
                    for corner2 in brick2_corners:
                        distance = np.sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)
                        if distance <= self.tolerance:
                            
                            contact_point = ((corner1[0] + corner2[0]) / 2, (corner1[1] + corner2[1]) / 2)
                            p_x, p_y = contact_point
                    
                            # dx1: направление от контакта к центру масс кирпичей
                            dx1 = r1_x - p_x
                            dx2 = r2_x - p_x

                            if dx1 * dx2 < 0: 
                                continue
                            
                            # вектор от угла2 к углу1
                            v_x = corner1[0] - corner2[0]
                            v_y = corner1[1] - corner2[1]
                            v_norm = np.sqrt(v_x**2 + v_y**2)
                            
                            if v_norm > 1e-10:
                                # теакция опоры (от кирпич2 к кирпич1)
                                n_x, n_y = v_x / v_norm, v_y / v_norm
                                n_global_ij = (n_x, n_y)
                                # трение между кирпичами
                                t_global_ij = (n_y, -n_x)
                            else:
                                # если углы совпадают
                                n_global_ij = (0.0, 1.0) 
                                t_global_ij = (-1.0, 0.0) 

                            contact = ContactPoint(
                                brick1_id=i,
                                brick2_id=j,
                                point=contact_point,
                                n_global=n_global_ij,
                                t_global=t_global_ij, # Добавляем тангенс
                                type='corner-corner',
                                distance=distance
                            )
                            contacts.append(contact)
                            flying_bricks.discard(i)
                            flying_bricks.discard(j)
                            
                # угол-ребро                
                # угол кирпича j на ребре кирпича i 
                for corner in brick1_corners:
                    for edge in brick2_edges:
                        distance = self.point_to_line_distance(corner, edge[0], edge[1])
                        if distance <= self.tolerance:
                            
                            # вычмисляем векторы сил
                            n_global_ij, t_global_ij = self._compute_normal_and_tangent(edge[0], edge[1], corner, i, j)

                            contact = ContactPoint(
                                brick1_id=i,
                                brick2_id=j,
                                point=corner, 
                                n_global=n_global_ij,
                                t_global=t_global_ij, 
                                type='corner-edge',
                                distance=distance
                            )
                            contacts.append(contact)
                            flying_bricks.discard(i)
                            flying_bricks.discard(j)
                            
                # Угол кирпича i на ребре кирпича j 
                for corner in brick2_corners:
                    for edge in brick1_edges:
                        distance = self.point_to_line_distance(corner, edge[0], edge[1])
                        if distance <= self.tolerance:                            
                            # вычисляем векторы сил
                            n_global_ji, t_global_ji = self._compute_normal_and_tangent(edge[0], edge[1], corner, j, i)
                            
                            # инвентируем верктор силы
                            n_global_ij = (-n_global_ji[0], -n_global_ji[1])
                            t_global_ij = (-t_global_ji[0], -t_global_ji[1])

                            contact = ContactPoint(
                                brick1_id=i,
                                brick2_id=j,
                                point=corner, 
                                n_global=n_global_ij,
                                t_global=t_global_ij,
                                type='corner-edge',
                                distance=distance
                            )
                            contacts.append(contact)
                            flying_bricks.discard(i)
                            flying_bricks.discard(j)
                            
                # проверяем перекрытия
                overlap_detected = self.check_overlap(brick1_corners, brick2_corners)
                if overlap_detected > 0:
                    overlaps.append((i, j, overlap_detected))
                    warnings.append(f"Перекрытие между кирпичами {i} и {j}")
        
        # проверяем кирпичи, которые висят в воздухе
        flying_list = list(flying_bricks)
        for brick_id in flying_list:
            warnings.append(f"Кирпич {brick_id} висит в воздухе")
        
        return Contact(
            contacts=contacts,
            overlaps=overlaps,
            flying_bricks=flying_list,
            underground_bricks=underground_bricks,
            warnings=warnings
        )
    
    def print_analysis(self, analysis: Contact):
        ground_contacts = [c for c in analysis.contacts if c.brick2_id == -1]
        brick_contacts = [c for c in analysis.contacts if c.brick2_id != -1]
        
        print(f"\nконтакты с землей: {len(ground_contacts)}")
        for contact in ground_contacts:
            print(f" кирпич {contact.brick1_id} - земля: {contact.type} в ({contact.point[0]:.3f}, {contact.point[1]:.3f}) [N={contact.n_global}, T={contact.t_global}]")
        
        print(f"\nконтакты между кирпичами: {len(brick_contacts)}")
        for contact in brick_contacts:
            print(f"  кирпич {contact.brick1_id} - кирпич {contact.brick2_id}: "
                  f"{contact.type} в ({contact.point[0]:.3f}, {contact.point[1]:.3f}) [N={contact.n_global}, T={contact.t_global}]")
        
        print(f"\nперекрытия: {len(analysis.overlaps)}")
        for brick1, brick2, area in analysis.overlaps:
            print(f" кирпич{brick1} - кирпич{brick2}: площадь~{area:.6f}")
        
        print(f"\nкирпичи в воздухе: {len(analysis.flying_bricks)}")
        if analysis.flying_bricks:
            print(f" id:{analysis.flying_bricks}")
        
        print(f"\nК=кирпичи в земле: {len(analysis.underground_bricks)}")
        if analysis.underground_bricks:
            print(f" id:{analysis.underground_bricks}")
        
        print(f"\nварнинги: {len(analysis.warnings)}")
        for warning in analysis.warnings:
            print(f" {warning}")

def analyze_contacts(config: BrickConfig, tolerance=1e-6) -> Contact:
    analyzer = ContactAnalyzer(config, tolerance)
    return analyzer.find_contacts(config)