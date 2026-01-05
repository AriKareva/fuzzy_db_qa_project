from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np


class Defuzzificator:
    def __init__(self, summarizer):
        self.quality_term_data = summarizer.fuzzificator.parser.quality

    def _find_funcs_intersections(self, tolerance=1e-6):
        quality_funcs = {}
        for term_name, fuzzy_set in self.quality_term_data.items():
            if fuzzy_set.x_range is not None and fuzzy_set.y_range is not None:
                quality_funcs[term_name] = {
                    'x': fuzzy_set.x_range,
                    'y': fuzzy_set.y_range
                }
        
        if len(quality_funcs) < 2:
            print(f"Недостаточно функций для поиска пересечений: {len(quality_funcs)}")
            return []
        
        # ШАГ 2: Интерполируем все функции
        interpolated_funcs = {}
        for term_name, data in quality_funcs.items():
            interpolated_funcs[term_name] = interpolate.interp1d(
                data['x'], data['y'], kind='linear', 
                bounds_error=False, fill_value=0
            )
        
        # ШАГ 3: Ищем пересечения между всеми парами функций
        intersections = []
        func_names = list(interpolated_funcs.keys())
        
        for i in range(len(func_names)):
            for j in range(i + 1, len(func_names)):
                name1 = func_names[i]
                name2 = func_names[j]
                f1 = interpolated_funcs[name1]
                f2 = interpolated_funcs[name2]
                
                # Определяем диапазон поиска
                x1_min, x1_max = quality_funcs[name1]['x'].min(), quality_funcs[name1]['x'].max()
                x2_min, x2_max = quality_funcs[name2]['x'].min(), quality_funcs[name2]['x'].max()
                
                x_min = max(x1_min, x2_min)
                x_max = min(x1_max, x2_max)
                
                if x_max <= x_min:
                    continue  # Нет пересечения по диапазону
                
                # Разность функций
                def difference(x):
                    return f1(x) - f2(x)
                
                # Грубый поиск: ищем смены знака
                x_coarse = np.linspace(x_min, x_max, 100)
                y_diff = [difference(x) for x in x_coarse]
                
                # Для каждой смены знака применяем бинарный поиск
                for k in range(len(y_diff) - 1):
                    if y_diff[k] * y_diff[k+1] < 0:  # Смена знака
                        
                        # Бинарный поиск
                        a = x_coarse[k]
                        b = x_coarse[k+1]
                        
                        while abs(b - a) > tolerance:
                            mid = (a + b) / 2
                            
                            if difference(a) * difference(mid) < 0:
                                b = mid
                            else:
                                a = mid
                        
                        x_intersect = float((a + b) / 2)
                        y_intersect = float(f1(x_intersect))
                        
                        intersections.append([round(float(x_intersect), 2), round(float(y_intersect), 2)])
        
        print(f"Найдено {len(intersections)} пересечений между функциями")
        for idx, (x, y) in enumerate(intersections, 1):
            print(f"   [{idx}] x={x:.3f}, y={y:.3f}")
        
        return intersections

    def _get_all_funcs_points(self):
        all_points_data = []
        for set in self.quality_term_data:
            for i in range(len(self.quality_term_data[set].x_range)):
                x = self.quality_term_data[set].x_range[i]
                y = self.quality_term_data[set].y_range[i]
                all_points_data.append((x, y))

        return all_points_data

    def set_outer_points(self):
        all_points_data = self._get_all_funcs_points()
        intersections = self._find_funcs_intersections()
        all_points_data += intersections
        # print(all_points_data)
        outer_points = defaultdict()
        for x, y in all_points_data:
            if x not in outer_points.keys() or x in outer_points.keys() and y > outer_points[x]:
                outer_points[x] = y

        self.outer_points = tuple(outer_points.items())

        return list(intersections)
    
    def find_x_axis_intersections(self):
        points = self.outer_points
        intersections = []

        if not points or len(points) < 2:
            return intersections

        # проверяем пары соседей
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            # точка на оси X
            if y1 == 0:
                intersections.append([round(float(x1), 2), 0.0])

            # сегмент полностью на оси X
            if y1 == 0 and y2 == 0:
                intersections.append([round(float(x2), 2), 0.0])
            # пересечение по смене знака
            elif y1 * y2 < 0:
                t = -y1 / (y2 - y1)
                x = x1 + t * (x2 - x1)
                intersections.append([round(float(x), 2), 0.0])

        # отдельно проверяем ПОСЛЕДНЮЮ точку
        last_x, last_y = points[-1]
        if last_y == 0:
            intersections.append([round(float(last_x), 2), 0.0])

        # убираем дубли
        # intersections = set(intersections)
        return intersections

    def get_gravity_center(self, points):
        n = len(points)
        sum_x = 0
        sum_y = 0
        for x, y in points:
            sum_x += x
            sum_y += y

        target_x, target_y = round(float(sum_x)/n, 3), round(float(sum_y)/n, 3)
        
        return [target_x, target_y]
    
    def visualize_polygon(self, vertices=None, centroid=None):
        x_range = []
        y_range = []
        points_x_y = self.outer_points

        plt.figure(figsize=(12, 8))
        color = "#1900FF"
        
        for x, y in points_x_y:
            x_range.append(x)
            y_range.append(y)

        plt.axhline(0, color='black', linewidth=1)  # ось X
        plt.axvline(0, color='black', linewidth=1)  # ось Y
        plt.plot(x_range, y_range, color=color, linewidth=2)

        if vertices:
            vx = [p[0] for p in vertices]
            vy = [p[1] for p in vertices]
            plt.scatter(vx, vy, color='green', s=40, zorder=5)

        if centroid is not None:
            cx, cy = centroid
            plt.scatter([cx], [cy], color='red', s=70, marker='x', zorder=6, label='Центр масс')
            plt.text(cx, cy, f"  ({cx:.3f}, {cy:.3f})", color='red', va='top')

        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()