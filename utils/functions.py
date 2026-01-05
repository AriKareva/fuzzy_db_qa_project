from typing import List, Tuple, Optional
import numpy as np


from typing import List

def l_func(params, x):
    a, b = params
    if x <= a:
        return 1.0
    elif x >= b:
        return 0.0
    else:
        return max(0.0, min(1.0, float(round((b - x) / (b - a), 3))))

def r_func(params, x):
    a, b = params
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return max(0.0, min(1.0, float(round((x - a) / (b - a), 3))))

def trapezoid_func(params, x):
    a, b, c, d = params
    if x <= a or x > d:
        return 0.0
    elif a < x <= b:
        return max(0.0, min(1.0, float(round((x - a) / (b - a), 3))))
    elif b < x <= c:
        return 1.0
    elif c < x <= d:
        return max(0.0, min(1.0, float(round((d - x) / (d - c), 3))))
    else:
        return 0.0

def triangle_func(params, x):
    a, b, c = params
    if x <= a or x > c:
        return 0.0
    elif a < x <= b:
        return max(0.0, min(1.0, float(round((x - a) / (b - a), 3))))
    elif b < x <= c:
        return max(0.0, min(1.0, float(round((c - x) / (c - b), 3))))
    else:
        return 0.0
