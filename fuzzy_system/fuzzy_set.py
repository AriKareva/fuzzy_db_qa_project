import numpy as np


class FuzzySet:
    def __init__(self, func, func_params, is_out=False):
        self.func = func
        self.func_params = func_params
        self.set_func_values(func, func_params, is_out)

    def set_func_values(self, func, params, is_out, e=0.0005):
        if is_out:
            y_range = []
            x_range = np.arange(0.0, 1+e, e)
            for x in x_range:
                y_range.append(func(params, x))

            self.x_range = x_range
            self.y_range = y_range

        else:
            self.x_range = None
            self.y_range = None

    def set_alpha(self, alpha):
        if self.y_range is not None:
            y_range = self.y_range
            self.y_range = [min(y, alpha) for y in y_range]

    def set_extra_points(self, alpha):
        max_x = -1
        min_x = np.inf
        extra_points = []
        for i in range(len(self.x_range)):
            if self.y_range[i] == alpha and self.x_range[i] > max_x:
                max_x = self.x_range[i]
            if self.y_range[i] == alpha and self.x_range[i] < min_x:
                min_x = self.x_range[i]

        extra_points.append([round(float(max_x), 2), round(float(alpha), 2)])
        extra_points.append([round(float(min_x), 2), round(float(alpha), 2)])
        return extra_points