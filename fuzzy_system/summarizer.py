import matplotlib.pyplot as plt


class Summarizer:
    def __init__(self, fuzzificator):
        self.fuzzificator = fuzzificator

    def activation(self, rule, fuzzified_input):
        if not rule.get('is_valid', False):
            return 0
        
        if not rule.get('conditions'):
            return 0
        
        # Получаем значения для каждого условия
        condition_values = []
        
        for condition in rule['conditions']:
            param_name = condition['parameter'].strip().lower()
            term_name = condition['term'].strip().lower()
            
            # Ищем параметр в входных данных
            param_found = None
            for param_key in fuzzified_input.keys():
                if param_key.lower() == param_name:
                    param_found = param_key
                    break
            
            if param_found is None:
                return 0
            
            # Ищем терм в параметре
            term_found = None
            for term_key in fuzzified_input[param_found].keys():
                if term_key.lower() == term_name:
                    term_found = term_key
                    break
            
            if term_found is None:
                return 0
            
            # Получаем степень принадлежности
            membership = fuzzified_input[param_found][term_found]
            condition_values.append(membership)
        
        # Активация = min всех условий
        activation_val = min(condition_values) if condition_values else 0.0
        
        return activation_val
    
    def aggregation(self, activation_results):
        extra_points = []
        for set_k, set_v in activation_results.items():
            cur_alpha = max(activation_results.get(set_k, None))
            self.fuzzificator.parser.quality[set_k].set_alpha(cur_alpha)
            extra_points += self.fuzzificator.parser.quality[set_k].set_extra_points(alpha=cur_alpha)
        return extra_points

    def visualize_sets(self):
        quality_data = self.fuzzificator.parser.quality
        plt.figure(figsize=(16, 8))
        colors_dict = {
            'очень_плохое': "#4E0404",
            'плохое': '#FF6B6B',
            'нормальное': '#FFD93D',
            'хорошее': "#D0E300",
            'очень_хорошее': '#6BCB77'
        }
        
        for i, data in quality_data.items():
            color = colors_dict[i]
            if not data:
                continue
            name = i
            x_range = quality_data[i].x_range
            y_range = quality_data[i].y_range
            
            
            plt.plot(x_range, y_range, 
                    label=name, 
                    color=color, 
                    linewidth=2)

        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()