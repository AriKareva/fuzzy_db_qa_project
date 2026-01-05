from fuzzy_system.rule_parser import RuleParser


class Fuzzificator:
    def __init__(self, parser : RuleParser):
        self.parser = parser

    def fuzzify_input_data(self, input_data):
        output_data = {}

        for param_name, value in input_data.items():
            if param_name not in self.parser.all_fuzzy_sets:
                print(f"Параметр '{param_name}' не найден в all_fuzzy_sets")
                print(f"Доступные параметры: {list(self.parser.all_fuzzy_sets.keys())}")
                continue
            
            fuzzy_sets = self.parser.all_fuzzy_sets[param_name]
            output_data[param_name] = {}
            
            for term_name, fuzzy_set in fuzzy_sets.items():
                membership = fuzzy_set.func(fuzzy_set.func_params, value)
                output_data[param_name][term_name] = round(membership, 3)
        
        return output_data

    def fuzzify_final_output(self, x : int):
        fuzzy_out = {}
        fuzzy_sets = self.parser.quality
        for k, v in fuzzy_sets.items():
            fuzzy_out[k] = fuzzy_sets[k].func(fuzzy_sets[k].func_params, x)
        return fuzzy_out
    
    def beautiful_output(self, fuzzy_out):
        max_out = 0
        k_max_out = ''
        for k, v in fuzzy_out.items():
            if max_out < v:
                max_out = v
                k_max_out = k
        return f'Качество БД - {k_max_out}'
