import json
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn import light_palette

from utils.functions import triangle_func
from fuzzy_system.fuzzy_set import FuzzySet


class RuleParser:
    def __init__(self, rules_file='utils/rules.json'):
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)

        self.rules_data = rules_data['db_quality_rules']
        self.input_terms = ['мало', 'средне', 'много']
        self.output_terms = ['очень_плохое', 'плохое', 'нормальное', 'хорошее', 'очень_хорошее']
        self.oo_min = 0.0
        self.oo_max = 1.0
    
        self.composite_pk = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0.0, 0.15, 0.3]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.25, 0.3, 0.5]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.4, 0.6, 1.0]
            )
        }

        self.not_int_pk = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0.0, 0.15, 0.3]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.25, 0.3, 0.5]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.4, 0.6, 1.0]
            )
        }

        self.memory_waste_char = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0, 0.05, 0.1]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.08, 0.15, 0.2]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.15, 0.2, 1.0]
            )
        }

        self.fnpz_tbl = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0, 0.05, 0.1]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.08, 0.15, 0.2]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.15, 0.2, 1.0]
            )
        }

        self.trz_tbl = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0, 0.05, 0.1]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.08, 0.15, 0.2]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.15, 0.2, 1.0]
            )
        }

        self.default_upd = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0.0, 0.1, 0.15]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.08, 0.2, 0.4]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.3, 0.5, 1.0]
            )
        }

        self.default_del = {
            'мало': FuzzySet(
                func=triangle_func,
                func_params=[0.0, 0.1, 0.15]
            ),
            'средне': FuzzySet(
                func=triangle_func,
                func_params=[0.08, 0.2, 0.4]
            ),
            'много': FuzzySet(
                func=triangle_func,
                func_params=[0.3, 0.5, 1.0]
            )
        }


        self.quality = {
            'очень_плохое': FuzzySet(
                func=triangle_func,
                func_params=[0.0, 0.15, 0.3],
                is_out=True
            ),
            'плохое': FuzzySet(
                func=triangle_func,
                func_params=[0.25, 0.3, 0.35],
                is_out=True
            ),
            'нормальное': FuzzySet(
                func=triangle_func,
                func_params=[0.32, 0.4, 0.48],
                is_out=True
            ),
            'хорошее': FuzzySet(
                func=triangle_func,
                func_params=[0.4, 0.5, 0.6],
                is_out=True
            ),
            'очень_хорошее': FuzzySet(
                func=triangle_func,
                func_params=[0.55, 0.75, 1.0],
                is_out=True
            )
        }


        self.all_fuzzy_sets = {
            'доля_составных_ключей': self.composite_pk,
            'доля_не_целочисленных_ключей': self.not_int_pk,
            'доля_char_атрибутов_с_растратой_пямяти': self.memory_waste_char,
            'доля_таблиц_без_on_update': self.default_upd,
            'доля_таблиц_без_on_delete': self.default_del,
            'качество': self.quality
        }

    def parse_rule(self, rule_data):
        rule = rule_data['rule']
        weight = rule_data['weight']
        rule = rule.strip().lower()
        pattern = r'если\s+([a-яё_\s]+?)\s+[–-]\s+([a-яё_\s]+?),?\s+то\s+([a-яё_\s]+?)\s+[–-]\s+([a-яё_\s]+?)$'
        
        match = re.match(pattern, rule)
        
        if not match:
            print(f"Ошибка парсинга правила: {rule}")
            return None, None, None, None
        
        input_var = match.group(1).strip()
        input_term = match.group(2).strip() 
        output_var = match.group(3).strip()
        output_term = match.group(4).strip()

        return input_var, input_term, output_var, output_term, weight

    def parse_rule_with_and(self, rule_data):
        rule = rule_data['rule']
        weight = rule_data['weight']
        rule = rule.strip().lower()
        main_pattern = r'если\s+(.+?)\s+то\s+(.+?)$'
        
        main_match = re.match(main_pattern, rule)
        
        if not main_match:
            return {
                'is_valid': False,
                'conditions': [],
                'output_var': None,
                'output_term': None,
                'error': f"Ошибка парсинга основной структуры правила: {rule}"
            }
        
        conditions_str = main_match.group(1)  # "параметр1 – терм1 И параметр2 – терм2 И ..."
        output_str = main_match.group(2)      # "качество – терм"
        
        # Разделяем по оператору И
        condition_parts = re.split(r'\s+и\s+', conditions_str)
        
        conditions = []
        
        # Паттерн для одного условия: "параметр – терм" или "параметр - терм"
        condition_pattern = r'([a-яё_\s]+?)\s+[–-]\s+([a-яё_\s]+?)$'
        
        for condition_part in condition_parts:
            condition_part = condition_part.strip()
            
            match = re.match(condition_pattern, condition_part)
            
            if not match:
                return {
                    'is_valid': False,
                    'conditions': [],
                    'output_var': None,
                    'output_term': None,
                    'error': f"Ошибка парсинга условия: '{condition_part}'"
                }
            
            param = match.group(1).strip()
            term = match.group(2).strip()
            
            conditions.append({
                'parameter': param,
                'term': term
            })

        output_pattern = r'([a-яё_\s]+?)\s+[–-]\s+([a-яё_\s]+?)$'
        output_match = re.match(output_pattern, output_str)
        
        if not output_match:
            return {
                'is_valid': False,
                'conditions': conditions,
                'output_var': None,
                'output_term': None,
                'error': f"Ошибка парсинга выхода: '{output_str}'"
            }
        
        output_var = output_match.group(1).strip()
        output_term = output_match.group(2).strip()
        
        return {
            'is_valid': True,
            'conditions': conditions,
            'output_var': output_var,
            'output_term': output_term,
            'weight': weight,
            'error': None
        }

    # построение матриц

    def build_matrices_from_parsed_rules(self, parsed_rules):
            matrices = {}
            param_rules = {}
            
            for parsed in parsed_rules:
                if not parsed['is_valid']:
                    continue
                    
                conditions = parsed['conditions']
                if not conditions:
                    continue
            
                first_param = conditions[0]['parameter']
                
                if first_param not in param_rules:
                    param_rules[first_param] = {}
                
                first_term = conditions[0]['term']
                output_term = parsed['output_term']
                weight = parsed['weight'] or 1.0
                
                param_rules[first_param][first_term] = {
                    'output_term': output_term,
                    'weight': weight
                }
            
            for param_name, param_data in param_rules.items():
                matrix = np.zeros((len(self.input_terms), len(self.output_terms)))
                
                for i, input_term in enumerate(self.input_terms):
                    if input_term in param_data:
                        output_term = param_data[input_term]['output_term']
                        weight = param_data[input_term]['weight']
                        
                        if output_term in self.output_terms:
                            j = self.output_terms.index(output_term)
                            matrix[i, j] = weight
                
                matrices[param_name] = {
                    'input_terms': self.input_terms,
                    'output_terms': self.output_terms,
                    'matrix': matrix.tolist()
                }
            
            return matrices

    def get_relation_matrices(self):
            parsed_rules = []
            
            for rule_data in self.rules_data:
                parsed = self.parse_rule_with_and(rule_data)
                parsed_rules.append(parsed)
            
            matrices = self.build_matrices_from_parsed_rules(parsed_rules)
            return matrices

    def matrices_to_dataframes(self, matrices):
            dataframes = {}
            
            for param_name, data in matrices.items():
                input_terms = data['input_terms']
                output_terms = data['output_terms']
                matrix = np.array(data['matrix'])
                df = pd.DataFrame(
                    matrix,
                    index=input_terms,
                    columns=output_terms
                )
                df = df.round(2)
                
                dataframes[param_name] = df
            
            return dataframes

    def print_all_dataframes(self, dataframes):
                for param_name, df in dataframes.items():
                    print(f"{param_name}")
                    print(df)
                    print()

    def visualize_matrices(self, matrices, transposed=False, figsize=(15, 10)):
            n_matrices = len(matrices)
            fig, axes = plt.subplots(2, n_matrices//2 + n_matrices%2, figsize=figsize)
            if n_matrices == 1:
                axes = [axes]
            elif n_matrices == 2:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for idx, (param_name, data) in enumerate(matrices.items()):
                ax = axes[idx]
                
                if transposed:
                    sns.heatmap(
                        data['matrix'],
                        annot=True, fmt='.2f',
                        cmap=light_palette('blue'),
                        ax=ax,
                        xticklabels=self.input_terms,
                        yticklabels=self.output_terms,
                        vmin=0, vmax=1
                    )
                    ax.set_xlabel(f'{param_name.replace('_', ' ')}', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Термы качества', fontsize=12, fontweight='bold')

                else:
                    sns.heatmap(
                        data['matrix'],
                        annot=True, fmt='.2f',
                        cmap=light_palette('blue'),
                        ax=ax,
                        xticklabels=self.output_terms,
                        yticklabels=self.input_terms,
                        vmin=0, vmax=1
                    )
                
                    ax.set_ylabel(f'{param_name.replace('_', ' ')}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Качество', fontsize=12, fontweight='bold')
            
            # Убираем лишние subplot'ы
            for i in range(n_matrices, len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            plt.show()


    def parse_conjunction_rule(self, rule_data):
        rule_text = rule_data['rule'].lower().strip()
        weight = rule_data.get('weight', 1.0)
        
        input_terms = ['мало', 'средне', 'много']
        output_terms = ['очень_плохое', 'плохое', 'нормальное', 'хорошее', 'очень_хорошее']
        
        match = re.search(r'если\s+(.+?)\s+то\s+(.+?)$', rule_text)
        if not match:
            raise ValueError(f"Не удалось распарсить правило: {rule_text}")
        
        conditions_str = match.group(1)
        output_str = match.group(2)
        
        condition_parts = re.split(r'\s+и\s+', conditions_str)
        conditions = []
        param_names = []
        
        for part in condition_parts:
            cond_match = re.search(r'([\w_]+)\s+[–-]\s+([\w_]+)', part)
            if cond_match:
                param = cond_match.group(1).strip()
                term = cond_match.group(2).strip()
                conditions.append((param, term))
                param_names.append(param)
        
        if len(conditions) != 2:
            raise ValueError(f"Ожидается 2 условия, получено {len(conditions)}")
        
        output_match = re.search(r'([\w_]+)\s+[–-]\s+([\w_]+)', output_str)
        if not output_match:
            raise ValueError(f"Не удалось распарсить выход: {output_str}")
        
        output_term = output_match.group(2).strip()
        
        param1, term1 = conditions[0]
        param2, term2 = conditions[1]
        
        if term1 not in input_terms:
            raise ValueError(f"Терм '{term1}' не найден в {input_terms}")
        if term2 not in input_terms:
            raise ValueError(f"Терм '{term2}' не найден в {input_terms}")
        
        term1_idx = input_terms.index(term1)
        term2_idx = input_terms.index(term2)
        
        if output_term not in output_terms:
            raise ValueError(f"Выходной терм '{output_term}' не найден в {output_terms}")
        
        output_idx = output_terms.index(output_term)
        
        row_idx = term1_idx * 3 + term2_idx
        
        matrix = np.zeros((9, 5))
        matrix[row_idx, output_idx] = weight
        
        key = f"{param1}_И_{param2}"
        
        result = {
            key: {
                'param1': param1,
                'param2': param2,
                'input_terms': input_terms,
                'output_terms': output_terms,
                'matrix': matrix.tolist()
            }
        }
        
        return result

    def aggregate_rules(self, rules_data):
        aggregated = {}
        
        for rule_data in rules_data:
            try:
                parsed = self.parse_conjunction_rule(rule_data)
                
                for key, data in parsed.items():
                    if key not in aggregated:
                        aggregated[key] = data
                    else:
                        old_matrix = np.array(aggregated[key]['matrix'])
                        new_matrix = np.array(data['matrix'])
                        aggregated[key]['matrix'] = np.maximum(old_matrix, new_matrix).tolist()
            
            except ValueError as e:
                print(f"⚠️  {e}")
        
        return aggregated

    def matrix_to_dataframe(self, result):
        dataframes = {}
        
        for key, data in result.items():
            param1 = data['param1']
            param2 = data['param2']
            input_terms = data['input_terms']
            output_terms = data['output_terms']
            matrix = data['matrix']
            
            # Создаём индексы строк: все комбинации (param1, param2)
            row_index = []
            for t1 in input_terms:
                for t2 in input_terms:
                    row_index.append(f"({t1}, {t2})")
            
            # Создаём DataFrame
            df = pd.DataFrame(
                matrix,
                index=pd.Index(row_index, name=f"{param1} × {param2}"),
                columns=output_terms
            )
            
            dataframes[key] = df
        
        return dataframes

    def visualize_fuzzy_matrix(self, param_name, data, ax=None, figsize=(10, 10)):
        # Создаём новый axes если не передан
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Проверяем, является ли data DataFrame или dict
        if isinstance(data, pd.DataFrame):
            # Работаем с DataFrame напрямую
            matrix_df = data
            matrix = matrix_df.values
            output_terms = matrix_df.columns.tolist()
            row_labels = matrix_df.index.tolist()
        else:
            # Работаем со словарём (старый формат)
            matrix = np.array(data['matrix']) if isinstance(data['matrix'], list) else data['matrix']
            output_terms = data.get('output_terms', ['очень_плохое', 'плохое', 'нормальное', 'хорошее', 'очень_хорошее'])
            input_terms = data.get('input_terms', ['мало', 'средне', 'много'])
            row_labels = [f"({input_terms[i]}, {input_terms[j]})" for i in range(3) for j in range(3)]
        
        # Визуализируем
        sns.heatmap(
            matrix,
            annot=True, fmt='.2f',
            cmap=light_palette('blue'),
            ax=ax,
            xticklabels=output_terms,
            yticklabels=row_labels,
            vmin=0, vmax=1,
            cbar_kws={'label': 'Вес'}
        )
        
        ax.set_ylabel(f'{param_name.replace("_", " ")}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Качество', fontsize=12, fontweight='bold')
        
        # Ротация меток осей
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        
        return ax

    def matrix_composition(self, R: np.ndarray, S: np.ndarray) -> np.ndarray:
        R = np.array(R)
        S = np.array(S)
        
        # Транспонируем S: качество(5) → проблемы2(3)
        S_T = S.T  # 3×5 → 5×3
        
        # Проверяем совместимость: R столбцы == S_T строки
        if R.shape[1] != S_T.shape[0]:
            raise ValueError(f"Несовместимые размеры: R{ R.shape }, S_T{ S_T.shape }")
        
        m, n = R.shape      # m=3, n=5
        n2, p = S_T.shape   # n2=5, p=3
        
        P = np.zeros((m, p))
        
        # Классический цикл "строкой на столбец"
        for i in range(m):
            for j in range(p):
                # min по k: строка R[i] × столбец S_T[:,j]
                candidates = np.minimum(R[i, :], S_T[:, j])
                P[i, j] = np.max(candidates)
        
        return P
    
    def fuzzify_composition_result(self, weigth):
        membership = 0
        term = ''
        for k, v in self.quality.items():
            self.quality[k] 
            cur_membership = self.quality[k].func(self.quality[k].func_params, weigth)
            if cur_membership > membership:
                term = k
                membership = cur_membership

        return term, membership

    def generate_rules_from_composition(self, param1_name, param2_name, matrices, threshold=0.5):
        R = np.array(matrices[param1_name]['matrix'])
        S = np.array(matrices[param2_name]['matrix'])
        
        # Свёртка
        P = self.matrix_composition(R, S)
        rules = []
        
        for i in range(3):
            for j in range(3):
                weight = P[i, j]
                term, membership = self.fuzzify_composition_result(weight)
                if weight >= threshold:
                    rule = f"ЕСЛИ {param1_name} – {self.input_terms[i]} И {param2_name} – {self.input_terms[j]} ТО качество – {term}"
                    rules.append({'rule': rule, 'weight': membership})
        
        return {
            'composition_matrix': P.tolist(),
            'rules': rules,
            'threshold': threshold
        }
    
    def transpose_relation_matrix(self, matrix_data) :
        matrix = np.array(matrix_data['matrix'])
        
        return {
            'input_terms': matrix_data['output_terms'], 
            'output_terms': matrix_data['input_terms'], 
            'matrix': matrix.T.tolist() 
        }

    def transpose_all_matrices(self, matrix):
        transposed_matrices = {}
        
        for param_name, matrix_data in matrix.items():
            transposed = self.transpose_relation_matrix(matrix_data)
            transposed_matrices[param_name] = transposed
    
        self.transposed_matrices = transposed_matrices
        return transposed_matrices

    def find_complement(self, fuzzy_set):
        operation: str = 'standard'
        
        is_series = isinstance(fuzzy_set, pd.Series)
        
        if is_series:
            fuzzy_dict = fuzzy_set.to_dict()
        else:
            fuzzy_dict = fuzzy_set.copy()
        
        complement = {}
        
        if operation == 'standard':
            for element, membership in fuzzy_dict.items():
                complement[element] = 1 - membership
        
        else:
            raise ValueError(f"Неизвестная операция: {operation}. ")
        
        if is_series:
            return pd.Series(complement)
        else:
            return complement
