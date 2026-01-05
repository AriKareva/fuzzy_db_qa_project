import matplotlib.pyplot as plt
from fuzzy_system.defuzzificator import Defuzzificator
from fuzzy_system.fuzzificator import Fuzzificator
from fuzzy_system.rule_parser import RuleParser
from fuzzy_system.summarizer import Summarizer


print('Работа нечеткой системы оценки качества БД')
parser = RuleParser()
matrices = parser.get_relation_matrices()
print(matrices)

dfs = parser.matrices_to_dataframes(matrices)
parser.print_all_dataframes(dfs)

parser.visualize_matrices(matrices)


# Генерация правил
n1 = 'доля_составных_ключей'
n2 = 'доля_char_атрибутов_с_растратой_пямяти'
pair_result = parser.generate_rules_from_composition(n1, n2, matrices=matrices)

for rule in pair_result['rules']:
    print(rule)


rules_data = pair_result['rules']
conjuction_matrices = parser.aggregate_rules(rules_data)
conjuction_matrices_df = parser.matrix_to_dataframe(conjuction_matrices)


ax = parser.visualize_fuzzy_matrix(
    param_name='доля_составных_ключей_И_доля_char_атрибутов_с_растратой_пямяти',
    data=conjuction_matrices_df['доля_составных_ключей_И_доля_char_атрибутов_с_растратой_пямяти']
)
plt.show()



# Дополнение

# Создаём матрицу правил
matrix_df = dfs['доля_составных_ключей']
title = 'доля_составных_ключей'

print("\nИсходная матрица правил:")
print(matrix_df)

# Получаем дополнение для каждого элемента матрицы
matrix_complement = matrix_df.applymap(lambda x: 1 - x)

print("\nДополнение матрицы правил:")
print(matrix_complement)


matrices = parser.get_relation_matrices()
matrices

transposed = parser.transpose_all_matrices(matrices)
transposed


dfs_transposed = parser.matrices_to_dataframes(transposed)
parser.print_all_dataframes(dfs_transposed)


parser = RuleParser('rules.json')

parsed_rules = []

for rule_data in parser.rules_data:
    parsed_rule = parser.parse_rule_with_and(rule_data)
    parsed_rules.append(parsed_rule)
    print(parsed_rule)



input_data = {
    'доля_составных_ключей': 0.0,
    'доля_не_целочисленных_ключей': 0.0,
    # 'доля_неатомарных_атрибутов': 0.4,
    'доля_char_атрибутов_с_растратой_пямяти': 0.0,
    'доля_таблиц_без_on_update': 0.0,
    'доля_таблиц_без_on_delete': 0.0
}


fuzzificator = Fuzzificator(parser)
summarizer = Summarizer(fuzzificator=fuzzificator)
defuzzificator = Defuzzificator(summarizer=summarizer)

output_data = fuzzificator.fuzzify_input_data(input_data)

min_by_rule = {}

for parsed_rule in parsed_rules:
    min_rule_val = summarizer.activation(rule=parsed_rule, fuzzified_input=output_data)
    term = parsed_rule['output_term']
    if term in min_by_rule.keys():
        min_by_rule[term].append(min_rule_val)
    else:
        min_by_rule[term] = [min_rule_val]
        print(term, min_rule_val)


summarizer.visualize_sets()
extra_points = summarizer.aggregation(min_by_rule)
summarizer.visualize_sets()

intersections = defuzzificator.set_outer_points()
print(intersections)

defuzzificator.visualize_polygon()
x_intersections = defuzzificator.find_x_axis_intersections()
polygon_vertices = x_intersections
polygon_vertices += intersections
polygon_vertices += extra_points


defuzzificator.visualize_polygon(vertices=polygon_vertices)
target = defuzzificator.get_gravity_center(polygon_vertices)

defuzzificator.visualize_polygon(vertices=polygon_vertices, centroid=target)


x = target[0]
fuzzy_out = fuzzificator.fuzzify_final_output(x)
print(fuzzy_out)

fuzzificator.beautiful_output(fuzzy_out)