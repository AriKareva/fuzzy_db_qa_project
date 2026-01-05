import matplotlib.pyplot as plt
from .defuzzificator import Defuzzificator
from .fuzzificator import Fuzzificator
from .rule_parser import RuleParser
from .summarizer import Summarizer



class FuzzyLogicSystem:
    def __init__(self):
        self.parser = RuleParser()
        self.fuzzificator = Fuzzificator(self.parser)
        self.summarizer = Summarizer(fuzzificator=self.fuzzificator)
        self.defuzzificator = Defuzzificator(summarizer=self.summarizer)


    def evaluate_quality(self, input_data):

        print('Работа нечеткой системы оценки качества БД')

        parsed_rules = []

        for rule_data in self.parser.rules_data:
            parsed_rule = self.parser.parse_rule_with_and(rule_data)
            parsed_rules.append(parsed_rule)
            print(parsed_rule)


        output_data = self.fuzzificator.fuzzify_input_data(input_data)

        min_by_rule = {}

        for parsed_rule in parsed_rules:
            min_rule_val = self.summarizer.activation(rule=parsed_rule, fuzzified_input=output_data)
            term = parsed_rule['output_term']
            if term in min_by_rule.keys():
                min_by_rule[term].append(min_rule_val)
            else:
                min_by_rule[term] = [min_rule_val]
                print(term, min_rule_val)


        self.summarizer.visualize_sets()
        extra_points = self.summarizer.aggregation(min_by_rule)
        self.summarizer.visualize_sets()

        intersections = self.defuzzificator.set_outer_points()
        print(intersections)

        self.defuzzificator.visualize_polygon()
        x_intersections = self.defuzzificator.find_x_axis_intersections()
        polygon_vertices = x_intersections
        polygon_vertices += intersections
        polygon_vertices += extra_points


        self.defuzzificator.visualize_polygon(vertices=polygon_vertices)
        target = self.defuzzificator.get_gravity_center(polygon_vertices)

        self.defuzzificator.visualize_polygon(vertices=polygon_vertices, centroid=target)


        x = target[0]
        fuzzy_out = self.fuzzificator.fuzzify_final_output(x)
        print(fuzzy_out)

        self.fuzzificator.beautiful_output(fuzzy_out)

        return {
            'fuzzy_out': fuzzy_out,
            'target': target
        }