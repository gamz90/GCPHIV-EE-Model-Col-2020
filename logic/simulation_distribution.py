import math
import os
import random
import pandas as pd
import numpy as np
from root import DIR_INPUT, DIR_OUTPUT
import datetime as dt


def recurrent_assignation(info_df: pd.DataFrame):
    """
    Method that turns a dataframe to a dictionary considering the keys according to the number of characteristic columns
    :param info_df: Dataframe to transform.
    :return: Created dictionary corresponding.
    """
    columns = list(info_df.columns)
    columns.remove('BASE_VALUE')
    columns.remove('INF_VALUE')
    columns.remove('SUP_VALUE')
    result_dict = dict()
    if len(columns) >= 2:
        for value in info_df[columns[0]].unique():
            result_dict[value] = recurrent_assignation(info_df[info_df[columns[0]] == value].drop(columns=columns[0]))
    else:
        for value in info_df[columns[0]].unique():
            value_dict = dict()
            for value_type in ['BASE_VALUE', 'INF_VALUE', 'SUP_VALUE']:
                value_dict[value_type] = info_df[info_df[columns[0]] == value][value_type].sum()
            result_dict[value] = value_dict
    return result_dict


def get_info(dictionary: dict, params_to_look: dict, value: str = 'BASE_VALUE'):
    """
    Method that obtains the value from a dictionary created with the recurrent assignation method.
    :param dictionary: Dictionary of the given conditions.
    :param params_to_look: Parameters with the specific values.
    :param value: Value of the scenario to consider.
    :return: Resulting value of the query.
    """
    levels = dictionary['levels']
    cur_dict = dictionary['values']
    for level in levels:
        cur_dict = cur_dict[params_to_look.get(level, 'ALL')]
    return cur_dict[value]


def generate_triangular(values: dict, n_iterations: int):
    """
    Generation of the random value scenarios, considering triangular distributions
    :param values: Range of values (inferior - min, base - mode, sup - max) values for the parameter.
    :param n_iterations: number of iterations to simulate, that require same number of simulated scenarios.
    :return: range of possible values for the scenario.
    """
    if values['INF_VALUE'] != values['SUP_VALUE']:
        return np.random.triangular(values['INF_VALUE'], values['BASE_VALUE'], values['SUP_VALUE'], n_iterations)
    else:
        return np.ones(n_iterations)*values['BASE_VALUE']


class DistributionModel(object):
    def __init__(self):
        """
        Model that simulates the patients receiving any defined ART with a set of parameters based on simulation
        according to ranges.
        """
        self.patients = dict()
        self.age_groups = dict()
        general_prob_df = pd.read_csv(DIR_INPUT + 'population_info.csv')
        columns = list(general_prob_df.columns)
        columns.remove('BASE_VALUE')
        columns.remove('INF_VALUE')
        columns.remove('SUP_VALUE')
        self.general_info = {'values': recurrent_assignation(general_prob_df), 'levels': columns}
        self.age_groups = list(general_prob_df.AGE.unique())
        self.age_groups.remove('ALL')
        self.sexes = list(general_prob_df.SEX.unique())
        self.sexes.remove('ALL')
        self.medications = dict()
        self.leaving_states = ['dead', 'failure', 'adverse_reaction', 'dead_ar']
        self.adverse_info = pd.read_csv(DIR_INPUT + 'adverse_general.csv')
        self.random_values = dict()

    def load_medication(self, medication_name: str):
        """
        Process that loads the medication info to the model set of parameters
        :param medication_name: Medication scheme to load
        """
        medication_df = pd.read_csv(DIR_INPUT + medication_name + '.csv')
        columns = list(medication_df.columns)
        columns.remove('BASE_VALUE')
        columns.remove('INF_VALUE')
        columns.remove('SUP_VALUE')
        self.medications[medication_name] = {'values': recurrent_assignation(medication_df), 'levels': columns}
        self.medications[medication_name]['adverse_probability'] = None
        try:
            medication_adverse_df = pd.read_csv(DIR_INPUT + medication_name + '_adverse_probability.csv')
            self.medications[medication_name]['adverse_probability'] = medication_adverse_df
        except:
            self.medications[medication_name]['adverse_probability'] = None

    def calculate_qaly_cost(self, viral_charge: bool, medication_name: str, age: str, chronic: int, acute: bool,
                            dead: bool, step_length: int, i: int):
        """
        :param dead: boolean that indicates if the patient died during the period
        :param medication_name: Name of the considered medication.
        :param viral_charge: boolean value indicating if patient has a high viral charge
        :param age: quinquennial age group of the patient
        :param chronic: trinary variable 0 if no chronic event, 1 if developed during period and 2 if it was a previous
        :param acute: boolean variable indicating the occurrence of an acute event
        :param step_length: duration of the step to calculate related qaly
        :param i: iteration from where to obtain the random values
        :return: dictionary with associated qaly value and cost
        """
        adverse = self.random_values[medication_name]['adverse']
        result_vector = {'qaly': self.random_values[medication_name]['base_qaly'][age][i]/12,
                         'test_cost': self.random_values[medication_name]['test_cost'][i], 'adverse_cost': 0,
                         'medication_cost': 0, 'total_cost': 0}
        if np.isnan(result_vector['qaly']):
            print('base_qaly')

        if viral_charge:
            result_vector['qaly'] = result_vector['qaly'] * self.random_values[medication_name]['high_test_qaly'][i]
            if np.isnan(result_vector['qaly']) or result_vector['qaly'] > 1:
                print('high_test_qaly')
            result_vector['test_cost'] = result_vector['test_cost'] + self.random_values[medication_name][
                'test_cost_high_charge'][i]
        result_vector['medication_cost'] = self.random_values[medication_name]['adherent_month_cost'][i] * (
            step_length / 2 if dead else step_length)

        if chronic == 1:
            result_vector['adverse_cost'] = adverse['chronic']['Immediate_Cost'][i]
            if dead:
                pass
            else:
                chronic_months = step_length / 2 - 1
                qaly_chronic_immediate = result_vector['qaly'] * (1 - adverse['chronic']['Immediate_QALY'][i])
                if np.isnan(qaly_chronic_immediate) or qaly_chronic_immediate > 1:
                    print(medication_name, 'qaly_chronic_immediate')
                qaly_chronic_long_term = result_vector['qaly'] * (1 - adverse['chronic']['Chronic_QALY'][i])
                if np.isnan(qaly_chronic_long_term) or qaly_chronic_long_term > 1:
                    print(medication_name, 'qaly_chronic_long_term')
                result_vector['qaly'] = (result_vector['qaly'] * step_length / 2 + qaly_chronic_immediate +
                                         qaly_chronic_long_term * chronic_months)
                result_vector['adverse_cost'] = result_vector['adverse_cost'] + adverse['chronic']['Chronic_Cost'][i] \
                                                * chronic_months
            result_vector['total_cost'] = result_vector['test_cost'] + result_vector['adverse_cost'] + result_vector[
                'medication_cost']
            return result_vector
        elif chronic == 2:
            result_vector['qaly'] = result_vector['qaly'] * (1 - adverse['chronic']['Chronic_QALY'][i])
            if np.isnan(result_vector['qaly']) or result_vector['qaly'] > 1:
                print(medication_name, 'qaly_chronic_long_term')
            result_vector['adverse_cost'] = adverse['chronic']['Chronic_Cost'][i] * (step_length / 2 if dead
                                                                                              else step_length)
        if acute:
            qaly_acute_immediate = 0 if dead else result_vector['qaly'] * (1 - adverse['acute']['Immediate_QALY'][i])
            if np.isnan(qaly_acute_immediate) or qaly_acute_immediate > 1:
                print(medication_name, 'qaly_acute_immediate')
            result_vector['qaly'] = result_vector['qaly'] * (step_length / 2 if dead else step_length - 1) + \
                                    qaly_acute_immediate
            result_vector['adverse_cost'] = result_vector['adverse_cost'] + adverse['acute']['Immediate_Cost'][i]
        else:
            result_vector['qaly'] = result_vector['qaly'] * (step_length/2 if dead else step_length)
        result_vector['total_cost'] = result_vector['test_cost'] + result_vector['adverse_cost'] + result_vector[
            'medication_cost']
        return result_vector

    def initial_state(self, efficiency: float):
        """
        Create the initial state of the patient, according to the incidence and initial efficiency of the medication.
        :param efficiency: number that indicates the percentage of population that will result in a low viral charge
        due to medication for the first control test.
        :return: Initial state of the patient in the tracked variables.
        """
        if random.random() < efficiency:
            n_tests = 0
        else:
            n_tests = 1
        random_value = random.random()
        for age in self.age_groups:
            for sex in self.sexes:
                random_value -= self.general_info['values']['incidence'][age][sex]['BASE_VALUE']
                if random_value < 0.0:
                    month_age = int(age[-2:]) * 60 + math.floor(random.random() * 60)
                    return {'tests': n_tests, 'chronic': 0, 'acute': False, 'age': month_age,
                            'sex': sex, 'treatment': 2}
        month_age = int(self.age_groups[len(self.age_groups) - 1][-2:]) * 60 + math.floor(random.random() * 60)
        return {'tests': n_tests, 'chronic': 0, 'acute': False, 'age': month_age,
                'sex': self.sexes[len(self.sexes) - 1], 'treatment': 2}

    def simulate_step(self, state: dict, i: int, medication_name: str, inflation_rate: float = 1.0,
                      max_high_tests: int = 2):
        """
        The simulation of the possible events between two tests.
        :param max_high_tests: maximum value of consecutive tests with high viral charge before considering a
        virological failure
        :param state: current state of the patient.
        :param i: iteration value to consider amongst the simulated scenarios.
        :param inflation_rate: discount rate considered to deprecate values.
        :param medication_name: Name of the considered medication.
        :return: state of the patient after the considered step.
        """
        result_state = state.copy()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        time_for_step = 2 if state['tests'] > 0 else 6
        death_prob = 1 - (1 - self.random_values[medication_name]['month_d_r'][age_group][state['sex']][i]) ** \
                     time_for_step
        # GENERAL DEATH
        result_state['acute'] = False
        if random.random() < death_prob:
            result_state['tests'] = 'dead'
            result_state['age'] += time_for_step / 2
            result_state['treatment'] += time_for_step / 2
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step,
                                                 medication_name=medication_name, dead=True)
            for gc in qaly_cost:
                result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
            return result_state
        #  Chronic event can occur
        if state['chronic'] == 0:
            occurrence = 1 - (1 - self.random_values[medication_name]['adverse']['chronic']['probability'][i]) ** \
                         time_for_step
            if random.random() < occurrence:  # Chronic reaction occurs
                # Immediate change death due to chronic reaction
                result_state['chronic'] = 1
                if random.random() < self.random_values[medication_name]['adverse']['chronic']['Immediate_Death'][i]:
                    result_state['tests'] = 'dead_ar'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0),
                                                         age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step,
                                                         medication_name=medication_name, dead=True)
                    for gc in qaly_cost:
                        result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                    return result_state
                else:
                    # Immediate change due to chronic reaction
                    r_val = self.random_values[medication_name]['adverse']['chronic']['Immediate_Change'][i]
                    if random.random() < r_val:
                        result_state['tests'] = 'adverse_reaction'
                        result_state['age'] += time_for_step / 2
                        result_state['treatment'] += time_for_step / 2
                        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                             chronic=(state['chronic'] + result_state['chronic']),
                                                             acute=result_state['acute'], i=i, dead=False,
                                                             step_length=time_for_step, medication_name=medication_name)
                        for gc in qaly_cost:
                            result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                        return result_state
        else:
            death = 1 - (1 - self.random_values[medication_name]['adverse']['chronic']['Chronic_Death'][i]) ** \
                    time_for_step
            if random.random() < death:  # Death due to chronic reaction aftermath
                result_state['tests'] = 'dead_ar'
                result_state['age'] += time_for_step / 2
                result_state['treatment'] += time_for_step / 2
                qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                     chronic=(state['chronic'] + result_state['chronic']),
                                                     acute=result_state['acute'], i=i, step_length=time_for_step,
                                                     medication_name=medication_name, dead=True)
                for gc in qaly_cost:
                    result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                return result_state
            else:
                change = 1 - (1 - self.random_values[medication_name]['adverse']['chronic']['Chronic_Change'][i]) ** \
                         time_for_step
                if random.random() < change:  # Death due to chronic reaction aftermath
                    result_state['tests'] = 'adverse_reaction'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step,
                                                         medication_name=medication_name, dead=False)
                    for gc in qaly_cost:
                        result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                    return result_state
        if (not state['acute']) and result_state['chronic'] == state['chronic']:  # Acute event can occur
            occurrence = 1 - (1 - self.random_values[medication_name]['adverse']['acute']['probability'][i]) ** \
                         time_for_step
            if random.random() < occurrence:
                result_state['acute'] = True  # Acute reaction occurs
                # Death due to acute reaction
                if random.random() < self.random_values[medication_name]['adverse']['acute']['Immediate_Death'][i]:
                    result_state['tests'] = 'dead_ar'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step,
                                                         medication_name=medication_name, dead=True)
                    for gc in qaly_cost:
                        result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                    return result_state
                else:
                    # Change due to acute reaction
                    if random.random() < self.random_values[medication_name]['adverse']['acute']['Immediate_Change'][i]:
                        result_state['tests'] = 'adverse_reaction'
                        result_state['age'] += time_for_step / 2
                        result_state['treatment'] += time_for_step / 2
                        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                             chronic=(state['chronic'] + result_state['chronic']),
                                                             acute=result_state['acute'], i=i, dead=False,
                                                             step_length=time_for_step, medication_name=medication_name)
                        for gc in qaly_cost:
                            result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                        return result_state
        result_state['age'] += time_for_step
        result_state['treatment'] += time_for_step
        efficiency_dict = self.random_values[medication_name]['efficiency']
        efficiency = 0
        ef2 = None
        if len(efficiency_dict) > 1:
            for ef in efficiency_dict:
                if int(ef) <= int(state['treatment']):
                    efficiency = efficiency_dict[ef][i]
                    ef2 = ef
        else:
            efficiency = efficiency_dict[list(efficiency_dict.keys())[0]][i]
        if efficiency >= 1:
            print('perfect efficiency', ef2, i, self.random_values[medication_name]['efficiency'], efficiency)
        result_state['tests'] = 0 if random.random() < efficiency else state['tests'] + 1

        if result_state['tests'] >= max_high_tests:
            result_state['tests'] = 'failure'
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step,
                                                 medication_name=medication_name, dead=False)
        else:
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(result_state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step,
                                                 medication_name=medication_name, dead=False)
        if np.isnan(qaly_cost['qaly']):
            print(state, i, qaly_cost['qaly'], qaly_cost)
        for gc in qaly_cost:
            result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
        return result_state

    def simulate_medication(self, iteration: int, medication_name: str, inflation_rate: float, max_high_tests: int = 2):
        """
        Model iteration, starting from the moment a patient has his or her first ART control, up to the moment of
        discontinuing the medication.
        :param max_high_tests: maximum value of tests with a high viral load before considering a virological failure
        :param iteration: Iteration value to consider.
        :param inflation_rate: Discount rate considered.
        :param medication_name: Name of the considered medication.
        :return: Summary of the simulation of the patient with the synthesized relevant information.
        """
        states = list()
        efficiency_dict = self.random_values[medication_name]['efficiency']
        efficiency = efficiency_dict[list(efficiency_dict.keys())[0]][iteration]
        state = self.initial_state(efficiency=efficiency)
        high_tests = 0
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group, chronic=0,
                                             dead=False, acute=False, step_length=2, i=iteration,
                                             medication_name=medication_name)
        for gc in qaly_cost:
            state[gc] = qaly_cost[gc]
        states.append(state)
        while states[len(states) - 1]['tests'] not in self.leaving_states:
            high_tests += 1 if states[len(states) - 1]['tests'] > 0 else 0
            states.append(self.simulate_step(state=states[len(states) - 1], i=iteration, inflation_rate=inflation_rate,
                                             medication_name=medication_name, max_high_tests=max_high_tests))
        change_reason = states[len(states) - 1]['tests']
        if change_reason == 'failure':
            high_tests += 1
        if change_reason != 'dead' or change_reason != 'dead_ar':
            if change_reason == 'failure':
                states[len(states) - 1]['test_cost'] += self.random_values[medication_name]['switch_cost'][
                                                            iteration] / inflation_rate ** states[len(states) - 1][
                                                            'treatment']
                states[len(states) - 1]['total_cost'] += self.random_values[medication_name]['switch_cost'][
                                                             iteration] / inflation_rate ** states[len(states) - 1][
                                                             'treatment']
            states[len(states) - 1]['qaly'] *= self.random_values[medication_name]['switch_qaly'][iteration]
        states = pd.DataFrame.from_dict(data=states, orient='columns')
        return {'qaly': states.qaly.sum(), 'treatment': states.treatment.max(), 'costs': states.total_cost.sum(),
                'test_cost': states.test_cost.sum(), 'adverse_cost': states.adverse_cost.sum(),
                'medication_cost': states.medication_cost.sum(), 'acute_events': len(states[states.acute]),
                'chronic_event': states.chronic.max(), 'exit_reason': change_reason, 'n_high_tests': high_tests,
                'percentage_high_tests': high_tests / len(states), 'starting_age': states.age.min()}

    def parallel_simulation(self, medication_name: str, group: str = 'Unique', n_simulations: int = 1000,
                            inflation_rate: float = 1.0, max_high_tests: int = 2):
        """
        Process that accumulates the results of the considered simulations from the given scenario.
        :param max_high_tests: maximum value of tests with a high viral load before considering a virological failure
        :param medication_name: Medication scheme received by the patient. Must be a str
        :param group: group to which the medication scheme belongs.
        :param n_simulations: number of simulations for the model.
        :param inflation_rate: discount rate considered.
        """
        file_name = DIR_OUTPUT + 'results_d_' + medication_name + '.csv'
        if os.path.exists(file_name):
            print(medication_name, dt.datetime.now())
        else:
            if medication_name in self.medications.keys():
                pass
            else:
                self.load_medication(medication_name)
            if 'switch_phase' in self.medications.keys():
                pass
            else:
                self.load_medication('switch_phase')
            self.generate_random(n_iterations=n_simulations, medication_name=medication_name)
            return_list = list()
            for i in range(n_simulations):
                return_list.append(self.simulate_medication(iteration=i, medication_name=medication_name,
                                                            inflation_rate=inflation_rate,
                                                            max_high_tests=max_high_tests))
            return_list = pd.DataFrame(return_list)
            return_list.reset_index(drop=False, inplace=True)
            return_list.sort_values(by='starting_age', inplace=True)
            return_list['medication_name'] = medication_name
            return_list.rename(columns={'index': 'iteration'}, inplace=True)
            return_list.iteration = return_list.iteration + 1
            return_list['group'] = group
            return_list['discount_rate'] = '3.5'
            return_list['treatment'] = return_list['treatment'] / 12
            return_list.to_csv(file_name, index=False)
            del self.random_values[medication_name]
            print(medication_name, dt.datetime.now())

    def generate_random(self, n_iterations: int, medication_name: str):
        """
        Generates the random values for the simulations of the medication scheme.
        :param n_iterations: number of required iterations
        :param medication_name: Medication scheme received by the patient. Must be a str
        """
        # Month_death_rate
        random_values = dict()
        month_d_r = dict()
        base_qaly = dict()

        for age in self.age_groups:
            age_g = dict()
            for sex in self.sexes:
                age_g[sex] = generate_triangular(values=self.general_info['values']['month_d_r'][age][sex],
                                                      n_iterations=n_iterations)
            base_qaly[age] = generate_triangular(values=self.general_info['values']['base_qaly'][age]['ALL'],
                                                      n_iterations=n_iterations)
            month_d_r[age] = age_g
        random_values['month_d_r'] = month_d_r
        random_values['base_qaly'] = base_qaly
        random_values['high_test_qaly'] = generate_triangular(values=self.general_info['values']['high_test_qaly'][
            'ALL']['ALL'], n_iterations=n_iterations)
        random_values['switch_cost'] = generate_triangular(values=self.general_info['values']['switch_cost']['ALL'][
            'ALL'], n_iterations=n_iterations)
        random_values['switch_qaly'] = generate_triangular(values=self.general_info['values']['switch_qaly']['ALL'][
            'ALL'], n_iterations=n_iterations)
        probabilities = self.medications[medication_name]['adverse_probability']
        chronic = {'Immediate_Death': np.zeros(n_iterations), 'Immediate_Change': np.zeros(n_iterations),
                   'Immediate_QALY': np.zeros(n_iterations), 'Immediate_Cost': np.zeros(n_iterations),
                   'Chronic_Death': np.zeros(n_iterations), 'Chronic_Change': np.zeros(n_iterations),
                   'Chronic_QALY': np.zeros(n_iterations), 'Chronic_Cost': np.zeros(n_iterations)}
        acute = {'Immediate_Death': np.zeros(n_iterations), 'Immediate_Change': np.zeros(n_iterations),
                 'Immediate_QALY': np.zeros(n_iterations), 'Immediate_Cost': np.zeros(n_iterations)}
        if probabilities is not None:
            chronic_df = probabilities[probabilities['TYPE'] == 'C']
            probability = 1
            weights = dict()
            for cond in chronic_df['NAME'].unique():
                prob = chronic_df[chronic_df['NAME'] == cond]['PROBABILITY'].mean()
                if prob > 0:
                    weights[cond] = prob
                    probability *= (1 - weights[cond])
            probability = 1 - probability
            if probability > 0:
                for cond in weights:
                    weights[cond] = weights[cond] / probability
                    adverse_df = self.adverse_info[self.adverse_info['NAME'] == cond]
                    for k in chronic:
                        try:
                            chronic[k] = chronic[k] + weights[cond] * generate_triangular(values=dict(
                                adverse_df[adverse_df['PROBABILITY'] == k].mean()), n_iterations=n_iterations)
                        except Exception as e:
                            print(k, adverse_df[adverse_df['PROBABILITY'] == k].mean(), e)
            chronic['probability'] = probability*np.ones(n_iterations)
            acute_df = probabilities[probabilities['TYPE'] == 'A']
            probability = 1
            weights = dict()
            for cond in acute_df['NAME'].unique():
                prob = acute_df[acute_df['NAME'] == cond]['PROBABILITY'].mean()
                if prob > 0:
                    weights[cond] = prob
                    probability *= (1 - weights[cond])
            probability = 1 - probability
            if probability > 0:
                for cond in weights:
                    if probability > 0:
                        weights[cond] = weights[cond] / probability
                    adverse_df = self.adverse_info[self.adverse_info['NAME'] == cond]
                    for k in acute:
                        try:
                            acute[k] += weights[cond] * generate_triangular(values=dict(
                                adverse_df[adverse_df['PROBABILITY'] == k].mean()), n_iterations=n_iterations)
                        except Exception as e:
                            print(cond, k, adverse_df[adverse_df['PROBABILITY'] == k].mean(), acute[k], e)
            acute['probability'] = probability*np.ones(n_iterations)
        else:
            chronic['probability'] = np.zeros(n_iterations)
            acute['probability'] = np.zeros(n_iterations)
        random_values['adverse'] = {'chronic': chronic, 'acute': acute}
        random_values['adherent_month_cost'] = generate_triangular(values=self.medications[medication_name][
            'values']['adherent_month_cost']['ALL'], n_iterations=n_iterations)
        random_values['test_cost'] = generate_triangular(values=self.medications[medication_name][
            'values']['test_cost']['ALL'], n_iterations=n_iterations)
        random_values['test_cost_high_charge'] = generate_triangular(values=self.medications[medication_name][
            'values']['test_cost_high_charge']['ALL'], n_iterations=n_iterations)
        efficiency = dict()
        eff_dict = self.medications[medication_name]['values']['efficiency']
        for cat in eff_dict:
            efficiency[cat] = generate_triangular(values=eff_dict[cat], n_iterations=n_iterations)
        random_values['efficiency'] = efficiency
        self.random_values[medication_name] = random_values
