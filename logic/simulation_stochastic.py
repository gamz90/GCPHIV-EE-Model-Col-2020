import math
import random
import pandas as pd
import numpy as np
from root import DIR_INPUT, DIR_OUTPUT
import multiprocessing


def recurrent_assignation(info_df: pd.DataFrame):
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
    levels = dictionary['levels']
    cur_dict = dictionary['values']
    for level in levels:
        cur_dict = cur_dict[params_to_look.get(level, 'ALL')]
    return cur_dict[value]


def generate_triangular(values: dict, n_iterations: int):
    if values['INF_VALUE'] != values['SUP_VALUE']:
        return np.random.triangular(values['INF_VALUE'], values['BASE_VALUE'], values['SUP_VALUE'], n_iterations)
    else:
        return np.ones(n_iterations)*values['BASE_VALUE']


class Model(object):
    def __init__(self):
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
        self.leaving_states = ['dead', 'failure', 'adverse_reaction']
        self.adverse_info = pd.read_csv(DIR_INPUT + 'adverse_general.csv')
        self.random_values = None

    def load_medication(self, medication_name: str):
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

    def calculate_qaly_cost(self, viral_charge: bool, age: str, chronic: int, acute: bool, step_length: int,
                            i: int, main_medication: bool = True):
        """
        :param main_medication:
        :param viral_charge: boolean value indicating if patient has a high viral charge
        :param age: quinquennial age group of the patient
        :param chronic: trinary variable 0 if no chronic event, 1 if developed during period and 2 if it was a previous
        :param acute: boolean variable indicating the occurrence of an acute event
        :param step_length: duration of the step to calculate related qaly
        :param i: iteration from where to obtain the random values
        :param main_medication: boolean recording if the study is a patient in main phase or after switch
        :return: dictionary with associated qaly value and cost
        """
        qaly = self.random_values['base_qaly'][age][i]/12
        if np.isnan(qaly):
            print('base_qaly')
        cost = self.random_values['test_cost'][i]
        adverse = self.random_values['adverse'] if main_medication else self.random_values['switch_phase']
        if viral_charge:
            qaly *= self.random_values['high_test_qaly'][i]
            if np.isnan(qaly):
                print('high_test_qaly')
            cost += self.random_values['test_cost_high_charge'][i]
        if step_length > 6:
            cost = cost * 2
        cost += self.random_values['adherent_month_cost'][i] * step_length
        if chronic == 1:
            chronic_months = step_length / 2 - 1 if (step_length in [2, 6]) else 0
            qaly_chronic_immediate = qaly * (1 - adverse['chronic']['Immediate_QALY'][i])
            if np.isnan(qaly_chronic_immediate):
                print('Chronic,Immediate_QALY')
            qaly_chronic_long_term = qaly * (1 - adverse['chronic']['Chronic_QALY'][i])
            if np.isnan(qaly_chronic_long_term):
                print('Chronic,Chronic_QALY')
            qaly = (qaly * step_length / 2 + qaly_chronic_immediate + qaly_chronic_long_term * chronic_months)
            cost += adverse['chronic']['Immediate_Cost'][i] + adverse['chronic']['Chronic_Cost'][i] * \
                    chronic_months
            return {'qaly': qaly, 'cost': cost}
        elif chronic == 2:
            qaly = qaly * (1 - adverse['chronic']['Chronic_QALY'][i])
            if np.isnan(qaly):
                print('Chronic,Chronic_QALY')
            cost += adverse['chronic']['Chronic_Cost'][i] * step_length
        if acute:
            qaly_acute_immediate = qaly * (1 - adverse['acute']['Immediate_QALY'][i])
            if np.isnan(qaly_acute_immediate):
                print('acute,Immediate_QALY', adverse['acute']['Immediate_QALY'][i])
            qaly = qaly * (step_length - 1) + qaly_acute_immediate
            cost += adverse['acute']['Immediate_Cost'][i]
        return {'qaly': qaly, 'cost': cost}

    def initial_state(self):
        """
        :return: Initial state of simulation
        """
        n_tests = math.floor(random.random() * 3)
        if n_tests == 3:
            n_tests -= 1
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

    def simulate_step(self, state: dict, i: int, inflation_rate: float = 1.0):
        result_state = state.copy()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        time_for_step = 2 if state['tests'] > 0 else 6
        death_prob = 1 - (1 - self.random_values['month_d_r'][age_group][state['sex']][i]) ** time_for_step
        # GENERAL DEATH
        result_state['acute'] = False
        if random.random() < death_prob:
            result_state['tests'] = 'dead'
            result_state['age'] += time_for_step / 2
            result_state['treatment'] += time_for_step / 2
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step)
            result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
            result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
            return result_state
        #  Chronic event can occur
        if state['chronic'] == 0:
            occurrence = 1 - (1 - self.random_values['adverse']['chronic']['probability'][i]) ** time_for_step
            if random.random() < occurrence:  # Chronic reaction occurs
                # Immediate change death due to chronic reaction
                result_state['chronic'] = 1
                if random.random() < self.random_values['adverse']['chronic']['Immediate_Death'][i]:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0),
                                                         age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step)
                    result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
                    result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
                    return result_state
                else:
                    # Immediate change due to chronic reaction
                    if random.random() < self.random_values['adverse']['chronic']['Immediate_Change'][i]:
                        result_state['tests'] = 'adverse_reaction'
                        result_state['age'] += time_for_step / 2
                        result_state['treatment'] += time_for_step / 2
                        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                             chronic=(state['chronic'] + result_state['chronic']),
                                                             acute=result_state['acute'], i=i,
                                                             step_length=time_for_step)
                        result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
                        result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
                        return result_state
        else:
            death = 1 - (1 - self.random_values['adverse']['chronic']['Chronic_Death'][i]) ** time_for_step
            if random.random() < death:  # Death due to chronic reaction aftermath
                result_state['tests'] = 'dead'
                result_state['age'] += time_for_step / 2
                result_state['treatment'] += time_for_step / 2
                qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                chronic=(state['chronic'] + result_state['chronic']),
                                                acute=result_state['acute'], i=i, step_length=time_for_step)
                result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
                result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
                return result_state
            else:
                change = 1 - (1 - self.random_values['adverse']['chronic']['Chronic_Change'][i]) ** time_for_step
                if random.random() < change:  # Death due to chronic reaction aftermath
                    result_state['tests'] = 'adverse_reaction'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step)
                    result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
                    result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
                    return result_state
        if (not state['acute']) and result_state['chronic'] == state['chronic']:  # Acute event can occur
            occurrence = 1 - (1 - self.random_values['adverse']['acute']['probability'][i]) ** time_for_step
            if random.random() < occurrence:
                result_state['acute'] = True  # Acute reaction occurs
                # Death due to acute reaction
                if random.random() < self.random_values['adverse']['acute']['Immediate_Death'][i]:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step)
                    result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
                    result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
                    return result_state
                else:
                    # Change due to acute reaction
                    if random.random() < self.random_values['adverse']['acute']['Immediate_Change'][i]:
                        result_state['tests'] = 'adverse_reaction'
                        result_state['age'] += time_for_step / 2
                        result_state['treatment'] += time_for_step / 2
                        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                             chronic=(state['chronic'] + result_state['chronic']),
                                                             acute=result_state['acute'], i=i,
                                                             step_length=time_for_step)
                        result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
                        result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
                        return result_state
        result_state['age'] += time_for_step
        result_state['treatment'] += time_for_step
        efficiency_dict = self.random_values['efficiency']
        efficiency = 0
        if len(efficiency_dict) > 1:
            for ef in efficiency_dict:
                if int(ef) <= int(state['treatment']):
                    efficiency = efficiency_dict[ef][i]
        else:
            efficiency = efficiency_dict['ALL'][i]
        efficiency = 1 - (1 - efficiency) ** time_for_step
        result_state['tests'] = 0 if random.random() < efficiency else result_state['tests'] + 1

        if result_state['tests'] >= 2:
            result_state['tests'] = 'failure'
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step)
        else:
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(result_state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step)
        if np.isnan(qaly_cost['qaly']):
            print(state, i, qaly_cost['qaly'], qaly_cost)
        result_state['qaly'] = float(qaly_cost['qaly'])/(inflation_rate**state['treatment'])
        result_state['cost'] = float(qaly_cost['cost'])/(inflation_rate**state['treatment'])
        return result_state

    def simulate_change(self, state: dict, i: int, inflation_rate: float = 1.0):
        result_state = state.copy()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        time_for_step = 12
        result_state['acute'] = False
        death_prob = 1 - (1 - self.random_values['month_d_r'][age_group][state['sex']][i]) ** time_for_step
        # GENERAL DEATH
        if random.random() < death_prob:
            result_state['tests'] = 'dead'
            result_state['age'] += time_for_step / 2
            result_state['treatment'] += time_for_step / 2
            qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], i=i, step_length=time_for_step,
                                                 main_medication=False)
            result_state['qaly'] = qaly_cost['qaly'] * self.random_values['switch_qaly'][i]/(
                    inflation_rate**state['treatment'])
            result_state['cost'] = qaly_cost['cost']/(inflation_rate**state['treatment'])
            return result_state
        if state['chronic'] == 0:  # Chronic event can occur
            occurrence = 1 - (1 - self.random_values['switch_phase']['chronic']['probability'][i]) ** time_for_step
            if random.random() < occurrence:  # Chronic reaction occurs
                # Immediate change death due to chronic reaction
                result_state['chronic'] = 1
                if random.random() < self.random_values['switch_phase']['chronic']['Immediate_Death'][i]:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step,
                                                         main_medication=False)
                    result_state['qaly'] = qaly_cost['qaly']/(inflation_rate**state['treatment'])
                    result_state['cost'] = qaly_cost['cost']/(inflation_rate**state['treatment'])
                    return result_state
        else:
            death = 1 - (1 - self.random_values['switch_phase']['chronic']['Chronic_Death'][i]) ** time_for_step
            if random.random() < death:  # Death due to chronic reaction
                result_state['tests'] = 'dead'
                result_state['age'] += time_for_step / 2
                result_state['treatment'] += time_for_step / 2
                qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                     chronic=(state['chronic'] + result_state['chronic']),
                                                     acute=result_state['acute'], i=i, step_length=time_for_step,
                                                     main_medication=False)
                result_state['qaly'] = qaly_cost['qaly']/(inflation_rate**state['treatment'])
                result_state['cost'] = qaly_cost['cost']/(inflation_rate**state['treatment'])
                return result_state
        if (not state['acute']) and result_state['chronic'] == state['chronic']:  # Acute event can occur
            occurrence = 1 - (1 - self.random_values['switch_phase']['acute']['probability'][i]) ** time_for_step
            if random.random() < occurrence:
                result_state['acute'] = True  # Acute reaction occurs
                # Death due to acute reaction
                if random.random() < self.random_values['switch_phase']['acute']['Immediate_Death'][i]:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], i=i, step_length=time_for_step,
                                                         main_medication=False)
                    result_state['qaly'] = qaly_cost['qaly']/(inflation_rate**state['treatment'])
                    result_state['cost'] = qaly_cost['cost']/(inflation_rate**state['treatment'])
                    return result_state
        result_state['age'] += time_for_step
        result_state['treatment'] += time_for_step
        qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                             chronic=(state['chronic'] + result_state['chronic']),
                                             acute=result_state['acute'], i=i, step_length=time_for_step,
                                             main_medication=False)
        result_state['qaly'] = qaly_cost['qaly']/(inflation_rate**state['treatment'])
        result_state['cost'] = qaly_cost['cost']/(inflation_rate**state['treatment'])
        return result_state

    def simulate_medication(self, result_list: list, iteration: int, inflation_rate: float,
                            project_switch: bool = False):
        states = list()
        state = self.initial_state()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group, chronic=0,
                                             acute=False, step_length=2, i=iteration)
        state['qaly'] = float(qaly_cost['qaly'])
        state['cost'] = float(qaly_cost['cost'])
        states.append(state)
        while states[len(states) - 1]['tests'] not in self.leaving_states:
            states.append(self.simulate_step(state=states[len(states) - 1], i=iteration, inflation_rate=inflation_rate))
        change_reason = states[len(states) - 1]['tests']
        if change_reason != 'dead':
            states[len(states) - 1]['cost'] += self.random_values['switch_cost'][iteration]
            states[len(states) - 1]['qaly'] *= self.random_values['switch_qaly'][iteration] /\
                                               (inflation_rate**states[len(states) - 1]['treatment'])
            states[len(states) - 1]['cost'] /= inflation_rate**states[len(states) - 1]['treatment']
        if project_switch:
            while states[len(states) - 1]['tests'] != 'dead':
                states.append(self.simulate_change(state=states[len(states) - 1], i=iteration))
        states = pd.DataFrame.from_dict(data=states, orient='columns')
        states['exit_reason'] = change_reason
        result_list.append(states)

    def parallel_simulation(self, medication_name: str, group: str = 'Unique', n_simulations: int = 1000,
                            inflation_rate: float = 1.0, project_switch: bool = False):
        if medication_name not in self.medications.keys():
            self.load_medication(medication_name)
        if 'switch_phase' not in self.medications.keys():
            self.load_medication('switch_phase')
        self.generate_random(n_iterations=n_simulations, medication_name=medication_name)
        cores = multiprocessing.cpu_count()*3
        manager = multiprocessing.Manager()
        return_list = manager.list()
        args = list()
        for i in range(n_simulations):
            args.append((return_list, i, inflation_rate, project_switch))
        with multiprocessing.Pool(processes=cores) as pool:
            pool.starmap(self.simulate_medication, args)
        result_list = list()
        for i in range(n_simulations):
            df = return_list[i]
            result_list.append({'medication_name': medication_name, 'qaly': df.qaly.sum(),
                                'treatment': df.treatment.max(), 'costs': df.cost.sum(),
                                'acute_events': len(df[df.acute]), 'chronic_event': df.chronic.max(),
                                'exit_reason': df.exit_reason.unique()[0]})
        return_list = pd.DataFrame(result_list)
        return_list.reset_index(drop=False, inplace=True)
        return_list.rename(columns={'index': 'iteration'}, inplace=True)
        return_list.iteration = return_list.iteration + 1
        return_list['group'] = group
        return_list['discount_rate'] = '5'
        return_list.to_csv(DIR_OUTPUT + 'results_s_' + medication_name + '.csv', index=False)

    def generate_random(self, n_iterations: int, medication_name: str):
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
        eff_dict = self.medications[medication_name]['values']['test_cost']
        for cat in eff_dict:
            efficiency[cat] = generate_triangular(values=eff_dict[cat], n_iterations=n_iterations)
        random_values['efficiency'] = efficiency
        #  TO DO SWITCH
        self.random_values = random_values
