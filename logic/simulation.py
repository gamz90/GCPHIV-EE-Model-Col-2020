import math
import random
import pandas as pd
from root import DIR_INPUT, DIR_OUTPUT
import datetime as dt
import numpy as np


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


class Model(object):
    def __init__(self):
        """
        Model that simulates the patients receiving any defined ART with a defined set of parameters.
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
        try:
            medication_adverse_df = pd.read_csv(DIR_INPUT + medication_name + '_adverse_probability.csv')
            self.medications[medication_name]['adverse_probability'] = medication_adverse_df
        except:
            self.medications[medication_name]['adverse_probability'] = None

    def initial_state(self, efficiency: float, value: str = 'BASE_VALUE'):
        """
        Create the initial state of the patient, according to the incidence and initial efficiency of the medication.
        :param efficiency: number that indicates the percentage of population that will result in a low viral charge
        due to medication for the first control test.
        :param value: Value for the scenario of the considered incidence.
        :return: Initial state of the patient in the tracked variables.
        """
        if random.random() < efficiency:
            n_tests = 0
        else:
            n_tests = 1
        random_value = random.random()
        for age in self.age_groups:
            for sex in self.sexes:
                random_value -= self.general_info['values']['incidence'][age][sex][value]
                if random_value <= 0.0:
                    month_age = int(age[-2:]) * 60 + math.floor(random.random() * 60)
                    if month_age < 18*12:
                        month_age = 18*12
                    return {'tests': n_tests, 'chronic': 0, 'acute': False, 'age': month_age,
                            'sex': sex, 'treatment': 2}
        month_age = int(self.age_groups[len(self.age_groups) - 1][-2:]) * 60 + math.floor(random.random() * 60)
        return {'tests': n_tests, 'chronic': 0, 'acute': False, 'age': month_age,
                'sex': self.sexes[len(self.sexes) - 1], 'treatment': 2}

    def calculate_probabilities(self, medication_name: str, scenario: dict):
        """
        Calculate the probabilities related to the adverse event fot the medication.
        :param medication_name: Name of the medication scheme to analyse.
        :param scenario: Scenario to consider in for the characteristics.
        :return: Dictionary with the cost, qaly and probabilities related for both acute and chronic reactions.
        """
        probabilities = self.medications[medication_name]['adverse_probability']
        chronic = {'Immediate_Death': 0, 'Immediate_Change': 0, 'Immediate_QALY': 0, 'Immediate_Cost': 0,
                   'Chronic_Death': 0, 'Chronic_Change': 0, 'Chronic_QALY': 0, 'Chronic_Cost': 0}
        acute = {'Immediate_Death': 0, 'Immediate_Change': 0, 'Immediate_QALY': 0, 'Immediate_Cost': 0}
        if probabilities is not None:
            chronic_df = probabilities[probabilities['TYPE'] == 'C']
            probability = 1
            weights = dict()
            for cond in chronic_df['NAME'].unique():
                weights[cond] = chronic_df[chronic_df['NAME'] == cond]['PROBABILITY'].mean()
                probability *= (1 - weights[cond])
            probability = 1-probability
            for cond in chronic_df['NAME'].unique():
                weights[cond] = weights[cond]/probability
                adverse_df = self.adverse_info[self.adverse_info['NAME'] == cond]
                for k in chronic:
                    chronic[k] += weights[cond] *\
                                  adverse_df[adverse_df['PROBABILITY'] == k][scenario.get(k, 'BASE_VALUE')].mean()
            chronic['probability'] = probability
            acute_df = probabilities[probabilities['TYPE'] == 'A']
            probability = 1
            weights = dict()
            for cond in acute_df['NAME'].unique():
                weights[cond] = acute_df[acute_df['NAME'] == cond]['PROBABILITY'].mean()
                probability *= (1 - weights[cond])
            probability = 1 - probability
            for cond in acute_df['NAME'].unique():
                if probability > 0:
                    weights[cond] = weights[cond] / probability
                adverse_df = self.adverse_info[self.adverse_info['NAME'] == cond]
                for k in acute:
                    acute[k] += weights[cond] * \
                                  adverse_df[adverse_df['PROBABILITY'] == k][scenario.get(k, 'BASE_VALUE')].mean()
            acute['probability'] = probability
        else:
            chronic['probability'] = 0
            acute['probability'] = 0
        return {'chronic': chronic, 'acute': acute}

    def simulate_medication(self, medication_name: str, scenario: dict, adverse_information: dict,
                            inflation_rate: float = 1.0, insert_switch: bool = False, max_high_tests: int = 2):
        """
        Model iteration, starting from the moment a patient has his or her first ART control, up to the moment of
        discontinuing the medication.
        :param max_high_tests: maximum value of tests with a high viral load before considering a virological failure
        :param medication_name: Name of the considered medication.
        :param scenario: Scenario for the variables.
        :param adverse_information: Relevant adverse reaction relevant information.
        :param inflation_rate: Discount rate considered.
        :param insert_switch: Boolean indicating if the model inserts future switch monthly costs
        :return: Summary of the simulation of the patient with the synthesized relevant information.
        """
        states = list()
        high_tests = 0
        efficiency_dict = self.medications[medication_name]['values']['efficiency']
        efficiency = efficiency_dict[list(efficiency_dict.keys())[0]][scenario.get('efficiency', 'BASE_VALUE')]
        state = self.initial_state(efficiency=efficiency, value=scenario.get('incidence', 'BASE_VALUE'))
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group, chronic=0,
                                             acute=False, scenario=scenario, step_length=2, dead=False,
                                             adverse_information=adverse_information, medication_name=medication_name)
        for gc in qaly_cost:
            state[gc] = qaly_cost[gc]
        states.append(state)
        while states[len(states) - 1]['tests'] not in self.leaving_states:
            high_tests += 1 if states[len(states) - 1]['tests'] > 0 else 0
            states.append(self.simulate_step(state=states[len(states) - 1], medication_name=medication_name,
                                             scenario=scenario, adverse_information=adverse_information,
                                             inflation_rate=inflation_rate, max_high_tests=max_high_tests))
        change_reason = states[len(states)-1]['tests']
        if change_reason == 'failure':
            high_tests += 1
        month = 0
        if change_reason != 'dead' and change_reason != 'dead_ar':
            chronic = states[len(states)-1]['chronic']
            if insert_switch:
                dead = False
                cost = self.general_info['values']['switch_cost']['ALL']['ALL'][
                            scenario.get('switch_cost', 'BASE_VALUE')]/(inflation_rate **
                                                                        states[len(states) - 1]['treatment'])
                qaly = 0
                while not dead:
                    age_group = math.floor((states[len(states) - 1]['age']+month) / 60)
                    age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
                    death_prob = 1 - (1 - self.general_info['values']['month_d_r'][age_group][state['sex']][
                        scenario.get('month_d_r', 'BASE_VALUE')])
                    # GENERAL DEATH
                    if random.random() < death_prob:
                        dead = True
                    else:
                        cost += self.general_info['values']['switch_month_cost']['ALL']['ALL'][
                            scenario.get('switch_cost', 'BASE_VALUE')]/(inflation_rate **
                                                                              (states[len(states) - 1]['treatment'] +
                                                                               month))
                        if chronic == 1:
                            cost += adverse_information['chronic']['Chronic_Cost'] / (inflation_rate **
                                                                              (states[len(states) - 1]['treatment'] +
                                                                               month))
                        qaly += self.general_info['values']['base_qaly'][age_group]['ALL'][
                                    scenario.get('base_qaly', 'BASE_VALUE')] * self.general_info['values'][
                            'switch_qaly']['ALL']['ALL'][
                            scenario.get('switch_qaly', 'BASE_VALUE')] / (12 * (inflation_rate ** (
                                states[len(states) - 1]['treatment']+month)))
                    if chronic == 1:
                        qaly = qaly * (1 - adverse_information['chronic']['Chronic_QALY'])
                    month += 1
                states[len(states)-1]['total_cost'] += cost
                states[len(states) - 1]['qaly'] += qaly
            else:
                if change_reason == 'failure':
                    states[len(states) - 1]['test_cost'] += self.general_info['values']['switch_cost']['ALL']['ALL'][
                        scenario.get('switch_cost', 'BASE_VALUE')] / inflation_rate ** states[len(states) - 1][
                        'treatment']
                    states[len(states) - 1]['total_cost'] += self.general_info['values']['switch_cost']['ALL']['ALL'][
                                                                scenario.get('switch_cost',
                                                                             'BASE_VALUE')] / inflation_rate ** \
                                                            states[len(states) - 1][
                                                                'treatment']
                states[len(states) - 1]['qaly'] *= self.general_info['values']['switch_qaly']['ALL']['ALL'][
                            scenario.get('switch_qaly', 'BASE_VALUE')]
        states = pd.DataFrame.from_dict(data=states, orient='columns')
        return {'qaly': states.qaly.sum(), 'treatment': states.treatment.max(), 'costs': states.total_cost.sum(),
                'test_cost': states.test_cost.sum(), 'adverse_cost': states.adverse_cost.sum(),
                'medication_cost': states.medication_cost.sum(), 'acute_events': len(states[states.acute]),
                'chronic_event': states.chronic.max(), 'exit_reason': change_reason, 'n_high_tests': high_tests,
                'percentage_high_tests': high_tests/len(states), 'starting_age': states.age.min(),
                'time_to_death': states.treatment.max() + month}

    def simulate_step(self, state: dict, medication_name: str, adverse_information: dict, scenario: dict,
                      inflation_rate: float = 1.0, max_high_tests: int = 2):
        """
        The simulation of the possible events between two tests.
        :param max_high_tests: maximum value of tests with a high viral load before considering a virological failure
        :param state: current state of the patient.
        :param medication_name: medication scheme considered in the simulation.
        :param adverse_information: adverse reactions' information for the specified medication scheme
        :param scenario: values to consider in the scenario for each variable
        :param inflation_rate: discount rate considered to deprecate values
        :return: state of the patient after the considered step.
        """
        result_state = state.copy()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        time_for_step = 2 if state['tests'] > 0 else 6
        death_prob = 1-(1-self.general_info['values']['month_d_r'][age_group][state['sex']][
            scenario.get('month_d_r', 'BASE_VALUE')]) ** time_for_step
        # GENERAL DEATH
        result_state['acute'] = False
        if random.random() < death_prob:
            result_state['tests'] = 'dead'
            result_state['age'] += time_for_step/2
            result_state['treatment'] += time_for_step/2
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], scenario=scenario,
                                                 step_length=time_for_step, adverse_information=adverse_information,
                                                 medication_name=medication_name, dead=True)
            for gc in qaly_cost:
                result_state[gc] = qaly_cost[gc]/(inflation_rate ** state['treatment'])
            return result_state
        if state['chronic'] == 0:  # Chronic event can occur
            occurrence = 1 - (1 - adverse_information['chronic']['probability']) ** time_for_step
            if random.random() < occurrence:  # Chronic reaction occurs
                # Immediate change death due to chronic reaction
                result_state['chronic'] = 1
                if random.random() < adverse_information['chronic']['Immediate_Death']:
                    result_state['tests'] = 'dead_ar'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0),
                                                         age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], dead=True,
                                                         scenario=scenario,
                                                         step_length=time_for_step,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    for gc in qaly_cost:
                        result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                    return result_state
                else:
                    # Change due to chronic reaction
                    if random.random() < adverse_information['chronic']['Immediate_Change']:
                        result_state['tests'] = 'adverse_reaction'
                        result_state['age'] += time_for_step / 2
                        result_state['treatment'] += time_for_step / 2
                        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0),
                                                             age=age_group,
                                                             chronic=(state['chronic'] + result_state['chronic']),
                                                             acute=result_state['acute'],
                                                             scenario=scenario,
                                                             step_length=time_for_step, dead=False,
                                                             adverse_information=adverse_information,
                                                             medication_name=medication_name)
                        for gc in qaly_cost:
                            result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                        return result_state
        else:
            death = 1-(1 - adverse_information['chronic']['Chronic_Death'])**time_for_step
            if random.random() < death:  # Death due to chronic reaction
                result_state['tests'] = 'dead_ar'
                result_state['age'] += time_for_step / 2
                result_state['treatment'] += time_for_step / 2
                qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                     chronic=(state['chronic'] + result_state['chronic']),
                                                     acute=result_state['acute'], scenario=scenario, dead=True,
                                                     step_length=time_for_step, adverse_information=adverse_information,
                                                     medication_name=medication_name)
                for gc in qaly_cost:
                    result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                return result_state
            else:
                change = 1 - (1 - adverse_information['chronic']['Chronic_Change'])**time_for_step
                if random.random() < change:  # Change due to chronic reaction
                    result_state['tests'] = 'adverse_reaction'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], scenario=scenario,
                                                         step_length=time_for_step, dead=False,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    for gc in qaly_cost:
                        result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                    return result_state
        if (not state['acute']) and result_state['chronic'] == state['chronic']:  # Acute event can occur
            occurrence = 1 - (1 - adverse_information['acute']['probability']) ** time_for_step
            if random.random() < occurrence:
                result_state['acute'] = True  # Acute reaction occurs
                # Death due to acute reaction
                if random.random() < adverse_information['acute']['Immediate_Death']:
                    result_state['tests'] = 'dead_ar'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], scenario=scenario,
                                                         step_length=time_for_step, dead=True,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    for gc in qaly_cost:
                        result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                    return result_state
                else:
                    # Change due to acute reaction
                    if random.random() < adverse_information['acute']['Immediate_Change']:
                        result_state['tests'] = 'adverse_reaction'
                        result_state['age'] += time_for_step / 2
                        result_state['treatment'] += time_for_step / 2
                        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                             chronic=(state['chronic'] + result_state['chronic']),
                                                             acute=result_state['acute'], scenario=scenario,
                                                             step_length=time_for_step, dead=False,
                                                             adverse_information=adverse_information,
                                                             medication_name=medication_name)
                        for gc in qaly_cost:
                            result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
                        return result_state
        result_state['age'] += time_for_step
        result_state['treatment'] += time_for_step
        efficiency_dict = self.medications[medication_name]['values']['efficiency']
        efficiency = 0
        if len(efficiency_dict) > 1:
            for ef in efficiency_dict:
                if int(ef) <= int(state['treatment']):
                    efficiency = efficiency_dict[ef][scenario.get('efficiency', 'BASE_VALUE')]
        else:
            efficiency = efficiency_dict[list(efficiency_dict.keys())[0]][scenario.get('efficiency', 'BASE_VALUE')]
        result_state['tests'] = 0 if random.random() < efficiency else result_state['tests'] + 1

        if result_state['tests'] >= max_high_tests:
            result_state['tests'] = 'failure'
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                             chronic=(state['chronic']+result_state['chronic']),
                                             acute=result_state['acute'], scenario=scenario, dead=False,
                                             step_length=time_for_step, adverse_information=adverse_information,
                                             medication_name=medication_name)
        else:
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(result_state['tests'] > 0), age=age_group,
                                             chronic=(state['chronic'] + result_state['chronic']),
                                             acute=result_state['acute'], scenario=scenario, dead=False,
                                             step_length=time_for_step, adverse_information=adverse_information,
                                             medication_name=medication_name)
        for gc in qaly_cost:
            result_state[gc] = qaly_cost[gc] / (inflation_rate ** state['treatment'])
        return result_state

    def calculate_qaly_cost(self, viral_charge: bool, age: str, chronic: int, acute: bool, dead: bool, step_length: int,
                            adverse_information: dict, medication_name: str, scenario: dict):
        """
        :param dead: boolean that indicates if the patient died during the period
        :param viral_charge: boolean value indicating if patient has a high viral charge
        :param age: quinquennial age group of the patient
        :param chronic: trinary variable 0 if no chronic event, 1 if developed during period and 2 if it was a previous
        :param acute: boolean variable indicating the occurrence of an acute event
        :param step_length: duration of the step to calculate related qaly
        :param adverse_information: dictionary that contains the information for the adverse conditions considered
        :param medication_name: name of medication to recollect specific information
        :param scenario: scenario of chosen information.
        :return: dictionary with associated qaly value and cost
        """
        result_vector = {'qaly': self.general_info['values']['base_qaly'][age]['ALL'][scenario.get('base_qaly',
                                                                                                   'BASE_VALUE')] / 12,
                         'test_cost': self.medications[medication_name]['values']['test_cost']['ALL'][scenario.get(
                             'test_cost', 'BASE_VALUE')], 'adverse_cost': 0, 'medication_cost': 0, 'total_cost': 0}
        if np.isnan(result_vector['qaly']) or result_vector['qaly'] > 1:
            print('base_qaly')

        if viral_charge:
            result_vector['qaly'] = result_vector['qaly']*self.general_info['values']['high_test_qaly']['ALL'][
                'ALL'][scenario.get('high_test_qaly', 'BASE_VALUE')]
            if np.isnan(result_vector['qaly']) or result_vector['qaly'] > 1:
                print('high_test_qaly')
            result_vector['test_cost'] = result_vector['test_cost'] + self.medications[medication_name]['values'][
                'test_cost_high_charge']['ALL'][scenario.get('test_cost', 'BASE_VALUE')]
        result_vector['medication_cost'] = self.medications[medication_name]['values']['adherent_month_cost']['ALL'][
                    scenario.get('month_cost', 'BASE_VALUE')]*(step_length/2 if dead else step_length)
        if chronic == 1:
            result_vector['adverse_cost'] = adverse_information['chronic']['Immediate_Cost']
            if dead:
                pass
            else:
                chronic_months = step_length/2-1
                qaly_chronic_immediate = result_vector['qaly']*(1-adverse_information['chronic']['Immediate_QALY'])
                if np.isnan(qaly_chronic_immediate) or qaly_chronic_immediate > 1:
                    print(medication_name, 'qaly_chronic_immediate')
                qaly_chronic_long_term = result_vector['qaly'] * (1 - adverse_information['chronic']['Chronic_QALY'])
                if np.isnan(qaly_chronic_long_term) or qaly_chronic_long_term > 1:
                    print(medication_name, 'qaly_chronic_long_term')
                result_vector['qaly'] = (result_vector['qaly']*step_length/2+qaly_chronic_immediate +
                                         qaly_chronic_long_term*chronic_months)
                result_vector['adverse_cost'] = result_vector['adverse_cost'] + adverse_information['chronic'][
                    'Chronic_Cost'] * chronic_months
            result_vector['total_cost'] = result_vector['test_cost'] + result_vector['adverse_cost'] + result_vector[
                'medication_cost']
            return result_vector
        elif chronic == 2:
            result_vector['qaly'] = result_vector['qaly'] * (1 - adverse_information['chronic']['Chronic_QALY'])
            if np.isnan(result_vector['qaly']) or result_vector['qaly'] > 1:
                print(medication_name, 'qaly_chronic_long_term')
            result_vector['adverse_cost'] = adverse_information['chronic']['Chronic_Cost'] * (step_length/2 if dead
                                                                                              else step_length)
        if acute:
            qaly_acute_immediate = 0 if dead else result_vector['qaly'] * (1 - adverse_information['acute'][
                'Immediate_QALY'])
            if np.isnan(qaly_acute_immediate) or qaly_acute_immediate > 1:
                print(medication_name, 'qaly_acute_immediate')
            result_vector['qaly'] = result_vector['qaly']*((step_length/2 if dead else step_length)-1) + \
                                    qaly_acute_immediate
            result_vector['adverse_cost'] = result_vector['adverse_cost'] + adverse_information['acute'][
                'Immediate_Cost']
        else:
            result_vector['qaly'] = result_vector['qaly'] * (step_length/2 if dead else step_length)
        result_vector['total_cost'] = result_vector['test_cost'] + result_vector['adverse_cost'] + result_vector[
            'medication_cost']
        return result_vector

    def parallel_simulation(self, medication_name: str, scenario: dict, group: str = 'Unique',
                            n_simulations: int = 1000, inflation_rate: tuple = ('None', 1.0),
                            insert_switch: bool = False, max_high_tests: int = 2):
        """
        Process that accumulates the results of the considered simulations from the given scenario.
        :param max_high_tests: maximum amount of tests with a high viral load before declaring virological failure
        :param medication_name: Medication scheme received by the patient. Must be a str
        :param scenario: Specified scenario for each of the considered variables of the model, those that are not
        specified will be considered in base value.
        :param group: group to which the medication scheme belongs.
        :param n_simulations: number of simulations for the model.
        :param inflation_rate: discount rate considered.
        :param insert_switch: Boolean indicating if the model inserts future switch monthly costs
        """
        if medication_name not in self.medications.keys():
            self.load_medication(medication_name)
        adverse_information = self.calculate_probabilities(medication_name=medication_name, scenario=scenario)
        result_list = list()
        for i in range(n_simulations):
            result_list.append(self.simulate_medication(medication_name=medication_name, scenario=scenario,
                                          adverse_information=adverse_information, inflation_rate=inflation_rate[1],
                                          insert_switch=insert_switch, max_high_tests=max_high_tests))
        return_list = pd.DataFrame(result_list)
        return_list.reset_index(drop=False, inplace=True)
        return_list.rename(columns={'index': 'iteration'}, inplace=True)
        return_list.iteration = return_list.iteration+1
        return_list['group'] = group
        return_list['discount_rate'] = inflation_rate[0]
        return_list['switch_cost'] = scenario['switch_cost']
        return_list['medication_name'] = medication_name
        return_list['treatment'] = return_list['treatment']/12
        return_list['time_to_death'] = return_list['time_to_death']/12
        return_list.sort_values(by='starting_age', ascending=True, inplace=True)
        file_name = DIR_OUTPUT + 'results_' + medication_name + '_'+inflation_rate[0] + '_' + scenario['switch_cost']
        if insert_switch:
            file_name += '_future_lines'
        return_list.to_csv(file_name + '.csv', index=False)
        print(medication_name, scenario, inflation_rate, insert_switch, dt.datetime.now())
