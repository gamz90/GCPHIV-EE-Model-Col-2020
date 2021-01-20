import math
import random
import pandas as pd
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
        self.results = list()

    def load_medication(self, medication_name: str):
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

    def initial_state(self, value: str = 'BASE_VALUE'):
        n_tests = math.floor(random.random() * 3)
        if n_tests == 3:
            n_tests -= 1
        random_value = random.random()
        for age in self.age_groups:
            for sex in self.sexes:
                random_value -= self.general_info['values']['incidence'][age][sex][value]
                if random_value < 0.0:
                    month_age = int(age[-2:]) * 60 + math.floor(random.random() * 60)
                    return {'tests': n_tests, 'chronic': 0, 'acute': False, 'age': month_age,
                            'sex': sex, 'treatment': 2}
        month_age = int(self.age_groups[len(self.age_groups) - 1][-2:]) * 60 + math.floor(random.random() * 60)
        return {'tests': n_tests, 'chronic': 0, 'acute': False, 'age': month_age,
                'sex': self.sexes[len(self.sexes) - 1], 'treatment': 2}

    def calculate_probabilities(self, medication_name: str, scenario: dict):
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

    def simulate_medication(self, medication_name: str, consolidated_results: list, scenario: dict,
                            adverse_information: dict):
        states = list()
        state = self.initial_state(value=scenario.get('incidence', 'BASE_VALUE'))
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group, chronic=0,
                                             acute=False, scenario=scenario, step_length=2,
                                             adverse_information=adverse_information, medication_name=medication_name)
        state['qaly'] = qaly_cost['qaly']
        state['cost'] = qaly_cost['cost']
        states.append(state)
        while states[len(states) - 1]['tests'] not in self.leaving_states:
            states.append(self.simulate_step(state=states[len(states) - 1], medication_name=medication_name,
                                             scenario=scenario, adverse_information=adverse_information))
        change_reason = states[len(states)-1]['tests']
        if change_reason != 'dead':
            states[len(states)-1]['cost'] += self.general_info['values']['switch_cost']['ALL'][
                'ALL'][scenario.get('switch_cost', 'BASE_VALUE')]
#        while states[len(states)-1]['tests'] != 'dead':
#            states.append(self.simulate_change(state=states[len(states)-1], scenario=scenario,
#                                               adverse_information=adverse_information))
        states = pd.DataFrame.from_dict(data=states, orient='columns')
        states['exit_reason'] = change_reason
        states.set_index(keys='treatment', inplace=True)
        consolidated_results.append(states)

    def simulate_step(self, state: dict, medication_name: str, adverse_information: dict, scenario: dict):
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
                                                 medication_name=medication_name)
            result_state['qaly'] = qaly_cost['qaly']
            result_state['cost'] = qaly_cost['cost']
            return result_state
        if state['chronic'] == 0:  # Chronic event can occur
            occurrence = 1 - (1 - adverse_information['chronic']['probability']) ** time_for_step
            if random.random() < occurrence:  # Chronic reaction occurs
                # Immediate change death due to chronic reaction
                result_state['chronic'] = 1
                if random.random() < adverse_information['chronic']['Immediate_Death']:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0),
                                                         age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'],
                                                         scenario=scenario,
                                                         step_length=time_for_step,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    result_state['qaly'] = qaly_cost['qaly']
                    result_state['cost'] = qaly_cost['cost']
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
                                                             step_length=time_for_step,
                                                             adverse_information=adverse_information,
                                                             medication_name=medication_name)
                        result_state['qaly'] = qaly_cost['qaly']
                        result_state['cost'] = qaly_cost['cost']
                        return result_state
        else:
            death = 1-(1 - adverse_information['chronic']['Chronic_Death'])**time_for_step
            if random.random() < death:  # Death due to chronic reaction
                result_state['tests'] = 'dead'
                result_state['age'] += time_for_step / 2
                result_state['treatment'] += time_for_step / 2
                qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                     chronic=(state['chronic'] + result_state['chronic']),
                                                     acute=result_state['acute'], scenario=scenario,
                                                     step_length=time_for_step, adverse_information=adverse_information,
                                                     medication_name=medication_name)
                result_state['qaly'] = qaly_cost['qaly']
                result_state['cost'] = qaly_cost['cost']
                return result_state
            else:
                change = 1 - (1 - adverse_information['chronic']['Chronic_Change'])**time_for_step
                if random.random() < change:  # Death due to chronic reaction
                    result_state['tests'] = 'adverse_reaction'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], scenario=scenario,
                                                         step_length=time_for_step,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    result_state['qaly'] = qaly_cost['qaly']
                    result_state['cost'] = qaly_cost['cost']
                    return result_state
        if (not state['acute']) and result_state['chronic'] == state['chronic']:  # Acute event can occur
            occurrence = 1 - (1 - adverse_information['acute']['probability']) ** time_for_step
            if random.random() < occurrence:
                result_state['acute'] = True  # Acute reaction occurs
                # Death due to acute reaction
                if random.random() < adverse_information['acute']['Immediate_Death']:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], scenario=scenario,
                                                         step_length=time_for_step,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    result_state['qaly'] = qaly_cost['qaly']
                    result_state['cost'] = qaly_cost['cost']
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
                                                             step_length=time_for_step,
                                                             adverse_information=adverse_information,
                                                             medication_name=medication_name)
                        result_state['qaly'] = qaly_cost['qaly']
                        result_state['cost'] = qaly_cost['cost']
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
            efficiency = efficiency_dict['ALL'][scenario.get('efficiency', 'BASE_VALUE')]
        efficiency = 1 - (1-efficiency)**time_for_step
        result_state['tests'] = 0 if random.random() < efficiency else result_state['tests'] + 1

        if result_state['tests'] > 2 or (result_state['tests'] == 2):
            result_state['tests'] = 'failure'
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(state['tests'] > 0), age=age_group,
                                             chronic=(state['chronic']+result_state['chronic']),
                                             acute=result_state['acute'], scenario=scenario,
                                             step_length=time_for_step, adverse_information=adverse_information,
                                             medication_name=medication_name)
        else:
            qaly_cost = self.calculate_qaly_cost(viral_charge=bool(result_state['tests'] > 0), age=age_group,
                                             chronic=(state['chronic'] + result_state['chronic']),
                                             acute=result_state['acute'], scenario=scenario,
                                             step_length=time_for_step, adverse_information=adverse_information,
                                             medication_name=medication_name)
        result_state['qaly'] = qaly_cost['qaly']
        result_state['cost'] = qaly_cost['cost']
        return result_state

    def calculate_qaly_cost(self, viral_charge: bool, age: str, chronic: int, acute: bool, step_length: int,
                            adverse_information: dict, medication_name: str, scenario: dict):
        """
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
        qaly = self.general_info['values']['base_qaly'][age]['ALL'][scenario.get('base_qaly', 'BASE_VALUE')]
        cost = self.medications[medication_name]['values']['test_cost']['ALL'][scenario.get('test_cost', 'BASE_VALUE')]
        if viral_charge:
            qaly *= self.general_info['values']['high_test_qaly']['ALL']['ALL'][scenario.get('high_test_qaly',
                                                                                                    'BASE_VALUE')]
            cost += self.medications[medication_name]['values']['test_cost_high_charge']['ALL'][
                scenario.get('test_cost', 'BASE_VALUE')]
        cost += self.medications[medication_name]['values']['adherent_month_cost']['ALL'][
                    scenario.get('month_cost', 'BASE_VALUE')]*step_length
        if step_length > 6:
            cost = cost*2
        if chronic == 1:
            chronic_months = step_length/2-1 if (step_length in [2, 6]) else 0
            qaly_chronic_immediate = qaly*(1-adverse_information['chronic']['Immediate_QALY'])
            qaly_chronic_long_term = qaly * (1 - adverse_information['chronic']['Chronic_QALY'])
            qaly = (qaly*step_length/2+qaly_chronic_immediate+qaly_chronic_long_term*chronic_months)
            cost += adverse_information['chronic']['Immediate_Cost'] + adverse_information['chronic']['Chronic_Cost'] *\
                    chronic_months
            return {'qaly': qaly, 'cost': cost}
        elif chronic == 2:
            qaly = qaly * (1 - adverse_information['chronic']['Chronic_QALY'])
            cost += adverse_information['chronic']['Chronic_Cost'] * step_length
        if acute:
            qaly_acute_immediate = qaly * (1 - adverse_information['acute']['Immediate_QALY'])
            qaly = qaly*(step_length-1)+qaly_acute_immediate
            cost += adverse_information['acute']['Immediate_Cost']
        return {'qaly': qaly, 'cost': cost}

    def parallel_simulation(self, medication_name: str, scenario: dict, group: str = 'Unique',
                            n_simulations: int = 1000):
        if medication_name not in self.medications.keys():
            self.load_medication(medication_name)
        if 'switch_phase' not in self.medications.keys():
            self.load_medication('switch_phase')
        adverse_information = self.calculate_probabilities(medication_name=medication_name, scenario=scenario)
        cores = multiprocessing.cpu_count() - 1
        manager = multiprocessing.Manager()
        return_list = manager.list()
        args = list()
        for i in range(n_simulations):
            args.append((medication_name, return_list, scenario, adverse_information))
        with multiprocessing.Pool(processes=cores) as pool:
            pool.starmap(self.simulate_medication, args)
        result_list = list()
        for i in range(n_simulations):
            df = return_list[i]
            result_list.append({'medication_name': medication_name, 'qaly': df.qaly.sum(), 'costs': df.cost.sum(),
                                'acute_events': len(df[df.acute]), 'chronic_event': df.chronic.max(),
                                'exit_reason': df.exit_reason.unique()[0]})
        return_list = pd.DataFrame(result_list)
        return_list.reset_index(drop=False, inplace=True)
        return_list.rename(columns={'index': 'iteration'}, inplace=True)
        return_list.iteration = return_list.iteration+1
        return_list['group'] = group
        self.results.append(return_list)

    def simulate_change(self, state: dict, adverse_information: dict, scenario: dict):
        medication_name = 'switch_phase'
        result_state = state.copy()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(min(age_group, 16)) if age_group >= 10 else 'e0' + str(age_group)
        time_for_step = 12
        result_state['acute'] = False
        death_prob = 1 - (1 - self.general_info['values']['month_d_r'][age_group][state['sex']][
            scenario.get('month_d_r', 'BASE_VALUE')]) ** time_for_step
        # GENERAL DEATH
        if random.random() < death_prob:
            result_state['tests'] = 'dead'
            result_state['age'] += time_for_step / 2
            result_state['treatment'] += time_for_step / 2
            qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                 chronic=(state['chronic'] + result_state['chronic']),
                                                 acute=result_state['acute'], scenario=scenario,
                                                 step_length=time_for_step, adverse_information=adverse_information,
                                                 medication_name=medication_name)
            result_state['qaly'] = qaly_cost['qaly']*self.general_info['values']['switch_qaly']['ALL']['ALL'][
                scenario.get('base_qaly', 'BASE_VALUE')]
            result_state['cost'] = qaly_cost['cost']
            return result_state

        if state['chronic'] == 0:  # Chronic event can occur
            occurrence = 1 - (1 - adverse_information['chronic']['probability']) ** time_for_step
            if random.random() < occurrence:  # Chronic reaction occurs
                # Immediate change death due to chronic reaction
                result_state['chronic'] = 1
                if random.random() < adverse_information['chronic']['Immediate_Death']:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=True,
                                                         age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'],
                                                         scenario=scenario,
                                                         step_length=time_for_step,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    result_state['qaly'] = qaly_cost['qaly']
                    result_state['cost'] = qaly_cost['cost']
                    return result_state
        else:
            death = 1 - (1 - adverse_information['chronic']['Chronic_Death']) ** time_for_step
            if random.random() < death:  # Death due to chronic reaction
                result_state['tests'] = 'dead'
                result_state['age'] += time_for_step / 2
                result_state['treatment'] += time_for_step / 2
                qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                     chronic=(state['chronic'] + result_state['chronic']),
                                                     acute=result_state['acute'], scenario=scenario,
                                                     step_length=time_for_step, adverse_information=adverse_information,
                                                     medication_name=medication_name)
                result_state['qaly'] = qaly_cost['qaly']
                result_state['cost'] = qaly_cost['cost']
                return result_state
        if (not state['acute']) and result_state['chronic'] == state['chronic']:  # Acute event can occur
            occurrence = 1 - (1 - adverse_information['acute']['probability']) ** time_for_step
            if random.random() < occurrence:
                result_state['acute'] = True  # Acute reaction occurs
                # Death due to acute reaction
                if random.random() < adverse_information['acute']['Immediate_Death']:
                    result_state['tests'] = 'dead'
                    result_state['age'] += time_for_step / 2
                    result_state['treatment'] += time_for_step / 2
                    qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                                         chronic=(state['chronic'] + result_state['chronic']),
                                                         acute=result_state['acute'], scenario=scenario,
                                                         step_length=time_for_step,
                                                         adverse_information=adverse_information,
                                                         medication_name=medication_name)
                    result_state['qaly'] = qaly_cost['qaly']
                    result_state['cost'] = qaly_cost['cost']
                    return result_state
        result_state['age'] += time_for_step
        result_state['treatment'] += time_for_step
        qaly_cost = self.calculate_qaly_cost(viral_charge=True, age=age_group,
                                             chronic=(state['chronic'] + result_state['chronic']),
                                             acute=result_state['acute'], scenario=scenario,
                                             step_length=time_for_step, adverse_information=adverse_information,
                                             medication_name=medication_name)
        result_state['qaly'] = qaly_cost['qaly']
        result_state['cost'] = qaly_cost['cost']
        return result_state

    def export_results(self, file_name: str = 'results'):
        pd.concat(self.results).to_csv(DIR_OUTPUT + file_name + '.csv', index=False)
