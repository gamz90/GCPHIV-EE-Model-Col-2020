import math
import random
import pandas as pd
from root import DIR_INPUT


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

    def load_medication(self, medication_name: str):
        medication_df = pd.read_csv(DIR_INPUT + medication_name + '.csv')
        columns = list(medication_df.columns)
        columns.remove('BASE_VALUE')
        columns.remove('INF_VALUE')
        columns.remove('SUP_VALUE')
        self.medications[medication_name] = {'values': recurrent_assignation(medication_df), 'levels': columns}
        medication_adverse_df = pd.read_csv(DIR_INPUT + medication_name + '_adverse.csv')
        self.medications[medication_name]['adverse'] = medication_adverse_df

    def initial_state(self, value: str = 'BASE_VALUE'):
        n_tests = math.floor(random.random() * 3)
        if n_tests == 3:
            n_tests -= 1
        adherent = 1 if random.random() < 0.5 else 0
        random_value = random.random()
        for age in self.age_groups:
            for sex in self.sexes:
                random_value -= self.general_info['values']['incidence']['ALL']['ALL']['ALL'][age][sex][value]
                if random_value < 0.0:
                    month_age = int(age[-2:]) * 60 + math.floor(random.random() * 60)
                    return {'tests': n_tests, 'adherent': adherent, 'chronic': 0, 'acute': 0, 'age': month_age,
                            'sex': sex, 'treatment': 2}
        month_age = int(self.age_groups[len(self.age_groups) - 1][-2:]) * 60 + math.floor(random.random() * 60)
        return {'tests': n_tests, 'adherent': adherent, 'chronic': 0, 'acute': 0, 'age': month_age,
                'sex': self.sexes[len(self.sexes) - 1], 'treatment': 2}

    def simulate_medication(self, medication_name: str):
        if medication_name not in self.medications.keys():
            self.load_medication(medication_name)
        states = list()
        states.append(self.initial_state())
        print('INITIAL STATE: ', states[0])
        while states[len(states) - 1]['tests'] not in self.leaving_states:
            states.append(self.simulate_step(states[len(states) - 1], medication_name))
        for i in range(1, len(states)):
            print(states[i])
        return states

    def simulate_step(self, state: dict, medication_name: str):
        result_state = state.copy()
        age_group = math.floor(state['age'] / 60)
        age_group = 'e' + str(age_group) if age_group >= 10 else 'e0' + str(age_group)
        # GENERAL DEATH
        if random.random() < get_info(self.general_info, {'EVENT': 'month_d_r', 'AGE': age_group, 'SEX': state['sex']}):
            result_state['tests'] = 'dead'
            return result_state
        result_state['adherent'] = 1 if random.random() < get_info(self.medications[medication_name],
                                                                   {'EVENT': 'p_ad', 'ADHERENT':
                                                                       ['N', 'A'][state['adherent']]}) else 0
        adherent_key = ['N', 'A'][result_state['adherent']]
        adverse_df = self.medications[medication_name]['adverse']
        chronic_adverse_df = adverse_df[adverse_df.TYPE == 'C']
        occurrence_vector = chronic_adverse_df[(chronic_adverse_df.PROBABILITY == 'Occurrence') &
                                               (chronic_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                               (chronic_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                               (chronic_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                               (chronic_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                               (chronic_adverse_df.INF_TREATMENT <= state['treatment']) &
                                               (chronic_adverse_df.SUP_TREATMENT >= state['treatment'])]['BASE_VALUE']
        occurrence = occurrence_vector.prod()
        occurrence_vector = (1 - occurrence_vector) / occurrence
        if state['chronic'] == 0:  # Chronic event can occur
            if random.random() < 1-occurrence:  # Chronic reaction occurs
                death = 1 - float((chronic_adverse_df[(chronic_adverse_df.PROBABILITY == 'Immediate_Death') &
                                                          (chronic_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                          (chronic_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                          (chronic_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                          (chronic_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                          (chronic_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                          (chronic_adverse_df.SUP_TREATMENT >= state['treatment'])]
                                   ['BASE_VALUE']*occurrence_vector).prod())
                if random.random() < death:  # Death due to chronic reaction
                    result_state['tests'] = 'dead'
                    return result_state
                else:
                    change = 1 - float((chronic_adverse_df[(chronic_adverse_df.PROBABILITY == 'Immediate_Change') &
                                                          (chronic_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                          (chronic_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                          (chronic_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                          (chronic_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                          (chronic_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                          (chronic_adverse_df.SUP_TREATMENT >= state['treatment'])]
                                       ['BASE_VALUE'] * occurrence_vector).prod())
                    if random.random() < change:  # Death due to chronic reaction
                        result_state['tests'] = 'adverse_reaction'
                        return result_state
                    else:
                        result_state['chronic'] = 1
        else:
            death = 1 - float((chronic_adverse_df[(chronic_adverse_df.PROBABILITY == 'Chronic_Death') &
                                                  (chronic_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                  (chronic_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                  (chronic_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                  (chronic_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                  (chronic_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                  (chronic_adverse_df.SUP_TREATMENT >= state['treatment'])]
                               ['BASE_VALUE'] * occurrence_vector).prod())
            if random.random() < death:  # Death due to chronic reaction
                result_state['tests'] = 'dead'
                return result_state
            else:
                change = 1 - float((chronic_adverse_df[(chronic_adverse_df.PROBABILITY == 'Chronic_Change') &
                                                       (chronic_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                       (chronic_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                       (chronic_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                       (chronic_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                       (chronic_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                       (chronic_adverse_df.SUP_TREATMENT >= state['treatment'])]
                                    ['BASE_VALUE'] * occurrence_vector).prod())
                if random.random() < change:  # Death due to chronic reaction
                    result_state['tests'] = 'adverse_reaction'
                    return result_state
        if state['acute'] == 0:  # Acute event can occur
            acute_adverse_df = adverse_df[adverse_df.TYPE == 'A']
            occurrence_vector = acute_adverse_df[(acute_adverse_df.PROBABILITY == 'Occurrence') &
                                                 (acute_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                 (acute_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                 (acute_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                 (acute_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                 (acute_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                 (acute_adverse_df.SUP_TREATMENT >= state['treatment'])]['BASE_VALUE']
            occurrence = occurrence_vector.prod()
            occurrence_vector = (1 - occurrence_vector) / occurrence
            if random.random() < 1-occurrence:  # Chronic reaction occurs
                death = 1 - float((acute_adverse_df[(acute_adverse_df.PROBABILITY == 'Immediate_Death') &
                                                    (acute_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                    (acute_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                    (acute_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                    (acute_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                    (acute_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                    (acute_adverse_df.SUP_TREATMENT >= state['treatment'])]
                                   ['BASE_VALUE']*occurrence_vector).prod())
                if random.random() < death:  # Death due to chronic reaction
                    result_state['tests'] = 'dead'
                    return result_state
                else:
                    change = 1 - float((acute_adverse_df[(acute_adverse_df.PROBABILITY == 'Immediate_Change') &
                                                          (acute_adverse_df.ADHERENT.isin([adherent_key, 'ALL'])) &
                                                          (acute_adverse_df.EXAMS.isin([state['tests'], 'ALL'])) &
                                                          (acute_adverse_df.AGE.isin([age_group, 'ALL'])) &
                                                          (acute_adverse_df.SEX.isin([state['sex'], 'ALL'])) &
                                                          (acute_adverse_df.INF_TREATMENT <= state['treatment']) &
                                                          (acute_adverse_df.SUP_TREATMENT >= state['treatment'])]
                                       ['BASE_VALUE'] * occurrence_vector).prod())
                    if random.random() < change:  # Death due to chronic reaction
                        result_state['tests'] = 'adverse_reaction'
                        return result_state
                    else:
                        result_state['chronic'] = 1
        else:
            result_state['acute'] = 0
        if result_state['tests'] > 0:
            result_state['age'] += 2
            result_state['treatment'] += 2
        else:
            result_state['age'] += 6
            result_state['treatment'] += 6
        if result_state['adherent'] == 1:
            result_state['tests'] = 0 if random.random() < get_info(self.medications[medication_name],
                                                                {'EVENT': 'efficiency',
                                                                 'ADHERENT': ['N', 'A'][result_state['adherent']]}) \
                else result_state['tests'] + 1
        else:
            result_state['tests'] += 1
        if result_state['tests'] > 2 or (result_state['tests'] == 2 and result_state['adherent'] == 1):
            result_state['tests'] = 'failure'
        return result_state


m = Model()
print(m.general_info)
for i in range(10):
    print(i+1)
    m.simulate_medication('fake_med')
