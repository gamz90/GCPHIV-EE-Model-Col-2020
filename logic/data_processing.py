import pandas as pd
import seaborn as sns
import numpy as np
from root import DIR_OUTPUT
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataProcessing(object):
    def __init__(self, name_list: list, stochastic: bool = False, discount_rate: bool = False, switch_cost: str = None,
                 pregnancy_model: bool = False):
        result_list = list()
        for name in tqdm(name_list):
            if discount_rate:
                for dr in ['0', '3.5', '5', '12']:
                    file_name = DIR_OUTPUT + 'results_' + name + '_'
                    if pregnancy_model:
                        file_name += 'p_'
                    file_name += dr + '_'
                    if switch_cost is not None:
                        file_name += switch_cost
                    file_name += '.csv'
                    result_list.append(pd.read_csv(file_name))
            else:
                file_name = DIR_OUTPUT + 'results_'
                if stochastic:
                    file_name += 's_'
                file_name += name
                if pregnancy_model:
                    file_name += '_p'
                file_name += '.csv'
                result_list.append(pd.read_csv(file_name))
        self.results = pd.concat(result_list)
        file_name = DIR_OUTPUT + 'consolidate_results'
        if stochastic:
            file_name += '_s'
        if pregnancy_model:
            file_name += '_p'
        if switch_cost is not None:
            file_name += '_' + switch_cost
        file_name += '.csv'
        self.results.to_csv(file_name, index=False)
        self.stochastic = stochastic
        self.discount_rate = discount_rate
        self.switch_cost = switch_cost
        self.pregnancy_model = pregnancy_model

    def generate_dispersion(self, language='eng'):
        results = self.results.copy()
        qaly = 'QALY'
        costs = 'Costs'
        medication_name = 'Therapy'
        acute_events = 'Acute event'
        chronic_events = 'Chronic event'
        exit_reason = 'Exit reason'
        group = 'Group'
        discount_rate = 'Discount Rate (%)'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos'
            medication_name = 'Terapia'
            acute_events = 'Eventos agudos'
            chronic_events = 'Eventos crónicos'
            exit_reason = 'Motivo de salida'
            group = 'Grupo'
            discount_rate = 'Tasa de Descuento (%)'
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events,
                                'exit_reason': exit_reason, 'group': group, 'discount_rate': discount_rate},
                       inplace=True)
        markers = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]
        mark = dict()
        names = list(results[medication_name].unique())
        for i in range(len(names)):
            mark[names[i]] = markers[i % len(markers)]
        sns.set(style="darkgrid")
        sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=group, col=discount_rate,
                    style=medication_name, markers=mark, kind='scatter')
        graph_name = DIR_OUTPUT + "dispersion"
        if self.stochastic:
            graph_name += '_s'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        plt.clf()
        mark = list()
        for i in range(len(results[medication_name].unique())):
            mark.append(markers[i % len(markers)])
        size = results.iteration.max()*4 if discount_rate else results.iteration.max()
        r2 = results[[medication_name, exit_reason, 'iteration']].groupby([medication_name, exit_reason]).count()/size
        r2.reset_index(drop=False, inplace=True)
        r2 = r2.pivot_table('iteration', [medication_name], exit_reason)
        r2.fillna(0, inplace=True)
        results = results[[discount_rate, medication_name, group, qaly, costs, acute_events, chronic_events]].groupby([
            discount_rate, medication_name, group]).mean().reset_index(drop=False)
        results = results.merge(r2, left_on=medication_name, right_on=medication_name, suffixes=(None, '_percent'))
        sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=discount_rate,  style=medication_name,
                    markers=mark, kind='scatter', legend='full')
        file_name = DIR_OUTPUT + "average_results"
        graph_name = DIR_OUTPUT + "ce_plane"
        if self.stochastic:
            file_name += '_s'
            graph_name += '_s'
        if self.switch_cost is not None:
            file_name += '_' + self.switch_cost
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            file_name += '_p'
            graph_name += '_p'
        file_name += '.csv'
        graph_name += '.png'
        results.to_csv(file_name, index=False)
        plt.savefig(graph_name, bbox_inches="tight")
        plt.clf()

    def calculate_net_monetary_benefit(self, threshold: float, result_list: list):
        df = self.results[['medication_name', 'group', 'iteration', 'qaly', 'costs', 'discount_rate']].copy()
        df['nmb'] = df['qaly']*threshold - df['costs']
        max_df = df[['iteration', 'nmb']].groupby('iteration').max()
        max_df.reset_index(drop=False, inplace=True)
        n_iterations = max_df['iteration'].max()
        df = df.merge(max_df, left_on='iteration', right_on='iteration', suffixes=(None, '_max'))
        del max_df
        df['is_best'] = np.where(df['nmb'] == df['nmb_max'], 1, 0)
        df = df[['medication_name', 'group', 'is_best', 'discount_rate']].groupby([
            'medication_name', 'group', 'discount_rate']).sum().reset_index(drop=False)
        df['is_best'] = df['is_best']/n_iterations
        df['threshold'] = threshold
        result_list.append(df)

    def acceptability_curve(self, min_value: float, max_value: float, n_steps: int = 1000, language='eng'):
        step_size = (max_value-min_value)/(n_steps-1)
        cores = multiprocessing.cpu_count()
        manager = multiprocessing.Manager()
        acceptability_list = manager.list()
        args = list()
        for i in range(n_steps):
            args.append((min_value+step_size*i, acceptability_list))
        with multiprocessing.Pool(processes=cores) as pool:
            pool.starmap(self.calculate_net_monetary_benefit, args)
        df = pd.concat(acceptability_list)
        medication_name = 'Therapy'
        threshold = 'Threshold'
        probability = 'Probability'
        group = 'Group'
        name = 'acceptability_curve'
        discount_rate = 'Discount rate'
        if language == 'spa':
            medication_name = 'Terapia'
            threshold = 'Disponibilidad a pagar'
            probability = 'Probabilidad'
            name = 'curva_aceptabilidad'
            discount_rate = 'Tasa de Descuento'
            group = 'Grupo'
        df.rename(columns={'threshold': threshold, 'is_best': probability, 'medication_name': medication_name,
                           'group': group, 'discount_rate': discount_rate}, inplace=True)
        file_name = DIR_OUTPUT + name
        graph_name = DIR_OUTPUT + "acceptability_curve"
        if self.stochastic:
            file_name += '_s'
            graph_name += '_s'
        if self.switch_cost is not None:
            file_name += '_' + self.switch_cost
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            file_name += '_p'
            graph_name += '_p'
        file_name += '.csv'
        graph_name += '.png'
        df.to_csv(file_name, index=False)
        sns.set(style="darkgrid")
        sns.relplot(x=threshold, y=probability, hue=medication_name, data=df, row=group, col=discount_rate,
                    style=medication_name, kind='line')
        plt.savefig(graph_name, bbox_inches="tight")
        plt.clf()
