import pandas as pd
import seaborn as sns
import numpy as np
from root import DIR_OUTPUT
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataProcessing(object):
    def __init__(self, name_list: list):
        result_list = list()
        for name in tqdm(name_list):
            result_list.append(pd.read_csv(DIR_OUTPUT+'results_'+name+'.csv'))
        self.results = pd.concat(result_list)
        self.results.to_csv(DIR_OUTPUT+'consolidate_results.csv', index=False)

    def generate_dispersion(self, language='eng'):
        results = self.results.copy()
        qaly = 'QALY'
        costs = 'Costs'
        medication_name = 'Therapy'
        acute_events = 'Acute event'
        chronic_events = 'Chronic event'
        exit_reason = 'Exit reason'
        group = 'Group'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos'
            medication_name = 'Terapia'
            acute_events = 'Eventos agudos'
            chronic_events = 'Eventos crónicos'
            exit_reason = 'Motivo de salida'
            group = 'Grupo'
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events,
                                'exit_reason': exit_reason, 'group': group}, inplace=True)
        markers = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]
        mark = dict()
        names = list(results[medication_name].unique())
        for i in range(len(names)):
            mark[names[i]] = markers[i % len(markers)]
        sns.set(style="darkgrid")
        plot = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=group, style=medication_name,
                           markers=mark, kind='scatter')
        plt.savefig(DIR_OUTPUT+"dispersion.png", bbox_inches="tight")
        plt.clf()
        mark = list()
        for i in range(len(results[medication_name].unique())):
            mark.append(markers[i % len(markers)])
        results = results[[medication_name, group, qaly, costs]].groupby([medication_name, group]).mean().reset_index(
            drop=False)
        plot = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, style=medication_name, markers=mark,
                           kind='scatter', legend='full')
        plt.savefig(DIR_OUTPUT + "ce_plane.png", bbox_inches="tight")
        plt.clf()

    def calculate_net_monetary_benefit(self, threshold: float, result_list: list):
        df = self.results[['medication_name', 'group', 'iteration', 'qaly', 'costs']].copy()
        df['nmb'] = df['qaly']*threshold - df['costs']
        max_df = df[['iteration', 'nmb']].groupby('iteration').max()
        max_df.reset_index(drop=False, inplace=True)
        n_iterations = max_df['iteration'].max()
        df = df.merge(max_df, left_on='iteration', right_on='iteration', suffixes=(None, '_max'))
        del max_df
        df['is_best'] = np.where(df['nmb'] == df['nmb_max'], 1, 0)
        df = df[['medication_name', 'group', 'is_best']].groupby(['medication_name', 'group']).sum().reset_index(
            drop=False)
        df['is_best'] = df['is_best']/n_iterations
        df['threshold'] = threshold
        result_list.append(df)

    def acceptability_curve(self, min_value: float, max_value: float, n_steps: int = 1000, language='eng'):
        step_size = (max_value-min_value)/(n_steps-1)
        cores = 1
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
        if language == 'spa':
            medication_name = 'Terapia'
            threshold = 'Disponibilidad a pagar'
            probability = 'Probabilidad'
            name = 'curva_aceptabilidad'
            group = 'Grupo'
        df.rename(columns={'threshold': threshold, 'is_best': probability, 'medication_name': medication_name,
                           'group': group}, inplace=True)

        df.to_csv(DIR_OUTPUT+name+'.csv', index=False)
        sns.set(style="darkgrid")
        plot = sns.relplot(x=threshold, y=probability, hue=medication_name, data=df, row=group, style=medication_name,
                           kind='line')
        plt.savefig(DIR_OUTPUT + "acceptability_curve.png", bbox_inches="tight")
        plt.clf()
