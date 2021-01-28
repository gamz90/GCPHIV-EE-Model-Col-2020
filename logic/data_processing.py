import pandas as pd
import seaborn as sns
import numpy as np
from root import DIR_OUTPUT
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt
import math
import warnings
warnings.filterwarnings("ignore")


class DataProcessing(object):
    def __init__(self, name_list: list, group_name: str = 'ALL', stochastic: bool = False, discount_rate: bool = False, switch_cost: str = None,
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
        file_name = DIR_OUTPUT + 'consolidate_results_'+group_name
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
        self.group_name = group_name

    def generate_dispersion(self, language: str = 'eng', n_iterations: int = 50000, has_discount_rate: bool = False):
        results = self.results.copy()
        qaly = 'QALY'
        costs = 'Costs'
        medication_name = 'Therapy'
        acute_events = 'Acute event'
        chronic_events = 'Chronic event'
        exit_reason = 'Exit reason'
        group = 'Group'
        discount_rate = 'Discount rate (%)'
        adverse_reaction = 'Adverse Reaction'
        failure = 'Failure'
        dead = 'Dead'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos'
            medication_name = 'Terapia'
            acute_events = 'Eventos agudos'
            chronic_events = 'Eventos cronicos'
            exit_reason = 'Motivo de salida'
            group = 'Grupo'
            discount_rate = 'Tasa de descuento (%)'
            adverse_reaction = 'Evento adverso'
            failure = 'Falla terapeutica'
            dead = 'Muerte'
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
        g = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=group, col=discount_rate,
                    style=medication_name, markers=mark, kind='scatter')
        graph_name = DIR_OUTPUT + 'dispersion_'+self.group_name
        if self.stochastic:
            graph_name += '_s'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        mark = list()
        for i in range(len(results[medication_name].unique())):
            mark.append(markers[i % len(markers)])
        size = n_iterations*4 if has_discount_rate else n_iterations
        r2 = results[[medication_name, exit_reason, 'iteration']].groupby([medication_name, exit_reason]).count()/size
        r2.reset_index(drop=False, inplace=True)
        r2 = r2.pivot_table('iteration', [medication_name], exit_reason)
        r2.fillna(0, inplace=True)
        r2.rename(columns={'adverse_reaction': adverse_reaction, 'failure': failure, 'dead': dead}, inplace=True)
        results = results[[discount_rate, medication_name, group, qaly, costs, acute_events, chronic_events]].groupby([
            discount_rate, medication_name, group]).mean().reset_index(drop=False)
        results = results.merge(r2, left_on=medication_name, right_on=medication_name, suffixes=(None, '_percent'))
        sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=discount_rate,  style=medication_name,
                    markers=mark, kind='scatter', legend='full')
        file_name = DIR_OUTPUT + 'average_results_'+self.group_name
        graph_name = DIR_OUTPUT + 'ce_plane_'+self.group_name
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
        print('Exported:', file_name, dt.datetime.now())
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        g = sns.pairplot(data=results[[medication_name, qaly, costs, acute_events, chronic_events, failure]], hue=medication_name, markers=mark, corner=True)
        g.add_legend()
        graph_name = DIR_OUTPUT + 'pair_grid_' + self.group_name
        if self.stochastic:
            graph_name += '_s'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        graph_name = DIR_OUTPUT + 'ce_plane_' + self.group_name
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
        sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=discount_rate, style=medication_name,
                    markers=mark, kind='scatter', legend='full')
        file_name = DIR_OUTPUT + 'average_results_' + self.group_name
        graph_name = DIR_OUTPUT + 'ce_plane_' + self.group_name
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
        print('Exported:', file_name, dt.datetime.now())
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        results = results[results[discount_rate] == 5.0]
        results.drop(columns={discount_rate, group}, inplace=True)
        columns_graph = list(results.columns)
        columns_graph.remove(medication_name)
        columns_graph.remove(adverse_reaction)
        columns_graph.remove(failure)
        columns_graph.remove(dead)
        n_cols = math.ceil(len(columns_graph)/2)
        f, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(25, 25), sharey=False, sharex=False)
        i = 0
        j = 0
        for column in columns_graph:
            results.sort_values(by=column, ascending=True, inplace=True)
            results.plot(x=medication_name, y=column, kind='barh', ax=ax[j][i], subplots=True, sharey=False, sharex=False, title=column, legend=False)
            i += 1
            if i == n_cols:
                i = 0
                j = j+1
        graph_name = DIR_OUTPUT + 'bar_plot_' + self.group_name
        if self.stochastic:
            graph_name += '_s'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        results = results[[medication_name, failure, adverse_reaction, dead]].set_index(medication_name)
        results.sort_values(by=[failure, adverse_reaction], ascending=[True, True], inplace=True)
        ax = results.plot(kind='barh', title=exit_reason, legend=True, stacked=True, figsize=(13, 13))
        ax.set_xlim(0, 1)
        graph_name = DIR_OUTPUT + exit_reason + '_' + self.group_name
        if self.stochastic:
            graph_name += '_s'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

    def calculate_net_monetary_benefit(self, threshold: float, result_list: list, n_iterations: int):
        df = self.results[['medication_name', 'group', 'iteration', 'qaly', 'costs', 'discount_rate']].copy()
        df['nmb'] = df['qaly']*threshold - df['costs']
        max_df = df[['discount_rate', 'iteration', 'nmb']].groupby(['discount_rate', 'iteration']).max()
        max_df.reset_index(drop=False, inplace=True)
        df = df.merge(max_df, left_on=['iteration', 'discount_rate'], right_on=['iteration', 'discount_rate'], suffixes=(None, '_max'))
        del max_df
        df['is_best'] = np.where(df['nmb'] == df['nmb_max'], 1, 0)
        df = df[['medication_name', 'group', 'is_best', 'discount_rate']].groupby(['medication_name', 'group', 'discount_rate']).sum().reset_index(drop=False)
        df['is_best'] = df['is_best']/n_iterations
        df['threshold'] = threshold
        result_list.append(df)

    def acceptability_curve(self, min_value: float, max_value: float, n_steps: int = 1000, language:str ='eng', n_iterations: int = 50000, step_value: float = None):
        print('Calculations starting - Please hold the use of the computer until it finishes', dt.datetime.now())
        step_size = (max_value - min_value) / (n_steps - 1)
        acceptability_list = list()
        for i in tqdm(range(n_steps)):
            self.calculate_net_monetary_benefit(min_value+step_size*i, acceptability_list, n_iterations)
        print('Consolidating information', dt.datetime.now())
        df = pd.concat(acceptability_list)
        print('Starting graph generation', dt.datetime.now())
        medication_name = 'Therapy'
        threshold = 'Threshold'
        probability = 'Probability'
        group = 'Group'
        name = 'acceptability_curve'
        discount_rate = 'Discount rate'
        pib_name = 'GDP'
        if language == 'spa':
            medication_name = 'Terapia'
            threshold = 'Disponibilidad a pagar'
            probability = 'Probabilidad'
            name = 'curva_aceptabilidad'
            discount_rate = 'Tasa de Descuento'
            group = 'Grupo'
            pib_name = 'PIB'
        df.rename(columns={'threshold': threshold, 'is_best': probability, 'medication_name': medication_name,
                           'group': group, 'discount_rate': discount_rate}, inplace=True)
        mp = df[probability].max()
        max_prob = 0
        while max_prob <= mp:
            max_prob += 0.1
        insert_threshold = list()
        val_to_add = step_value
        pib_value = 1
        groups = list(df[group].unique())
        discount_rates = list(df[discount_rate].unique())
        file_name = DIR_OUTPUT + name + '_' + self.group_name
        graph_name = DIR_OUTPUT + 'acceptability_curve_'+self.group_name
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
        while val_to_add < max_value:
            for gr in groups:
                for dr in discount_rates:
                    insert_threshold.append({threshold: val_to_add, probability: 0, medication_name: str(pib_value) + ' ' + pib_name, group: gr, discount_rate: dr})
                    insert_threshold.append({threshold: val_to_add+1, probability: max_prob, medication_name: str(pib_value) + ' ' + pib_name, group: gr, discount_rate: dr})
            val_to_add += step_value
            pib_value += 1
        pib_df = pd.DataFrame(insert_threshold)
        df = pd.concat([df, pib_df], ignore_index=True)
        df.to_csv(file_name, index=False)
        sns.set(style="darkgrid")
        sns.relplot(x=threshold, y=probability, hue=medication_name, data=df, row=group, col=discount_rate, style=medication_name, kind='line')
        plt.ylim(0, max_prob)
        plt.xlim(min_value, max_value)
        plt.savefig(graph_name, bbox_inches="tight")

        plt.clf()
        print('Aceptability curve generated')
