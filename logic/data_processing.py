import pandas as pd
import seaborn as sns
import numpy as np
from root import DIR_OUTPUT
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt
import math
import warnings
from pyexcelerate import Workbook
import matplotlib.ticker as mtick
warnings.filterwarnings("ignore")


class DataProcessing(object):
    def __init__(self, name_list: list, group_name: str = 'ALL', distribution: bool = False,
                 discount_rate: bool = False, switch_cost: str = None, pregnancy_model: bool = False,
                 insert_switch: bool = False, currency: str = 'COP'):
        """
        :param name_list:
        :param group_name:
        :param distribution:
        :param discount_rate:
        :param switch_cost:
        :param pregnancy_model:
        :param insert_switch:
        """
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
                    if insert_switch:
                        file_name += '_future_lines'
                    file_name += '.csv'
                    result_list.append(pd.read_csv(file_name))
            else:
                file_name = DIR_OUTPUT + 'results_'
                if distribution:
                    file_name += 'd_'
                file_name += name
                if pregnancy_model:
                    file_name += '_p'
                if insert_switch:
                    file_name += '_future_lines'
                file_name += '.csv'
                result_list.append(pd.read_csv(file_name))
        self.results = pd.concat(result_list)
        file_name = DIR_OUTPUT + 'consolidate_results_'+group_name
        if distribution:
            file_name += '_d'
        if pregnancy_model:
            file_name += '_p'
        if switch_cost is not None:
            file_name += '_' + switch_cost
        if insert_switch:
            file_name += '_future_lines'
        file_name += '.csv'
        self.results.to_csv(file_name, index=False)
        self.distribution = distribution
        self.discount_rate = discount_rate
        self.switch_cost = switch_cost
        self.pregnancy_model = pregnancy_model
        self.group_name = group_name
        self.insert_switch = insert_switch
        self.currency = currency

    def generate_histograms(self, language: str = 'eng'):
        """
        Generates general information graphs for the study.
        :param language: language of the exportation for the names of the graphs.
        """
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        results = self.results.copy()
        qaly = 'QALY'
        costs = 'Costs (' + self.currency + ')'
        medication_name = 'Therapy'
        treatment = 'Treatment Duration (years)'
        acute_events = 'Acute reactions (number of events)'
        chronic_events = 'Chronic reactions'
        exit_reason = 'Exit reason'
        group = 'Group'
        discount_rate = 'Discount rate (%)'
        n_high_tests = 'High load tests (number of events)'
        probability = 'Probability'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos (' + self.currency + ')'
            medication_name = 'Terapia'
            treatment = 'Duracion Tratamiento (años)'
            acute_events = 'Eventos agudos (número de eventos)'
            chronic_events = 'Eventos cronicos'
            exit_reason = 'Motivo de salida'
            group = 'Grupo'
            discount_rate = 'Tasa de descuento (%)'
            n_high_tests = 'Pruebas altas (número total)'
            probability = 'Probabilidad'
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events,
                                'exit_reason': exit_reason, 'group': group, 'discount_rate': discount_rate,
                                'treatment': treatment, 'n_high_tests': n_high_tests},
                       inplace=True)
        sns.set(style="darkgrid")
        # Histograms
        results = results[(results[discount_rate] == '3.5') | (results[discount_rate] == 3.5)][[
            medication_name, group, qaly, costs, acute_events, chronic_events, treatment, n_high_tests]]
        results[treatment] = results[treatment]/12
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(45, 25), sharey=False, sharex=False)
        i = 0
        j = 0
        for column in [qaly, costs, acute_events, chronic_events, treatment, n_high_tests]:
            g = sns.histplot(data=results, x=column, hue=medication_name, stat='probability', common_norm=False,
                             fill=False, cumulative=True, element='poly', ax=ax[j][i])
            if column == costs:
                xlabels = ['$' + '{:,.2f}'.format(x) + 'M' for x in g.get_xticks() / 1000000]
                g.set_xticklabels(xlabels)
            if column == acute_events:
                xlabels = [(x if round(x) == x else '') for x in g.get_xticks()]
                g.set_xticklabels(xlabels)
            if column == chronic_events:
                print(g.get_xticks())
                xlabels = list()
                for x in g.get_xticks():
                    if x == 0:
                        if language == 'spa':
                            xlabels.append('Sin evento')
                        else:
                            xlabels.append('No reaction')
                    elif round(x, 2) == 1.0:
                        print('entró')
                        if language == 'spa':
                            xlabels.append('Con evento')
                        else:
                            xlabels.append('Reaction')
                    else:
                        xlabels.append('')
                g.set_xticklabels(xlabels)
            g.set(ylim=(0, 1.05))
            ylabels = ['{:,.2f}'.format(x) + '%' for x in g.get_yticks() * 100]
            g.set_yticklabels(ylabels)
            g.set_ylabel(probability)
            ax_act = ax[j][i]
            ax_act.set_title(column)
            i += 1
            if i == 3:
                i = 0
                j = j + 1
        graph_name = DIR_OUTPUT + 'histograms_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

    def generate_dispersion(self, gdp: float, language: str = 'eng', n_iterations: int = 50000,
                            has_discount_rate: bool = False):
        """
        Generates general information graphs for the study.
        :param gdp: Gross domestic product to use as reference
        :param language: language of the exportation for the names of the graphs.
        :param n_iterations: number of iterations in the original data.
        :param has_discount_rate: boolean indicating if the model has several discount rates or only one factor.
        """
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        results = self.results.copy()
        qaly = 'QALY'
        costs = 'Costs' + ' (' + self.currency + ')'
        test_costs = 'Testing costs'
        reaction_costs = 'Reactions costs'
        treatment_costs = 'Treatment costs'
        medication_name = 'Therapy'
        treatment = 'Treatment Duration'
        acute_events = 'Acute event'
        chronic_events = 'Chronic event'
        exit_reason = 'Exit reason'
        group = 'Group'
        discount_rate = 'Discount rate (%)'
        adverse_reaction = 'Adverse Reaction'
        failure = 'Failure'
        dead = 'Dead'
        dead_ar = 'AR Death'
        n_high_tests = 'High tests'
        time_to_death = 'Time to death'
        pib_name = 'GDP'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos' + ' (' + self.currency + ')'
            test_costs = 'Costo de Seguimiento'
            reaction_costs = 'Costos de Eventos Adversos'
            treatment_costs = 'Costos de la Terapia'
            medication_name = 'Terapia'
            treatment = 'Duracion Tratamiento'
            acute_events = 'Eventos agudos'
            chronic_events = 'Eventos cronicos'
            exit_reason = 'Motivo de salida'
            group = 'Grupo'
            discount_rate = 'Tasa de descuento (%)'
            adverse_reaction = 'Evento adverso'
            failure = 'Falla terapeutica'
            dead = 'Muerte'
            dead_ar = 'Muerte por EA'
            n_high_tests = 'Pruebas altas'
            time_to_death = 'Tiempo a muerte'
            pib_name = 'PIB'
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events, 'group': group,
                                'exit_reason': exit_reason, 'discount_rate': discount_rate, 'treatment': treatment,
                                'n_high_tests': n_high_tests, 'time_to_death': time_to_death, 'test_cost': test_costs,
                                'adverse_cost': reaction_costs, 'medication_cost': treatment_costs},
                       inplace=True)
        markers = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]
        mark = dict()
        names = list(results[medication_name].unique())
        for i in range(len(names)):
            mark[names[i]] = markers[i % len(markers)]
        sns.set(style="darkgrid")
        g = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=group, col=discount_rate,
                    style=medication_name, markers=mark, kind='scatter')
        for ax in g.axes.flat:
            ylabels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set_yticklabels(ylabels)  # set new labels
        graph_name = DIR_OUTPUT + 'dispersion_'+self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        g = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results[results[discount_rate==3.5]], row=group,
                        style=medication_name, markers=mark, kind='scatter')
        for ax in g.axes.flat:
            ylabels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set_yticklabels(ylabels)  # set new labels
        graph_name = DIR_OUTPUT + 'dispersion_3.5_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
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
        r2.rename(columns={'adverse_reaction': adverse_reaction, 'failure': failure, 'dead': dead, 'dead_ar': dead_ar},
                  inplace=True)
        for c in [adverse_reaction, failure, dead, dead_ar]:
            if c not in list(r2.columns):
                r2[c] = 0.0
        results = results[[discount_rate, medication_name, group, qaly, costs, acute_events, chronic_events,
                           treatment, n_high_tests, treatment_costs, test_costs, reaction_costs
                           ]].groupby([discount_rate, medication_name, group]).mean()
        results = results.reset_index(drop=False)
        results = results.merge(r2, left_on=medication_name, right_on=medication_name, suffixes=(None, '_percent'))
        file_name = DIR_OUTPUT + 'average_results_' + self.group_name
        graph_name = DIR_OUTPUT + 'ce_plane_' + self.group_name
        if self.distribution:
            file_name += '_d'
            graph_name += '_d'
        if self.switch_cost is not None:
            file_name += '_' + self.switch_cost
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            file_name += '_p'
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
            file_name += '_future_lines'
        file_name += '.csv'
        results.to_csv(file_name, index=False)
        print('Exported:', file_name, dt.datetime.now())
        max_qaly = math.ceil(results[qaly].max())
        max_cost = results[costs].max()
        power = 0
        while max_cost > 10:
            max_cost /= 10
            power += 1
        max_cost = math.ceil(max_cost) * (10**power)
        step_down = max_cost/5
        while max_cost - step_down >= results[costs].max():
            max_cost -= step_down
        plane_lines = list()
        for ite in range(1, 4):
            qaly_lim = min(max_qaly, max_cost/(ite*gdp))
            cost_lim = min(max_cost, max_qaly*ite*gdp)
            qaly_step = qaly_lim/100
            cost_step = cost_lim/100
            for step in range(101):
                plane_lines.append({pib_name: str(ite) + pib_name, qaly: qaly_step*step, costs: cost_step*step})
        plane_lines = pd.DataFrame(plane_lines)
        for dr in list(results[discount_rate].unique()):
            fig, ax = plt.subplots()
            df_graph = results[results[discount_rate] == dr].copy()
            sns.lineplot(data=plane_lines, x=qaly, y=costs, hue=pib_name, style=pib_name, legend='full',
                             ax=ax)
            sns.scatterplot(data=df_graph, x=qaly, y=costs, hue=medication_name, style=medication_name,
                                markers=mark, legend='full', ax=ax)
            ax.set_ylim(0, max_cost)
            ax.set_xlim(0, max_qaly)
            ax.set_title(discount_rate + ': ' + str(dr))
            ylabels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set_yticklabels(ylabels)
            # Put the legend out of the figure
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            graph_name_e = graph_name + '_DR' + str(dr) + '.png'
            plt.savefig(graph_name_e, bbox_inches="tight")
            print('Exported:', graph_name_e, dt.datetime.now())
            plt.clf()
        results = results[(results[discount_rate] == '3.5') | (results[discount_rate] == 3.5)]
        results.drop(columns={discount_rate, group}, inplace=True)
        columns_graph = list(results.columns)
        exit_reasons = [medication_name]
        columns_graph.remove(medication_name)
        if adverse_reaction in columns_graph:
            columns_graph.remove(adverse_reaction)
            exit_reasons.append(adverse_reaction)
        if failure in columns_graph:
            columns_graph.remove(failure)
            exit_reasons.append(failure)
        if dead in columns_graph:
            columns_graph.remove(dead)
            exit_reasons.append(dead)
        if dead_ar in columns_graph:
            columns_graph.remove(dead_ar)
            exit_reasons.append(dead_ar)
        if reaction_costs in columns_graph:
            columns_graph.remove(reaction_costs)
        if treatment_costs in columns_graph:
            columns_graph.remove(treatment_costs)
        if test_costs in columns_graph:
            columns_graph.remove(test_costs)
        n_cols = math.ceil(len(columns_graph)/2)
        f, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(n_cols*15, 25), sharey=False, sharex=False)
        i = 0
        j = 0
        for column in columns_graph:
            results.sort_values(by=column, ascending=True, inplace=True)
            if column == costs:
                rc = results[[medication_name, column]].copy()
                rc[column] = rc[column]/1000000
                rc.plot(x=medication_name, y=column, kind='barh', ax=ax[j][i], subplots=True, sharey=False,
                             sharex=False, title=column, legend=False)
                ax[j][i].xaxis.set_major_formatter(tick)
                del rc
            else:
                results.plot(x=medication_name, y=column, kind='barh', ax=ax[j][i], subplots=True, sharey=False,
                             sharex=False, title=column, legend=False)
            i += 1
            if i == n_cols:
                i = 0
                j = j+1
        f.suptitle('Bar plot')
        graph_name = DIR_OUTPUT + 'bar_plot_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        df_graph = results[[medication_name, treatment_costs, test_costs, reaction_costs, costs]].copy()
        df_graph = df_graph.set_index(medication_name)
        df_graph = df_graph/1000000
        df_graph.sort_values(by=[costs, treatment_costs, test_costs, reaction_costs], ascending=True, inplace=True)
        ax = df_graph[[treatment_costs, test_costs, reaction_costs]].plot(kind='barh', title=costs, legend=True,
                                                                        stacked=True, figsize=(13, 13))
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        ax.xaxis.set_label_text('(' + self.currency + ')')
        graph_name = DIR_OUTPUT + 'costs_total' + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

        for col in [treatment_costs, test_costs, reaction_costs]:
            df_graph[col] = df_graph[col]/df_graph[costs]*100
        df_graph.sort_values(by=[treatment_costs, test_costs, reaction_costs], ascending=True, inplace=True)
        ax = df_graph[[treatment_costs, test_costs, reaction_costs]].plot(kind='barh', title=costs + ' (%)',
                                                                          legend=True, stacked=True, figsize=(13, 13))
        ax.set_xlim(0, 100)
        fmt = '{x:,.2f}' + '%'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        if language == 'spa':
            ax.yaxis.set_label_text('Proporcion')
        else:
            ax.yaxis.set_label_text('Proportion')
        graph_name = DIR_OUTPUT + 'costs_percent' + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

        results = results[exit_reasons].set_index(medication_name)
        results = results[[failure, adverse_reaction, dead, dead_ar]]*100
        results.sort_values(by=[failure, adverse_reaction, dead], ascending=[True, True, True], inplace=True)
        ax = results.plot(kind='barh', title=exit_reason, legend=True, stacked=True, figsize=(13, 13))
        ax.set_xlim(0, 100)
        fmt = '{x:,.2f}' + '%'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        if language == 'spa':
            ax.yaxis.set_label_text('Proporcion')
        else:
            ax.yaxis.set_label_text('Proportion')
        graph_name = DIR_OUTPUT + exit_reason + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

    def calculate_net_monetary_benefit(self, threshold: float, n_iterations: int):
        """
        Calculates the percentage of best value considering the Net Monetary Benefit with a given threshold.
        :param threshold: value to consider for the value of 1 QALY of improvement
        :param n_iterations: number of iterations for each scenario, to consider
        :return: Dataframe with the resulting percentage of acceptance for each set, with the given threshold.
        """
        df = self.results[['medication_name', 'group', 'iteration', 'qaly', 'costs', 'discount_rate']].copy()
        df['nmb'] = df['qaly']*threshold - df['costs']
        max_df = df[['discount_rate', 'iteration', 'nmb']].groupby(['discount_rate', 'iteration']).max()
        max_df.reset_index(drop=False, inplace=True)
        df = df.merge(max_df, left_on=['iteration', 'discount_rate'], right_on=['iteration', 'discount_rate'],
                      suffixes=(None, '_max'))
        del max_df
        df['is_best'] = np.where(df['nmb'] == df['nmb_max'], 1, 0)
        df = df[['medication_name', 'group', 'is_best', 'discount_rate']].groupby([
            'medication_name', 'group', 'discount_rate']).sum().reset_index(drop=False)
        df['is_best'] = df['is_best']/n_iterations
        df['threshold'] = threshold
        return df

    def acceptability_curve(self, min_value: float, max_value: float, n_steps: int = 1000, language: str = 'eng',
                            n_iterations: int = 50000, step_value: float = None):
        """
        Creates the associated graph for the percentage of acceptance of every option, with a considered range of
        thresholds and a number of intermediate steps to calculate.
        :param min_value: Minimum threshold to consider in the iterations.
        :param max_value: Maximum threshold to consider in the iterations.
        :param n_steps: Number of divisions to consider the different thresholds for the graph.
        :param language: Language to consider in the associated graphs (axis names and legends)
        :param n_iterations: Number of iterations that were considered in the model.
        :param step_value: Value of GDP to show reference thresholds
        """
        print('Calculations starting - Please hold the use of the computer until it finishes', dt.datetime.now())
        step_size = (max_value - min_value) / (n_steps - 1)
        acceptability_list = list()
        for i in tqdm(range(n_steps)):
            acceptability_list.append(self.calculate_net_monetary_benefit(min_value+step_size*i, n_iterations))
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
        gbp_value = 1
        groups = list(df[group].unique())
        discount_rates = list(df[discount_rate].unique())
        file_name = DIR_OUTPUT + name + '_' + self.group_name
        graph_name = DIR_OUTPUT + 'acceptability_curve_'+self.group_name
        if self.distribution:
            file_name += '_d'
            graph_name += '_d'
        if self.switch_cost is not None:
            file_name += '_' + self.switch_cost
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            file_name += '_p'
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
            file_name += '_future_lines'
        file_name += '.csv'
        graph_name += '.png'
        df.to_csv(file_name, index=False)
        while val_to_add < max_value:
            for gr in groups:
                for dr in discount_rates:
                    insert_threshold.append({threshold: val_to_add, probability: 0,
                                             medication_name: str(gbp_value) + ' ' + pib_name, group: gr,
                                             discount_rate: dr})
                    insert_threshold.append({threshold: val_to_add+1, probability: max_prob,
                                             medication_name: str(gbp_value) + ' ' + pib_name, group: gr,
                                             discount_rate: dr})
            val_to_add += step_value
            gbp_value += 1
        pib_df = pd.DataFrame(insert_threshold)
        df = pd.concat([df, pib_df], ignore_index=True)
        df.to_csv(file_name, index=False)
        sns.set(style="darkgrid")
        sns.relplot(x=threshold, y=probability, hue=medication_name, data=df, row=group, col=discount_rate,
                    style=medication_name, kind='line')
        plt.ylim(0, max_prob)
        plt.xlim(min_value, max_value)
        plt.savefig(graph_name, bbox_inches="tight")
        plt.clf()
        print('Aceptability curve generated')

    def export_net_monetary_benefit(self, thresholds: dict, language: str = 'eng'):
        """
        Calculates the percentage of best value considering the Net Monetary Benefit with a given threshold.
        :param thresholds: values to consider for the value of 1 QALY of improvement
        :param language: language of exportation
        :return: Dataframe with the resulting percentage of acceptance for each set, with the given threshold.
        """
        df = self.results[['medication_name', 'discount_rate', 'qaly', 'costs']].copy()
        df = df.groupby(['medication_name', 'discount_rate']).mean().reset_index(drop=False)
        discount_rates = list(df['discount_rate'].unique())
        n_col = len(thresholds)
        n_row = len(discount_rates)
        medication_name = 'Therapy'
        discount_rate = 'Discount rate'
        pib_name = 'GDP'
        file_name = 'net_monetary_benefit'
        graph_title = 'Net Monetary Benefit'
        qaly = 'QALY'
        costs = 'Costs'
        initials = 'MNB'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos'
            medication_name = 'Terapia'
            discount_rate = 'Tasa de Descuento'
            pib_name = 'PIB'
            file_name = 'beneficio_monetario_neto'
            graph_title = 'Beneficio Monetario Neto'
            initials = 'BMN'
        df.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name, 'discount_rate':
            discount_rate}, inplace=True)
        f, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(45, 25), sharey=False, sharex=False)
        if n_row == 1:
            f, ax = plt.subplots(ncols=n_col, figsize=(40, 15), sharey=False, sharex=False)
        i = 0
        j = 0
        wb = Workbook()
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        for key in thresholds:
            df2 = df.copy()
            threshold_value = thresholds[key]
            col_name = initials + ':' + pib_name + key
            df2[col_name] = (df2[qaly]*threshold_value - df2[costs])/1000000
            df2.sort_values(by=col_name, ascending=False, inplace=True)
            for dr in discount_rates:
                df3 = df2[df2[discount_rate] == dr].copy()
                df3.sort_values(by=col_name, ascending=True, inplace=True)
                subplot_title = col_name + '|' + discount_rate + ':' + str(dr)
                df3.rename(columns={col_name: subplot_title}, inplace=True)
                if n_row == 1:
                    df3.plot(x=medication_name, y=subplot_title, kind='barh', ax=ax[i],
                                                       subplots=True, sharex=False, legend=False)
                    ax[i].xaxis.set_major_formatter(tick)
                    ax[i].xaxis.set_label_text(costs + ' (' + self.currency + ')')
                else:
                    df3.plot(x=medication_name, y=subplot_title, kind='barh', ax=ax[j][i],
                                                   subplots=True, sharex=False, legend=False)
                    ax[j][i].xaxis.set_major_formatter(tick)
                    ax[j][i].xaxis.set_label_text(costs + ' (' + self.currency + ')')
                values = [df2[df2[discount_rate] == dr].columns] + list(df2[df2[discount_rate] == dr].values)
                wb.new_sheet('DR'+str(dr)+pib_name+key, data=values)
                j += 1
                if j == n_row:
                    j = 0
                    i += 1
        f.suptitle(graph_title)
        graph_name = DIR_OUTPUT + file_name + '_' + self.group_name
        file_name = DIR_OUTPUT + file_name + '_' + self.group_name
        if self.distribution:
            file_name += '_d'
            graph_name += '_d'
        if self.switch_cost is not None:
            file_name += '_' + self.switch_cost
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            file_name += '_p'
            graph_name += '_p'
        if self.insert_switch:
            graph_name += '_future_lines'
            file_name += '_future_lines'
        file_name += '.xlsx'
        graph_name += '.png'
        wb.save(file_name)
        print('Excel ', file_name, 'exported')
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

        if n_row > 1:
            f, ax = plt.subplots(ncols=n_col, figsize=(40, 15), sharey=False, sharex=False)
            i = 0
            df = df[(df[discount_rate] == '3.5') | (df[discount_rate] == 3.5)]
            for key in thresholds:
                df2 = df.copy()
                threshold_value = thresholds[key]
                col_name = initials + ':' + pib_name + key
                df2[col_name] = (df2[qaly]*threshold_value - df2[costs])/1000000
                df2.sort_values(by=col_name, ascending=True, inplace=True)
                subplot_title = col_name + '|' + discount_rate + ':' + str(3.5)
                df2.rename(columns={col_name: subplot_title}, inplace=True)
                df2.plot(x=medication_name, y=subplot_title, kind='barh', ax=ax[i],
                                                       subplots=True, sharex=False, legend=False)
                ax[i].xaxis.set_major_formatter(tick)
                ax[i].xaxis.set_label_text(costs + ' (' + self.currency + ')')
                i += 1
            f.suptitle(graph_title)
            graph_name = graph_name[:-4] + '_3.5.png'
            plt.savefig(graph_name, bbox_inches="tight")
            print('Exported:', graph_name, dt.datetime.now())
            plt.clf()
