import os

import pandas as pd
import seaborn as sns
import numpy as np
from root import DIR_OUTPUT, DIR_INPUT
from mpl_toolkits.mplot3d import Axes3D
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
                 currency: str = 'COP', language: str = 'eng'):
        """
        :param language: Language considered for the extraction and processing of information.
        :param currency: String that indicates what type of currency the model is considering
        :param name_list: List of technologies to consider in the analysis
        :param group_name: Name of the set of technologies to analyze
        :param distribution: Boolean indicating if the model had distribution associated or fixed parameters
        :param discount_rate: Boolean indicating if the model had multiple discount rates or not.
        :param switch_cost: Str indicating the value (BASE, INF or SUP) of the switch cost, None if a distribution model
        :param pregnancy_model: Boolean indicating if the model corresponded to the pregnant subgroup or general
        population.

        """
        file_name = DIR_OUTPUT + 'consolidate_results_' + group_name
        if distribution:
            file_name += '_d'
        if pregnancy_model:
            file_name += '_p'
        if switch_cost is not None:
            file_name += '_' + switch_cost
        file_name += '.csv'
        if os.path.exists(file_name):
            self.results = pd.read_csv(file_name)
        else:
            consolidated_name_file = file_name
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
                    if distribution:
                        file_name += 'd_'
                    file_name += name
                    if pregnancy_model:
                        file_name += '_p'
                    file_name += '.csv'
                    result_list.append(pd.read_csv(file_name))
            self.results = pd.concat(result_list)
            self.results.to_csv(consolidated_name_file, index=False)
        self.distribution = distribution
        self.discount_rate = discount_rate
        self.switch_cost = switch_cost
        self.pregnancy_model = pregnancy_model
        self.group_name = group_name
        self.currency = currency
        self.language = language
        qaly = 'QALY'
        costs = 'Costs'
        medication_name = 'Therapy'
        treatment = 'Treatment Duration'
        treatment_units = 'Years'
        acute_events = 'Acute reactions'
        acute_events_unit = 'Total amount'
        chronic_events = 'Chronic reactions'
        chronic_events_units = 'Occurrence'
        exit_reason = 'Exit reason'
        group = 'Group'
        discount_rate = 'Discount rate (%)'
        n_high_tests = 'High load tests'
        probability = 'Probability'
        test_costs = 'Testing costs'
        reaction_costs = 'Reactions costs'
        treatment_costs = 'Treatment costs'
        pregnancy_risks = 'Pregnancy related events'
        pregnancy_cost = 'Pregnancy risks costs'
        adverse_reaction = 'Adverse Reaction'
        failure = 'Failure'
        dead = 'Dead'
        dead_ar = 'Adverse reaction related death'
        time_to_death = 'Time to death'
        gdp_name = 'GDP'
        histogram_graph = 'histograms_'
        average_results_name = 'average_results_'
        acceptability_curve_name = 'acceptability_curve'
        threshold = 'Threshold'
        net_monetary_benefit_name = 'net_monetary_benefit'
        net_monetary_benefit = 'Net monetary benefit'
        net_monetary_benefit_initials = 'NMB'
        bmn_graph_name = 'net_monetary_benefit_ip_sensibility'
        bmn_graph_title = 'Net Monetary Benefit Sensibility'
        other = 'Other'
        international_sensibility = 'international_sensibility_'
        month_cost = 'Monthly cost'
        costs_total = 'costs_total'
        costs_percent = 'costs_percent'
        ce_plane = 'ce_plane_'
        if language == 'spa':
            qaly = 'AVAC'
            costs = 'Costos'
            medication_name = 'Terapia'
            treatment = 'Duracion Tratamiento'
            treatment_units = 'Años'
            acute_events = 'Eventos agudos'
            acute_events_unit = 'Cantidad total'
            chronic_events = 'Eventos cronicos'
            chronic_events_units = 'Ocurrencia'
            exit_reason = 'Motivo de salida'
            group = 'Grupo'
            discount_rate = 'Tasa de descuento (%)'
            n_high_tests = 'Pruebas altas'
            probability = 'Probabilidad'
            test_costs = 'Costo de Seguimiento'
            reaction_costs = 'Costos de eventos adversos'
            treatment_costs = 'Costos de la terapia'
            pregnancy_risks = 'Eventos relacionados con el embarazo'
            pregnancy_cost = 'Costos de riesgos de embarazo'
            adverse_reaction = 'Evento adverso'
            failure = 'Falla terapeutica'
            dead = 'Muerte'
            dead_ar = 'Muerte por eventos adversos'
            time_to_death = 'Tiempo a muerte'
            gdp_name = 'PIB'
            histogram_graph = 'histogramas_'
            average_results_name = 'resultados_promedio_'
            threshold = 'Disponibilidad a pagar'
            acceptability_curve_name = 'curva_aceptabilidad'
            net_monetary_benefit_name = 'beneficio_monetario_neto'
            net_monetary_benefit = 'Beneficio monetario neto'
            net_monetary_benefit_initials = 'BMN'
            bmn_graph_name = 'beneficio_monetario_neto_sensibilidad_ip'
            bmn_graph_title = 'Sensibilidad de Beneficio Monetario Neto'
            other = 'Otros'
            international_sensibility = 'sensibilidad_international_'
            month_cost = 'Costo mensual'
            costs_total = 'costos_totales'
            costs_percent = 'costos_porcentaje'
            ce_plane = 'plano_ce_'
        self.names = {'qaly': qaly, 'costs': costs, 'medication_name': medication_name, 'treatment': treatment,
                      'acute_events': acute_events, 'chronic_events': chronic_events, 'exit_reason': exit_reason,
                      'group': group, 'discount_rate': discount_rate, 'n_high_tests': n_high_tests, 'other': other,
                      'probability': probability, 'test_costs': test_costs, 'reaction_costs': reaction_costs,
                      'treatment_costs': treatment_costs, 'pregnancy_risks': pregnancy_risks, 'threshold': threshold,
                      'pregnancy_cost': pregnancy_cost, 'adverse_reaction': adverse_reaction, 'failure': failure,
                      'dead': dead, 'dead_ar': dead_ar, 'time_to_death': time_to_death, 'gdp_name': gdp_name,
                      'histogram_graph': histogram_graph, 'average_results_name': average_results_name,
                      'acceptability_curve_name': acceptability_curve_name, 'bmn_graph_name': bmn_graph_name,
                      'net_monetary_benefit': net_monetary_benefit, 'bmn_graph_title': bmn_graph_title,
                      'net_monetary_benefit_name': net_monetary_benefit_name, 'month_cost': month_cost,
                      'international_sensibility': international_sensibility, 'costs_total': costs_total,
                      'net_monetary_benefit_initials': net_monetary_benefit_initials, 'costs_percent': costs_percent,
                      'ce_plane': ce_plane}
        self.units = {'qaly': qaly, 'costs': currency, 'treatment': treatment_units, 'acute_events': acute_events_unit,
                      'chronic_events': chronic_events_units, 'n_high_tests': acute_events_unit}

        results = self.results.copy()
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events, 'group': group,
                                'exit_reason': exit_reason, 'discount_rate': discount_rate, 'treatment': treatment,
                                'n_high_tests': n_high_tests, 'time_to_death': time_to_death, 'test_cost': test_costs,
                                'adverse_cost': reaction_costs, 'medication_cost': treatment_costs}, inplace=True)
        if self.pregnancy_model:
            results.rename(columns={'pregnancy_risk_cost': pregnancy_cost, 'pregnancy_risks': pregnancy_risks},
                           inplace=True)
        n_iterations = results['iteration'].max()+1
        size = n_iterations * 4 if self.discount_rate else n_iterations
        r2 = results[[medication_name, exit_reason, 'iteration']].groupby([medication_name, exit_reason]).count()/size
        r2.reset_index(drop=False, inplace=True)
        r2 = r2.pivot_table('iteration', [medication_name], exit_reason)
        r2.fillna(0, inplace=True)
        r2.rename(columns={'adverse_reaction': adverse_reaction, 'failure': failure, 'dead': dead, 'dead_ar': dead_ar},
                  inplace=True)
        for c in [adverse_reaction, failure, dead, dead_ar]:
            if c not in list(r2.columns):
                r2[c] = 0.0
        list_cols = [discount_rate, medication_name, group, qaly, costs, acute_events, chronic_events, treatment,
                     n_high_tests, treatment_costs, test_costs, reaction_costs]
        if self.pregnancy_model:
            list_cols.append(pregnancy_cost)
            list_cols.append(pregnancy_risks)
        results = results[list_cols].groupby([discount_rate, medication_name, group]).mean()
        results = results.reset_index(drop=False)
        results = results.merge(r2, left_on=medication_name, right_on=medication_name, suffixes=(None, '_percent'))
        file_name = DIR_OUTPUT + average_results_name + self.group_name
        if self.distribution:
            file_name += '_d'
        if self.switch_cost is not None:
            file_name += '_' + self.switch_cost
        if self.pregnancy_model:
            file_name += '_p'
        file_name += '.csv'
        results.to_csv(file_name, index=False)
        self.average_results = results
        print('Exported:', file_name, dt.datetime.now())

    def generate_dispersion_graph(self):
        results = self.results[['qaly', 'costs', 'medication_name', 'group', 'discount_rate']].copy()
        qaly = self.names['qaly']
        costs = self.names['costs']
        medication_name = self.names['medication_name']
        group = self.names['group']
        discount_rate = self.names['discount_rate']
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name, 'group': group,
                                'discount_rate': discount_rate}, inplace=True)
        markers = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]
        mark = dict()
        names = list(results[medication_name].unique())
        for i in range(len(names)):
            mark[names[i]] = markers[i % len(markers)]
        sns.set(style="darkgrid")
        g = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=group, col=discount_rate,
                        style=medication_name, markers=mark, kind='scatter')
        for ax in g.axes.flat:
            y_labels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set_yticklabels(y_labels)  # set new labels
            ax.set_ylabel(costs + '(' + self.currency + ')')
        graph_name = DIR_OUTPUT + 'dispersion_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        if not self.distribution:
            sns.set(style="darkgrid")
            results = results[(results[discount_rate] == '3.5') | (results[discount_rate] == 3.5)]
            g = sns.relplot(x=qaly, y=costs, hue=medication_name, data=results, row=group,
                            style=medication_name, markers=mark, kind='scatter')
            for ax in g.axes.flat:
                ylabels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks() / 1000000]
                ax.set_yticklabels(ylabels)  # set new labels
                ax.set_ylabel(costs + ' (' + self.currency + ')')
            graph_name = DIR_OUTPUT + 'dispersion_3.5_' + self.group_name
            if self.switch_cost is not None:
                graph_name += '_' + self.switch_cost
            if self.pregnancy_model:
                graph_name += '_p'
            graph_name += '.png'
            plt.savefig(graph_name, bbox_inches="tight")
            print('Exported:', graph_name, dt.datetime.now())
            plt.clf()

    def generate_bar_plots(self):
        """
        Generates general information graphs for the study.
        """
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        results = self.results.copy()
        qaly = self.names['qaly']
        costs = self.names['costs']
        test_costs = self.names['test_costs']
        reaction_costs = self.names['reaction_costs']
        treatment_costs = self.names['treatment_costs']
        pregnancy_risks = self.names['pregnancy_risks']
        pregnancy_cost = self.names['pregnancy_cost']
        medication_name = self.names['medication_name']
        treatment = self.names['treatment'] + ' (' + self.units['treatment'] + ')'
        acute_events = self.names['acute_events']
        chronic_events = self.names['chronic_events']
        exit_reason = self.names['exit_reason']
        group = self.names['group']
        discount_rate = self.names['discount_rate']
        adverse_reaction = self.names['adverse_reaction']
        failure = self.names['failure']
        dead = self.names['dead']
        dead_ar = self.names['dead_ar']
        n_high_tests = self.names['n_high_tests']
        time_to_death = self.names['time_to_death']
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events, 'group': group,
                                'exit_reason': exit_reason, 'discount_rate': discount_rate, 'treatment': treatment,
                                'n_high_tests': n_high_tests, 'time_to_death': time_to_death, 'test_cost': test_costs,
                                'adverse_cost': reaction_costs, 'medication_cost': treatment_costs},
                       inplace=True)
        if self.pregnancy_model:
            results.rename(columns={'pregnancy_risk_cost': pregnancy_cost, 'pregnancy_risks': pregnancy_risks},
                           inplace=True)
        markers = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]
        mark = dict()
        names = list(results[medication_name].unique())
        for i in range(len(names)):
            mark[names[i]] = markers[i % len(markers)]
        sns.set(style="darkgrid")
        results = self.average_results
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
        if pregnancy_cost in columns_graph:
            columns_graph.remove(pregnancy_cost)
        n_cols = math.ceil(len(columns_graph) / 2)
        f, ax = plt.subplots(nrows=2, ncols=n_cols, figsize=(n_cols * 15, 25), sharey=False, sharex=False)
        i = 0
        j = 0
        for column in columns_graph:
            results.sort_values(by=column, ascending=True, inplace=True)
            if column == costs:
                rc = results[[medication_name, column]].copy()
                rc[column] = rc[column] / 1000000
                rc.plot(x=medication_name, y=column, kind='barh', ax=ax[j][i], subplots=True, sharey=False,
                        sharex=False, title=column + ' (' + self.currency + ')', legend=False)
                ax[j][i].xaxis.set_major_formatter(tick)
                ax[j][i].xaxis.set_label_text(self.currency)
                del rc
            else:
                results.plot(x=medication_name, y=column, kind='barh', ax=ax[j][i], subplots=True, sharey=False,
                             sharex=False, title=column, legend=False)
            i += 1
            if i == n_cols:
                i = 0
                j = j + 1
        f.suptitle('Bar plot')
        graph_name = DIR_OUTPUT + 'bar_plot_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        costs_cols = [medication_name, treatment_costs, test_costs, reaction_costs, costs]
        if self.pregnancy_model:
            costs_cols.append(pregnancy_cost)
        df_graph = results[costs_cols].copy()
        costs_cols.remove(medication_name)
        costs_cols.remove(costs)
        df_graph = df_graph.set_index(medication_name)
        df_graph = df_graph / 1000000
        df_graph.sort_values(by=[costs, treatment_costs, test_costs, reaction_costs], ascending=True, inplace=True)
        ax = df_graph[costs_cols].plot(kind='barh', title=costs, legend=True,
                                       stacked=True, figsize=(13, 13))
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        ax.xaxis.set_label_text(self.currency)
        graph_name = DIR_OUTPUT + self.names['costs_total'] + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

        for col in costs_cols:
            df_graph[col] = df_graph[col] / df_graph[costs] * 100
        df_graph.sort_values(by=[treatment_costs, test_costs, reaction_costs], ascending=True, inplace=True)
        ax = df_graph[costs_cols].plot(kind='barh', title=costs + ' (%)', legend=True, stacked=True, figsize=(13, 13))
        ax.set_xlim(0, 100)
        fmt = '{x:,.2f}' + '%'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        if self.language == 'spa':
            ax.xaxis.set_label_text('Proporcion')
            ax.yaxis.set_label_text('TAR')
        else:
            ax.xaxis.set_label_text('Proportion')
            ax.yaxis.set_label_text('ART')
        graph_name = DIR_OUTPUT + self.names['costs_percent'] + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

        results = results[exit_reasons].set_index(medication_name)
        results = results[[failure, adverse_reaction, dead, dead_ar]] * 100
        results.sort_values(by=[failure, adverse_reaction, dead], ascending=[True, True, True], inplace=True)
        ax = results.plot(kind='barh', title=exit_reason, legend=True, stacked=True, figsize=(13, 13))
        ax.set_xlim(0, 100)
        fmt = '{x:,.2f}' + '%'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        if self.language == 'spa':
            ax.xaxis.set_label_text('Proporcion')
            ax.yaxis.set_label_text('TAR')
        else:
            ax.xaxis.set_label_text('Proportion')
            ax.yaxis.set_label_text('ART')
        graph_name = DIR_OUTPUT + exit_reason + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

    def generate_histograms(self):
        """
        Generates general information graphs for the study.
        """
        results = self.results.copy()
        qaly = self.names['qaly']
        costs = self.names['costs']
        medication_name = self.names['medication_name']
        treatment = self.names['treatment'] + ' (' + self.units['treatment'] + ')'
        acute_events = self.names['acute_events'] + ' (' + self.units['acute_events'] + ')'
        chronic_events = self.names['chronic_events'] + ' (' + self.units['chronic_events'] + ')'
        exit_reason = self.names['exit_reason']
        group = self.names['group']
        discount_rate = self.names['discount_rate']
        n_high_tests = self.names['n_high_tests'] + ' (' + self.units['n_high_tests'] + ')'
        probability = self.names['probability']
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'acute_events': acute_events, 'chronic_event': chronic_events,
                                'exit_reason': exit_reason, 'group': group, 'discount_rate': discount_rate,
                                'treatment': treatment, 'n_high_tests': n_high_tests},
                       inplace=True)
        sns.set(style="darkgrid")
        # Histograms
        results = results[(results[discount_rate] == '3.5') | (results[discount_rate] == 3.5)][[
            medication_name, group, qaly, costs, acute_events, chronic_events, treatment, n_high_tests]]
        results[treatment] = results[treatment]
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(45, 25), sharey=False, sharex=False)
        i = 0
        j = 0
        for column in [qaly, costs, acute_events, chronic_events, treatment, n_high_tests]:
            g = sns.histplot(data=results, x=column, hue=medication_name, stat='probability', common_norm=False,
                             fill=False, cumulative=True, element='poly', ax=ax[j][i])
            if column == costs:
                xlabels = ['$' + '{:,.2f}'.format(x) + 'M' for x in g.get_xticks() / 1000000]
                g.set_xticklabels(xlabels)
                g.set_xlabel(costs + '(' + self.currency + ')')
            if column == acute_events:
                xlabels = [(x if round(x) == x else '') for x in g.get_xticks()]
                g.set_xticklabels(xlabels)
            if column == chronic_events:
                xlabels = list()
                for x in g.get_xticks():
                    if x == 0:
                        if self.language == 'spa':
                            xlabels.append('Sin evento')
                        else:
                            xlabels.append('No reaction')
                    elif round(x, 2) == 1.0:
                        if self.language == 'spa':
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
        graph_name = DIR_OUTPUT + self.names['histogram_graph'] + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()

    def generate_ce_plane(self, gdp: float):
        """
        Generates general information graphs for the study.
        :param gdp: Gross domestic product to use as reference
        """
        results = self.results.copy()
        qaly = self.names['qaly']
        costs = self.names['costs']
        medication_name = self.names['medication_name']
        group = self.names['group']
        discount_rate = self.names['discount_rate']
        gdp_name = self.names['gdp_name']
        results.rename(columns={'qaly': qaly, 'costs': costs, 'medication_name': medication_name,
                                'group': group, 'discount_rate': discount_rate}, inplace=True)
        markers = [".", "o", "v", "^", "<", ">", "s", "p", "P", "*", "X", "D"]
        mark = dict()
        names = list(results[medication_name].unique())
        for i in range(len(names)):
            mark[names[i]] = markers[i % len(markers)]
        sns.set(style="darkgrid")
        results = self.average_results
        graph_name = DIR_OUTPUT + self.names['ce_plane'] + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
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
                plane_lines.append({gdp_name: str(ite) + gdp_name, qaly: qaly_step*step, costs: cost_step*step})
        plane_lines = pd.DataFrame(plane_lines)
        for dr in list(results[discount_rate].unique()):
            sns.set(style="darkgrid")
            fig, ax = plt.subplots()
            df_graph = results[results[discount_rate] == dr].copy()
            sns.lineplot(data=plane_lines, x=qaly, y=costs, hue=gdp_name, style=gdp_name, legend='full',
                             ax=ax)
            sns.scatterplot(data=df_graph, x=qaly, y=costs, hue=medication_name, style=medication_name,
                                markers=mark, legend='full', ax=ax)
            ax.set_ylim(0, max_cost)
            ax.set_xlim(0, max_qaly)
            ax.set_title(discount_rate + ': ' + str(dr))
            ax.set_ylabel(costs + '(' + self.currency + ')')
            y_labels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks() / 1000000]
            ax.set_yticklabels(y_labels)
            # Put the legend out of the figure
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            graph_name_e = graph_name + '_DR' + str(dr) + '.png'
            plt.savefig(graph_name_e, bbox_inches="tight")
            print('Exported:', graph_name_e, dt.datetime.now())
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

    def acceptability_curve(self, min_value: float, max_value: float, n_steps: int = 1000,
                            n_iterations: int = 50000, step_value: float = 21088903.3):
        """
        Creates the associated graph for the percentage of acceptance of every option, with a considered range of
        thresholds and a number of intermediate steps to calculate.
        :param min_value: Minimum threshold to consider in the iterations.
        :param max_value: Maximum threshold to consider in the iterations.
        :param n_steps: Number of divisions to consider the different thresholds for the graph.
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
        medication_name = self.names['medication_name']
        threshold = self.names['threshold']
        probability = self.names['probability']
        group = self.names['group']
        name = self.names['acceptability_curve_name']
        discount_rate = self.names['discount_rate']
        gdp_name = self.names['gdp_name']
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
        graph_name = DIR_OUTPUT + name + '_' + self.group_name
        if self.distribution:
            file_name += '_d'
            graph_name += '_d'
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
                    insert_threshold.append({threshold: val_to_add, probability: 0,
                                             medication_name: str(gbp_value) + ' ' + gdp_name, group: gr,
                                             discount_rate: dr})
                    insert_threshold.append({threshold: val_to_add+1, probability: max_prob,
                                             medication_name: str(gbp_value) + ' ' + gdp_name, group: gr,
                                             discount_rate: dr})
            val_to_add += step_value
            gbp_value += 1
        pib_df = pd.DataFrame(insert_threshold)
        df = pd.concat([df, pib_df], ignore_index=True)
        df.to_csv(file_name, index=False)
        sns.set(style="darkgrid")
        g = sns.relplot(x=threshold, y=probability, hue=medication_name, data=df, row=group, col=discount_rate,
                    style=medication_name, kind='line')
        for ax in g.axes.flat:
            x_labels = ['$' + '{:,.0f}'.format(x) + 'M' for x in ax.get_xticks() / 1000000]
            x_labels2 = list()
            i = 0
            for x in x_labels:
                if i == 0:
                    x_labels2.append(x)
                else:
                    x_labels2.append('')
                i = abs(i-1)
            ax.set_xlim(min_value, max_value)
            ax.set_xticklabels(x_labels2)  # set new labels
            ax.set_xlabel(threshold + '(' + self.currency + ')')
            y_labels = ['{:,.0f}'.format(x) + '%' for x in ax.get_yticks() * 100]
            ax.set_yticklabels(y_labels)
            ax.set_ylim(0, max_prob)
        plt.savefig(graph_name, bbox_inches="tight")
        plt.clf()
        print('Aceptability curve generated')

    def export_net_monetary_benefit(self, thresholds: dict):
        """
        Calculates the percentage of best value considering the Net Monetary Benefit with a given threshold.
        :param thresholds: values to consider for the value of 1 QALY of improvement
        :return: Dataframe with the resulting percentage of acceptance for each set, with the given threshold.
        """
        df = self.results[['medication_name', 'discount_rate', 'qaly', 'costs']].copy()
        df = df.groupby(['medication_name', 'discount_rate']).mean().reset_index(drop=False)
        discount_rates = list(df['discount_rate'].unique())
        n_col = len(thresholds)
        n_row = len(discount_rates)
        medication_name = self.names['medication_name']
        threshold = self.names['threshold']
        discount_rate = self.names['discount_rate']
        gdp_name = self.names['gdp_name']
        file_name = self.names['net_monetary_benefit_name']
        graph_title = self.names['net_monetary_benefit']
        qaly = self.names['qaly']
        costs = self.names['costs']
        initials = self.names['net_monetary_benefit_initials']

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
            col_name = threshold + ':' + key + ' ' + gdp_name
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
                    ax[i].xaxis.set_label_text(initials + ' (' + self.currency + ')')
                else:
                    df3.plot(x=medication_name, y=subplot_title, kind='barh', ax=ax[j][i],
                                                   subplots=True, sharex=False, legend=False)
                    ax[j][i].xaxis.set_major_formatter(tick)
                    ax[j][i].xaxis.set_label_text(initials + ' (' + self.currency + ')')
                values = [df2[df2[discount_rate] == dr].columns] + list(df2[df2[discount_rate] == dr].values)
                wb.new_sheet('DR'+str(dr)+gdp_name+key, data=values)
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
                col_name = threshold + ':' + key + ' ' + gdp_name
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

    def international_price_sensibility(self, min_price: float = 27044, max_price: float = 2000000,
           n_steps: int = 50, gdp: float = 21088903.3):
        dor_base_prices = {'TAF_FTC__DOR': 3215059.74, 'TDF_FTC__DOR': 26595.35285}
        treatments = {'DOR': ['TAF_FTC__DOR', 'TDF_FTC__DOR'], 'TAF_FTC_BIC': ['TAF_FTC_BIC']}
        base_prices = dict()
        for sets in treatments:
            for treatment in treatments[sets]:
                medication_df = pd.read_csv(DIR_INPUT + treatment + '.csv')
                base_prices[treatment] = float(medication_df[medication_df['EVENT'] == 'adherent_month_cost'][
                                                    'BASE_VALUE'].mean())
        min_price_dor = min_price - min(dor_base_prices.values())
        min_price_bic = min_price
        step_size = (max_price - min_price) / (n_steps - 1)
        medication_name = self.names['medication_name']
        discount_rate = self.names['discount_rate']
        bmn_graph_name = self.names['bmn_graph_name']
        graph_title = self.names['bmn_graph_title']
        qaly = self.names['qaly']
        costs = self.names['costs']
        initials = self.names['net_monetary_benefit_initials']
        treatment_costs = self.names['treatment_costs']
        month_cost = self.names['month_cost']
        net_monetary_benefit = self.names['net_monetary_benefit']
        probability = self.names['probability']
        other = self.names['other']

        # Phase 1: MNB calculation

        # Load average values
        df_average = self.average_results.copy()
        df_average = df_average[df_average[discount_rate].isin(['3.5', 3.5])]
        df_average = df_average[df_average[medication_name].isin(['TAF_FTC__DOR', 'TDF_FTC__DOR', 'TAF_FTC_BIC'])]
        df_average = df_average[[medication_name, costs, qaly, treatment_costs]]
        df_average['base_prices'] = df_average[medication_name].apply(lambda x: base_prices.get(x))
        bmn_list = list()
        for step in tqdm(range(n_steps)):
            df_steps = df_average.copy()
            treatment_value = min_price + step*step_size
            df_steps[month_cost] = treatment_value
            df_steps[costs] = df_steps[costs] + df_steps[treatment_costs]*(treatment_value/df_steps['base_prices']-1)
            df_steps[initials] = (df_steps[qaly]*gdp-df_steps[costs])/1000000
            df_steps[month_cost] = df_steps[month_cost]/1000000
            bmn_list.append(df_steps[[medication_name, month_cost, initials]])
        df_average = pd.concat(bmn_list)
        del bmn_list
        df_average = df_average.pivot_table(values=initials, index=month_cost, columns=medication_name)
        f, ax = plt.subplots()
        df_average.plot.line(ax=ax)
        fmt = '$' + '{x:,.2f}' + 'M'
        tick = mtick.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)
        ax.yaxis.set_major_formatter(tick)
        ax.yaxis.set_label_text(net_monetary_benefit + ' (' + self.currency + ')')
        ax.xaxis.set_label_text(month_cost + ' (' + self.currency + ')')
        ax.set_title(graph_title)

        graph_name = DIR_OUTPUT + bmn_graph_name + '_' + self.group_name
        if self.distribution:
            graph_name += '_d'
        if self.switch_cost is not None:
            graph_name += '_' + self.switch_cost
        if self.pregnancy_model:
            graph_name += '_p'
        graph_name += '.png'
        plt.savefig(graph_name, bbox_inches="tight")
        print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
        del df_average

        # Step 2: 3 D Sensibility

        results = self.results[self.results['discount_rate'].isin(['3.5', 3.5])].copy()
        results.drop(columns='discount_rate', inplace=True)

        static_results = results[~results['medication_name'].isin(['TAF_FTC__DOR', 'TDF_FTC_DOR', 'TAF_FTC_BIC'])]
        dynamic_results = results[results['medication_name'].isin(['TAF_FTC__DOR', 'TDF_FTC_DOR', 'TAF_FTC_BIC'])]
        del results
        static_results = static_results[['iteration', 'medication_name', 'costs', 'qaly']]
        static_results[initials] = (static_results['qaly'] * gdp - static_results['costs']) / 1000000
        static_results = static_results[['iteration', 'medication_name', initials]]
        dynamic_results = dynamic_results[['iteration', 'medication_name', 'costs', 'medication_cost', 'qaly']]
        dynamic_results['base_prices'] = dynamic_results['medication_name'].apply(lambda x: base_prices.get(x))
        dor_results = list()
        for i in range(n_steps):
            dor_price = min_price_dor + step_size * i
            for treatment in treatments['DOR']:
                df_steps = dynamic_results[dynamic_results['medication_name'] == treatment].copy()
                df_steps['DOR'] = dor_price
                treatment_value = dor_price + dor_base_prices[treatment]
                df_steps['costs'] = df_steps['costs'] + df_steps['medication_cost'] * (
                        treatment_value / df_steps['base_prices'] - 1)
                df_steps[initials] = (df_steps['qaly'] * gdp - df_steps['costs']) / 1000000
                df_steps['DOR'] = df_steps['DOR'] / 1000000
                dor_results.append(df_steps[['iteration', 'medication_name', 'DOR', initials]])
        dor_results = pd.concat(dor_results)
        dor_results['TAF_FTC_BIC'] = 0
        bic_results = list()
        for i in range(n_steps):
            bic_price = min_price_bic + step_size * i
            treatment = 'TAF_FTC_BIC'
            df_steps = dynamic_results[dynamic_results['medication_name'] == treatment].copy()
            df_steps['TAF_FTC_BIC'] = bic_price
            treatment_value = bic_price
            df_steps['costs'] = df_steps['costs'] + df_steps['medication_cost'] * (
                    treatment_value / df_steps['base_prices'] - 1)
            df_steps[initials] = (df_steps['qaly'] * gdp - df_steps['costs']) / 1000000
            df_steps['TAF_FTC_BIC'] = df_steps['TAF_FTC_BIC'] / 1000000
            bic_results.append(df_steps[['iteration', 'medication_name', 'TAF_FTC_BIC', initials]])
        bic_results = pd.concat(bic_results)
        bic_results['DOR'] = 0
        final_results = list()
        for i in tqdm(range(n_steps)):
            dor_price = (min_price_dor + step_size * i)/1000000
            dor_df = dor_results[dor_results['DOR'] == dor_price]
            static_results['DOR'] = dor_price
            bic_results['DOR'] = dor_price
            for j in range(n_steps):
                bic_price = (min_price_bic + step_size * j)/1000000
                bic_df = bic_results[bic_results['TAF_FTC_BIC'] == bic_price]
                dor_df['TAF_FTC_BIC'] = bic_price
                static_results['TAF_FTC_BIC'] = bic_price
                consolidated_df = pd.concat([dor_df.copy(), bic_df, static_results])
                max_df = consolidated_df[['TAF_FTC_BIC', 'DOR', 'iteration', initials]].groupby([
                    'TAF_FTC_BIC', 'DOR', 'iteration']).max()
                max_df.reset_index(drop=False, inplace=True)
                consolidated_df = consolidated_df.merge(max_df, left_on='iteration', right_on='iteration',
                                                        suffixes=(None, '_max'))
                del max_df
                consolidated_df['is_best'] = np.where(consolidated_df[initials] == consolidated_df[initials+'_max'],
                                                      1, 0)
                consolidated_df = consolidated_df[['TAF_FTC_BIC', 'DOR', 'medication_name', 'is_best']].groupby([
                    'TAF_FTC_BIC', 'DOR', 'medication_name']).mean().reset_index(drop=False)
                consolidated_df.sort_values(by='is_best', ascending=False, inplace=True)
                final_results.append(consolidated_df)
        print('Consolidating results:', dt.datetime.now())
        final_results = pd.concat(final_results)
        final_results['is_best'] = final_results['is_best']*100
        print('Results have been consolidated, starting graph design')

        fig = plt.figure(figsize=(15, 8))
        ax = fig.gca(projection='3d')
        i = 0
        medication_list = list(final_results['medication_name'].unique())
        colors = ['Red', 'Green', 'Blue']
        markers = ["*", "+", "x", ">", "s", "p", "P", "*", "X", "D"]
        alphas = [1, 1, 1]
        entered = False
        for medication in tqdm(medication_list):
            cur_df = final_results[final_results['medication_name'] == medication]
            if medication in ['TAF_FTC__DOR', 'TDF_FTC__DOR', 'TAF_FTC_BIC']:
                ax.plot(cur_df['TAF_FTC_BIC'], cur_df['DOR'], cur_df['is_best'], linewidth=0, marker=markers[i],
                                label=medication, color=colors[i], alpha=alphas[i])
                i = i+1
            else:
                if entered:
                    ax.plot(cur_df['TAF_FTC_BIC'], cur_df['DOR'], cur_df['is_best'], linewidth=0, marker='.',
                            color='Gray', alpha=0.25)
                else:
                    ax.plot(cur_df['TAF_FTC_BIC'], cur_df['DOR'], cur_df['is_best'], linewidth=0, marker='.',
                            color='Gray', alpha=0.25, label=other)
                    entered = True

        ax._facecolors2d = ax._facecolor
        ax.set_xlabel('TAF_FTC_BIC (' + self.currency + ')', rotation=150, labelpad=15)
        ax.set_ylabel('DOR (' + self.currency + ')', labelpad=15)
        ax.set_zlabel(probability, rotation=180, labelpad=20)
        y_labels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_yticks()]
        ax.set_yticklabels(y_labels)  # set new labels
        x_labels = ['$' + '{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()]
        ax.set_xticklabels(x_labels)  # set new labels
        z_labels = ['{:,.2f}'.format(x) + '%' for x in ax.get_zticks()]
        ax.set_zticklabels(z_labels)  # set new labels
        ax.legend()
        angle1 = 20
        for angle2 in [40, 80]:
            # Set the angle of the camera
            ax.view_init(angle1, angle2)
            graph_name = DIR_OUTPUT + self.names['international_sensibility'] + self.group_name
            if self.distribution:
                graph_name += '_d'
            if self.switch_cost is not None:
                graph_name += '_' + self.switch_cost
            if self.pregnancy_model:
                graph_name += '_p'
            graph_name += '_' + str(angle1) + '_' + str(angle2)
            graph_name += '.png'
            plt.savefig(graph_name)
            print('Exported:', graph_name, dt.datetime.now())
        plt.clf()
