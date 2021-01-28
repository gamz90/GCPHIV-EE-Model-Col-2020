import pandas as pd
from logic.simulation import Model
from logic.simulation_stochastic import Model as DModel
from logic.data_processing import DataProcessing
from root import DIR_INPUT
import datetime as dt
import multiprocessing

print(dt.datetime.now())
lists = pd.read_csv(DIR_INPUT+'Esquemas.csv')
n_scenario = {'incidence': 'BASE_VALUE', 'base_qaly': 'BASE_VALUE', 'high_test_qaly': 'BASE_VALUE',
              'test_cost': 'BASE_VALUE', 'adherence': 'BASE_VALUE'}
n_iterations = 50000
pib = 21088903.3
n_steps = 100
ac_min_value = 20000000
ac_max_value = 100000000
inflation_rates = {'0': 1.0, '3.5': 1.00287089871907664, '5': 1.00407412378364835, '12': 1.00948879293458305}
tuple_list = list()
cores = multiprocessing.cpu_count()
treatment_list = {'ALL': list(lists['Treatment'].unique()), 'SECOND_GROUP': list(lists[lists['Second_group'] == 1]['Treatment'].unique()), 'LOCAL': list(lists[lists['International_price'] == 0]['Treatment'].unique()), 'SECOND_GROUP_LOCAL': list(lists[(lists['Second_group'] == 1) & (lists['International_price'] == 0)]['Treatment'].unique())}
m = Model()
for x in treatment_list['ALL']:
    for inflation_rate in inflation_rates:
        ir = (inflation_rate, inflation_rates[inflation_rate])
        for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
            group = lists[lists['Treatment'] == x]['Group'].unique()[0]
            tuple_list.append((x, scenario, group, n_iterations, ir, False))
print('Calculations starting - Please hold the use of the computer until it finishes', dt.datetime.now())
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)
m = DModel()
tuple_list = list()
for x in lists['Treatment'].unique():
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    tuple_list.append((x, group, n_iterations, (inflation_rates['5']), False))
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)
del m
for treatments in treatment_list:
    print(treatments)
    for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
        print(scenario, dt.datetime.now())
        analysis = DataProcessing(name_list=treatment_list[treatments], group_name=treatments, stochastic=False, discount_rate=True,
                                  switch_cost=scenario['switch_cost'])
        print(dt.datetime.now())
        analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=n_steps, language='spa',
                                     n_iterations=n_iterations, step_value=pib)
        print(dt.datetime.now())
        analysis.generate_dispersion(language='spa', n_iterations=n_iterations, has_discount_rate=True)
        del analysis
    print('Distribution model', dt.datetime.now())
    analysis = DataProcessing(name_list=treatment_list[treatments], group_name=treatments, stochastic=True)
    print(dt.datetime.now())
    analysis.generate_dispersion(language='spa', n_iterations=n_iterations, has_discount_rate=False)
    print(dt.datetime.now())
    analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=n_steps, language='spa', n_iterations=n_iterations, step_value=pib)
    print(dt.datetime.now())
print(dt.datetime.now())
