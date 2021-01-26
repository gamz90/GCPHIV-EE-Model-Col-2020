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
ac_min_value = 21088903.3
ac_max_value = 100000000
inflation_rates = {'0': 1.0, '3.5': 1.00287089871907664, '5': 1.00407412378364835, '12': 1.00948879293458305}
tuple_list = list()
cores = multiprocessing.cpu_count()
treatment_list = list(lists['Treatment'].unique())
m = Model()
for x in treatment_list:
    for inflation_rate in inflation_rates:
        ir = (inflation_rate, inflation_rates[inflation_rate])
        for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
            group = lists[lists['Treatment'] == x]['Group'].unique()[0]
            tuple_list.append((x, scenario, group, n_iterations, ir, False))
print('Calculations starting - Please hold the use of the computer until it finishes', dt.datetime.now())
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)
del m

for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
    print(scenario, dt.datetime.now())
    analysis = DataProcessing(name_list=list(lists['Treatment'].unique()), stochastic=False, discount_rate=True,
                              switch_cost=scenario['switch_cost'])
    print(dt.datetime.now())
    analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=100, language='spa', n_iterations=n_iterations)
    print(dt.datetime.now())
    analysis.generate_dispersion(language='spa', n_iterations=n_iterations)
    del analysis
print(dt.datetime.now())

m = DModel()
tuple_list = list()
for x in lists['Treatment'].unique():
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    tuple_list.append((x, group, n_iterations, (inflation_rates['5']), False))

with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)
del m
print(dt.datetime.now())
analysis = DataProcessing(name_list=list(lists['Treatment'].unique()), stochastic=True)
print(dt.datetime.now())
analysis.generate_dispersion(language='spa', n_iterations=n_iterations)
print(dt.datetime.now())
analysis.acceptability_curve(min_value=21088903.3, max_value=100000000, n_steps=100, language='spa', n_iterations=n_iterations)
print(dt.datetime.now())
