import pandas as pd
from logic.simulation import Model
from logic.simulation_stochastic import Model as StochasticModel
from logic.data_processing import DataProcessing
from root import DIR_INPUT
import datetime as dt
from tqdm import tqdm

m = Model()
print(dt.datetime.now())
lists = pd.read_csv(DIR_INPUT+'Esquemas.csv')
n_scenario = {'incidence': 'BASE_VALUE', 'base_qaly': 'BASE_VALUE', 'high_test_qaly': 'BASE_VALUE',
              'test_cost': 'BASE_VALUE', 'adherence': 'BASE_VALUE'}
n_iterations = 50000
ac_min_value = 21088903.3
ac_max_value = 100000000
inflation_rates = {'0': 0, '3.5': 0.00287089871907664, '5': 0.00407412378364835, '12': 0.00948879293458305}
tuple_list = list()
for x in lists['Treatment'].unique():
    for inflation_rate in inflation_rates:
        for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
            tuple_list.append((x, inflation_rate, scenario))
for new_tuple in tqdm(tuple_list):
    x, inflation_rate, scenario = new_tuple
    print('\n', x, inflation_rate, scenario, dt.datetime.now())
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    m.parallel_simulation(medication_name=x, group=group, n_simulations=n_iterations, scenario=scenario,
                                  inflation_rate=(inflation_rate, 1+inflation_rates[inflation_rate]))
for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
    print(scenario, dt.datetime.now())
    analysis = DataProcessing(name_list=list(lists['Treatment'].unique()), stochastic=False, discount_rate=True,
                              switch_cost=scenario['switch_cost'])
    print(dt.datetime.now())
    analysis.generate_dispersion(language='spa')
    print(dt.datetime.now())
    analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=100, language='spa')
print(dt.datetime.now())