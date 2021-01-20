import pandas as pd
from logic.simulation_stochastic import Model
from logic.data_processing import DataProcessing
from root import DIR_INPUT
import datetime as dt
from tqdm import tqdm

m = Model()
lists = pd.read_csv(DIR_INPUT+'Esquemas.csv')
print(dt.datetime.now())
n_scenario = {'incidence': 'BASE_VALUE', 'base_qaly': 'BASE_VALUE', 'high_test_qaly': 'BASE_VALUE',
              'test_cost': 'BASE_VALUE', 'adherence': 'BASE_VALUE'}
n_iterations = 100000
for x in tqdm(lists['Treatment'].unique()):
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    m.parallel_simulation(medication_name=x, group=group, n_simulations=n_iterations)
analysis = DataProcessing(list(lists['Treatment'].unique()))
analysis.generate_dispersion(language='spa')
analysis.acceptability_curve(min_value=21088903.3, max_value=100000000, n_steps=100, language='spa')
print(dt.datetime.now())
