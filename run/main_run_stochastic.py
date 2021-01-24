import pandas as pd
from logic.simulation_stochastic import Model
from logic.data_processing import DataProcessing
from root import DIR_INPUT
import datetime as dt
from tqdm import tqdm

m = Model()
lists = pd.read_csv(DIR_INPUT+'Esquemas.csv')
print(dt.datetime.now())
n_iterations = 50000
inflation_rates = {'0': 0, '3.5': 0.00287089871907664, '5': 0.00407412378364835, '12': 0.00948879293458305}
for x in tqdm(lists['Treatment'].unique()):
    print(x)
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    m.parallel_simulation(medication_name=x, group=group, n_simulations=n_iterations,
                          inflation_rate=(1+inflation_rates['5']))
print(dt.datetime.now())
analysis = DataProcessing(name_list=list(lists['Treatment'].unique()), stochastic=True)
print(dt.datetime.now())
analysis.generate_dispersion(language='spa')
print(dt.datetime.now())
analysis.acceptability_curve(min_value=21088903.3, max_value=100000000, n_steps=100, language='spa')
print(dt.datetime.now())
