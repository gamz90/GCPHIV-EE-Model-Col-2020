import pandas as pd
from logic.simulation import Model
from logic.simulation_distribution import DistributionModel
from logic.data_processing import DataProcessing
from logic.simulation_p import PregnancyModel
from logic.simulation_p_distribution import PregnancyDistributionModel
from root import DIR_INPUT
import datetime as dt
import multiprocessing

print(dt.datetime.now())
lists = pd.read_csv(DIR_INPUT+'Esquemas.csv')
n_iterations = 50000
gdp = 21088903.3
thresholds = dict()
for i in range(1, 4):
    thresholds[str(i)] = gdp*i
n_steps = 100
ac_min_value = 20000000
ac_max_value = 100000000
inflation_rates = {'0': 1.0, '3.5': 1.00287089871907664, '5': 1.00407412378364835, '12': 1.00948879293458305}
cores = multiprocessing.cpu_count()
lists.sort_values(by='Treatment', inplace=True)
treatment_list = {'ALL': list(lists['Treatment'].unique()),
                  'LOCAL': list(lists[lists['International_price'] == 0]['Treatment'].unique()),
                  'PREGNANT': list(lists[lists['Pregnant'] == 1]['Treatment'].unique()),
                  'PREGNANT_LOCAL': list(lists[(lists['Pregnant'] == 1) & (lists['International_price'] == 0)][
                                             'Treatment'].unique()),
                  }
m = Model()
tuple_list = list()
for x in treatment_list['ALL']:
    for inflation_rate in inflation_rates:
        ir = (inflation_rate, inflation_rates[inflation_rate])
        for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
            group = lists[lists['Treatment'] == x]['Group'].unique()[0]
            tuple_list.append((x, scenario, group, n_iterations, ir))
print('Static parameters model Calculations starting - Please hold the use of the computer until it finishes',
      dt.datetime.now())
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)

m = DistributionModel()
tuple_list = list()
for x in treatment_list['ALL']:
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    tuple_list.append((x, group, n_iterations, inflation_rates['3.5']))
print('Distribution model Calculations starting - Please hold the use of the computer until it finishes',
      dt.datetime.now())
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)
del m
m = PregnancyModel()
tuple_list = list()
for x in treatment_list['PREGNANT']:
    for inflation_rate in inflation_rates:
        ir = (inflation_rate, inflation_rates[inflation_rate])
        for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
            group = lists[lists['Treatment'] == x]['Group'].unique()[0]
            tuple_list.append((x, scenario, group, n_iterations, ir))
print('Pregnancy model Calculations starting - Please hold the use of the computer until it finishes',
      dt.datetime.now())
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)

m = PregnancyDistributionModel()
tuple_list = list()
for x in treatment_list['PREGNANT']:
    group = lists[lists['Treatment'] == x]['Group'].unique()[0]
    tuple_list.append((x, group, n_iterations, inflation_rates['3.5']))
print('Pregnancy distribution model calculations starting - Please hold the use of the computer until it finishes',
      dt.datetime.now())
with multiprocessing.Pool(processes=cores) as pool:
    pool.starmap(m.parallel_simulation, tuple_list)
del m

for treatments in ['ALL', 'LOCAL']:
    print(treatments)
    for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
        print(scenario, dt.datetime.now())
        analysis = DataProcessing(name_list=treatment_list[treatments], group_name=treatments, distribution=False,
                                  discount_rate=True, switch_cost=scenario['switch_cost'], language='spa')
        print(dt.datetime.now(), 'Loaded information')
        analysis.generate_bar_plots()
        print(dt.datetime.now(), 'Generated bar plots')
        analysis.generate_histograms()
        print(dt.datetime.now(), 'Generated histograms')
        analysis.generate_ce_plane(gdp=gdp)
        print(dt.datetime.now(), 'Generated ce_planes')
        analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=n_steps,
                                     n_iterations=n_iterations, step_value=gdp)
        print(dt.datetime.now(), 'Generated aceptability curve')
        analysis.export_net_monetary_benefit(thresholds=thresholds)
        print(dt.datetime.now(), 'Generated net monetary benefit')
        analysis.generate_dispersion_graph()
        print(dt.datetime.now(), 'Generated dispersion graph')
        if treatments == 'ALL' and scenario['switch_cost'] == 'BASE_VALUE':
            analysis.international_price_sensibility(min_price=27044, max_price=3000000, n_steps=25,
                                                     gdp=21088903.3)
            print(dt.datetime.now(), 'Generated international price sensibility')
        del analysis
    print('Distribution model', dt.datetime.now())
    analysis = DataProcessing(name_list=treatment_list[treatments], group_name=treatments, distribution=True,
                              language='spa')
    print(dt.datetime.now(), 'Loaded information')
    analysis.generate_bar_plots()
    print(dt.datetime.now(), 'Generated bar plots')
    analysis.generate_histograms()
    print(dt.datetime.now(), 'Generated histograms')
    analysis.generate_ce_plane(gdp=gdp)
    print(dt.datetime.now(), 'Generated ce_planes')
    analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=n_steps,
                                 n_iterations=n_iterations, step_value=gdp)
    print(dt.datetime.now(), 'Generated aceptability curve')
    analysis.export_net_monetary_benefit(thresholds=thresholds)
    print(dt.datetime.now(), 'Generated net monetary benefit')
    analysis.generate_dispersion_graph()
    print(dt.datetime.now(), print(dt.datetime.now(), 'Generated dispersion graph'))
    if treatments == 'ALL':
        print(dt.datetime.now())
        analysis.international_price_sensibility(min_price=27044, max_price=3000000, n_steps=25,
                                                 gdp=21088903.3)
        print(dt.datetime.now(), 'Generated international price sensibility')

for treatments in ['PREGNANT', 'PREGNANT_LOCAL']:
    print(treatments)
    for scenario in [{'switch_cost': 'BASE_VALUE'}, {'switch_cost': 'INF_VALUE'}, {'switch_cost': 'SUP_VALUE'}]:
        print(scenario, dt.datetime.now())
        analysis = DataProcessing(name_list=treatment_list[treatments], group_name=treatments, distribution=False,
                                  discount_rate=True, switch_cost=scenario['switch_cost'],
                                  pregnancy_model=True, language='spa')
        print(dt.datetime.now(), 'Loaded information')
        analysis.generate_bar_plots()
        print(dt.datetime.now(), 'Generated bar plots')
        analysis.generate_histograms()
        print(dt.datetime.now(), 'Generated histograms')
        analysis.generate_ce_plane(gdp=gdp)
        print(dt.datetime.now(), 'Generated ce_planes')
        analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=n_steps,
                                     n_iterations=n_iterations, step_value=gdp)
        print(dt.datetime.now(), 'Generated aceptability curve')
        analysis.export_net_monetary_benefit(thresholds=thresholds)
        print(dt.datetime.now(), 'Generated net monetary benefit')
        analysis.generate_dispersion_graph()
        print(dt.datetime.now(), print(dt.datetime.now(), 'Generated dispersion graph'))
        if treatments == 'PREGNANT' and scenario['switch_cost'] == 'BASE_VALUE':
            print(dt.datetime.now())
            analysis.international_price_sensibility(min_price=27044, max_price=3000000, n_steps=25,
                                                     gdp=21088903.3)
            print(dt.datetime.now(), 'Generated international price sensibility')
        del analysis
    print('Distribution model', dt.datetime.now())
    analysis = DataProcessing(name_list=treatment_list[treatments], group_name=treatments, distribution=True,
                              pregnancy_model=True, language='spa')
    print(dt.datetime.now(), 'Loaded information')
    analysis.generate_bar_plots()
    print(dt.datetime.now(), 'Generated bar plots')
    analysis.generate_histograms()
    print(dt.datetime.now(), 'Generated histograms')
    analysis.generate_ce_plane(gdp=gdp)
    print(dt.datetime.now(), 'Generated ce_planes')
    analysis.acceptability_curve(min_value=ac_min_value, max_value=ac_max_value, n_steps=n_steps,
                                 n_iterations=n_iterations, step_value=gdp)
    print(dt.datetime.now(), 'Generated aceptability curve')
    analysis.export_net_monetary_benefit(thresholds=thresholds)
    print(dt.datetime.now(), 'Generated net monetary benefit')
    analysis.generate_dispersion_graph()
    print(dt.datetime.now(), print(dt.datetime.now(), 'Generated dispersion graph'))
    if treatments == 'PREGNANT':
        print(dt.datetime.now())
        analysis.international_price_sensibility(min_price=27044, max_price=3000000, n_steps=25,
                                                 gdp=21088903.3)
        print(dt.datetime.now(), 'Generated international price sensibility')
print(dt.datetime.now())
