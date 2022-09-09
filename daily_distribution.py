import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### ========================================= ###
def parse_columns(name: str) -> int:
    return int(name.split('_')[0])

def cast_float(col: pd.Series) -> pd.Series:
    return col.str.replace(',', '.').astype(float)

def format_data(data: pd.DataFrame) -> pd.DataFrame:
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    
    dfs = []
    for year in years:
        cols = [c for c in data.columns if parse_columns(c) == year]
        tmp = data[cols]
        tmp['year'] = year
        tmp.set_index('year', append=True, inplace=True)
        tmp.columns = ['tipo_dia', 'spot_diario']
        tmp['spot_diario'] = cast_float(tmp['spot_diario'])
        dfs.append(tmp)
    
    return pd.concat(dfs)

# Plots of the chosen parameter on a horuly basis for the diffent months
def make_hourly_plot(data: pd.DataFrame, var: str, months: list[str]) -> None:
    fig, axes = plt.subplots(nrows=12, sharex=True, figsize=(12, 30))
    for idx, ax in enumerate(axes):
        tmp = data.loc[data['Mes'] == idx+1]
        p = sns.lineplot(data=tmp, x='Hora', y=var, hue='tipo_dia', ax=ax)
        ax.set_title(months[idx])
        
        for ind, label in enumerate(p.get_xticklabels()):
            if ind % 4 == 0:  
                label.set_visible(True)
            else:
                label.set_visible(False)

    fig.savefig(f'Datos\imgs\hourly profiles\hourly_{var}_price_data_by_months.png')

# Plots of the chosen parameter on a horuly basis for the diffent months
def make_hourly_plot_by_years(data: pd.DataFrame, var: str, months: list[str]) -> None:
    fig, axes = plt.subplots(nrows=12, sharex=True, figsize=(12, 30))
    for idx, ax in enumerate(axes):
        tmp = data.loc[data['Mes'] == idx+1]
        p = sns.lineplot(data=tmp, x='Hora', y=var, hue='year', ax=ax, palette='tab10', ci=None)
        ax.set_title(months[idx])

        
        for ind, label in enumerate(p.get_xticklabels()):
            if ind % 4 == 0:  
                label.set_visible(True)
            else:
                label.set_visible(False)

    fig.savefig(f'Datos\imgs\hourly profiles\year profiles\hourly_{var}_price_data_by_months_and_year.png')

#### ======================================================== ####

prices = pd.read_csv(r"Datos\Precio\hourly_price_extraction.csv", delimiter=';')
prices.set_index(['Mes','Día','Hora'], inplace=True)    
data_formatted = format_data(prices)

### Daily mean value
data_daily = data_formatted.reset_index() \
                            .groupby(['year','Mes','Día'])['spot_diario'] \
                            .agg('mean') \
                            .reset_index()

data_factors = pd.merge(data_formatted.reset_index(), data_daily, on=['year','Mes','Día'], suffixes=['_hourly','_daily'], how='outer')

# Price factor calculation
data_factors['price_factor'] = data_factors['spot_diario_hourly'] / data_factors['spot_diario_daily']
data_factors_summary = data_factors.groupby(['Mes', 'Hora', 'tipo_dia'])['price_factor'] \
                                    .agg(['mean','std']) \
                                    .reset_index()
data_factors_summary['low_bound'] = data_factors_summary['mean'] - data_factors_summary['std']
data_factors_summary['high_bound'] = data_factors_summary['mean'] + data_factors_summary['std']
data_factors_summary.to_csv(r"Datos\HourlyPriceProfiles\hourly_price_factors.csv", sep=";", decimal=',')

######### Processing data ###############
# Grouping takes into account the years, to look for clear differences between years
data_summary = data_formatted.reset_index() \
                            .groupby(['tipo_dia','year','Mes','Hora'])['spot_diario'] \
                            .agg(["mean","std","min","max"]) \
                            .reset_index()

data_summary.to_csv(r"Datos\HourlyPriceProfiles\hourly_prices_by_years.csv", sep=";", decimal=',')

months = [datetime.date(1900, m, 1).strftime('%B') for m in range(1,13)]
months = data_summary['Mes'].unique()
    
data_summary_2 = data_formatted.reset_index() \
                            .groupby(['tipo_dia','Mes','Hora'])['spot_diario'] \
                            .agg(["mean","std","min","max"]) \
                            .reset_index()
data_summary_2.to_csv(r"Datos\HourlyPriceProfiles\hourly_prices_by_months.csv", sep=";", decimal=',')

data_summary_daily = data_formatted.reset_index() \
                            .groupby(['tipo_dia','Mes'])['spot_diario'] \
                            .agg(["mean","std","min","max"]) \
                            .reset_index()
data_summary_daily.to_csv(r"Datos\HourlyPriceProfiles\daily_prices_by_months.csv", sep=";", decimal=',')


################### Price factors ######################

data_expanded = pd.merge(data_summary_2, data_summary_daily, on=['tipo_dia', 'Mes'], suffixes=['_hourly', '_daily'])
data_expanded['price_factor'] = data_expanded['mean_hourly'] / data_expanded['mean_daily']
data_expanded.to_csv(r"Datos\HourlyPriceProfiles\daily_hourly_prices_by_months.csv", sep=";", decimal=',')

data_expanded.head()

################## Plots ########################
g = sns.FacetGrid(data=data_summary,
            row='Mes', 
            col='tipo_dia')
g.map_dataframe(sns.lineplot, x='Hora', y='mean', hue='year')

cats = ['mean','std','min','max']
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
fig, axes = plt.subplots(ncols=4, sharex=True, figsize=(24, 8))
for idx, ax in enumerate(axes):
    p = sns.barplot(data=data_summary_daily, x='Mes', y=cats[idx], hue='tipo_dia', ax=ax)
fig.savefig('Datos\imgs\daily_data_by_month_and_day_type.png')

print("Plotting hourly profiles by months")
[make_hourly_plot(data_summary_2, var, MONTHS) for var in cats]

print("Plotting hourly profiles by months and years")
[make_hourly_plot_by_years(data_summary, var, MONTHS) for var in cats]
