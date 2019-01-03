import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt


def load_data(path, column_mappings = {'Time': 'ds', 'prod08.ld5': 'y'}):
    data = pd.read_csv(path)

    if column_mappings is not None:
        data = data.rename(columns = column_mappings)
    return data

def fix_nan_data(df):
    for index, row in df.iterrows():
        if row['y'].strip() == 'No Data':
            row.loc['y'] = None
            df.loc[index] = row

def run():
    # df = load_data('./data/example_wp_log_peyton_manning.csv')
    # df = load_data('./data/santaba-demo4.csv')
    # df = load_data('./data/relayserver-CPU-1-month.csv', { 'Time': 'ds', 'cpu_usagePercentCores': 'y' })
    df = load_data('./data/relayserver-CPU-3-month.csv', { 'Time': 'ds', 'cpu_usagePercentCores': 'y' })
    # df = load_data('./data/HTTPS-Response_Time.csv', { 'Time': 'ds', 'Page load time': 'y' })
    
    fix_nan_data(df)

    df.head()

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods = 0)
    future.tail()
    
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = m.plot(forecast)
    plt.title('Relayserver CPU 1 Month', verticalalignment='baseline')

    for i in range(len(forecast.values)):
        p = forecast.values[i]
        current = df['y'][i]
        ds = p[0]
        lower = p[2]
        upper = p[3]

        if current is None or lower is None or upper is None:
            continue

        current = float(current)
        if current > upper or current < lower:
            plt.plot(ds, current, 'ro')

    plt.show()

run()

