import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

def load_data(path):
    data = pd.read_csv(path)
    return data

def run():
    df = load_data('./data/example_wp_log_peyton_manning.csv')
    df.head()

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=0)
    future.tail()
    
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    print(forecast)

    fig1 = m.plot(forecast)
    plt.show()
    # fig2 = m.plot_components(forecast)

run()

