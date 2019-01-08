from fbprophet import Prophet

try:
    from matplotlib import pyplot as plt
    from matplotlib.dates import MonthLocator, num2date
    from matplotlib.ticker import FuncFormatter
except ImportError:
    logger.error('Importing matplotlib failed. Plotting will not work.')

class MyProphet(Prophet):
    def plot(self, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6)):
        m = self
        if ax is None:
            fig = plt.figure(facecolor='w', figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        fcst_t = fcst['ds'].dt.to_pydatetime()
        ax.plot(m.history['ds'].dt.to_pydatetime(), m.history['y'])
        # ax.plot(fcst_t, fcst['yhat'], ls='-', c='#0072B2')
        if 'cap' in fcst and plot_cap:
            ax.plot(fcst_t, fcst['cap'], ls='--', c='k')
        if m.logistic_floor and 'floor' in fcst and plot_cap:
            ax.plot(fcst_t, fcst['floor'], ls='--', c='k')
        if uncertainty:
            ax.fill_between(fcst_t, fcst['yhat_lower'], fcst['yhat_upper'],
                            color='#0072B2', alpha=0.2)
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig
        