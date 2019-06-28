from examples.residential_mg_with_pv_and_dewhs.plotting.plotting_helper import *
from examples.residential_mg_with_pv_and_dewhs.tariff_generator import TariffGenerator
from examples.residential_mg_with_pv_and_dewhs.modelling.parameters import grid_param_struct
from datetime import datetime as DateTime
from matplotlib.ticker import FormatStrFormatter

plot_with_tex(True)
##LOAD DATA FRAMES ##

tariff_gen = TariffGenerator(low_off_peak=48.40, low_stnd=76.28, low_peak=110.84, high_off_peak=55.90,
                             high_stnd=102.95, high_peak=339.77)

time_0 = DateTime(2018, 12, 10)
steps = 96*14

cost_profile = tariff_gen.get_price_vector(time_0, steps,
                                           grid_param_struct.control_ts) / 100

cont_profile = pd.DataFrame(cost_profile, index=pd.date_range(time_0,periods=steps, freq='15Min'))

# Plot
fig, axes = get_fig_axes_A4(1, 1, v_scale=1 / 5, h_scale=1, sharex='all')

ax0 = axes[0]

cont_profile.plot(drawstyle="steps-post", ax=ax0, lw=lineWidthNorm)


## Legends

ax0.legend().remove()

##PLOT LABELLING#
ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax0.set_ylabel(tex_s(r'$c_{g,\text{imp}}\;[\si{R\per\kWh}]$'), wrap=True)

fig.savefig(FIG_SAVE_PATH + "plot_grid_import_prices.pdf", bbox_inches='tight')
