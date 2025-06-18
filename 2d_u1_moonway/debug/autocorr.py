# %%
from utils import *
import matplotlib.pyplot as plt

topo = np.concatenate((np.zeros(990), np.ones(34)))

lattice_size = 16
volume = lattice_size ** 2
beta = 6

max_lag = 20


auto_chi = auto_from_chi(topo, max_lag, beta, volume)

auto_def = auto_by_def(topo, max_lag)

plt.plot(auto_chi, label='chi')
plt.plot(auto_def, label='def')
plt.legend()
plt.show()


# %%
