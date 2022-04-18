import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

###############
#CODE NON MIS EN FORME _ UNIQUEMENT POUR DEMONSTRATION
###############




gamma = 70e-3
g = 9.81
rho = 1000 # kg/m3
facteur = 1.65# facteur d'agrandissement 


f = np.array([17.15, 22.69, 31.71, 40.34, 50.59, 60.40, 70.26, 78.99])
lambda_ecran = np.array([21.17, 17.08, 12.93, 10.80, 9.161, 7.97, 7.41, 6.5])*1e-3
lambda_reel = lambda_ecran/facteur

lambda_melde = np.array([32, 30.6, 26.1, 23.33, 21.75, 21.5])*1e-3/facteur
f_melde = np.array([10.9630, 11.9868, 14.3733, 16.517, 18.1258, 20.3474])

F = np.append(f_melde, f)
A = np.append(lambda_melde, lambda_reel)


k = 2*np.pi/A
w = 2*np.pi * F
kb = 2*np.pi/lambda_melde
wb = 2*np.pi * f_melde


def fit(x, a, b):
    return a*x + b




err = 2


popt, pcov = curve_fit(fit, k*k, w*w/k)
g_exp = popt[1]
gamma_exp = popt[0]*rho*1000
s = "g = {:.2f} $m/s^2$\n $\gamma$ = {:.2f} $mN/m$".format(g_exp, gamma_exp)




####################

# x = np.arange(-0.5e6, 3e6, 1)

# fig,ax = plt.subplots(figsize=(9,5))


# # plt.plot(k*k, w*w/k, "o")
# ax.plot(x, fit(x, *popt), '--', color='orangered', label=r'$\phi(k^2) = \frac{\gamma}{\rho}k^2 + g$')
# axins = zoomed_inset_axes(ax,6,bbox_to_anchor=(575, 225))
# # axins.plot(kb*kb, wb*wb/kb)
# axins.errorbar(kb*kb, wb*wb/kb, yerr=err, fmt='o', markersize = 5, capsize=2, color = 'navy')

# axins.grid(alpha=0.5)
# axins.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


# ax.errorbar(k*k, w*w/k, yerr=0, fmt='o', markersize = 5, capsize=2, color = 'navy')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(-0.5e6,155, s, bbox=props)
# ax.set_xlabel('$k^2$')
# ax.set_ylabel('$\omega^2/k$')
# ax.set_ylim([-80, 220])
# ax.legend(loc='best')
# ax.grid(alpha=0.5)
# mark_inset(ax,axins,loc1=1,loc2=3)
# plt.show()





plt.figure()
plt.errorbar(k, w, yerr=0, fmt='o', markersize = 5, capsize=2, color = 'navy')
plt.errorbar(kb, wb, yerr=0, fmt='o', markersize = 5, capsize=2, color = 'navy')
# plt.plot(k, w/k, "o")
# plt.xlim(0, 1800)
# plt.ylim(0, 0.7)

def fit2(x, g, gamma, rho):
    return np.sqrt((gamma/rho)*x + g/x)

def fit3(x, g, gamma, rho):
    return np.sqrt((gamma/rho)*x**3 + g*x)



x = np.arange(1,2000, 1)
# popt, pcov = curve_fit(fit2, k, w/k)

plt.plot(x, fit3(x, g, gamma, rho), '--', color='orangered', label='Courbe théorique')



plt.xlabel('$k$')
plt.ylabel('$\omega$')
plt.legend()
plt.grid(alpha=0.5)
# plt.plot(x, fit2(x, *popt), '--', color='orangered')


# plt.savefig('CodeTP3_w.pdf')


