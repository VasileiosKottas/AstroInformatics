import numpy as np

# StarBurst
t_e = np.log10([2.e7, 3.5e7])
age = np.log10([5.e6, 3.e7])
tau_v = np.log10([51., 250.])
fsb = [-5.,5.]
polt = np.log10([ 800., 1200. ])
# Spheroid
tvv = np.log10([ 0.1, 15.])
psi = np.log10([ 1.1, 16.9])
cirr_tau = np.log10([ 1.26e8, 7.9e9 ])
iview = np.log10([ 0.1, 89.9 ])
fsph = [-5.,5.]
# AGN1
theta_1 = np.log10([ 16., 58. ])
tau_uv = np.log10([ 260., 1490. ])
theta_v = np.log10([3, 90.0])
r2tor1 = np.log10([21 , 100])
fagn = [-5.,5.]
# AGN2
ct= np.log10([ 31., 69. ])
rm= np.log10([ 11., 149. ])
ta = np.log10([2., 6.])
thfr06 = np.log10([1., 10.])
# AGN3
oa = np.log10([ 21., 75.])
rr = np.log10([ 10., 30. ])
tt = np.log10([ 3., 11. ])
thst16 = np.log10([1., 89.])
# AGN4
vc = np.log10([ 16., 770.])
ac = np.log10([ 1., 440. ])
ad = np.log10([ 50., 499. ])
th = np.log10([0.1, 10])
# Polar Dust
fpol = [-5., 5.]