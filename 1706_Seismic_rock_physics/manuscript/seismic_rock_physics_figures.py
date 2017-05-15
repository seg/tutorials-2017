import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def vrh(f, M1, M2):
    '''
    Simple Voigt-Reuss-Hill bounds for 2-components mixture, (C) aadm 2017

    INPUT
    f: volumetric fraction of mineral 1
    M1: elastic modulus mineral 1
    M2: elastic modulus mineral 2

    OUTPUT
    M_Voigt: upper bound or Voigt average
    M_Reuss: lower bound or Reuss average
    M_VRH: Voigt-Reuss-Hill average
    '''
    M_Voigt = f*M1 + (1-f)*M2
    M_Reuss = 1 / (f/M1 + (1-f)/M2)
    M_VRH = (M_Voigt+M_Reuss)/2
    return M_Voigt, M_Reuss, M_VRH


def vels(K_DRY, G_DRY, K0, D0, Kf, Df, phi):
    '''
    Calculates velocities and densities of saturated rock via Gassmann
    equation, (C) aadm 2015

    INPUT
    K_DRY, G_DRY: dry rock bulk & shear modulus in GPa
    K0, D0: mineral bulk modulus and density in GPa
    Kf, Df: fluid bulk modulus and density in GPa
    phi: porosity
    '''
    rho = D0*(1-phi)+Df*phi
    K = K_DRY + (1-K_DRY/K0)**2 / ((phi/Kf) + ((1-phi)/K0) - (K_DRY/K0**2))
    vp = np.sqrt((K+4./3*G_DRY)/rho)*1e3
    vs = np.sqrt(G_DRY/rho)*1e3
    return vp, vs, rho, K


def hertzmindlin(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
    Hertz-Mindlin model
    written by aadm (2015) from Rock Physics Handbook, p.246

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1 = dry pack with perfect adhesion
       0 = dry frictionless pack
    '''
    P /= 1e3  # converts pressure in same units as solid moduli (GPa)
    PR0 = (3*K0-2*G0)/(6*K0+2*G0)  # poisson's ratio of mineral mixture
    K_HM = (P*(Cn**2*(1-phic)**2*G0**2) / (18*np.pi**2*(1-PR0)**2))**(1/3)
    G_HM = ((2+3*f-PR0*(1+3*f))/(5*(2-PR0))) * ((P*(3*Cn**2*(1-phic)**2*G0**2)/(2*np.pi**2*(1-PR0)**2)))**(1/3)
    return K_HM, G_HM


def softsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
    Soft-sand (uncemented) model
    written by aadm (2015) from Rock Physics Handbook, p.258

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1 = dry pack with perfect adhesion
       0 = dry frictionless pack
    '''
    K_HM, G_HM = hertzmindlin(K0, G0, phi, phic, Cn, P, f)
    K_DRY = -4/3*G_HM + (((phi/phic)/(K_HM+4/3*G_HM)) + ((1-phi/phic)/(K0+4/3*G_HM)))**-1
    tmp = G_HM/6*((9*K_HM+8*G_HM) / (K_HM+2*G_HM))
    G_DRY = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY


def stiffsand(K0, G0, phi, phic=0.4, Cn=8.6, P=10, f=1):
    '''
    Stiff-sand model
    written by aadm (2015) from Rock Physics Handbook, p.260

    INPUT
    K0, G0: mineral bulk & shear modulus in GPa
    phi: porosity
    phic: critical porosity (default 0.4)
    Cn: coordination nnumber (default 8.6)
    P: confining pressure in MPa (default 10)
    f: shear modulus correction factor
       1 = dry pack with perfect adhesion
       0 = dry frictionless pack
    '''
    K_HM, G_HM = hertzmindlin(K0, G0, phi, phic, Cn, P, f)
    K_DRY = -4/3*G0 + (((phi/phic)/(K_HM+4/3*G0)) + ((1-phi/phic)/(K0+4/3*G0)))**-1
    tmp = G0/6*((9*K0+8*G0) / (K0+2*G0))
    G_DRY = -tmp + ((phi/phic)/(G_HM+tmp) + ((1-phi/phic)/(G0+tmp)))**-1
    return K_DRY, G_DRY


def rpt(model='soft', vsh=0.0, fluid='gas', phic=0.4, Cn=8, P=10, f=1, display=True):
    phi = np.linspace(0.01, phic, 10)
    sw = np.linspace(0, 1, 10)
    xx = np.empty((phi.size, sw.size))
    yy = np.empty((phi.size, sw.size))
    (K_hc, RHO_hc) = (K_g, RHO_g) if fluid == 'gas' else (K_o, RHO_o)
    _, _, K0 = vrh(vsh, K_sh, K_qz)
    _, _, MU0 = vrh(vsh, MU_sh, MU_qz)
    RHO0 = vsh*RHO_sh+(1-vsh)*RHO_qz
    if model == 'soft':
        Kdry, MUdry = softsand(K0, MU0, phi, phic, Cn, P, f)
    elif model == 'stiff':
        Kdry, MUdry = stiffsand(K0, MU0, phi, phic, Cn, P, f)
    for i, val in enumerate(sw):
        _, K_f, _ = vrh(val, K_b, K_hc)
        RHO_f = val*RHO_b + (1-val)*RHO_hc
        vp, vs, rho, _ = vels(Kdry, MUdry, K0, RHO0, K_f, RHO_f, phi)
        xx[:, i] = vp*rho
        yy[:, i] = vp/vs
    opt1 = {'backgroundcolor': '0.9'}
    opt2 = {'ha': 'right', 'backgroundcolor': '0.9'}
    if display:
        plt.figure(figsize=(7, 7))
        plt.plot(xx, yy, '-ok', alpha=0.3)
        plt.plot(xx.T, yy.T, '-ok', alpha=0.3)
        for i, val in enumerate(phi):
            plt.text(xx[i, -1], yy[i, -1]+.01, '$\phi = {:.02f}$'.format(val), **opt1)
        plt.text(xx[-1, 0]-100, yy[-1, 0], '$S_w = {:.02f}$'.format(sw[0]), **opt2)
        plt.text(xx[-1, -1]-100, yy[-1, -1], '$S_w = {:.02f}$'.format(sw[-1]), **opt2)
        plt.xlabel('Ip'), plt.ylabel('Vp/Vs')
        plt.title('RPT {} (N:G = {}, fluid = {})'.format(model.upper(), 1-vsh, fluid))
    return xx, yy


# define basic styles for plotting log curves (sty0), sand (sty1) and shale (sty2)
sty0 = {'lw': 1, 'color': 'k', 'ls': '-'}
sty1 = {'marker': 'o', 'color': 'g',
        'ls': 'none', 'ms': 6, 'mec': 'none', 'alpha': 0.5}
sty2 = {'marker': 'o', 'color': 'r',
        'ls': 'none', 'ms': 6, 'mec': 'none', 'alpha': 0.5}

# define labels for subplots
subplotlabels = ['(A)', '(B)', '(C)']

# load well data, setup some initial parameters
L = pd.read_csv('../qsiwell5.csv', index_col=0)
z1, z2 = 2100, 2250
cutoff_sand = 0.3
cutoff_shale = 0.5

# define filters to select sand (ss) and shale (sh)
ss = (L.index >= z1) & (L.index <= z2) & (L.VSH <= cutoff_sand)
sh = (L.index >= z1) & (L.index <= z2) & (L.VSH >= cutoff_shale)


# ----------------------------------
# --> FIGURE 1
# ----------------------------------

f = plt.subplots(figsize=(12, 5))
ax0 = plt.subplot2grid((1, 9), (0, 0), colspan=1)  # shale volume curve
ax1 = plt.subplot2grid((1, 9), (0, 1), colspan=1)  # ip curve
ax2 = plt.subplot2grid((1, 9), (0, 2), colspan=1)  # vp/vs curve
ax3 = plt.subplot2grid((1, 9), (0, 3), colspan=3)  # crossplot phi - vp
ax4 = plt.subplot2grid((1, 9), (0, 6), colspan=3)  # crossplot ip - vp/vs

ax0.plot(L.VSH[ss], L.index[ss], **sty1)
ax0.plot(L.VSH[sh], L.index[sh], **sty2)
ax0.plot(L.VSH, L.index, **sty0)
ax0.locator_params(axis='x', nbins=2)
ax0.set_ylabel('Depth')
ax0.set_xlabel('VSH')
ax1.plot(L.IP[ss], L.index[ss], **sty1)
ax1.plot(L.IP[sh], L.index[sh], **sty2)
ax1.plot(L.IP, L.index,  **sty0)
ax1.locator_params(axis='x', nbins=2)
ax1.set_xlabel('$I_\mathrm{P}$')
ax1.set_xlim(4e3, 8e3)
ax2.plot(L.VPVS[ss], L.index[ss], **sty1)
ax2.plot(L.VPVS[sh], L.index[sh], **sty2)
ax2.plot(L.VPVS, L.index, **sty0)
ax2.locator_params(axis='x', nbins=2)
ax2.set_xlabel('$V_\mathrm{P} / V_\mathrm{S}$')
ax2.set_xlim(1.5, 3)
ax3.plot(L.PHIE[ss], L.VP[ss], **sty1)
ax3.set_xlim(0, 0.4),  ax3.set_ylim(2e3, 4e3)
ax3.set_xlabel('$\phi_\mathrm{e}$ ')
ax3.set_ylabel('$V_\mathrm{P}$')
ax4.plot(L.VP*L.RHO[ss], L.VP/L.VS[ss], **sty1)
ax4.plot(L.VP*L.RHO[sh], L.VP/L.VS[sh], **sty2)
ax4.set_xlim(4e3, 8e3),  ax4.set_ylim(1.5, 3)
ax4.set_xlabel('$I_\mathrm{P}$')
ax4.set_ylabel('$V_\mathrm{P} / V_\mathrm{S}$')
for aa in [ax0, ax1, ax2]:
    aa.set_ylim(z2, z1)
for aa in [ax0, ax1, ax2, ax3, ax4]:
    aa.tick_params(which='major', labelsize=8)
for aa in [ax1, ax2]:
    aa.set_yticklabels([])
# to place the labels on the outside comment following 3 lines
# and increase subplots_adjust wspace to 1
for aa in [ax3, ax4]:
    aa.yaxis.set_label_coords(0.08, 0.5)
    aa.xaxis.set_label_coords(0.5, 0.06)
# ax0.text(0, 1.05, '(A)', fontsize=16, weight='bold', transform=ax0.transAxes)
# ax3.text(0, 1.05, '(B)', fontsize=16, weight='bold', transform=ax3.transAxes)
# ax4.text(0, 1.05, '(C)', fontsize=16, weight='bold', transform=ax4.transAxes)
plt.subplots_adjust(wspace=.5, left=0.05, right=0.95, top=.85)

plt.savefig('Figure_1_AADM.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_1_AADM.pdf', dpi=300, bbox_inches='tight')

# ----------------------------------
# --> FIGURE 2
# ----------------------------------

RHO_qz, K_qz = 2.6, 37
MU_qz, RHO_sh = 44, 2.8
K_sh, MU_sh = 15, 5
RHO_b, K_b = 1.1, 2.8
RHO_o, RHO_g = 0.8, 0.2
K_o, K_g = 0.9, 0.06

Cn = 8
phic = 0.4

phi = np.linspace(0.01, 0.4)
K0, MU0, RHO0 = K_qz, MU_qz, RHO_qz

Kdry, MUdry = softsand(K0, MU0, phi, phic, Cn, P=45)
vp_ssm0, vs_ssm0, rho_ssm0, _ = vels(Kdry, MUdry, K0, RHO0, K_b, RHO_b, phi)

Kdry, MUdry = stiffsand(K0, MU0, phi, phic, Cn, P=45)
vp_sti0, vs_sti0, rho_sti0, _ = vels(Kdry, MUdry, K0, RHO0, K_b, RHO_b, phi)

NG = np.linspace(0.6, 1.0, 5)

f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
ax[0].plot(phi, vp_ssm0, '-k')
ax[0].plot(phi, vp_sti0, ':k')
ax[0].text(0.15, 3500, 'Soft Sand', fontsize=14, ha='right', style='italic')
ax[0].text(0.28, 3750, 'Stiff Sand', fontsize=14, ha='left', style='italic')
for i in NG:
    _, _, K0 = vrh(i, K_qz, K_sh)
    _, _, MU0 = vrh(i, MU_qz, MU_sh)
    RHO0 = i*RHO_qz+(1-i)*RHO_sh
    Kdry, MUdry = softsand(K0, MU0, phi, phic=.5, Cn=12, P=45)
    vp_ssm, vs_ssm, rho_ssm, _ = vels(Kdry, MUdry, K0, RHO0, K_b, RHO_b, phi)
    Kdry, MUdry = stiffsand(K0, MU0, phi, phic=.4, Cn=8, P=45)
    vp_sti, vs_sti, rho_sti, _ = vels(Kdry, MUdry, K0, RHO0, K_b, RHO_b, phi)
    ax[1].plot(phi, vp_ssm, '-k', label='N:G = {:.2f}'.format(i), alpha=i-.4)
    ax[2].plot(phi, vp_sti, '-k', label='N:G = {:.2f}'.format(i), alpha=i-.4)
for i, aa in enumerate(ax):
    if i is not 0:
        aa.legend(fontsize=8, loc=3)
    aa.plot(L.PHIE[ss], L.VP[ss], **sty1, label='')
    aa.set_xlim(0, 0.4), aa.set_ylim(2e3, 4e3)
    aa.set_xlabel('$\phi_\mathrm{e}$ ')
    aa.set_ylabel('$V_\mathrm{P}$')
    aa.yaxis.set_label_coords(0.08, 0.5)
    aa.xaxis.set_label_coords(0.5, 0.06)
    aa.tick_params(which='major', labelsize=8)
    # aa.text(0, 1.05, subplotlabels[i], fontsize=16, weight='bold', transform=aa.transAxes)
ax[0].set_title('Soft and Stiff Sand models')
ax[1].set_title('Soft Sand model')
ax[2].set_title('Stiff Sand model')
plt.subplots_adjust(wspace=.15, left=0.05, right=0.95, top=.85)

plt.savefig('Figure_2_AADM.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_2_AADM.pdf', dpi=300, bbox_inches='tight')

# ----------------------------------
# --> FIGURE 3
# ----------------------------------

phic = 0.5
ip_rpt0, vpvs_rpt0 = rpt(model='soft', vsh=0.6, fluid='oil', phic=phic, Cn=12, P=45, f=.3, display=False)
ip_rpt1, vpvs_rpt1 = rpt(model='soft', vsh=0.8, fluid='oil', phic=phic, Cn=12, P=45, f=.3, display=False)
phi = np.linspace(0.01, phic, 10)
sw = np.linspace(0, 1, 10)

opt1 = {'fontsize': 8, 'ha': 'left', 'va': 'bottom', 'weight': 'bold', 'backgroundcolor': '.9'}
opt2 = {'fontsize': 8, 'ha': 'right', 'va': 'bottom', 'weight': 'bold', 'rotation': 'vertical'}

f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
# ax[0].plot(ip_rpt0, vpvs_rpt0, 'sk', mew=0, alpha=0.5)
ax[1].plot(ip_rpt0, vpvs_rpt0, '-sk', mew=0, alpha=0.3, ms=5)
ax[1].plot(ip_rpt0.T, vpvs_rpt0.T, '-k', alpha=0.3)
ax[2].plot(ip_rpt1, vpvs_rpt1, '-sb', mew=0, alpha=0.3, ms=5)
ax[2].plot(ip_rpt1.T, vpvs_rpt1.T, '-b', alpha=0.3)
xx = ip_rpt0
yy = vpvs_rpt0
ax[0].plot(xx, yy, '-sk', alpha=0.3, mew=0, ms=5)
ax[0].plot(xx.T, yy.T, '-k', alpha=0.3)
for i, val in enumerate(phi):
    ax[0].text(xx[i, -1], yy[i, -1], '$\phi = {:.02f}$'.format(val), **opt1)
ax[0].text(xx[-1, 0]-100, yy[-1, 0], '$S_w = {:.02f}$'.format(sw[0]), **opt2)
ax[0].text(xx[-1, -1]-100, yy[-1, -1], '$S_w = {:.02f}$'.format(sw[-1]), **opt2)
for i, aa in enumerate(ax):
    if i is not 0:
        aa.plot(L.VP[ss]*L.RHO[ss], L.VP[ss]/L.VS[ss], **sty1)
    aa.set_xlim(1e3, 12e3), aa.set_ylim(1.6, 2.8)
    aa.set_xlabel('$I_\mathrm{P}$')
    aa.set_ylabel('$V_\mathrm{P} / V_\mathrm{S}$')
    aa.yaxis.set_label_coords(0.08, 0.5)
    aa.xaxis.set_label_coords(0.5, 0.06)
    aa.tick_params(which='major', labelsize=8)
    # aa.text(0, 1.05, subplotlabels[i], fontsize=16, weight='bold', transform=aa.transAxes)
plt.subplots_adjust(wspace=.15, left=0.05, right=0.95, top=.85)
plt.savefig('Figure_3_AADM.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_3_AADM.pdf', dpi=300, bbox_inches='tight')


# ----------------------------------
# --> FIGURE 4
# ----------------------------------

from bruges.reflection import shuey2
from bruges.filters import ricker

# the properties of the upper layer are an average from
# the shale points in the well
vp0, vs0, rho0 = L[['VP', 'VS', 'RHO']][sh].mean().values

# the properties of the lower layer come from RPM
phi, vsh = .15, .8
_, _, K0 = vrh(vsh, K_qz, K_sh)
_, _, MU0 = vrh(vsh, MU_qz, MU_sh)
RHO0 = vsh*RHO_qz+(1-vsh)*RHO_sh
Kdry, MUdry = softsand(K0, MU0, phi, phic=.5, Cn=12, P=45, f=.3)
vp1, vs1, rho1, _ = vels(Kdry, MUdry, K0, RHO0, K_g, RHO_g, phi)

n_samples = 500
interface = int(n_samples/2)
ang = np.arange(31)
wavelet = ricker(.25, 0.001, 25)

model_ip, model_vpvs, rc0, rc1 = (np.zeros(n_samples) for _ in range(4))
model_z = np.arange(n_samples)
model_ip[:interface] = vp0*rho0
model_ip[interface:] = vp1*rho1
model_vpvs[:interface] = np.true_divide(vp0, vs0)
model_vpvs[interface:] = np.true_divide(vp1, vs1)

avo = shuey2(vp0, vs0, rho0, vp1, vs1, rho1, ang)
rc0[interface] = avo[0]
rc1[interface] = avo[-1]
synt0 = np.convolve(rc0, wavelet, mode='same')
synt1 = np.convolve(rc1, wavelet, mode='same')
clip = np.max(np.abs([synt0, synt1]))
clip += clip*.2

opt3 = {'color': 'k', 'linewidth': 4}
opt4 = {'linewidth': 0, 'alpha': 0.5}

f = plt.subplots(figsize=(10, 4))
ax0 = plt.subplot2grid((1, 7), (0, 0), colspan=1)  # ip
ax1 = plt.subplot2grid((1, 7), (0, 1), colspan=1)  # vp/vs
ax2 = plt.subplot2grid((1, 7), (0, 2), colspan=1)  # synthetic @ 0 deg
ax3 = plt.subplot2grid((1, 7), (0, 3), colspan=1)  # synthetic @ 30 deg
ax4 = plt.subplot2grid((1, 7), (0, 4), colspan=3)  # avo curve

ax0.plot(model_ip, model_z, **opt3)
ax0.locator_params(axis='x', nbins=1)
ax0.set_xlabel('$I_\mathrm{P}$')
ax0.set_xlim(4500, 7500)

ax1.plot(model_vpvs, model_z, **opt3)
ax1.set_xlabel('$V_\mathrm{P} / V_\mathrm{S}$')
ax1.set_xlim(1., 3.)

ax2.plot(synt0, model_z, **opt3)
ax2.fill_betweenx(model_z, 0, synt0, where=synt0>0, facecolor='black', **opt4)
ax2.set_xlim(-clip, clip)
ax2.set_xlabel('angle = {:.0f}'.format(ang[0]))
ax2.set_xticklabels([])

ax3.plot(synt1, model_z, **opt3)
ax3.fill_betweenx(model_z, 0, synt1, where=synt1>0, facecolor='black', **opt4)
ax3.set_xlim(-clip, clip)
ax3.set_xlabel('angle = {:.0f}'.format(ang[-1]))
ax3.set_xticklabels([])

ax4.plot(ang, avo, **opt3)
# ax4.axhline(0, color='k', lw=2)
ax4.hlines(y=0, xmin=0, xmax=30, color='k', lw=2)
ax4.set_xlabel('Angle of Incidence')
ax4.set_ylabel('Amplitude', )
ax4.tick_params(which='major', labelsize=8)
ax4.yaxis.set_label_coords(0.08, 0.5)
ax4.xaxis.set_label_coords(0.5, 0.06)
ax4.set_ylim(-.1, .2)
ax4.set_xlim(0, 30)

for aa in [ax0, ax1, ax2, ax3]:
    aa.set_ylim(350, 150)
    aa.tick_params(which='major', labelsize=8)
    aa.set_yticklabels([])
    aa.set_yticks([])
plt.subplots_adjust(wspace=.6, left=0.05, right=0.95, top=.85)

plt.savefig('Figure_4_AADM.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure_4_AADM.pdf', dpi=300, bbox_inches='tight')
