import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from SimPEG import Utils, Solver


def simulateMT(mesh, sigma, frequency, rtype="app_res"):
    """
       Compute apparent resistivity and phase at each frequency.
       Return apparent resistivity and phase for rtype="app_res",
       or impedance for rtype="impedance"
    """

    # Angular frequency (rad/s)
    def omega(freq):
        return 2*np.pi*freq

    # make sure we are working with numpy arrays
    if type(frequency) is float:
        frequency = np.r_[frequency]  # make it a list to loop over later if it is just a scalar
    elif type(frequency) is list:
        frequency = np.array(frequency)

    # Grad
    mesh.setCellGradBC([['dirichlet', 'dirichlet']]) # Setup boundary conditions
    Grad = mesh.cellGrad # Gradient matrix

    # MfMu
    mu = np.ones(mesh.nC)*mu_0 # magnetic permeability values for all cells
    Mmu = Utils.sdiag(mesh.aveCC2F * mu)

    # Mccsigma
    sigmahat = sigma  # quasi-static assumption
    Msighat = Utils.sdiag(sigmahat)

    # Div
    Div = mesh.faceDiv # Divergence matrix

    # Right Hand Side
    B = mesh.cellGradBC  # a matrix for boundary conditions
    Exbc = np.r_[0., 1.] # boundary values for Ex

    # Right-hand side
    rhs = np.r_[
        -B*Exbc,
        np.zeros(mesh.nC)
    ]

    # loop over frequencies
    Zxy = []
    for freq in frequency:

        # A-matrix
        A = sp.vstack([
            sp.hstack([Grad, 1j*omega(freq)*Mmu]), # Top row of A matrix
            sp.hstack((Msighat, Div)) # Bottom row of A matrix
        ])

        Ainv = Solver(A) # Factorize A matrix
        sol = Ainv*rhs   # Solve A^-1 rhs = sol
        Ex = sol[:mesh.nC] # Extract Ex from solution vector u
        Hy = sol[mesh.nC:mesh.nC+mesh.nN] # Extract Hy from solution vector u

        Zxy.append(- 1./Hy[-1]) # Impedance at the surface

    # turn it into an array
    Zxy = np.array(Zxy)

    if rtype.lower() == "impedance":
        return Zxy

    elif rtype.lower() == "app_res":
        app_res = abs(Zxy)**2 / (mu_0*omega(frequency))
        app_phase = np.rad2deg(np.arctan(Zxy.imag / Zxy.real))
        return app_res, app_phase

    else:
        raise Exception("rtype must be 'impedance' or 'app_res', not {}".format(rtype.lower()))
