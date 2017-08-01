from SimPEG import (
    Problem, Utils, Maps, Props, Mesh, Tests, Survey, Solver as SimpegSolver
    )
import numpy as np
import scipy.sparse as sp
import properties
from scipy.constants import mu_0


class MT1DSurvey(Survey.BaseSurvey):

    def __init__(self, srcList, **kwargs):
        self.srcList = srcList
        Survey.BaseSurvey.__init__(self, **kwargs)
        self.getUniqFrequency()

    @property
    def nFreq(self):
        if getattr(self, '_nFreq', None) is None:
            self._nFreq = len(self.frequency)
        return self._nFreq

        self.getUniqueTimes()

    def getUniqFrequency(self):
        frequency_rx = []

        rxcount = 0
        for src in self.srcList:
            for rx in src.rxList:
                frequency_rx.append(rx.frequency)
                rxcount += 1
        freqs_temp = np.hstack(frequency_rx)
        self.frequency = np.unique(freqs_temp)

        # TODO: Generalize this so that user can omit specific datum at
        # certain frequencies
        if (len(freqs_temp) != rxcount * self.nFreq):
            raise Exception("# of Frequency of each Rx should be same!")

    @property
    def P0(self):
        """
            Evaluation matrix at surface
        """
        if getattr(self, '_P0', None) is None:
            P0 = sp.coo_matrix(
                (
                    np.r_[1.], (np.r_[0], np.r_[2*self.mesh.nC])),
                shape=(1, 2 * self.mesh.nC + 1)
                )
            self._P0 = P0.tocsr()
        return self._P0

    def eval(self, f):
        """
        Project fields to receiver locations

        :param Fields f: fields object
        :rtype: numpy.ndarray
        :return: data
        """
        data = Survey.Data(self)
        for src in self.srcList:
            for rx in src.rxList:
                data[src, rx] = rx.eval(src, f, self.P0)
        return data

    def evalDeriv(self):
        raise Exception('Use Receivers to project fields deriv.')

    def setMesh(self, sigma=0.1, max_depth_core=3000., ncell_per_skind=10, n_skind=2, core_meshType="linear", max_hz_core=None):

        """
        Set 1D Mesh based using skin depths

        """
        rho = 1./sigma
        fmin, fmax = self.frequency.min(), self.frequency.max()
        print (
            (">> Smallest cell size = %d m") % (500*np.sqrt(rho/fmax) / ncell_per_skind)
            )
        print (
            (">> Padding distance = %d m") % (500*np.sqrt(rho/fmin) * n_skind)
            )
        cs = 500*np.sqrt(rho/fmax) / ncell_per_skind
        length_bc = 500*np.sqrt(100/fmin) * n_skind

        if core_meshType == "linear":

            max_hz_core = cs

        elif core_meshType == "log":

            if max_hz_core is None:
                max_hz_core  = cs * 10

            ncz = 2
            hz_core = np.logspace(np.log10(cs), np.log10(max_hz_core), ncz)

            while hz_core.sum() < max_depth_core:
                ncz += 1
                hz_core = np.logspace(np.log10(cs), np.log10(max_hz_core), ncz)

        npad = 1
        blength = max_hz_core*1.3**(np.arange(npad)+1)

        while blength < length_bc:
            npad += 1
            blength = (max_hz_core*1.3**(np.arange(npad)+1)).sum()
        print (
            (">> # of padding cells %d") % (npad)
        )

        if core_meshType == "linear":
            ncz = int(max_depth_core / cs)
            hz = [(cs, npad, -1.3), (cs, ncz)]
        elif core_meshType == "log":
            hz_pad = max_hz_core * 1.3**(np.arange(npad)+1)
            hz = np.r_[hz_pad[::-1], hz_core[::-1]]

        print (
            (">> # of core cells cells %d") % (ncz)
            )
        mesh = Mesh.TensorMesh([hz], x0='N')

        return mesh


class MT1DSrc(Survey.BaseSrc):
    """
    Source class for MT1D
    We assume a boundary condition of Ex (z=0) = 1
    """
    loc = np.r_[0.]


class ZxyRx(Survey.BaseRx):

    def __init__(self, locs, component=None, frequency=None):
        self.component = component
        self.frequency = frequency
        Survey.BaseRx.__init__(self, locs, rxType=None)

    def eval(self, src, f, P0):
        Zxy = - 1./(P0*f)
        if self.component == "real":
            return Zxy.real
        elif self.component == "imag":
            return Zxy.imag
        elif self.component == "both":
            return np.r_[Zxy.real, Zxy.imag]
        else:
            raise NotImplementedError('must be real, imag or both')

    def evalDeriv(self, f, freq, P0, df_dm_v=None, v=None, adjoint=False):

        if adjoint:

            if self.component == "real":
                PTvr = (P0.T*v).astype(complex)
                dZr_dfT_v = Utils.sdiag((1./(f**2)))*PTvr
                return dZr_dfT_v
            elif self.component == "imag":
                PTvi = P0.T*v*-1j
                dZi_dfT_v = Utils.sdiag((1./(f**2)))*PTvi
                return dZi_dfT_v
            elif self.component == "both":
                PTvr = (P0.T*np.r_[v[0]]).astype(complex)
                PTvi = P0.T*np.r_[v[1]]*-1j
                dZr_dfT_v = Utils.sdiag((1./(f**2)))*PTvr
                dZi_dfT_v = Utils.sdiag((1./(f**2)))*PTvi
                return dZr_dfT_v + dZi_dfT_v
            else:
                raise NotImplementedError('must be real, imag or both')

        else:

            dZd_dm_v = P0 * (Utils.sdiag(1./(f**2)) * df_dm_v)

            if self.component == "real":
                return dZd_dm_v.real
            elif self.component == "imag":
                return dZd_dm_v.imag
            elif self.component == "both":
                return np.r_[dZd_dm_v.real, dZd_dm_v.imag]
            else:
                raise NotImplementedError('must be real, imag or both')

    @property
    def nD(self):
        if self.component == "both":
            return len(self.frequency) * 2
        else:
            return len(self.frequency)


class AppResPhaRx(ZxyRx):

    def __init__(self, locs, component=None, frequency=None):
        super(AppResPhaRx, self).__init__(locs, component, frequency)

    def eval(self, src, f, P0):
        Zxy = - 1./(P0*f)
        omega = 2*np.pi*self.frequency
        if self.component == "appres":
            appres = abs(Zxy)**2 / (mu_0*omega)
            return appres
        elif self.component == "phase":
            phase = np.rad2deg(np.arctan(Zxy.imag / Zxy.real))
            return phase
        elif self.component == "both":
            appres = abs(Zxy)**2 / (mu_0*omega)
            phase = np.rad2deg(np.arctan(Zxy.imag / Zxy.real))
            return np.r_[appres, phase]
        else:
            raise NotImplementedError('must be appres, phase or both')

    def evalDeriv(self, f, freq, P0, df_dm_v=None, v=None, adjoint=False):

        Zxy = - 1./(P0*f)
        omega = 2*np.pi*freq

        if adjoint:

            dZa_dZ = Zxy / abs(Zxy)
            dappres_dZa = 2. * abs(Zxy) / (mu_0*omega)
            dappres_dZ = (dappres_dZa * dZa_dZ)

            if self.component == "appres":
                dappres_dZT_v = dappres_dZ.conj() * np.r_[v]
                dappres_dfT_v = Utils.sdiag((1./(f**2)))*(P0.T*dappres_dZT_v)
                return dappres_dfT_v
            elif self.component == "phase":
                return np.zeros_like(dappres_dfT_v)
            elif self.component == "both":
                dappres_dZT_v = dappres_dZ.conj() * np.r_[v[0]]
                dappres_dfT_v = Utils.sdiag((1./(f**2)))*(P0.T*dappres_dZT_v)
                return dappres_dfT_v
            else:
                raise NotImplementedError('must be real, imag or both')

        else:

            dZ_dm_v = P0 * (Utils.sdiag(1./(f**2)) * df_dm_v)

            dZa_dZ = Zxy.conjugate() / abs(Zxy)
            dappres_dZa = 2. * abs(Zxy) / (mu_0*omega)
            dappres_dZ = dappres_dZa * dZa_dZ
            dappres_dm_v = (dappres_dZ * dZ_dm_v).real

            if self.component == "appres":
                return dappres_dm_v
            elif self.component == "phase":
                return np.zeros_like(dappres_dm_v)
            elif self.component == "both":
                return np.r_[dappres_dm_v, np.zeros_like(dappres_dm_v)]
            else:
                raise NotImplementedError('must be appres, phase or both')


class MT1DProblem(Problem.BaseProblem):
    """
    1D Magnetotelluric problem under quasi-static approximation

    """

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    rho, rhoMap, rhoDeriv = Props.Invertible(
        "Electrical resistivity (Ohm-m)"
    )

    Props.Reciprocal(sigma, rho)

    mu = Props.PhysicalProperty(
        "Magnetic Permeability (H/m)",
        default=mu_0
    )

    surveyPair = Survey.BaseSurvey  #: The survey to pair with.
    dataPair = Survey.Data  #: The data to pair with.

    mapPair = Maps.IdentityMap  #: Type of mapping to pair with

    Solver = SimpegSolver  #: Type of solver to pair with
    solverOpts = {}  #: Solver options

    verbose = False
    f = None

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)
        # Setup boundary conditions
        mesh.setCellGradBC([['dirichlet', 'dirichlet']])

    @property
    def deleteTheseOnModelUpdate(self):
        if self.verbose:
            print ("Delete Matrices")
        toDelete = []
        if self.sigmaMap is not None or self.rhoMap is not None:
            toDelete += ['_MccSigma', '_Ainv', '_ATinv']
        return toDelete

    @property
    def Exbc(self):
        """
            Boundary value for Ex
        """
        if getattr(self, '_Exbc', None) is None:
            self._Exbc = np.r_[0., 1.]
        return self._Exbc

    ####################################################
    # Mass Matrices
    ####################################################

    @property
    def MccSigma(self):
        """
        Diagonal matrix for \\(\\sigma\\).
        """
        if getattr(self, '_MccSigma', None) is None:
            self._MccSigma = Utils.sdiag(self.sigma)
        return self._MccSigma

    def MccSigmaDeriv(self, u):
        """
        Derivative of MccSigma with respect to the model
        """
        if self.sigmaMap is None:
            return Utils.Zero()

        return (
            Utils.sdiag(u) * self.sigmaDeriv
        )

    @property
    def MccEpsilon(self):
        """
        Diagonal matrix for \\(\\epsilon\\).
        """
        if getattr(self, '_MccEpsilon', None) is None:
            self._MccEpsilon = Utils.sdiag(self.epsilon)
        return self._MccEpsilon

    @property
    def MfMu(self):
        """
        Edge inner product matrix for \\(\\mu\\).
        """
        if getattr(self, '_MMfMu', None) is None:
            self._MMfMu = Utils.sdiag(
                self.mesh.aveCC2F * self.mu * np.ones(self.mesh.nC)
                )
        return self._MMfMu

    ####################################################
    # Physics?
    ####################################################

    def getA(self, freq):
        """
            .. math::

                \mathbf{A} =
                \begin{bmatrix}
                    \mathbf{Grad} & \imath \omega \mathbf{M}^{f2cc}_{\mu} \\[0.3em]
                   \mathbf{M}^{cc}_{\hat{\sigma}} & \mathbf{Div}           \\[0.3em]
                \end{bmatrix}

        """

        Div = self.mesh.faceDiv
        Grad = self.mesh.cellGrad
        omega = 2*np.pi*freq
        A = sp.vstack(
            (
                sp.hstack((Grad, 1j*omega*self.MfMu)),
                sp.hstack((self.MccSigma, Div))
            )
        )
        return A

    @property
    def Ainv(self):
        if getattr(self, '_Ainv', None) is None:
            if self.verbose:
                print ("Factorize A matrix")
            self._Ainv = []
            for freq in self.survey.frequency:
                self._Ainv.append(self.Solver(self.getA(freq)))
        return self._Ainv

    @property
    def ATinv(self):
        if getattr(self, '_ATinv', None) is None:
            if self.verbose:
                print ("Factorize AT matrix")
            self._ATinv = []
            for freq in self.survey.frequency:
                self._ATinv.append(self.Solver(self.getA(freq).T))
        return self._ATinv

    def getADeriv_sigma(self, freq, f, v, adjoint=False):
        Ex = f[:self.mesh.nC]
        dMcc_dsig = self.MccSigmaDeriv(Ex)
        if adjoint:
            return sp.hstack(
                (Utils.spzeros(self.mesh.nC, self.mesh.nN), dMcc_dsig.T)
                ) * v
        else:
            return np.r_[np.zeros(self.mesh.nC+1), dMcc_dsig*v]

    def getRHS(self, freq):
        """
            .. math::

                \mathbf{rhs} =
                \begin{bmatrix}
                     - \mathbf{B}\mathbf{E}_x^{bc} \\ [0.3em]
                    \boldsymbol{0} \\[0.3em]
                \end{bmatrix}$

        """
        B = self.mesh.cellGradBC
        RHS = np.r_[-B*self.Exbc, np.zeros(self.mesh.nC)]
        return RHS

    def fields(self, m=None):
        if self.verbose:
            print (">> Compute fields")

        if m is not None:
            self.model = m

        f = np.zeros(
            (int(self.mesh.nC*2+1), self.survey.nFreq), dtype="complex"
            )

        for ifreq, freq in enumerate(self.survey.frequency):
            f[:, ifreq] = self.Ainv[ifreq] * self.getRHS(freq)
        return f

    def Jvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)

        Jv = []

        for src in self.survey.srcList:
            for rx in src.rxList:
                for ifreq, freq in enumerate(self.survey.frequency):
                    dA_dm_f_v = self.getADeriv_sigma(freq, f[:, ifreq], v)
                    df_dm_v = - (self.Ainv[ifreq] * dA_dm_f_v)
                    Jv.append(
                        rx.evalDeriv(
                            f[:, ifreq], freq, self.survey.P0, df_dm_v=df_dm_v
                            )
                        )
        return np.hstack(Jv)

    def Jtvec(self, m, v, f=None):

        if f is None:
            f = self.fields(m)
        # Ensure v is a data object.

        if not isinstance(v, self.dataPair):
            v = self.dataPair(self.survey, v)

        Jtv = np.zeros(m.size)

        for src in self.survey.srcList:
            for rx in src.rxList:
                for ifreq, freq in enumerate(self.survey.frequency):
                    if rx.component == "both":
                        v_temp = v[src, rx].reshape(
                            (self.survey.nFreq, 2)
                            )[ifreq, :]
                    else:
                        v_temp = v[src, rx][ifreq]

                    dZ_dfT_v = rx.evalDeriv(
                        f[:, ifreq], freq, self.survey.P0,
                        v=v_temp, adjoint=True
                        )

                    ATinvdZ_dfT = self.ATinv[ifreq]*dZ_dfT_v
                    Jtv += - self.getADeriv_sigma(
                        freq, f[:, ifreq], ATinvdZ_dfT, adjoint=True
                        ).real

        return Jtv
