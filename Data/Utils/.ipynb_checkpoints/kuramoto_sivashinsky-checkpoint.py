from numpy import pi
from scipy.fftpack import fft, ifft
import numpy as np

np.seterr(over='raise', invalid='raise')

class KS:
    #
    # Solution of the 1D Kuramoto-Sivashinsky equation
    #
    # u_t + u*u_x + u_xx + u_xxxx = 0,
    # with periodic BCs on x \in [0, 2*pi*L]: u(x+2*pi*L,t) = u(x,t).
    #
    # The nature of the solution depends on the system size L and on the initial
    # condition u(x,0).  Energy enters the system at long wavelengths via u_xx
    # (an unstable diffusion term), cascades to short wavelengths due to the
    # nonlinearity u*u_x, and dissipates via diffusion with u_xxxx.
    #
    # Spatial  discretization: spectral (Fourier)
    # Temporal discretization: exponential time differencing fourth-order Runge-Kutta
    # see AK Kassam and LN Trefethen, SISC 2005

    def __init__(self, L=16, N=128, dt=0.25, inhomo=0, mu_inhomo=None, lambda_inhomo=None, tend=50000, t_transient=2500, norm_steps=10, num_lyaps=26, nsteps=None, iout=1):
        #
        # Initialize
        self.rng = np.random.default_rng(seed=0)
        L  = float(L); dt = float(dt); tend = float(tend)
        if (nsteps is None):
            nsteps = int(tend/dt)
        else:
            nsteps = int(nsteps)
            # override tend
            tend = dt*nsteps
        #
        # save to self
        self.L      = L
        self.N      = N
        self.dx     = 2*pi*L/N
        self.dt     = dt
        self.inhomo = inhomo
        self.mu_inhomo = mu_inhomo
        self.lambda_inhomo = lambda_inhomo
        #
        self.t_transient = t_transient
        self.transient_steps = int(t_transient/dt)
        self.nsteps = nsteps
        self.iout   = iout  # no. of steps for norm of output
        self.nout   = int(nsteps/iout)
        #
        self.norm_steps = norm_steps   # no. of steps for norm of lyapunov calculation
        self.num_lyaps = num_lyaps
        #
        # precompute Fourier-related quantities
        self.setup_fourier()
        # set initial condition
        self.IC()
        # self.IC(testing=True)
        # initialize simulation arrays
        self.setup_timeseries()
        # precompute ETDRK4 scalar quantities:
        self.setup_etdrk4()
        # preset lyapunov exponentials:
        self.setup_lyapunov()

    def setup_fourier(self, coeffs=None):
        # self.x  = 2*pi*self.L*np.r_[0:self.N]/self.N
        self.x  = 2*pi*self.L*np.r_[-self.N/2 + 1:self.N/2 + 1]/self.N
        self.k  = np.r_[0:self.N/2, 0, -self.N/2+1:0]/self.L # Wave numbers
        # Fourier multipliers for the linear term Lu
        if (coeffs is None):
            # normal-form equation
            self.l = self.k**2 - self.k**4
        else:
            # altered-coefficients 
            self.l = -      coeffs[0]*np.ones(self.k.shape) \
                     -      coeffs[1]*1j*self.k             \
                     + (1 + coeffs[2])  *self.k**2          \
                     +      coeffs[3]*1j*self.k**3          \
                     - (1 + coeffs[4])  *self.k**4


    def IC(self, u0=None, v0=None, testing=False):
        #
        # Set initial condition, either provided by user or by "template"
        if (v0 is None):
            # IC provided in u0 or use template
            if (u0 is None):
                # set u0
                if testing:
                    # template from AK Kassam and LN Trefethen, SISC 2005
                    u0 = np.cos(self.x/self.L)*(1. + np.sin(self.x/self.L))
                else:
                    # random noise
                    u0 = (np.random.rand(self.N) -0.5)*0.01
                    # u0 = (np.random.rand(self.N)*2-1)*0.6
            else:
                # check the input size
                if (np.size(u0,0) != self.N):
                    print('Error: wrong IC array size')
                    return -1
                else:
                    # if ok cast to np.array
                    u0 = np.array(u0)
            # in any case, set v0:
            v0 = fft(u0)
        else:
            # the initial condition is provided in v0
            # check the input size
            if (np.size(v0,0) != self.N):
                print('Error: wrong IC array size')
                return -1
            else:
                # if ok cast to np.array
                v0 = np.array(v0)
                # and transform to physical space
                u0 = ifft(v0)
        #
        # and save to self
        self.u0  = u0
        self.v0  = v0
        self.v   = v0
        self.t   = 0.
        self.stepnum = 0
        self.ioutnum = 0 # [0] is the initial condition
        self.nontransient_stepnum = 0

    def setup_timeseries(self, nout=None):
        if (nout != None):
            self.nout = int(nout)
        # nout+1 so we store the IC as well
        self.vv = np.zeros([self.nout+1, self.N], dtype=np.complex64)
        self.tt = np.zeros(self.nout+1)
        #
        # store the IC in [0]
        self.vv[0,:] = self.v0
        self.tt[0]   = 0.


    def setup_etdrk4(self):
        self.E  = np.exp(self.dt*self.l)
        self.E2 = np.exp(self.dt*self.l/2.)
        self.M  = 16                                           # no. of points for complex means
        self.r  = np.exp(1j*pi*(np.r_[1:self.M+1]-0.5)/self.M) # roots of unity
        self.LR = self.dt*np.repeat(self.l[:,np.newaxis], self.M, axis=1) + np.repeat(self.r[np.newaxis,:], self.N, axis=0)
        self.Q  = self.dt*np.real(np.mean((np.exp(self.LR/2.) - 1.)/self.LR, 1))
        self.f1 = self.dt*np.real( np.mean( (-4. -    self.LR              + np.exp(self.LR)*( 4. - 3.*self.LR + self.LR**2) )/(self.LR**3) , 1) )
        self.f2 = self.dt*np.real( np.mean( ( 2. +    self.LR              + np.exp(self.LR)*(-2. +    self.LR             ) )/(self.LR**3) , 1) )
        self.f3 = self.dt*np.real( np.mean( (-4. - 3.*self.LR - self.LR**2 + np.exp(self.LR)*( 4. -    self.LR             ) )/(self.LR**3) , 1) )
        self.g  = -0.5j*self.k
        if self.inhomo == True:
            self.omega = 2*pi/self.lambda_inhomo
            self.p = self.mu_inhomo * np.cos(self.omega * self.x)
            self.px = -self.omega * self.mu_inhomo * np.sin(self.omega * self.x)
            self.pxx = -(self.omega ** 2) * self.p


    def setup_lyapunov(self):
        self.spectrum = None
        self.ky_point = None
        self.kydim = None
        self.Rii = np.zeros([self.num_lyaps, (self.nsteps - self.transient_steps) // self.norm_steps])
        self.Y, _ = np.linalg.qr(self.rng.random((self.N, self.num_lyaps)))


    def evolve_tangent_map(self):
        # Compute Hv
        rifftv = np.real(ifft(self.v))
        rifftY = np.real(ifft(self.Y))
        if self.inhomo == True:
            Hv = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * rifftY) + \
                    (1j * self.k[:, np.newaxis]) * fft(self.px[:, np.newaxis] * rifftY) + \
                        -fft(self.pxx[:, np.newaxis] * rifftY)
            # Compute a_ and Ha
            a_ = self.E2[:, np.newaxis] * self.Y + self.Q[:, np.newaxis] * Hv;    riffta_ = np.real(ifft(a_))
            Ha = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * riffta_) + \
                (1j * self.k[:, np.newaxis]) * fft(self.px[:, np.newaxis] * rifftY) + \
                - fft(self.pxx[:, np.newaxis] * rifftY)
            
            # Compute b_ and Hb
            b_ = self.E2[:, np.newaxis] * self.Y + self.Q[:, np.newaxis] * Ha;    rifftb_ = np.real(ifft(b_))
            Hb = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * rifftb_) + \
            (1j * self.k[:, np.newaxis]) * fft(self.px[:, np.newaxis] * rifftY) + \
            - fft(self.pxx[:, np.newaxis] * rifftY)

            # Compute c_ and Hc
            c_ = self.E2[:, np.newaxis] * a_ + self.Q[:, np.newaxis] * (2 * Hb - Hv);     rifftc_ = np.real(ifft(c_))
            Hc = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * rifftc_) + \
            (1j * self.k[:, np.newaxis]) * fft(self.px[:, np.newaxis] * rifftY) + \
            - fft(self.pxx[:, np.newaxis] * rifftY)
        else:
            Hv = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * rifftY)
            # Compute a_ and Ha
            a_ = self.E2[:, np.newaxis] * self.Y + self.Q[:, np.newaxis] * Hv;    riffta_ = np.real(ifft(a_))
            Ha = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * riffta_) 

            # Compute b_ and Hb
            b_ = self.E2[:, np.newaxis] * self.Y + self.Q[:, np.newaxis] * Ha;    rifftb_ = np.real(ifft(b_))
            Hb = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * rifftb_) 

            # Compute c_ and Hc
            c_ = self.E2[:, np.newaxis] * a_ + self.Q[:, np.newaxis] * (2 * Hb - Hv);     rifftc_ = np.real(ifft(c_))
            Hc = (-1j * self.k[:, np.newaxis]) * fft(rifftv[:, np.newaxis] * rifftc_) 
        # Update Y
        self.Y = self.E[:, np.newaxis] * self.Y + Hv * self.f1[:, np.newaxis] + 2 * (Ha + Hb) * self.f2[:, np.newaxis] + Hc * self.f3[:, np.newaxis]

        # Normalize tangent vectors and record normalization
        if self.nontransient_stepnum % self.norm_steps == 0:
            matQ, matR = np.linalg.qr(rifftY)
            self.Rii[:, self.nontransient_stepnum // self.norm_steps] = np.log(np.abs(np.diag(matR[:self.num_lyaps, :self.num_lyaps])))
            self.Y = fft(matQ[:, :self.num_lyaps])  


    def step(self):
        #
        # Computation is based on v = fft(u), so linear term is diagonal.
        # The time-discretization is done via ETDRK4
        # (exponential time differencing - 4th order Runge Kutta)
        #
        if self.inhomo == True:
            v = self.v;     rifftv = np.real(ifft(v));     Nv = self.g*fft(rifftv**2) + 2j*self.k*fft(rifftv*self.px) - fft(rifftv * self.pxx) + (self.k**2)*fft(rifftv*self.p)
            a = self.E2*v + self.Q*Nv;      riffta = np.real(ifft(a));     Na = self.g*fft(riffta**2) + 2j*self.k*fft(riffta*self.px) - fft(riffta * self.pxx) + (self.k**2)*fft(riffta*self.p)
            b = self.E2*v + self.Q*Na;      rifftb = np.real(ifft(b));     Nb = self.g*fft(rifftb**2) + 2j*self.k*fft(rifftb*self.px) - fft(rifftb * self.pxx) + (self.k**2)*fft(rifftb*self.p)
            c = self.E2*a + self.Q*(2.*Nb - Nv);  rifftc = np.real(ifft(c));    Nc = self.g*fft(rifftc**2) + 2j*self.k*fft(rifftc*self.px) - fft(rifftc * self.pxx) + (self.k**2)*fft(rifftc*self.p)
        else:
            v = self.v;                           Nv = self.g*fft(np.real(ifft(v))**2)
            a = self.E2*v + self.Q*Nv;            Na = self.g*fft(np.real(ifft(a))**2)
            b = self.E2*v + self.Q*Na;            Nb = self.g*fft(np.real(ifft(b))**2)
            c = self.E2*a + self.Q*(2.*Nb - Nv);  Nc = self.g*fft(np.real(ifft(c))**2)
        #
        self.v = self.E*v + Nv*self.f1 + 2.*(Na + Nb)*self.f2 + Nc*self.f3
        self.stepnum += 1
        self.t       += self.dt

        if self.stepnum > self.transient_steps:
            self.evolve_tangent_map()
            self.nontransient_stepnum += 1

        return self.v


    def simulate(self, nsteps=None, iout=None, restart=False, correction=[]):
        #
        # If not provided explicitly, get internal values
        if (nsteps is None):
            nsteps = self.nsteps
        else:
            nsteps = int(nsteps)
            self.nsteps = nsteps
        if (iout is None):
            iout = self.iout
            nout = self.nout
        else:
            self.iout = iout
        if restart:
            # update nout in case nsteps or iout were changed
            nout      = int(nsteps/iout)
            self.nout = nout
            # reset simulation arrays with possibly updated size
            self.setup_timeseries(nout=self.nout)
        #
        # advance in time for nsteps 
        if (correction==[]):
            for n in range(1,self.nsteps+1):
                try:
                    self.step()
                except FloatingPointError:
                    #
                    # something exploded
                    # cut time series to last saved solution and return
                    self.nout = self.ioutnum
                    self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
                    self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
                    return -1
                if ( (self.iout>0) and (n%self.iout==0) ):
                    self.ioutnum += 1
                    self.vv[self.ioutnum,:] = self.v
                    self.tt[self.ioutnum]   = self.t
        else:
            # lots of code duplication here, but should improve speed instead of having the 'if correction' at every time step
            for n in range(1,self.nsteps+1):
                try:
                    self.step()
                    self.v += correction
                except FloatingPointError:
                    #
                    # something exploded
                    # cut time series to last saved solution and return
                    self.nout = self.ioutnum
                    self.vv.resize((self.nout+1,self.N)) # nout+1 because the IC is in [0]
                    self.tt.resize(self.nout+1)          # nout+1 because the IC is in [0]
                    return -1
                if ( (self.iout>0) and (n%self.iout==0) ):
                    self.ioutnum += 1
                    self.vv[self.ioutnum,:] = self.v
                    self.tt[self.ioutnum]   = self.t


    def fou2real(self):
        #
        # Convert from spectral to physical space
        self.uu = np.real(ifft(self.vv)).astype(np.float64)

    def calculate_lyapnov_exponents(self):
        # Compute the complex spectrum
        complex_spectrum = np.sum(self.Rii, axis=1) / (self.nsteps * self.dt)
        # Take the real part of the complex spectrum
        self.spectrum = np.real(complex_spectrum)
        # Compute the cumulative sum of the spectrum
        spec_sum = np.cumsum(self.spectrum)
        # Compute the absolute values of the cumulative sum
        abs_spec_sum = np.abs(spec_sum)
        # Find the minimum of the absolute values
        self.ky_point = np.min(abs_spec_sum)
        # Find the index of the first negative value in the cumulative sum
        self.kydim = np.where(spec_sum < 0)[0][0] if np.any(spec_sum < 0) else None


    def compute_Ek(self):
        #
        # compute all forms of kinetic energy
        #
        # Kinetic energy as a function of wavenumber and time
        self.compute_Ek_kt()
        # Time-averaged energy spectrum as a function of wavenumber
        self.Ek_k = np.sum(self.Ek_kt, 0)/(self.ioutnum+1) # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0]
        # Total kinetic energy as a function of time
        self.Ek_t = np.sum(self.Ek_kt, 1)
		# Time-cumulative average as a function of wavenumber and time
        self.Ek_ktt = np.cumsum(self.Ek_kt, 0) / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero
		# Time-cumulative average as a function of time
        self.Ek_tt = np.cumsum(self.Ek_t, 0) / np.arange(1,self.ioutnum+2)[:,None] # not self.nout because we might not be at the end; ioutnum+1 because the IC is in [0] +1 more because we divide starting from 1, not zero

    def compute_Ek_kt(self):
        try:
            self.Ek_kt = 1./2.*np.real( self.vv.conj()*self.vv / self.N ) * self.dx
        except FloatingPointError:
            #
            # probable overflow because the simulation exploded, try removing the last solution
            problem=True
            remove=1
            self.Ek_kt = np.zeros([self.nout+1, self.N]) + 1e-313
            while problem:
                try:
                    self.Ek_kt[0:self.nout+1-remove,:] = 1./2.*np.real( self.vv[0:self.nout+1-remove].conj()*self.vv[0:self.nout+1-remove] / self.N ) * self.dx
                    problem=False
                except FloatingPointError:
                    remove+=1
                    problem=True
        return self.Ek_kt


    def space_filter(self, k_cut=2):
        #
        # spatially filter the time series
        self.uu_filt  = np.zeros([self.nout+1, self.N])
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])    # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 0 # set to zero wavenumbers > k_cut
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt


    def space_filter_int(self, k_cut=2, N_int=10):
        #
        # spatially filter the time series
        self.N_int        = N_int
        self.uu_filt      = np.zeros([self.nout+1, self.N])
        self.uu_filt_int  = np.zeros([self.nout+1, self.N_int])
        self.x_int        = 2*pi*self.L*np.r_[0:self.N_int]/self.N_int
        for n in range(self.nout+1):
            v_filt = np.copy(self.vv[n,:])   # copy vv[n,:] (otherwise python treats it as reference and overwrites vv on the next line)
            v_filt[np.abs(self.k)>=k_cut] = 313e6
            v_filt_int = v_filt[v_filt != 313e6] * self.N_int/self.N
            self.uu_filt_int[n,:] = np.real(ifft(v_filt_int))
            v_filt[np.abs(self.k)>=k_cut] = 0
            self.uu_filt[n,:] = np.real(ifft(v_filt))
        #
        # compute u_resid
        self.uu_resid = self.uu - self.uu_filt
