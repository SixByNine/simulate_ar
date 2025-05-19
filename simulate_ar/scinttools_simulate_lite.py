
import numpy as np
from numpy import random
from numpy.random import randn
from numpy.fft import fft2, ifft2
from scipy.special import gamma
import scipy.constants as sc
import matplotlib.pyplot as plt


class Simulation():

    def __init__(self, mb2=2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                 inner=0.001, ns=256, nf=256, dlam=0.25, lamsteps=False,
                 seed=None, nx=None, ny=None, dx=None, dy=None, plot=False,
                 verbose=False, freq=1400, dt=30, mjd=60000, nsub=None,
                 efield=False, noise=None):
        """
        Electromagnetic simulator based on original code by Coles et al. (2010)

        mb2: Max Born parameter for strength of scattering
        rf: Fresnel scale
        ds (or dx,dy): Spatial step sizes with respect to rf
        alpha: Structure function exponent (Kolmogorov = 5/3)
        ar: Anisotropy axial ratio
        psi: Anisotropy orientation
        inner: Inner scale w.r.t rf - should generally be smaller than ds
        ns (or nx,ny): Number of spatial steps
        nf: Number of frequency steps.
        dlam: Fractional bandwidth relative to centre frequency
        lamsteps: Boolean to choose whether steps in lambda or freq
        seed: Seed number, or use "-1" to shuffle
        """

        self.mb2 = mb2
        self.rf = rf
        self.ds = ds
        self.dx = dx if dx is not None else ds
        self.dy = dy if dy is not None else ds
        self.alpha = alpha
        self.ar = ar
        self.psi = psi
        self.inner = inner
        self.nx = nx if nx is not None else ns
        self.ny = ny if ny is not None else ns
        self.nf = nf
        self.dlam = dlam
        self.lamsteps = lamsteps
        self.seed = seed

        # Now run simulation
        self.set_constants()
        if verbose:
            print('Computing screen phase')
        self.get_screen()
        if verbose:
            print('Getting intensity...')
        self.get_intensity(verbose=verbose)
        if nf > 1:
            if verbose:
                print('Computing dynamic spectrum')
            self.get_dynspec()
        if verbose:
            print('Getting impulse response...')
        self.get_pulse()
        if plot:
            self.plot_all()

        # Now prepare simulation for use with scintools, using physical units
        self.name =\
            'sim:mb2={0},ar={1},psi={2},dlam={3}'.format(self.mb2, self.ar,
                                                         self.psi, self.dlam)
        if lamsteps:
            self.name += ',lamsteps'

        self.header = [self.name, 'MJD0: {}'.format(mjd)]
        if efield:
            dyn = np.real(self.spe)
        else:
            dyn = self.spi
        dlam = self.dlam

        self.dt = dt
        self.freq = freq
        self.nsub = int(np.shape(dyn)[0]) if nsub is None else nsub
        self.nchan = int(np.shape(dyn)[1])
        # lams = np.linspace(1-self.dlam/2, 1+self.dlam/2, self.nchan)
        # freqs = np.divide(1, lams)
        # freqs = np.linspace(np.min(freqs), np.max(freqs), self.nchan)
        # self.freqs = freqs*self.freq/np.mean(freqs)
        if not lamsteps:
            self.df = self.freq*self.dlam/(self.nchan - 1)
            self.freqs = self.freq + np.arange(-self.nchan/2,
                                               self.nchan/2, 1)*self.df
        else:
            self.lam = sc.c/(self.freq*10**6)  # centre wavelength in m
            self.dl = self.lam*self.dlam/(self.nchan - 1)
            self.lams = self.lam + np.arange(-self.nchan/2,
                                             self.nchan/2, 1)*self.dl
            self.freqs = sc.c/self.lams/10**6  # in MHz
            self.freq = (np.max(self.freqs) - np.min(self.freqs))/2
        self.bw = max(self.freqs) - min(self.freqs)
        self.times = self.dt*np.arange(0, self.nsub)
        self.df = self.bw/self.nchan
        self.tobs = float(self.times[-1] - self.times[0])
        self.mjd = mjd
        if nsub is not None:
            dyn = dyn[0:nsub, :]
        self.dyn = np.transpose(dyn)

        # # Theoretical arc curvature
        V = self.ds / self.dt
        lambda0 = self.freq  # wavelength, c=1
        k = 2*np.pi/lambda0  # wavenumber
        L = self.rf**2 * k
        # Curvature to use for Dynspec object within scintools
        self.eta = L/(2 * V**2) / 10**6 / np.cos(psi * np.pi/180)**2
        c = 299792458.0  # m/s
        beta_to_eta = c*1e6/((self.freq*10**6)**2)
        # Curvature for wavelength-rescaled dynamic spectrum
        self.betaeta = self.eta / beta_to_eta

        return

    def set_constants(self):

        ns = 1
        lenx = self.nx*self.dx
        leny = self.ny*self.dy
        self.ffconx = (2.0/(ns*lenx*lenx))*(np.pi*self.rf)**2
        self.ffcony = (2.0/(ns*leny*leny))*(np.pi*self.rf)**2
        dqx = 2*np.pi/lenx
        dqy = 2*np.pi/leny
        # dqx2 = dqx*dqx
        # dqy2 = dqy*dqy
        a2 = self.alpha*0.5
        # spow = (1.0+a2)*0.5
        # ap1 = self.alpha+1.0
        # ap2 = self.alpha+2.0
        aa = 1.0+a2
        ab = 1.0-a2
        cdrf = 2.0**(self.alpha)*np.cos(self.alpha*np.pi*0.25)\
            * gamma(aa)/self.mb2
        self.s0 = self.rf*cdrf**(1.0/self.alpha)

        cmb2 = self.alpha*self.mb2 / (4*np.pi *
                                      gamma(ab)*np.cos(self.alpha *
                                                       np.pi*0.25)*ns)
        self.consp = cmb2*dqx*dqy/(self.rf**self.alpha)
        self.scnorm = 1.0/(self.nx*self.ny)

        # ffconlx = ffconx*0.5
        # ffconly = ffcony*0.5
        self.sref = self.rf**2/self.s0
        return

    def get_screen(self):
        """
        Get phase screen in x and y
        """
        random.seed(self.seed)  # Set the seed, if any

        nx2 = int(self.nx/2 + 1)
        ny2 = int(self.ny/2 + 1)

        w = np.zeros([self.nx, self.ny])  # initialize array
        dqx = 2*np.pi/(self.dx*self.nx)
        dqy = 2*np.pi/(self.dy*self.ny)

        # first do ky=0 line
        k = np.arange(2, nx2+1)
        w[k-1, 0] = self.swdsp(kx=(k-1)*dqx, ky=0)
        w[self.nx+1-k, 0] = w[k, 0]
        # then do kx=0 line
        ll = np.arange(2, ny2+1)
        w[0, ll-1] = self.swdsp(kx=0, ky=(ll-1)*dqy)
        w[0, self.ny+1-ll] = w[0, ll-1]
        # now do the rest of the field
        kp = np.arange(2, nx2+1)
        k = np.arange((nx2+1), self.nx+1)
        km = -(self.nx-k+1)
        for il in range(2, ny2+1):
            w[kp-1, il-1] = self.swdsp(kx=(kp-1)*dqx, ky=(il-1)*dqy)
            w[k-1, il-1] = self.swdsp(kx=km*dqx, ky=(il-1)*dqy)
            w[self.nx+1-kp, self.ny+1-il] = w[kp-1, il-1]
            w[self.nx+1-k, self.ny+1-il] = w[k-1, il-1]

        # done the whole screen weights, now generate complex gaussian array
        xyp = np.multiply(w, np.add(randn(self.nx, self.ny),
                                    1j*randn(self.nx, self.ny)))

        xyp = np.real(fft2(xyp))
        self.w = w
        self.xyp = xyp
        return

    def get_intensity(self, verbose=True):
        spe = np.zeros([self.nx, self.nf],
                       dtype=np.dtype(np.csingle)) + \
            1j*np.zeros([self.nx, self.nf],
                        dtype=np.dtype(np.csingle))
        for ifreq in range(0, self.nf):
            if verbose:
                if ifreq % round(self.nf/100) == 0:
                    print(int(np.floor((ifreq+1)*100/self.nf)), '%')
            if self.lamsteps:
                scale = 1.0 +\
                    self.dlam * (ifreq - 1 - (self.nf / 2)) / (self.nf)
            else:
                frfreq = 1.0 +\
                    self.dlam * (-0.5 + ifreq / self.nf)
                scale = 1 / frfreq
            scaled = scale
            xye = fft2(np.exp(1j * self.xyp * scaled))
            xye = self.frfilt3(xye, scale)
            xye = ifft2(xye)
            gam = 0
            spe[:, ifreq] = xye[:, int(np.floor(self.ny / 2))] / scale**gam

        xyi = np.real(np.multiply(xye, np.conj(xye)))

        self.xyi = xyi
        self.spe = spe
        return

    def get_dynspec(self):
        if self.nf == 1:
            print('no spectrum because nf=1')

        # dynamic spectrum
        spi = np.real(np.multiply(self.spe, np.conj(self.spe)))
        self.spi = spi

        self.x = np.linspace(0, self.dx*(self.nx), (self.nx))
        ifreq = np.linspace(0, self.nf-1, self.nf)
        lam_norm = 1.0 + self.dlam * (ifreq - 1 - (self.nf / 2)) / self.nf
        self.lams = lam_norm / np.mean(lam_norm)
        frfreq = 1.0 + self.dlam * (-0.5 + ifreq / self.nf)
        self.freqs = frfreq / np.mean(frfreq)
        return

    def get_pulse(self):
        """
        script to get the pulse shape vs distance x from spe

        you usually need a spectral window because the leading edge of the
        pulse response is very steep. it is also attractive to pad the spe file
        with zeros before FT of course this correlates adjacent samples in the
        pulse response
        """
        if not hasattr(self, 'spe'):
            self.get_intensity()

        # get electric field impulse response
        p = np.fft.fft(np.multiply(self.spe, np.blackman(self.nf)), 2*self.nf)
        p = np.real(p*np.conj(p))  # get intensity impulse response
        # shift impulse to middle of window
        self.pulsewin = np.transpose(np.roll(p, self.nf))

        # get phase delay from the phase screen
        # get units of 1/2BW from phase
        self.dm = self.xyp[:, int(self.ny/2)]*self.dlam/np.pi

    def swdsp(self, kx=0, ky=0):
        cs = np.cos(self.psi*np.pi/180)
        sn = np.sin(self.psi*np.pi/180)
        r = self.ar
        con = np.sqrt(self.consp)
        alf = -(self.alpha+2)/4
        # anisotropy parameters
        a = (cs**2)/r + r*sn**2
        b = r*cs**2 + sn**2/r
        c = 2*cs*sn*(1/r-r)
        q2 = a * np.power(kx, 2) + b * np.power(ky, 2) + c*np.multiply(kx, ky)
        # isotropic inner scale
        out = con*np.multiply(np.power(q2, alf),
                              np.exp(-(np.add(np.power(kx, 2),
                                              np.power(ky, 2))) *
                                     self.inner**2/2))
        return out

    def frfilt3(self, xye, scale):
        nx2 = int(self.nx / 2) + 1
        ny2 = int(self.ny / 2) + 1
        filt = np.zeros([nx2, ny2], dtype=np.dtype(np.csingle))
        q2x = np.linspace(0, nx2-1, nx2)**2 * scale * self.ffconx
        for ly in range(0, ny2):
            q2 = q2x + (self.ffcony * (ly**2) * scale)
            filt[:, ly] = np.cos(q2) - 1j * np.sin(q2)

        xye[0:nx2, 0:ny2] = np.multiply(xye[0:nx2, 0:ny2], filt[0:nx2, 0:ny2])
        xye[self.nx:nx2-1:-1, 0:ny2] = np.multiply(
            xye[self.nx:nx2-1:-1, 0:ny2], filt[1:(nx2 - 1), 0:ny2])
        xye[0:nx2, self.ny:ny2-1:-1] =\
            np.multiply(xye[0:nx2, self.ny:ny2-1:-1], filt[0:nx2, 1:(ny2-1)])
        xye[self.nx:nx2-1:-1, self.ny:ny2-1:-1] =\
            np.multiply(xye[self.nx:nx2-1:-1, self.ny:ny2-1:-1],
                        filt[1:(nx2-1), 1:(ny2-1)])
        return xye

    def plot_screen(self, subplot=False):
        if not hasattr(self, 'xyp'):
            self.get_screen()
        x_steps = np.linspace(0, self.dx*self.nx, self.nx)
        y_steps = np.linspace(0, self.dy*self.ny, self.ny)
        plt.pcolormesh(x_steps, y_steps, np.transpose(self.xyp))
        plt.title("Screen phase")
        plt.ylabel('$y/r_f$')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_intensity(self, subplot=False):
        # routine to plot intensity
        if not hasattr(self, 'xyi'):
            self.get_intensity()
        x_steps = np.linspace(0, self.dx*(self.nx), (self.nx))
        y_steps = np.linspace(0, self.dy*(self.ny), (self.ny))
        plt.pcolormesh(x_steps, y_steps, np.transpose(self.xyi))
        plt.title('Intensity / Mean')
        plt.ylabel('$y/r_f$')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_dynspec(self, subplot=False):
        if not hasattr(self, 'spi'):
            self.get_dynspec()

        if self.lamsteps:
            plt.pcolormesh(self.x, self.lams, np.transpose(self.spi))
            plt.ylabel(r'Wavelength $\lambda$')
        else:
            plt.pcolormesh(self.x, self.freqs, np.transpose(self.spi))
            plt.ylabel('Frequency f')
        plt.title('Dynamic Spectrum (Intensity/Mean)')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_efield(self, subplot=False):
        if not hasattr(self, 'spe'):
            self.get_intensity()

        if self.lamsteps:
            plt.pcolormesh(self.x, self.lams,
                           np.real(np.transpose(self.spe)))
            plt.ylabel(r'Wavelength $\lambda$')
        else:
            plt.pcolormesh(self.x, self.freqs,
                           np.real(np.transpose(self.spe)))
            plt.ylabel('Frequency f')
        plt.title('Electric field (Intensity/Mean)')
        plt.xlabel('$x/r_f$')
        if not subplot:
            plt.show()
        return

    def plot_delay(self, subplot=False):
        # get frequency to set the scale, enter in GHz
        Freq = self.freq/1000
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, self.dx*self.nx, self.nx),
                 -self.dm/(2*self.dlam*Freq))
        plt.ylabel('Group delay (ns)')
        plt.xlabel('$x/r_f$')
        plt.subplot(2, 1, 2)
        plt.plot(np.mean(self.pulsewin, axis=1))
        plt.ylabel('Intensity (arb)')
        plt.xlabel('Delay (arb)')
        plt.show()
        return

    def plot_pulse(self, subplot=False):
        # get frequency to set the scale, enter in GHz
        Freq = self.freq/1000
        lpw = np.log10(self.pulsewin)
        vmax = np.max(lpw)
        vmin = np.median(lpw) - 3
        plt.pcolormesh(np.linspace(0, self.dx*self.nx, self.nx),
                       (np.arange(0, 3*self.nf/2, 1) - self.nf/2) /
                       (2*self.dlam*Freq),
                       lpw[int(self.nf/2):, :], vmin=vmin, vmax=vmax)
        plt.colorbar
        plt.ylabel('Delay (ns)')
        plt.xlabel('$x/r_f$')
        plt.plot(np.linspace(0, self.dx*self.nx, self.nx),
                 -self.dm/(2*self.dlam*Freq), 'k')  # group delay=-phase delay
        plt.show()

    def plot_all(self):
        plt.figure(2)
        plt.subplot(2, 2, 1)
        self.plot_screen(subplot=True)
        plt.subplot(2, 2, 2)
        self.plot_intensity(subplot=True)
        plt.subplot(2, 1, 2)
        self.plot_dynspec(subplot=True)
        plt.show()

