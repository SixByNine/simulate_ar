import numpy as np
import matplotlib.pyplot as plt
import scipy.special 
from dataclasses import dataclass

@dataclass
class VonMisesComponent:
    phase: float
    k: float
    amp: float
    spectral_index: float = 0
    ref_freq: float = 1400.0

class VonMisesProfile:
    def __init__(self, vm_components):
        self.components = vm_components

    def set_flux(self, flux):
        current_flux = 0
        for vm in self.components:
            # ive is the exponentially scaled modified Bessel function of the first kind
            # Which is the integral of our von mises function
            # We want to set the flux to be the integral of the profile
            current_flux += vm.amp * scipy.special.ive(0,vm.k)
            # We want to set the flux to be the integral of the profile
        scale_factor = flux/current_flux
        for vm in self.components:
            vm.amp *= scale_factor


    def compute(self,phase, freq, epoch,nbin):
        freq = np.atleast_1d(freq)
        outprof = np.zeros((len(freq), nbin))
        phase_axis = np.arange(nbin)/nbin
        for vm in self.components:
            amp = vm.amp * (vm.ref_freq/freq)**vm.spectral_index
            outprof += np.outer(amp, np.exp(((np.cos(2*np.pi*(phase_axis - phase+vm.phase))-1) *vm.k)))

        return outprof


class SubPulseGenerator:
    def __init__(self, navg:int, profile_generator):
        self.subpulse_covariance_width = 0.1
        self.modulation_index = 0.5
        self.L = None
        self.profile_generator = profile_generator
        self.navg = navg

    def generate_L(self,nbin):
        self.nbin=nbin
        covariance_matrix = np.zeros((self.nbin, self.nbin))
        for i in range(self.nbin):
            for j in range(self.nbin):
                delta_phase = (i - j)/self.nbin
                covariance_matrix[i, j] = self.modulation_index * np.sinc(np.abs(delta_phase) / self.subpulse_covariance_width)

        self.L = np.linalg.cholesky(covariance_matrix+np.diag(1e-6*np.diag(covariance_matrix)))
        
    def compute(self,phase, freq, epoch,nbin):
        if self.L is None or self.nbin != nbin:
            self.generate_L(nbin)

        pulses = np.zeros(nbin)
        for i in range(self.navg):
            y = self.L.dot(np.random.normal(size=(nbin)))
            pulses += (1+y)
        return (pulses/self.navg) * self.profile_generator.compute(phase, freq, epoch,nbin)

    def set_parameters_from_data(self, stack, onmask,offmask):
        onstack = stack[:,onmask]
        npulse,nbin = stack.shape
        meanprof = np.mean(onstack,axis=0)
        white=np.median(np.std(stack[:,offmask],axis=1))
        print(white)

        phase = np.arange(nbin)/nbin
        onphase = phase[onmask]
        modulation_index = np.std(onstack,axis=0)/np.mean(onstack,axis=0)
        min_modulation_index = np.min(modulation_index)

        plt.plot(onphase,meanprof/np.amax(meanprof))
        plt.plot(onphase,np.std(onstack,axis=0)/np.mean(onstack,axis=0))
        plt.ylim(0,3)
        plt.axhline(min_modulation_index)
        plt.show()




        C = np.zeros((onstack.shape[1],onstack.shape[1]))
        meanprof=np.mean(onstack,axis=0)
        for i in range(onstack.shape[0]):
            p = (onstack[i] - meanprof)/meanprof
            C += np.outer(p,p)

        C /= onstack.shape[0]
        ibin=np.argmax(meanprof)
        cf=C[ibin]
        xx=np.arange(len(cf))

        def gauss(x, A, mu, sig):
            return A*np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)
        
        def gauss_sin(x, A, mu, sig,w):
            return A*np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)*np.cos(2*np.pi*(x-mu)/w)
        # def minimize(x):
        #     return np.sum((gauss_sin(xx, *x) - cf)**2)

        def sinc(x, A, x0,w):
            return A*np.sinc((x-x0)/w)

        def minimize(x):
            return np.sum((sinc(xx, *x) - cf)**2)
        
        res = opt.differential_evolution(minimize, bounds=[(0,4*np.amax(cf)),(0,len(cf)),(0,len(cf))],maxiter=1000,popsize=128,mutation=(0.5,1.2))
        print(res)  
        optP = res.x
        correlation_width=optP[2]/nbin
        print("Corr_width",correlation_width)
        plt.plot(xx,C[ibin])
        plt.plot(xx,sinc(xx, *optP),color='red')

        self.modulation_index = min_modulation_index
        self.subpulse_covariance_width = correlation_width
        self.L=None
        return min_modulation_index, white, correlation_width

