from dataclasses import dataclass
import numpy as np

@dataclass
class SimpleNoise:
    sefd: float = 1.0
    
    def apply(self,simulated_obs):
        for isub,subint in enumerate(simulated_obs.subints):
            noise = np.random.normal(scale=self.sefd, size=subint.data.shape)
            subint.data += noise

# @TODO: Need to add the DC value for the noise
# TODO: Gain as a function of frequency, other noise sources.
# TODO: RFI.

@dataclass
class TsysNoise:
    tsys: float = 18.0 # K
    gain: float = 2.8 # K/Jy
    tsky: float = 5.0 # K @ ref_freq
    tsky_index: float = 2
    tsky_ref_freq: float = 1400.0 # MHz

    def apply(self,simulated_obs):
        for isub,subint in enumerate(simulated_obs.subints):
            T = self.tsys + self.tsky * (subint.ssb_freq/self.tsky_ref_freq)**self.tsky_index
            chanbw= simulated_obs.obs_setup.bw/simulated_obs.obs_setup.nchan
            tobs = subint.tsub
            sefd = T/(self.gain*np.sqrt(2*chanbw*1e6*(tobs/simulated_obs.obs_setup.nbin)))
            for ichan in range(simulated_obs.obs_setup.nchan):
                for ipol in range(simulated_obs.obs_setup.npol):
                    subint.data[ipol,ichan] += np.random.normal(scale=sefd[ichan], size=subint.nbin)
