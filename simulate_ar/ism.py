from simulate_ar import scinttools_simulate_lite
from dataclasses import dataclass


@dataclass 
class SimpleISM:
    dm: float = 0.0

    def get_delays(self,ssb_freq):
        # Calculate the delays for the given frequencies
        dm_delay = self.dm/2.41e-4*ssb_freq**-2
        return dm_delay
    
    def apply(self, simulated_obs):
        return
    
    def generate(self, cfreq, bw ,nchan,nsub,tsub):
        return

@dataclass 
class ScintScatISM:
    dm: float = 0.0
    scint_bw: float = 0.0
    scint_time: float = 0.0
    ref_freq: float = 1400.0

    sim: scinttools_simulate_lite.Simulation = None

    def get_delays(self,ssb_freq):
        # Calculate the delays for the given frequencies
        dm_delay = self.dm/2.41e-4*ssb_freq**-2
        return dm_delay


    def apply(self, simulated_obs):
        cfreq = (simulated_obs.subints[0].ssb_freq[-1]+simulated_obs.subints[0].ssb_freq[0])/2
        bw = (simulated_obs.subints[0].ssb_freq[1]-simulated_obs.subints[0].ssb_freq[0])*simulated_obs.nchan
        dynspec = self.make_dynspec(cfreq, bw, simulated_obs.nchan, simulated_obs.nsub, simulated_obs.tsub)
        # apply dynamic spectrum to simulated observation

        for isub,subint in enumerate(simulated_obs.subints):
            subint.data *= dynspec[:, isub] # note it can make some extra subints.

        # scattering timescale from scintilation bandwidth
        tscat = 1/(2*np.pi*scint_bw) * (freq/self.ref_freq)**(-4.4)
        # apply scattering
        for isub, subint in enumerate(simulated_obs.subints):
            tscat_phase = tscat / subint.ssb_period
            phase = np.arange(subint.nbin)/subint.nbin
            exponentials = np.exp(-phase/tscat_phase)
            data_fft = np.fft.rfft(subint.data, axis=1)
            filter = np.fft.rfft(exponentials)
            data_fft *= filter
            subint.data = np.fft.irfft(data_fft, axis=1)
        
        
    def generate(self, cfreq, bw ,nchan,nsub,tsub):
        scint_bw = self.scint_bw*(cfreq/self.ref_freq)**(22/5)
        scint_time = self.scint_time*(cfreq/self.ref_freq)**(6/5)


        mb2 = 0.773*(cfreq/scint_bw)**(5/6)

        ds = 0.1/(scint_time/tsub)

        ns = max(nsub,1/ds)
        while ns % 2 != 0:
            ns += 1

        self.sim = scinttools_simulate_lite.Simulation(mb2=mb2, rf=1,ds=ds,inner=0.1*ds, ns=ns,nf=nchan,dlam=bw/cfreq,plot=True,verbose=True, freq=cfreq,dt=tsub)
        return self.sim.spi.T

        
        
        


