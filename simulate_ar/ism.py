from simulate_ar import scinttools_simulate_lite
from dataclasses import dataclass
import numpy as np

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
        bw = (simulated_obs.subints[0].ssb_freq[1]-simulated_obs.subints[0].ssb_freq[0])*simulated_obs.subints[0].nchan
        dynspec = self.generate(cfreq, bw, simulated_obs.subints[0].nchan, len(simulated_obs.subints), simulated_obs.subints[0].tsub)
        # apply dynamic spectrum to simulated observation



        for isub,subint in enumerate(simulated_obs.subints):
            # subbint.data is Npol x Nchan x Nbin
            # A slice of the dynamic spectrum, ds is 1-D size Nchan
            ds = dynspec[:, isub]
            # multiply the channel axis by the ds array
            # subint.data is Npol x Nchan x Nbin
            ds = ds.reshape((1, ds.shape[0], 1))

            
            subint.data *= ds # note it can make some extra subints.



        # scattering timescale from scintilation bandwidth
        # apply scattering
        # Here we don't use the actual model, but just a simple exponential decay
        # This is simply because we are lazy to understand how to generate a good 
        # impulse response for the scattering model.
        for isub, subint in enumerate(simulated_obs.subints):
            # scint bw is in MHz
            tscat = 1/(2*np.pi*self.scint_bw*1e6) * (subint.ssb_freq/self.ref_freq)**(-4.4)
            tscat_phase = tscat / subint.ssb_period
            phase = np.arange(subint.nbin)/subint.nbin
            
            exponentials = np.exp(-np.outer(phase,1/tscat_phase)).T
            filter = np.fft.rfft(exponentials,axis=1)
            for ipol in range(subint.npol):
                data_fft = np.fft.rfft(subint.data[ipol], axis=1)
                data_fft *= filter
                subint.data[ipol] = np.fft.irfft(data_fft, axis=1)
        
        
    def generate(self, cfreq, bw ,nchan,nsub,tsub):
        scint_bw = self.scint_bw*(cfreq/self.ref_freq)**(22/5)
        scint_time = self.scint_time*(cfreq/self.ref_freq)**(6/5)
        orig_nchan=nchan

        if scint_bw/(bw/nchan) < 0.01:
            # there are 100 scintils per band... maybe just give up.
            print("NOTICE: Scintillation is too small, just return a flat spectrum")
            return np.ones((nchan,nsub))



        mb2 = 0.773*(cfreq/scint_bw)**(5/6)

        ds = 0.1/(scint_time/tsub)

        ns = int(max(nsub,1/ds))
        if ns % 2 != 0:
            ns += 1

        freq_factor=1
        while nchan<100 or bw/nchan > 2*scint_bw:
            freq_factor *=2
            nchan *=2

        print("Calling scinttools...")
        print("Scintillation bandwidth: ",scint_bw)
        print("Scintillation time: ",scint_time)
        print(f"nchan={nchan}, nsub={nsub}, tsub={tsub}, ds={ds}, ns={ns}")

        self.sim = scinttools_simulate_lite.Simulation(mb2=mb2, rf=1,ds=ds,inner=0.1*ds, ns=ns,nf=nchan,dlam=bw/cfreq,plot=True,verbose=True, freq=cfreq,dt=tsub)

        raw_dynamic_spectrum = self.sim.spi

        dynamic_spectrum = np.mean(np.reshape(raw_dynamic_spectrum, (ns, orig_nchan,freq_factor)), axis=2).T
        dynamic_spectrum = dynamic_spectrum[:, :nsub]

        # self.sim.get_pulse()
        # impulse = self.sim.pulsewin
        # print(impulse.shape)
        # time_axis = (np.arange(nchan*2)-nchan)/(bw*1e6) # Need to double check this...
        # # time axis is at the reference frequency I guess?

        # plt.plot(time_axis,impulse[:,0]/np.max(impulse[:,0]))
        # tscat = 1/(2*np.pi*scint_bw*1e6)
        # plt.plot(time_axis[time_axis>0],np.exp(-time_axis[time_axis>0]/tscat))
        # plt.show()
        return dynamic_spectrum

        
        
        


