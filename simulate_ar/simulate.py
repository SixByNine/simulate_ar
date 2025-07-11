import astropy.coordinates as coord
import subprocess
import t2pred
import os,shutil
import tempfile
import numpy as np
import astropy.units as u
import json
import datetime
import astropy.io.fits as fits
from dataclasses import dataclass

from .pulsar import Pulsar
from .epoch import Epoch


def rotate_phs_1d(inprof, phase_shift):
    ff = np.fft.rfft(inprof)
    fr = ff * np.exp(1.0j * 2 * np.pi * np.arange(len(ff)) * phase_shift)
    return np.fft.irfft(fr)


@dataclass
class Observatory:
    name: str = "MeerKAT"
    tempo2_name: str = "MeerKAT"
    x: float = 5109360.133
    y: float = 2006852.586
    z: float = -3238948.127
    
    def get_site(self):
        return coord.EarthLocation(x=self.x*u.m, y=self.y*u.m, z=self.z*u.m)

@dataclass
class ObsSetup:
    start_epoch: Epoch
    observer: str = "Simulation"
    projid: str = "Simulation"
    nchan: int = 128
    nbin: int = 1024
    nsub: int = 30
    npol: int = 1 # currently only 1 pol supported!
    cfreq: float = 1400.0
    bw: float = 64.0
    tsub: float = 10.0

    def get_freq_table(self):
        chan_bw = self.bw / self.nchan
        fch1 = self.cfreq - self.bw/2 + chan_bw/2
        freq_table = np.arange(self.nchan) * chan_bw + fch1
        return freq_table

@dataclass
class Frontend:
    name: str = "Unknown"
    fd_poln: str = "LIN"
    fd_hand: int = -1
    fd_sang: float = 45.0
    fd_xyph: float = 0.0
    nrcvr: int = 2

@dataclass
class Backend:
    name: str = "Unknown"
    config: str = ""
    be_phase: int = 1
    be_dcc: int = 0
    be_delay: float = 0.0

@dataclass 
class Subint:
    ssb_freq: np.ndarray
    site_freq : np.ndarray
    ssb_epoch: Epoch
    site_epoch: Epoch
    ssb_period: float
    topo_period: float
    doppler_factor: float
    lst: str
    pointing: coord.SkyCoord
    par_angle: float
    altaz: coord.AltAz
    nbin: int
    nchan: int
    npol: int
    tsub: float
    # 
    data: np.ndarray
    weights: np.ndarray



@dataclass
class Simulation:
    observatory: Observatory
    obs_setup: ObsSetup
    frontend: Frontend
    backend: Backend
    pulsar: Pulsar


    def setup(self):
        earth_location = self.observatory.get_site()
        coordinate = self.pulsar.get_coordinate()
        # Make site predictor
        with  tempfile.TemporaryDirectory() as tempdir:
            print(f"Working in {tempdir}")
            temp_par = os.path.join(tempdir,"sim.par")
            shutil.copy(self.pulsar.parfile,temp_par)
            flo=self.obs_setup.cfreq - self.obs_setup.bw/2
            fhi=self.obs_setup.cfreq + self.obs_setup.bw/2
            predictor_buffer=1/86400.0
            predictor_length = self.obs_setup.tsub*self.obs_setup.nsub/86400.
            obs_epoch = self.obs_setup.start_epoch
            print("Call tempo2")
            print(f"tempo2 -pred \"{self.observatory.tempo2_name} {obs_epoch-predictor_buffer} {obs_epoch+predictor_length+predictor_buffer} {flo} {fhi} 12 2 3660\" -f sim.par")
            subprocess.call(f"tempo2 -pred \"{self.observatory.tempo2_name} {obs_epoch-predictor_buffer} {obs_epoch+predictor_length+predictor_buffer} {flo} {fhi} 12 2 3660\" -f sim.par",
                            shell=True,cwd=tempdir)
            
            pred_file = os.path.join(tempdir,"t2pred.dat")
            with open(pred_file,"r") as f:
                predlines=[]
                for line in f:
                    predlines.append(f"{line.strip()}")
                    print(f"'{predlines[-1]}'")
            site_predictor = t2pred.phase_predictor(pred_file)
            self.site_predictor = site_predictor
            self.predictor_lines = predlines

            # Compute subint start times to align with phase 0

            subint_epochs = []
            subint_toffs = []
            self.subints = []
            for isub in range(self.obs_setup.nsub):
                intitial_epoch = obs_epoch + isub*self.obs_setup.tsub/86400.0
                phase = self.site_predictor.getPrecisePhase(intitial_epoch.imjd,intitial_epoch.fmjd,self.obs_setup.cfreq)
                freq = self.site_predictor.getFrequency(intitial_epoch.imjd+intitial_epoch.fmjd,self.obs_setup.cfreq)
                offs = isub*self.obs_setup.tsub - phase/freq
                subint_toffs.append(offs)
                new_epoch = obs_epoch + offs/86400.0
                subint_epochs.append(new_epoch)
                # phase2 = self.site_predictor.getPrecisePhase(new_epoch.imjd,new_epoch.fmjd,self.obs_setup.cfreq)
                # print(phase, phase2,freq)

            # Compute SSB information.

        
            with open(os.path.join(tempdir,"trial.tim"),"w") as f:
                f.write("FORMAT 1\n")
                for isub,e in enumerate(subint_epochs):
                    f.write(f" inffreq_{isub} 0.0 {e} 1.0 {self.observatory.tempo2_name}\n")
                    f.write(f" 1freq_{isub} 1.0 {e} 1.0 {self.observatory.tempo2_name}\n")

            print(f"tempo2 -nofit -output general2 -f sim. trial.tim  -s \"{{sat}} {{bat}} {{freq}} {{freqssb}} ZZ\\n\"")
            subprocess.call(f"tempo2 -nofit -output general2 -f sim.par trial.tim  -s \"{{sat}} {{bat}} {{freq}} {{freqssb}} ZZ\\n\" | grep ZZ > trial.info",shell=True,cwd=tempdir)

            bary_epoch=[]
            doppler=[]
            with open(os.path.join(tempdir,"trial.info"),"r") as f:
                for isub in range(self.obs_setup.nsub):
                    e = f.readline().split()
                    barymjd_int = int(e[1].split(".")[0])
                    barymjd_frac = float("0."+e[1].split(".")[1])
                    bary_epoch.append(Epoch(barymjd_int,barymjd_frac))
                    e = f.readline().split() # the 1MHz one
                    doppler.append(float(e[3])/1e6)

            predictor_buffer=3600/86400.0 # Make the ssb predictor much longer than needed

            print(f"tempo2 -pred \"@ {obs_epoch-predictor_buffer} {obs_epoch+predictor_length+predictor_buffer} 1e11 1e13 12 2 3660\" -f sim.par")

            subprocess.call(f"tempo2 -pred \"@ {obs_epoch-predictor_buffer} {obs_epoch+predictor_length+predictor_buffer} 1e11 1e13 12 2 3660\" -f sim.par",shell=True,cwd=tempdir)

            self.ssb_predictor = t2pred.phase_predictor(os.path.join(tempdir,"t2pred.dat"))
            
            freq_table = self.obs_setup.get_freq_table()

            for isub in range(self.obs_setup.nsub):
                offs_sub=subint_toffs[isub]
                aux_dm=0
                aux_rm=0
                dat_freq = freq_table
                ssb_freq = freq_table*doppler[isub]
                epoch = subint_epochs[isub]
                ssb_epoch = bary_epoch[isub]

                bary_period = 1/self.ssb_predictor.getFrequency(bary_epoch[isub].imjd+bary_epoch[isub].fmjd, 1e12) 
                topo_period = 1/self.site_predictor.getFrequency(epoch.imjd+epoch.fmjd, self.obs_setup.cfreq)

                nowcoord = coordinate.apply_space_motion(new_obstime=epoch.to_astropy())
                atime = epoch.to_astropy()
                altaz = nowcoord.transform_to(coord.AltAz(obstime=atime, location=earth_location))
                lst = atime.sidereal_time('mean', earth_location)

                H = (lst - coordinate.ra).radian
                parallactic_angle = 180*np.arctan2(np.sin(H),
                       (np.tan(earth_location.lat.radian) *
                        np.cos(coordinate.dec.radian) -
                        np.sin(coordinate.dec.radian)*np.cos(H)))/np.pi

                subint = Subint(ssb_freq=ssb_freq,site_freq=dat_freq,
                                ssb_epoch=ssb_epoch,site_epoch=epoch,
                                ssb_period=bary_period,topo_period=topo_period,
                                doppler_factor=doppler[isub],lst=lst,
                                pointing=nowcoord,par_angle=parallactic_angle,
                                altaz=altaz,nbin=self.obs_setup.nbin,
                                nchan=self.obs_setup.nchan,npol=self.obs_setup.npol,tsub=self.obs_setup.tsub,
                                data=np.zeros((self.obs_setup.npol,self.obs_setup.nchan,self.obs_setup.nbin)),
                                weights=np.ones(self.obs_setup.nchan))
                self.subints.append(subint)
    

    def make_fits(self):
        try:
            internal_datdir=os.path.dirname(__file__)
        except:
            internal_datdir="."
        with open(os.path.join(internal_datdir,"psrfits_structure.json"), "r") as json_file:
            fits_struct = json.load(json_file)
        obs_struct=fits_struct['PRIMARY']

        overwrite_keys={}
        
        # Observation parameters
        overwrite_keys['DATE']=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        overwrite_keys['OBSERVER']=self.obs_setup.observer
        overwrite_keys['SRC_NAME']=self.pulsar.name()
        overwrite_keys['OBSFREQ']=self.obs_setup.cfreq
        overwrite_keys['OBSBW']=self.obs_setup.bw
        overwrite_keys['OBSNCHAN']=self.obs_setup.nchan
        overwrite_keys['RA'] = self.subints[0].pointing.ra.to_string(unit=u.hour, sep=':', pad=True)
        overwrite_keys['DEC'] = self.subints[0].pointing.dec.to_string(unit=u.deg, sep=':', pad=True)
        overwrite_keys['STT_CRD1'] = overwrite_keys['RA']
        overwrite_keys['STT_CRD2'] = overwrite_keys['DEC']
        overwrite_keys['STP_CRD1'] = overwrite_keys['RA']
        overwrite_keys['STP_CRD2'] = overwrite_keys['DEC']
        overwrite_keys['TRK_MODE'] = "TRACK"
        overwrite_keys['COORD_MD'] = "J2000"

        # Start epoch
        imjd,smjd,offs = self.obs_setup.start_epoch.get_fits_epoch()
        overwrite_keys['STT_IMJD']=imjd
        overwrite_keys['STT_SMJD']=smjd
        overwrite_keys['STT_OFFS']=offs

        # Observatory parameters
        overwrite_keys['TELESCOP']=self.observatory.name
        overwrite_keys['ANT_X'] = self.observatory.x
        overwrite_keys['ANT_Y'] = self.observatory.y
        overwrite_keys['ANT_Z'] = self.observatory.z

        # Frontend Parameters
        overwrite_keys['FRONTEND'] = self.frontend.name
        overwrite_keys['FE_POLN'] = self.frontend.fd_poln
        overwrite_keys['FE_HAND'] = self.frontend.fd_hand
        overwrite_keys['FE_SANG'] = self.frontend.fd_sang
        overwrite_keys['FE_XYPH'] = self.frontend.fd_xyph
        overwrite_keys['NRCVR'] = self.frontend.nrcvr

        # Backend parameters
        overwrite_keys['BACKEND']  = self.backend.name
        overwrite_keys['BECONFIG'] =self.backend.config
        overwrite_keys['BE_PHASE'] = self.backend.be_phase
        overwrite_keys['BE_DCC']   = self.backend.be_dcc
        overwrite_keys['BE_DELAY'] = self.backend.be_delay


        nchan=self.obs_setup.nchan
        nbin=self.obs_setup.nbin
        nsub=self.obs_setup.nsub
        npol=self.obs_setup.npol

        primary_hdu = fits.PrimaryHDU()
        for key in obs_struct:
            if key in overwrite_keys:
                primary_hdu.header[key] = overwrite_keys[key]
            else:
                primary_hdu.header[key] = obs_struct[key]

        l=[primary_hdu]
        for hduname in fits_struct.keys():
            if hduname == "PRIMARY":
                continue
            
            table_structure = fits_struct[hduname]['tab']
            if hduname == "SUBINT":
                table_structure['DATA']['format'] = f'{nchan*nbin*npol}I'
                table_structure['DAT_SCL']['format'] = f'{nchan*npol}E'
                table_structure['DAT_OFFS']['format'] = f'{nchan*npol}E'
                table_structure['DAT_WTS']['format'] = f'{nchan}E'
                table_structure['DAT_FREQ']['format'] = f'{nchan}E'
                nrows=nsub
            if hduname=="HISTORY":
                nrows=1
            cols = []
            for key, value in table_structure.items():
                col = fits.Column(name=key, format=value['format'], unit=value['unit'])
                cols.append(col)
            
            hdu = fits.BinTableHDU.from_columns(cols,nrows=nrows)
            hdu.header['EXTNAME'] = hduname
            for key in fits_struct[hduname]['hdr']:
                hdu.header[key] = fits_struct[hduname]['hdr'][key]
            if hduname == "SUBINT":
                hdu.header['NPOL']=npol
                if npol==1:
                    hdu.header["POL_TYPE"]= "AA+BB"
                if npol==4:
                    hdu.header["POL_TYPE"]= "IQUV"
                hdu.header["NBIN"]=nbin
                hdu.header["NCHAN"]=nchan
                hdu.header["CHAN_BW"] = self.obs_setup.bw/nchan
                hdu.header['REFFREQ'] = self.obs_setup.cfreq
                hdu.header['DM'] = self.pulsar.dm()
                hdu.header['RM'] = 0.0

            l.append(hdu)

        hdul = fits.HDUList(l)

        hdu=hdul['T2PREDICT']
        newhdu = fits.BinTableHDU.from_columns(hdu.columns, nrows=len(self.predictor_lines),character_as_bytes=True,fill=False)
        for iline in range(len(self.predictor_lines)):
            newhdu.data[iline] = (self.predictor_lines[iline].encode('UTF-8'),)
        newhdu.header=hdu.header
        hdul['T2PREDICT'] = newhdu

        hdu=hdul['PSRPARAM']
        newhdu = fits.BinTableHDU.from_columns(hdu.columns, nrows=len(self.pulsar.parfile_contents))
        for iline in range(len(self.pulsar.parfile_contents)):
            newhdu.data[iline] = (self.pulsar.parfile_contents[iline],)
        newhdu.header=hdu.header
        hdul['PSRPARAM'] = newhdu
        
        subint_table = hdul['SUBINT']

        for isub,si in enumerate(self.subints):
            
            prof = si.data*1e3 # PSRFITS data is in mJy, data are in Jy.

            profmax = np.reshape(np.amax(prof,axis=2),(npol,nchan,1))
            profmin = np.reshape(np.amin(prof,axis=2),(npol,nchan,1))
            rng = profmax - profmin
            scale = 1/rng
            prof = scale*(prof-profmin)
            # scale the profile
            prof = prof * (2**15 - 1)
            prof = prof.astype(np.int16)
            dat_offs = profmin.reshape(npol*nchan)
            dat_scl = rng.reshape(npol*nchan)/ (2**15 - 1)
            data = prof.flatten()

            ra_sub=si.pointing.ra.deg
            dec_sub=si.pointing.dec.deg
            glon_sub=si.pointing.galactic.l.deg
            glat_sub=si.pointing.galactic.b.deg
            par_ang=si.par_angle
            lst = si.lst.deg/360*86400.0
            period = si.topo_period
            fd_ang = 0 # TODO : WHAT ARe these
            pos_ang = 0 # TODO : WHAT ARe these
            tel_az = si.altaz.az.deg
            tel_zen = si.altaz.zen.deg
            aux_dm=0
            aux_rm=0
            tsub=si.tsub
            dat_freq = si.site_freq
            dat_wts = si.weights

            offs_sub = (si.site_epoch - self.obs_setup.start_epoch).to_seconds() # TODO: Check if this is accurate enough?

            data = [isub,tsub,offs_sub,period,lst,ra_sub,dec_sub,glon_sub,glat_sub,fd_ang,
                    pos_ang,par_ang,tel_az,tel_zen,aux_dm,aux_rm,dat_freq,dat_wts,dat_offs,dat_scl,data]
            
            subint_table.data[isub] = data



        return hdul
        

    def generate_data(self,generator,propagation_model):
            #def compute(self,phase, freq, epoch,nbin):
        for isub,subint in enumerate(self.subints):
            print(f"Generating subint {isub}/{len(self.subints)}")
            bary_epoch = subint.ssb_epoch
            phase0 = self.ssb_predictor.getPrecisePhase(bary_epoch.imjd, np.longdouble(bary_epoch.fmjd), 1e12)
            prof = generator.compute(-phase0, subint.ssb_freq, bary_epoch, subint.nbin)
            #print("phase",phase0)
            if len(prof.shape) == 2:
                prof = np.reshape(prof,(1,self.obs_setup.nchan,subint.nbin))
            if prof.shape != subint.data.shape:
                raise ValueError(f"Profile shape {prof.shape} does not match subint shape {subint.data.shape}")

            dm_delay = propagation_model.get_delays(subint.ssb_freq) # Maybe more parameters?
            for ichan in range(self.obs_setup.nchan):
                phase = self.ssb_predictor.getPrecisePhase(bary_epoch.imjd,np.longdouble(bary_epoch.fmjd-dm_delay[ichan]/86400.0), 1e12)
                #print("ichan,phase=",ichan,phase)
                for ipol in range(self.obs_setup.npol):
                    subint.data[ipol,ichan] = rotate_phs_1d(prof[ipol,ichan],phase-phase0)

        propagation_model.apply(self)

