from .epoch import Epoch
import astropy.coordinates as coord
import astropy.units as u


class Pulsar:
    def __init__(self, parfile:str):
        self.parfile=parfile
        self.read_parfile()

    def read_parfile(self):
        self.parfile_contents = []
        self.pulsar_params={}
        with open(self.parfile) as f:
            for line in f:
                e=line.split()
                if len(e) > 1:
                    self.pulsar_params[e[0]] = e[1]
                self.parfile_contents.append(line)

    def name(self):
        if "PSRJ" in self.pulsar_params:
            return self.pulsar_params["PSRJ"]
        elif "PSRB" in self.pulsar_params:
            return self.pulsar_params["PSRB"]
        elif "PSR" in self.pulsar_params:
            return self.pulsar_params["PSR"]
        else:
            return "UNKNOWN"
        
    def dm(self):
        if "DM" in self.pulsar_params:
            return float(self.pulsar_params["DM"])
        else:
            return 0.0

    def get_coordinate(self):
        if "POSEPOCH" in self.pulsar_params:
            posepoch = Epoch.from_string(self.pulsar_params['POSEPOCH'])
        elif "PEPOCH" in self.pulsar_params:
            posepoch = Epoch.from_string(self.pulsar_params['PEPOCH'])
        else:
            posepoch = Epoch(55000,0.0) # Default to 55000.0 MJD

        if "RAJ" in self.pulsar_params:
            ra = self.pulsar_params['RAJ']
            dec = self.pulsar_params['DECJ']
            if 'PMRA' in self.pulsar_params:
                pmra = float(self.pulsar_params['PMRA'])*u.mas/u.yr
            else:
                pmra = 0
            if 'PMDEC' in self.pulsar_params:
                pmdec = float(self.pulsar_params['PMDEC'])*u.mas/u.yr
            else:
                pmdec = 0
            return coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg), pm_ra_cosdec=pmra, pm_dec=pmdec, frame='icrs',obstime=posepoch.to_astropy(),distance=1*u.kpc) # Maybe get distance sometime
        
        # Probbaly this isn't right! Pint defines a custom PulsarEcliptic class
        if "ELAT" in self.pulsar_params:
            elon = self.pulsar_params['ELONG'] # UNITS?
            elat = self.pulsar_params['ELAT']
            if 'PMELONG' in self.pulsar_params:
                pmelon = float(self.pulsar_params['PMELONG'])*u.mas/u.yr
            else:
                pmelon = 0
            if 'PMELAT' in self.pulsar_params:
                pmelat =float(self.pulsar_params['PMELAT'])*u.mas/u.yr
            else:
                pmelon = 0
            return coord.HeliocentricMeanEcliptic(
                lon=elon, lat=elat, pm_lon_coslat=pmelon, pm_lat=pmelat,obstime=posepoch.to_astropy(),distance=1*u.kpc)


