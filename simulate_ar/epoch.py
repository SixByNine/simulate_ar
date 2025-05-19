import numpy as np
from astropy.time import Time

# A class that represents a mjd with integer part, seconds since midnight and fractional seconds
class Epoch:
    def __init__(self, imjd, fmjd):
        self.imjd = int(imjd)
        self.fmjd = fmjd
        self.clean()

    def clean(self):
        shift=np.floor(self.fmjd)
        self.imjd += int(shift)
        self.fmjd -= shift

    def get_fits_epoch(self):
        sec = self.fmjd * 86400.0
        return self.imjd, int(sec), sec - int(sec)
    
    def __str__(self):
        self.clean()
        fracpart=f"{self.fmjd:.14f}".lstrip("0")
        return f"{self.imjd:d}{fracpart}"
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other):
        if isinstance(other, Epoch):
            i= self.imjd + other.imjd
            f= self.fmjd + other.fmjd
            return Epoch(i,f)
        elif isinstance(other, (int, float)):
            intpart = int(other)
            fracpart = other - intpart
            f = self.fmjd + fracpart
            i = self.imjd + intpart
            return Epoch(i, f)
        else:
            raise TypeError(f"Unsupported type {type(other)} for addition with Epoch")
    def __sub__(self, other):
        if isinstance(other, Epoch):
            i= self.imjd - other.imjd
            f= self.fmjd - other.fmjd
            return Epoch(i,f)
        elif isinstance(other, (int, float)):
            intpart = int(other)
            fracpart = other - intpart
            f = self.fmjd - fracpart
            i = self.imjd - intpart
            return Epoch(i, f)
        else:
            raise TypeError(f"Unsupported type {type(other)} for subtraction with Epoch")
    def __eq__(self, other):
        if isinstance(other, Epoch):
            return self.imjd == other.imjd and self.fmjd == other.fmjd
        else:
            raise TypeError(f"Unsupported type {type(other)} for equality check with Epoch")
        


    
    def to_astropy(self):
        return Time(self.imjd+self.fmjd, format='mjd', scale='utc')
    

    def from_string(string):
        e=string.split(".")
        i = int(e[0])
        if len(e) > 0:
            f = float("0."+e[1])
        else:
            f=0
        return Epoch(i, f)
    
    def to_seconds(self):
        return self.imjd * 86400.0 + self.fmjd * 86400.0