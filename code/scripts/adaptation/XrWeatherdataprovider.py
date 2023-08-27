import xarray as xr
import numpy as np
import datetime
import pandas as pd
from pcse.util import reference_ET
import datetime as dt
from pcse.base import WeatherDataContainer, WeatherDataProvider


class XrWeatherObsGridDataProvider(WeatherDataProvider):

    def __init__(self, ELEVE=None, xrdata=None):
        WeatherDataProvider.__init__(self)

        if ELEVE is None or xrdata is None:
            msg = "Provide location of HDF weather data store and grid ID."
            raise RuntimeError(msg)
        # Read grid information and meteo time-series
        self.df_meteo                  =   xrdata.time.copy()
        self.df_meteo["LAT"]           =   xrdata.lat
        self.df_meteo["LON"]           =   xrdata.lon
        self.df_meteo["ELEV"]          =   np.float32(ELEVE)
        self.df_meteo['TMAX']          =   xrdata.tasmax   -   273.15   #Celsius
        self.df_meteo['TMIN']          =   xrdata.tasmin   -   273.15   #Celsius
        self.df_meteo['IRRAD']         =   xrdata.rsds  *   86400.0    #Wm-2 to 𝐽𝑚−2𝑑𝑎𝑦−1
        self.df_meteo['DAY']           =   xrdata.time.dt.strftime("%Y%m%d")
        self.df_meteo['VAP']           =   xrdata.vap   *   10.0     # kPa_to_hPa
        self.df_meteo['WIND']          =   xrdata.sfcWind    
        self.df_meteo['RAIN']          =   xrdata['pr'] *   86400.0/10. #kg/m2/s to cm/day
        self.df_meteo['SNOWDEPTH']     =   xr.full_like(xrdata.pr, -999)
        self.df_meteo['TEMP']          =   (xrdata.tasmax+xrdata.tasmin) * 0.5 - 273.15
        self.df_meteo['TMINRA']        =   self.df_meteo.TMIN.rolling(time=7, min_periods=1).mean().astype(np.float32)
        self.df_meteo['e0']            =   xr.full_like(xrdata.pr, -999)
        self.df_meteo['es0']           =   xr.full_like(xrdata.pr, -999)
        self.df_meteo['et0']           =   xr.full_like(xrdata.pr, -999)
        self.df_meteo["DTEMP"]         =   (0.5 * (self.df_meteo.TEMP + self.df_meteo.TMAX)).astype(np.float32)
        self.df_meteo                  =   self.df_meteo.to_dataframe()

        for i in range(len(self.df_meteo['DAY'])):
            #print(da.time.values[i])
            DAK=dt.datetime.strptime(self.df_meteo['DAY'][i], '%Y%m%d').date()
            self.df_meteo['e0'][i], self.df_meteo['es0'][i], self.df_meteo['et0'][i]= reference_ET(DAY=DAK, LAT=self.df_meteo['LAT'][i], ELEV=200.0,
                            TMIN=self.df_meteo['TMIN'][i],TMAX=self.df_meteo['TMAX'][i],
                            IRRAD=self.df_meteo['IRRAD'][i], VAP=self.df_meteo['VAP'][i], WIND=self.df_meteo['WIND'][i],
                            ANGSTA=0.18, ANGSTB=0.55,
                            ETMODEL='PM')
        # convert to cm/day
        self.df_meteo['e0']  =  self.df_meteo['e0']  / 10.0
        self.df_meteo['es0'] =  self.df_meteo['es0'] / 10.0
        self.df_meteo['et0'] =  self.df_meteo['et0'] / 10.0

        # Post-processing on meteo records
        # Metadata for weatherdataprovider
        self.description = "Weather data for grid "  
        self.latitude    = self.df_meteo["LAT"][0]
        self.longitude   = self.df_meteo["LON"][0]
        self.elevation   = self.df_meteo["ELEV"][0]
        self._first_date = min(self.df_meteo["DAY"])
        self._last_date  = max(self.df_meteo["DAY"])
        self.store       = self.df_meteo["DAY"]
        print(self.latitude)

        wdc = WeatherDataContainer(LAT=self.latitude, LON=self.longitude, ELEV=self.elevation, **self.df_meteo)
        self._store_WeatherDataContainer(wdc, self.df_meteo["DAY"])


        #print (self.df_meteo.values())
    def __call__(self, day):

        kday = self.check_keydate(day)
        df = self.df_meteo[self.df_meteo.index == kday.isoformat()]
        if len(df) == 0:
            msg = "Cannot find weather data for %s" % kday
            raise exc.WeatherDataProviderError(msg)
        return list(df.itertuples())[0]

    def export(self):
        #self.df_meteo=self.df_meteo.to_dataframe()
        return self.df_meteo.to_dict(orient='records')

