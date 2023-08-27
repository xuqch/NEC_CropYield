# -*- coding: utf-8 -*-
# Copyright (c) 2004-2015 Alterra, Wageningen-UR
# Allard de Wit (allard.dewit@wur.nl), April 2015
"""A weather data provider reading its data from Excel files.
"""
import os
import datetime as dt
import xlrd
import xarray as xr
import numpy as np
from pcse.base import WeatherDataContainer, WeatherDataProvider
from pcse.util import reference_ET, angstrom, check_angstromAB
from pcse.exceptions import PCSEError
from pcse.settings import settings
import pandas as pd
from datetime import datetime
# Conversion functions, note that we defined a second parameter 's' that is only there
# to catch the sheet which is needed in the function xlsdate_to_date(value, sheet).
NoConversion = lambda x, s: x
kJ_to_J = lambda x, s: x*1000.
kPa_to_hPa = lambda x, s: x*10.
mm_to_cm = lambda x, s: x/10.
#def xlsdate_to_date(value, sheet):
#    """Convert an excel date into a python date

#    :param value: A value from an excel cell
#    :param sheet: A reference to the excel sheet for getting the datemode
#    :return: a python date
#    """
#    year, month, day, hr, min, sec = xlrd.xldate_as_tuple(value, sheet.book.datemode)
#    return dt.date(year, month, day)

class NoDataError(PCSEError):
    pass


class OutOfRange(PCSEError):
    pass





class XarrayWeatherDataProvider(WeatherDataProvider):
    """Reading weather data from an xarray dataset.

    :param xls_fname: name of the xarray file to be read
    :param mising_snow_depth: the value that should use for missing SNOW_DEPTH values
    :param force_reload: bypass the cache file and reload data from the XLS file

    For reading weather data from file, initially only the CABOWeatherDataProvider
    was available that reads its data from a text file in the CABO Weather format.
    Nevertheless, building CABO weather files is tedious as for each year a new
    file must constructed. Moreover it is rather error prone and formatting
    mistakes are easily leading to errors.

    To simplify providing weather data to PCSE models, a new data provider
    was written that reads its data from simple excel files

    The ExcelWeatherDataProvider assumes that records are complete and does
    not make an effort to interpolate data as this can be easily
    accomplished in Excel itself. Only SNOW_DEPTH is allowed to be missing
    as this parameter is usually not provided outside the winter season.
    """

    obs_conversions = {
        "TMAX": NoConversion,
        "TMIN": NoConversion,
        "IRRAD": NoConversion,
        "DAY": NoConversion,
        "VAP": NoConversion,
        "WIND": NoConversion,
        "RAIN": NoConversion,
        "SNOWDEPTH": NoConversion
    }

    # row numbers where values start. Note that the row numbers are
    # zero-based, so add 1 to find the corresponding row in excel.
    #site_row = 8
    #label_row = 10
    #data_start_row = 12

    def __init__(self, ELEVE=None, xrdata=None):
        WeatherDataProvider.__init__(self)
        xrdata['DAY']= xrdata.time.dt.strftime("%Y%m%d")
        self.fp_xls_fname='./temp'
        self._read_header(xrdata)
        self._read_site_characteristics(ELEVE,xrdata)
        self._read_observations(xrdata)
        self._write_cache_file(self.fp_xls_fname)

    def _read_header(self, xrdata):

        country = "unknown"
        station = "unknown"
        desc    = "unknown"
        src     = "unknown"
        contact = "unknown"
        self.nodata_value = float(-999)
        self.description = [u"Weather data for:",
                            u"Country: %s" % country,
                            u"Station: %s" % station,
                            u"Description: %s" % desc,
                            u"Source: %s" % src,
                            u"Contact: %s" % contact]

    def _read_site_characteristics(self, ELEV,xrdata):
        self.latitude    =  float(xrdata["lat"].values)
        self.longitude   =  float(xrdata["lon"].values)
        self.elevation   =  float(ELEV)
        self._first_date =  min(xrdata["DAY"].values)
        self._last_date  =  max(xrdata["DAY"].values)
        #print(self._first_date,self._last_date)
        #self.store       =  dt.datetime.strptime(str(xrdata['DAY']), "%Y%m%d").date() #xrdata["DAY"]
        angstA           =  0.18 
        angstB           =  0.55 
        self.angstA, self.angstB = check_angstromAB(angstA, angstB)
        self.has_sunshine = False

    def _read_observations(self, xrdata):

        # First get the column labels
        xrdata=xrdata.to_dataframe()
        for i in range(len(xrdata.tasmax)):
            try:
                d = {}
                d['LON']          =    float(xrdata.lon[i])   #Celsius
                d['LAT']          =    float(xrdata.lat[i])   #Celsius
                d['TMAX']          =   float(xrdata.tasmax[i]   -   273.15)   #Celsius
                d['TMIN']          =   float(xrdata.tasmin[i]   -   273.15)   #Celsius
                d['IRRAD']         =   float(xrdata.rsds[i]  *   86400.0 )   #Wm-2 to 𝐽𝑚−2𝑑𝑎𝑦−1
                d['VAP']           =   float(xrdata.vap[i]   *   10.0)     # kPa_to_hPa
                d['WIND']          =   float(xrdata.sfcWind[i] )
                d['RAIN']          =   float(xrdata['pr'][i] *   86400.0/10.) #kg/m2/s to cm/day
                d['SNOWDEPTH']     =   None # -999.0
                #d['TEMP']          =   float((xrdata.tasmax[i]+xrdata.tasmin[i]) * 0.5 - 273.15)
                d['DAY']           =   dt.datetime.strptime(xrdata['DAY'][i], "%Y%m%d").date()
                d['time']          =   d['DAY']#dt.datetime.strptime(xrdata['DAY'][i], "%Y%m%d").date()
                #d['time']

                e0, es0, et0 = reference_ET( ELEV=self.elevation, ANGSTA=self.angstA,
                                            ANGSTB=self.angstB, **d)
                # convert to cm/day
                d["E0"] = e0/10.; d["ES0"] = es0/10.; d["ET0"] = et0/10.

                wdc = WeatherDataContainer( ELEV=self.elevation, **d)
                self._store_WeatherDataContainer(wdc, d["DAY"])
            
            except ValueError as e: # strange value in cell
                msg = "Failed reading row: %i. Skipping..." % (i + 1)
                self.logger.warn(msg)

            except NoDataError as e: # Missing value encountered
                msg = "No data value (%f) encountered at row %i. Skipping..." % (self.nodata_value, (i + 1))
                self.logger.warn(msg)

            except OutOfRange as e:
                self.logger.warn(e)

    def _load_cache_file(self, xls_fname):

         cache_filename = self._find_cache_file(xls_fname)
         if cache_filename is None:
             return False
         else:
             try:
                 self._load(cache_filename)
                 return True
             except:
                 return False

    def _find_cache_file(self, xls_fname):
        """Try to find a cache file for file name

        Returns None if the cache file does not exist, else it returns the full path
        to the cache file.
        """
        cache_filename = self._get_cache_filename(xls_fname)
        if os.path.exists(cache_filename):
            cache_date = os.stat(cache_filename).st_mtime
            xls_date = os.stat(xls_fname).st_mtime
            if cache_date > xls_date:  # cache is more recent then XLS file
                return cache_filename

        return None

    def _get_cache_filename(self, xls_fname):
        """Constructs the filename used for cache files given xls_fname
        """
        basename = os.path.basename(xls_fname)
        filename, ext = os.path.splitext(basename)

        tmp = "%s_%s.cache" % (self.__class__.__name__, filename)
        cache_filename = os.path.join(settings.METEO_CACHE_DIR, tmp)
        return cache_filename

    def _write_cache_file(self, xls_fname):

        cache_filename = self._get_cache_filename(xls_fname)
        try:
            self._dump(cache_filename)
        except (IOError, EnvironmentError) as e:
            msg = "Failed to write cache to file '%s' due to: %s" % (cache_filename, e)
            self.logger.warning(msg)
