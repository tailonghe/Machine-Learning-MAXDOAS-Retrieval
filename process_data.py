import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import glob
import metpy.calc
from metpy.units import units
import numpy as np
from datetime import datetime, timedelta

def dt_from_filename(fname):
    dtstr = fname.split('.')[0][-13:]
    dtstr = datetime.strptime(dtstr, '%Y%m%d_%H%M')
    return dtstr

def find_tidx(tarray, targ):
    curr = 0
    while tarray[curr] + timedelta(hours=1)< targ:
        curr += 1
    return curr

def to_meas(dt):
    return 'For_Tailong/p103_short_uv_aer/%s/general/meas_%s.dat'%(dt.strftime('%Y%m%d'), dt.strftime('%Y%m%d'))



flist = glob.glob('For_Tailong/p103_short_uv_tg/20180203/profiles/*dat')
final_alt = np.arange(0, 4.2, 0.2)*1e3

ref = datetime(2018, 2, 2)

for f in flist:
    dtnow = dt_from_filename(f) 

    df = pd.read_csv(to_meas(dtnow), delim_whitespace=True)
    dts = df['date'] + ' ' + df['time']
    dts = np.array([datetime.strptime(x, '%d/%m/%Y %H:%M:%S') for x in dts])
    dts = dts.reshape(-1, 9)
    meandts = np.max(dts, axis=-1)
    meastidx = -1
    found = False
    for i in range(meandts.shape[0]):
        if meandts[i] <= dtnow:
            meastidx = i
            found = True
    if found: 
        sza = df['SZA'].to_numpy().reshape(-1, 9)
        raa = df['rel_azim'].to_numpy().reshape(-1, 9)
        o4 = df['O4meas'].to_numpy().reshape(-1, 9)

        sza = sza[meastidx]/1e2
        raa = raa[meastidx]/1e2
        o4 = o4[meastidx]*1e2

        fh = nc.Dataset('ERA5_preslevels.nc')
        eralons = fh.variables['longitude'][:]
        eralats = fh.variables['latitude'][:]
        jidx = np.argmin(abs(eralats - 43.7267 ))
        iidx = np.argmin(abs(eralons - -79.4821 ))
        times = np.array(fh.variables['time'][:])
        times = times*timedelta(hours=1) + datetime(1900, 1, 1, 0, 0)
        tidx = find_tidx(times, dtnow)

        geop = fh.variables['z'][tidx, ::-1, jidx, iidx]
        tprof = fh.variables['t'][tidx, ::-1, jidx, iidx]/100
        qprof = fh.variables['q'][tidx, ::-1, jidx, iidx]*1e3   # g kg**-1
        uprof = fh.variables['u'][tidx, ::-1, jidx, iidx]/10
        vprof = fh.variables['v'][tidx, ::-1, jidx, iidx]/10
        geop = units.Quantity(geop, "m**2 s**-2")
        height = metpy.calc.geopotential_to_height(geop)
        height = np.array(height) - 76   # downsview's altitude above mean sea level is 76 m.
        fh.close()

        tprof = np.interp(final_alt, height, tprof)
        qprof = np.interp(final_alt, height, qprof)
        uprof = np.interp(final_alt, height, uprof)
        vprof = np.interp(final_alt, height, vprof)


        fh = nc.Dataset('ERA5_singlelevels.nc')
        eralons = fh.variables['longitude'][:]
        eralats = fh.variables['latitude'][:]
        jidx = np.argmin(abs(eralats - 43.7267 ))
        iidx = np.argmin(abs(eralons - -79.4821 ))
        times = np.array(fh.variables['time'][:])
        times = times*timedelta(hours=1) + datetime(1900, 1, 1, 0, 0)
        tidx = find_tidx(times, dtnow)
        d2m = fh.variables['d2m'][tidx, jidx, iidx]/100
        t2m = fh.variables['t2m'][tidx, jidx, iidx]/100
        skt = fh.variables['skt'][tidx, jidx, iidx]/100
        sp = fh.variables['sp'][tidx, jidx, iidx]/1e5  # bar
        fh.close()

        
        iid = f.split('\\')[-1][-17:-4]
        try:
            aerodf = pd.read_csv(aero_file(iid), delim_whitespace=True)
        except Exception as e:
            print(e)
            pass

        no2df = pd.read_csv(f, delim_whitespace=True)
        no2prof = no2df['retr_nd']
        no2err = no2df['err_r_nd']
        aeroprof = aerodf['retrieved']
        aeroerr = aerodf['err_retrieved']
        
        d2m = np.ones(9)*d2m
        t2m = np.ones(9)*d2m
        skt = np.ones(9)*d2m
        sp = np.ones(9)*d2m
        #print(np.mean(d2m), np.mean(t2m), np.mean(skt), np.mean(sp), np.mean(sza), np.mean(raa), np.mean(o4), np.mean(tprof), np.mean(qprof), np.mean(uprof), np.mean(vprof))
        print(dtnow, times[tidx], meandts[meastidx])
        np.savez('X/X_%s.npz'%(dtnow.strftime('%Y%m%d_%H%M%S')), d2m=d2m, t2m=t2m, skt=skt, 
                 sp=sp, sza=sza, raa=raa, o4=o4, tprof=tprof, qprof=qprof, uprof=uprof, vprof=vprof)
        np.savez('Y/Y_%s.npz'%(dtnow.strftime('%Y%m%d_%H%M%S')), aero=aeroprof, no2=no2prof/1e11)
    else:
        print('NOT FOUND!')
