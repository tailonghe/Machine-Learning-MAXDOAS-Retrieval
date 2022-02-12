import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import glob
import metpy.calc
from metpy.units import units
import numpy as np
from datetime import datetime, timedelta

CMD = os.getcwd()
DATAPATH = CMD + "/data/"  # raw retrieval + meteorological data
INPATH = CMD + "/input/"  # input to ML model
OUTPATH = CMD + "/output/"  # output from ML model
AEROPATH = CMD + "/data/aerosol/"
NO2PATH = CMD + "/data/no2/"
METPATH = CMD + "/data/met/"

chisq0 = 40  # chisq threshold
dof0 = 1.5  # dof threshold


def dt_from_filename(fname):
    dtstr = fname.split('.')[0][-13:]
    dtstr = datetime.strptime(dtstr, '%Y%m%d_%H%M')
    return dtstr


def find_tidx(tarray, targ):
    curr = 0
    while tarray[curr] + timedelta(hours=1) < targ:
        curr += 1
    return curr


def to_meas(dt):
    return AEROPATH + ('p103_short_uv/%s/general/meas_%s.dat'
                       % (dt.strftime('%Y%m%d'), dt.strftime('%Y%m%d')))


def aero_file(iid, type='profiles'):
    date = iid.split('_')[0]
    if 'prof' in type:
        filename = 'prof361nm_' + iid + '.dat'
    elif 'av' in type:
        filename = 'avk_ext_' + iid + '.dat'

    return AEROPATH + 'p103_short_uv/%s/%s/' % (date, type) + filename


def no2_file(iid, type='profiles'):
    date = iid.split('_')[0]
    if 'prof' in type:
        filename = 'NO2_prof_' + iid + '.dat'
    elif 'av' in type:
        filename = 'avk_' + iid + '.dat'
    return NO2PATH + 'p103_short_uv/%s/%s/' % (date, type) + filename


def get_indicators(iid):
    date = iid.split('_')[0]
    filepath = AEROPATH + 'p103_short_uv/%s/' % date + 'retrieval_details/'
    filename = 'retr_' + iid + '.dat'

    df = pd.read_csv(os.path.join(filepath, filename), sep=':',
                     encoding="ISO-8859-1",
                     index_col=False, names=['Info', 'Value'])
    assert df.loc[1, 'Info'] == 'Chisquare'
    chisq = df.loc[1, 'Value']

    assert df.loc[3, 'Info'] == "Degrees of freedom for signal"
    dof = df.loc[3, 'Value']

    return chisq, dof


flist = glob.glob(NO2PATH + 'p103_short_uv/*/profiles/*dat')
final_alt = np.arange(0, 4.2, 0.2) * 1e3

ref = datetime(2018, 2, 2)
o4_nan = 0
no2_nan = 0
aero_nan = 0
missing_aero = 0
missing_meas = 0
missing_retr = 0
notfound = 0
exclude = 0

for f in flist:

    iid = f.split('/')[-1][-17:-4]

    try:
        chisq, dof = get_indicators(iid)

    except Exception as e:
        print(e)
        missing_retr += 1
        continue

    dtnow = dt_from_filename(f)

    df = pd.read_csv(to_meas(dtnow), delim_whitespace=True)
    dts = df['date'] + ' ' + df['time']
    dts = np.array([datetime.strptime(x, '%d/%m/%Y %H:%M:%S') for x in dts])

    try:
        dts = dts.reshape(-1, 9)
    except Exception as e:
        print(e)
        missing_meas += 1
        continue

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

        sza = sza[meastidx] / 1e2
        raa = raa[meastidx] / 1e2
        o4 = o4[meastidx] * 1e2

        iid = f.split('/')[-1][-17:-11]

        fh = nc.Dataset(METPATH + 'ERA5_preslevels_%s.nc' % (iid))
        eralons = fh.variables['longitude'][:]
        eralats = fh.variables['latitude'][:]
        jidx = np.argmin(abs(eralats - 43.7267))
        iidx = np.argmin(abs(eralons - -79.4821))
        times = np.array(fh.variables['time'][:])
        times = times * timedelta(hours=1) + datetime(1900, 1, 1, 0, 0)
        tidx = find_tidx(times, dtnow)

        geop = fh.variables['z'][tidx, ::-1, jidx, iidx]
        tprof = fh.variables['t'][tidx, ::-1, jidx, iidx] / 100
        qprof = fh.variables['q'][tidx, ::-1, jidx, iidx] * 1e3  # g kg**-1
        uprof = fh.variables['u'][tidx, ::-1, jidx, iidx] / 10
        vprof = fh.variables['v'][tidx, ::-1, jidx, iidx] / 10
        geop = units.Quantity(geop, "m**2 s**-2")
        height = metpy.calc.geopotential_to_height(geop)
        height = np.array(height) - 76  # Downsview's altitude above mean sea level is 76 m.
        fh.close()

        tprof = np.interp(final_alt, height, tprof)
        qprof = np.interp(final_alt, height, qprof)
        uprof = np.interp(final_alt, height, uprof)
        vprof = np.interp(final_alt, height, vprof)

        fh = nc.Dataset(METPATH + 'ERA5_singlelevels_2018.nc')
        eralons = fh.variables['longitude'][:]
        eralats = fh.variables['latitude'][:]
        jidx = np.argmin(abs(eralats - 43.7267))  # Downsview's coordinate 43.7267° N, 79.4821° W
        iidx = np.argmin(abs(eralons - -79.4821))
        times = np.array(fh.variables['time'][:])
        times = times * timedelta(hours=1) + datetime(1900, 1, 1, 0, 0)
        tidx = find_tidx(times, dtnow)
        d2m = fh.variables['d2m'][tidx, jidx, iidx] / 100
        t2m = fh.variables['t2m'][tidx, jidx, iidx] / 100
        skt = fh.variables['skt'][tidx, jidx, iidx] / 100
        sp = fh.variables['sp'][tidx, jidx, iidx] / 1e5  # bar
        fh.close()

        iid = f.split('/')[-1][-17:-4]
        try:
            aerodf = pd.read_csv(aero_file(iid, 'profiles'), delim_whitespace=True)
            aeroavk_df = pd.read_csv(aero_file(iid, 'av_kernels'), delim_whitespace=True)
            no2avk_df = pd.read_csv(no2_file(iid, 'av_kernels'), delim_whitespace=True)
        except Exception as e:
            print(e)
            missing_aero += 1
            continue

        no2df = pd.read_csv(f, delim_whitespace=True)
        no2prof = no2df['retr_nd']
        no2err = no2df['err_r_nd']
        no2avk = no2avk_df['area']

        aeroprof = aerodf['retrieved']
        aeroerr = aerodf['err_retrieved']
        aeroavk = aeroavk_df['area']

        d2m = np.ones(9) * d2m
        t2m = np.ones(9) * t2m
        skt = np.ones(9) * skt
        sp = np.ones(9) * sp
        # print(np.mean(d2m), np.mean(t2m), np.mean(skt), np.mean(sp),
        # np.mean(sza), np.mean(raa), np.mean(o4), np.mean(tprof),
        # np.mean(qprof), np.mean(uprof), np.mean(vprof))

        print(dtnow, times[tidx], meandts[meastidx])

        if np.any(np.isnan(o4)):
            o4_nan += 1
            print("O4 measurement data contains Nan!")
            continue

        if np.any(np.isnan(aeroprof)):
            aero_nan += 1
            print("Aero profile data contains Nan!")
            continue

        if np.any(np.isnan(no2prof)):
            no2_nan += 1
            print("NO2 profile data contains Nan!")
            continue

        xpath = INPATH + 'X/'
        ypath = INPATH + 'Y/'
        np.savez(os.path.join(xpath, 'X_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                 d2m=d2m, t2m=t2m, skt=skt, sp=sp, sza=sza, raa=raa, o4=o4, tprof=tprof,
                 qprof=qprof, uprof=uprof, vprof=vprof, aeroavk=aeroavk, no2avk=no2avk)
        np.savez(os.path.join(ypath, 'Y_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                 aero=aeroprof, no2=no2prof / 1e11, chisq=chisq, dof=dof)

        if dof > dof0 and chisq < chisq0:

            xpath = INPATH + 'X_flag1/'
            ypath = INPATH + 'Y_flag1/'
            np.savez(os.path.join(xpath, 'X_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     d2m=d2m, t2m=t2m, skt=skt, sp=sp, sza=sza, raa=raa, o4=o4, tprof=tprof,
                     qprof=qprof, uprof=uprof, vprof=vprof, aeroavk=aeroavk, no2avk=no2avk)
            np.savez(os.path.join(ypath, 'Y_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     aero=aeroprof, no2=no2prof / 1e11, chisq=chisq, dof=dof)

        elif dof < dof0 and chisq > chisq0:

            xpath = INPATH + 'X_flag2/'
            ypath = INPATH + 'Y_flag2/'
            np.savez(os.path.join(xpath, 'X_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     d2m=d2m, t2m=t2m, skt=skt, sp=sp, sza=sza, raa=raa, o4=o4, tprof=tprof,
                     qprof=qprof, uprof=uprof, vprof=vprof, aeroavk=aeroavk, no2avk=no2avk)
            np.savez(os.path.join(ypath, 'Y_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     aero=aeroprof, no2=no2prof / 1e11, chisq=chisq, dof=dof)

        elif dof > dof0 and chisq > chisq0:

            xpath = INPATH + 'X_flag3/'
            ypath = INPATH + 'Y_flag3/'
            np.savez(os.path.join(xpath, 'X_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     d2m=d2m, t2m=t2m, skt=skt, sp=sp, sza=sza, raa=raa, o4=o4, tprof=tprof,
                     qprof=qprof, uprof=uprof, vprof=vprof, aeroavk=aeroavk, no2avk=no2avk)
            np.savez(os.path.join(ypath, 'Y_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     aero=aeroprof, no2=no2prof / 1e11, chisq=chisq, dof=dof)

        elif dof < dof0 and chisq < chisq0:

            xpath = INPATH + 'X_flag4/'
            ypath = INPATH + 'Y_flag4/'
            np.savez(os.path.join(xpath, 'X_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     d2m=d2m, t2m=t2m, skt=skt, sp=sp, sza=sza, raa=raa, o4=o4, tprof=tprof,
                     qprof=qprof, uprof=uprof, vprof=vprof, aeroavk=aeroavk, no2avk=no2avk)
            np.savez(os.path.join(ypath, 'Y_%s.npz' % (dtnow.strftime('%Y%m%d_%H%M%S'))),
                     aero=aeroprof, no2=no2prof / 1e11, chisq=chisq, dof=dof)

        else:
            pass

    else:
        print('NOT FOUND!')
        notfound += 1

print("Number of invalid O4 data: %s" % o4_nan)
print("Number of invalid NO2 data: %s" % no2_nan)
print("Number of invalid Aerosol data: %s" % aero_nan)
print("Number of missing Aerosol profiles: %s" % missing_aero)
print("Number of missing Aerosol retrieval info: %s" % missing_retr)
print("Number of missing Elevation measurement: %s" % missing_meas)
print("Number of Not Found: %s" % notfound)
print("Number of excluded data: %s" % exclude)