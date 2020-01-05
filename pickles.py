"""
This is the file for the sonic data.
"""

__author__ = "Astrid Myren, UiB"
__email__ = "astrid.myren@student.uib.no"

import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from windrose import WindAxes
import numpy as np


def rot_aws(wd):
    #Rotate wind direction with 180 degrees.
    wd = wd + 230 # 230
    # Make sure no wind direction value exeeds 360 or falls below 0
    if wd >= 360:
        wd = wd - 360
    elif wd < 0:
        wd = wd + 360
    return wd


def rot_sonic(wd):
    #Rotate wind direction with 110 degrees.
    wd = wd + 160 #160
    # Make sure no wind direction value exeeds 360 or falls below 0
    if wd >= 360:
        wd = wd - 360
    elif wd < 0:
        wd = wd + 360
    return wd


def get_u(u_h, wd):
    # transform wd unit (deg) to radians
    wd_rad = wd / 180 * np.pi

    return - u_h * np.sin(wd_rad)


def get_v(u_h, wd):

    wd_rad = wd / 180 * np.pi

    return - u_h * np.cos(wd_rad)


def mixing_ratio(temp, rel_hum, press):
    # Temperature temp in deg C
    # Pressure in hPa
    # RH in %
    # Guide to Meteorological Instruments and Methods of Observation (CIMO Guide)
    # (WMO, 2008)
    eWs =  6.112*np.exp((17.62*temp)/(243.12+temp)) # Saturation vapor pressure in hPa
    eW = rel_hum*eWs/100 # water vapor pressure
    W = 621.97*eW/((press-eW)*1000) # mixing ratio in g water/g dry air
    return W


def power_law(z, u_ref, z_ref, alpha):
    """
    power law approach to estimate the wind profile from a reference wind speed measurement

    Parameters
    ----------
    z : float
       chosen height levels the wind speed will be extrapolated to
    u_ref : float
        reference value of wind speed measured on reference level z_ref
    z_ref : float
        reference level
    alpha :float
       (alpha)-exponent value of power law

    Results
    -------
    u :float
        array of wind speed values corresponding to chosen height levels (z)
    """
    u = u_ref * (z / z_ref) ** alpha

    return u


def get_power_output(cp, d, u_0, rho=1.3, cut_in=3, rated=6, cut_out=25):
    """
    Estimate of power output from wind

    Parameters
    ----------
    cp : float
        power coefficient
    d : float
       rotor diameter
    u_0 : float
        wind speed (undisturbed flow)
    rho : float
        air density
    cut_in : float
        cut-in wind speed
    rated : float
        rated wind speed
    cut_out : float
        cut-out wind speed

    Results
    -------
    P : float
        power output from wind
    """
    # Estimate the Rotor Area from rotor diameter
    A = np.pi*(d/2)**2

    P = 0.5 * cp * A * rho * u_0 ** 3 / 1000

    if u_0 < cut_in or u_0 >= cut_out:
        P = 0
    elif rated < u_0 < cut_out:
        0.5 * cp * A * rho * rated **3/1000

    return P


sonic = r'C:\Users\astri\.PyCharm2019.2\wind_meas\sonic.pkl'
aws = r'C:\Users\astri\.PyCharm2019.2\wind_meas\AWS_processed.csv'

# Weather station
w_df = pd.read_csv(aws, skiprows=1, names=['time','press','temp','hum','wspd','wdir','prec','rad'])

w_df['datetime'] = pd.to_datetime(w_df['time'], unit='s', origin='unix')
w_df = w_df.set_index('datetime')
w_df = w_df.drop(columns=["time"])

w_df['wdir'] = w_df.apply(lambda row: rot_aws(row['wdir']), axis=1)
w_df['u'] = w_df.apply(lambda row: get_u(row['wspd'], row['wdir']), axis=1)
w_df['v'] = w_df.apply(lambda row: get_v(row['wspd'], row['wdir']), axis=1)


# Sonic data
s_df = pd.read_pickle(sonic)

s_df['Wind direction'] = s_df.apply(lambda row: rot_sonic(row['Wind direction']), axis=1)
s_df = s_df.drop(columns=['X Wind Speed (m/s)', 'Y Wind Speed (m/s)'])
s_df['u']= s_df.apply(lambda row: get_u(row['Horizontal wind speed'], row['Wind direction']), axis=1)
s_df['v']= s_df.apply(lambda row: get_v(row['Horizontal wind speed'], row['Wind direction']), axis=1)

# Get columns from AWS
s_df['press'] = w_df['press']
s_df['hum'] = w_df['hum']
s_df['prec'] = w_df['prec']


# Find potential temperature
po = 1000  # hPa
s_df['theta'] = (273.15 + s_df['Temperature (deg C)'])*(po/s_df['press'])**(0.286)  # potential temperature
s_df['mixing ratio'] = s_df.apply(lambda row: mixing_ratio(row['Temperature (deg C)'], row['hum'], row['press']), axis=1)
s_df['theta v'] = s_df['theta']*(1-0.61*s_df['mixing ratio'])


# Get rolling means and deviations from mean
# n = 10  # 10 min rolling mean
# s_df['u_mean'] = s_df['u'].rolling(n).mean()
# s_df['v_mean'] = s_df['v'].rolling(n).mean()
# s_df['w_mean'] = s_df['Z Wind Speed (m/s)'].rolling(n).mean()
#
# s_df['u prim'] = s_df['u'] - s_df['u_mean']
# s_df['v prim'] = s_df['v'] - s_df['v_mean']
# s_df['w prim'] = s_df['Z Wind Speed (m/s)'] - s_df['w_mean']
#
#
# s_df['uw prim'] = s_df['u prim']*s_df['w prim']
# s_df['uv prim'] = s_df['u prim']*s_df['v prim']
# s_df['vw prim'] = s_df['v prim']*s_df['w prim']

# Get covariances
uv_bar = s_df['u'].cov(s_df['v'])
uw_bar = s_df['u'].cov(s_df['Z Wind Speed (m/s)'])
vw_bar = s_df['v'].cov(s_df['Z Wind Speed (m/s)'])
wtheta_bar = s_df['theta v'].cov(s_df['Z Wind Speed (m/s)'])
theta_v_bar = s_df['theta v'].mean()
fric_u = (uw_bar**2+vw_bar**2)**1/4


# Monin-Obukov length
k =  0.4 # von Karman constant
g = 9.81 #m/s^2
L = (- theta_v_bar * fric_u**3)/(k * g * wtheta_bar)


# Power law for neutral conditions
alpha = 1/7
z = 120 #m
z_ref = 2 #m
s_df['wspd 120 m'] = s_df.apply(lambda row: power_law(z, row['Horizontal wind speed'], z_ref, alpha), axis=1)

d = 90 #m
Cp = 0.35
s_df['Power'] = s_df.apply(lambda row: get_power_output(Cp, d, row['wspd 120 m']), axis=1)

# s_df['Max power'] = s_df['Power']*.59


# # Compare sonic and aws u and v wind
# ax = plt.gca()
# s_df['u'].plot(kind='line', ax=ax)
# w_df['u'].plot(kind='line', ax=ax)
# plt.legend(loc='best')
# plt.show()
#
# ax = plt.gca()
# s_df['v'].plot(kind='line', ax=ax)
# w_df['v'].plot(kind='line', ax=ax)
# plt.legend(loc='best')
# plt.show()

plt.rcParams.update({'font.size': 22})

# Plots of sonic data
s_df['Wind direction'].plot(kind='line', linewidth=2)
plt.title('Sonic Anemometer: Wind direction')
plt.ylabel('Degrees from North')
plt.show()

s_df['Horizontal wind speed'].loc['2019-11-07 06:00:00':'2019-11-11 09:00:00'].plot(kind='line', linewidth=2)
s_df['wspd 120 m'].loc['2019-11-07 06:00:00':'2019-11-11 09:00:00'].plot(kind='line', linewidth=2)
plt.title('Sonic Anemometer: Horizontal wind speed (m/s)')
plt.ylabel('Wind speed (m/s)')
plt.show()

s_df['wspd 120 m'].loc['2019-11-07 06:00:00':'2019-11-11 09:00:00'].plot(kind='line', linewidth=2)
plt.title('Sonic Anemometer: Horizontal wind speed (m/s)')
plt.ylabel('Wind speed (m/s)')
plt.show()

# Plot of available power in wind
ax = plt.gca()
s_df['Power'].loc['2019-11-07 06:00:00':'2019-11-11 09:00:00'].plot(kind='line', ax=ax)
plt.title('Sonic Anemometer: Power Output (kW)')
plt.ylabel('Power (kW)')
plt.legend(loc='best')
plt.show()

ax3 = plt.gca()
s_df['Horizontal wind speed'].hist(ax=ax3, density=True)
#s_df['wspd 120 m'].hist(ax=ax3, density=True)
plt.title('Sonic Anemometer: Histogram of Wind Speeds')
plt.ylabel('Probability')
plt.xlabel('Wind speed (m/s)')
plt.show()

ax3 = plt.gca()
s_df['wspd 120 m'].hist(ax=ax3, density=True)
#s_df['wspd 120 m'].hist(ax=ax3, density=True)
plt.title('Sonic Anemometer: Histogram of Wind Speeds at 120 m')
plt.ylabel('Probability')
plt.xlabel('Wind speed (m/s)')
plt.show()

# Scatter plot of wind speed
plt.scatter(s_df['Horizontal wind speed'].loc['2019-11-05 00:00:00':'2019-11-10 23:59:00'], w_df['wspd'].loc['2019-11-05 00:00:00':'2019-11-10 23:59:00'])

#
# ax1 = plt.gca()
# s_df['X Wind Speed (m/s)'].plot(kind='line', ax=ax1)
# w_df['u'].plot(kind='line', ax=ax1)
# plt.legend(loc='best')
# plt.show()
#
# ax2 = plt.gca()
# s_df['X Wind Speed (m/s)'].plot(kind='line', ax=ax2)
# w_df['v'].plot(kind='line', ax=ax2)
# plt.legend(loc='best')
# plt.show()
#
# ax3 = plt.gca()
# w_df['wdir'].hist(ax=ax3)
# plt.show()
#
#
# ax4 = plt.gca()
# s_df['Wind direction'].plot(kind='line', ax=ax4)
# w_df['wdir'].plot(kind='line', ax=ax4)
# plt.legend(loc='best')
# plt.show()
#
ax = WindAxes.from_ax()
bins = np.arange(0, 6 + 1, 0.5)
bins = bins[1:]
ax, params = ax.pdf(s_df['Horizontal wind speed'], bins=bins)

ax = WindroseAxes.from_ax()
ax.bar(s_df['Wind direction'], s_df['Horizontal wind speed'], bins=np.arange(0, 3, 0.5), normed=True, opening=0.8, edgecolor='white')
plt.legend(loc=3, prop={'size': 20})

ax1 = WindroseAxes.from_ax()
ax1.bar(w_df['wdir'], w_df['wspd'], bins=np.arange(0, 3, 0.5), normed= True, opening=0.8, edgecolor='white')
plt.legend(loc=3, prop={'size': 20})

### Organize labels ###

