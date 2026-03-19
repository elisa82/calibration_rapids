def create_ini(lat,lon,depth,date,time,mw,stk,dip,rak,output_folder,SETTINGS_FILE,rapids_ini,vm,k,rt,rp,recordings_folder,IDs,lons,lats):
    if vm == 'GNDT_14':
        gf = 'gndt0014'
    IDs_text = ', '.join(IDs)
    lats_text = ', '.join(str(lat) for lat in lats)
    lons_text = ', '.join(str(lon) for lon in lons)

    with open(rapids_ini, "w", encoding="utf-8") as f:
        f.write(f"""output_folder = {output_folder}
settings_file = {SETTINGS_FILE}

fault_geolocation = from_hypo

#type_run: standard o test
type_run = standard

path_mseed = {os.path.dirname(recordings_folder)}

#velocity model
vel_model = {vm}

fault_type = point
slip_mode = Archuleta
lat_hypo = {lat}
lon_hypo = {lon}
depth_hypo = {depth}
strike = {stk} 
dip = {dip}
rake = {rak}
Mw = {mw}

# Computational params UCSB
gf = {gf}
freq_band_gf = LF
kappa = {k}
fmin_ucsb = 0.
Tp_Tr = 0.2
duration_ucsb = 60
#tenere fisso fmax a questo valore per evitare problemi con dt
fmax_ucsb = 20
STF = Archuleta
rise_time_relationship = {rt}
radiation_pattern = {rp}

# receivers
receivers_lat = [{lats_text}]
receivers_lon = [{lons_text}]
receivers_ID = [{IDs_text}]

#plot
fmax_filter = 19.5
time_max_plot = 60
interpolation_map = no
""")


def retrieve_focal_mechanisms_Mw(datetime_slo,lat_slo,lon_slo,depth_slo,ml_slo):
    from datetime import datetime, timedelta

    focal_mechanism_catalogue = r'/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/Focalmechanisms/catalogue1928-2023_Suganal2024.csv'
    df_mt = pd.read_csv(
        focal_mechanism_catalogue,
        sep=";",
        comment=None,
        header=0,
        na_values=['', '---']
    )

    df_mt.columns = df_mt.columns.str.strip().str.replace('-', '_')

    df_mt['datetime'] = pd.to_datetime(
        df_mt['Date'].astype(str).str.strip() + ' ' + df_mt['Time'].astype(str).str.strip(),
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )

    for col in ['Lat','Lon','Dep','Ml','MD','Ms','Mb','M','Mw']:
        df_mt[col] = pd.to_numeric(df_mt[col], errors='coerce')

    df_mt['Mw_new'] = df_mt['Mw']
    mask_missing_mw = df_mt['Mw'].isna() & df_mt['M'].notna()
    df_mt.loc[mask_missing_mw, 'Mw_new'] = df_mt.loc[mask_missing_mw, 'M']

    time_tol=30
    dist_tol=0.2
    mag_tol = 1.0

    mask_time = (df_mt["datetime"] - datetime_slo).abs() <= pd.Timedelta(seconds=time_tol)

    mask_space = (
        abs(df_mt["Lat"] - lat_slo) <= dist_tol
    ) & (
        abs(df_mt["Lon"] - lon_slo) <= dist_tol
    )

    mask_mag = abs(df_mt["Mw_new"] - ml_slo) <= mag_tol

    candidates = df_mt[mask_time & mask_space & mask_mag]
    candidates = df_mt[mask_time]

    if candidates.empty:
        return None

    pref = candidates[candidates["Pref"] == "P"]

    if not pref.empty:
        return pref.iloc[0]

    return candidates.iloc[0]


def retrieve_recordings(t0,name,parent):
    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime
    import matplotlib.pyplot as plt
    from obspy import Inventory
    import numpy as np
    from datetime import datetime, timedelta

    client = Client("http://158.110.30.217:8080")
    inv_all = None
    network_req = "*"
    station_req = "*" 
    location_req = "*"
    channel_req="HH?,BH?,HN?"

    st = client.get_waveforms(network_req, station_req, location_req, channel_req, t0, t0 + 60)
    st.detrend("demean") #remove the mean
    st.detrend("linear") #remove the trend
    inv = client.get_stations(network=network_req, station=station_req,channel=channel_req,level="response")
    st.remove_response(inventory=inv,output="VEL")
    fmin = 0.01
    fmax = 25.
    st.filter('bandpass',freqmin=fmin,freqmax=fmax,zerophase=True)
    filename = os.path.join(parent, f"{name}.mseed")
    st.write(filename, format="MSEED")

    networks = ",".join({tr.stats.network for tr in st})
    stations = ",".join({tr.stats.station for tr in st})

    inv = client.get_stations(network=networks, station=stations, level="station")

    station_coords = {}

    for net in inv:
        for sta in net:
            station_coords[(net.code, sta.code)] = (sta.latitude, sta.longitude)

    codes = np.array([sta for (net, sta) in station_coords.keys()])
    lats = np.array([lat for (lat, lon) in station_coords.values()])
    lons = np.array([lon for (lat, lon) in station_coords.values()])

    return codes, lats, lons


import pandas as pd
import os
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy import read_inventory
from datetime import datetime, timedelta
import glob


parent_folder = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration'
parent_recordings = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/Waveforms'
velocity_models = ['GNDT_14']
kappa = [0.]
rise_time = ['Somerville']
rad_pattern = ['fixed']

file_terremoti = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/test_EQ/test_earthquakes_arso_2.txt'
SETTINGS_FILE = f'/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/UrgentShake/settings.ini'
rows = []
with open(file_terremoti, encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            rows.append(parts[:10])

path_lista_events = os.path.join(parent_recordings,"lista_concordia.csv")
df = pd.read_csv(
    path_lista_events,
    sep=",",
    comment=None,  
    header=0       
)

df.columns = df.columns.str.strip().str.lstrip("#")
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

df_events = pd.DataFrame({
    "event_id": df['event_id'].astype(str),
    "datetime": df['datetime'],
    "year": df['datetime'].dt.year,
    "month": df['datetime'].dt.month,
    "day": df['datetime'].dt.day,
    "hour": df['datetime'].dt.hour,
    "minute": df['datetime'].dt.minute,
    "second": df['datetime'].dt.second,
    "lat": df['latitude'],
    "lon": df['longitude'],
    "depth": df['depth'],
    "Ml": df['magnitude']
})


lista_ini = []
lista_output_folders = []
for vm in velocity_models:
    for k in kappa:
        for rt in rise_time:
            for rp in rad_pattern:
                modello = f"{vm}_k0_{k}_Tr_{rt}_rp_{rp}"
                for i, row in df_events.iterrows():
                    recordings_folder = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'raw')
                    if os.path.exists(recordings_folder):
                        parent_folder_simulations = os.path.join(parent_folder, 'Simulations')
                        os.makedirs(parent_folder_simulations, exist_ok=True)
                        rapids_ini = os.path.join(parent_folder_simulations, f'rapids_{row['event_id']}_{modello}.ini')
                        output_folder = os.path.join(parent_folder_simulations, row['event_id'], modello)

                        mt = retrieve_focal_mechanisms_Mw(row['datetime'],row["lat"],row["lon"],row["depth"],row["Ml"])

                        ok = 'no'

                        if mt is not None:
                            lat = mt['Lat']
                            lon = mt['Lon']
                            depth = mt['Dep']
                            date = mt['Date']
                            time = mt['Time']
                            mw = mt['Mw_new']
                            stk = mt['Str1']
                            dip = mt['Dip1']
                            rak = mt['Rak1']
                            ok = "yes"

                        if row['datetime']==pd.Timestamp(datetime(2024, 3, 27, 21, 19, 0)):
                            #MT solutions; OGS Real Time Seismology (RTS) service: https://rts.crs.inogs.it/en/
                            lat = 46.3583	
                            lon = 12.808	
                            dep = 10.22
                            mw = 4.2
                            stk1 = 81	
                            dip1 = 47	
                            rak1 = 133
                            stk2 = 207	
                            dip2 = 58	
                            rak2 = 54
                            stk = stk1
                            dip = dip1
                            rak = rak1
                            ok = "yes"

                        if ok == "yes":
                            path_metadata = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'response')
                            path_recordings = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'raw')
                            stations = []
                            IDs = []
                            lons = []
                            lats = []

                            channels = ['HHN', 'BHN', 'HNN', 'HHE', 'BHE', 'HNE','HHZ', 'BHZ', 'HNZ']
                            for file in glob.glob(os.path.join(path_metadata, "*.xml")):
                                inv = read_inventory(file)
                                for network in inv:
                                    for station in network:
                                        code = f"{network.code}.{station.code}"
                                        lat_station = station.latitude
                                        lon_station = station.longitude
                                        dist_m, _, _ = gps2dist_azimuth(lat_station, lon_station, lat, lon)
                                        dist = dist_m/1000.
                                        if dist < 100.:
                                            channel_exists = False
                                            for ch in channels:
                                                mseed_file = os.path.join(path_recordings, f"{code}..{ch}.mseed")
                                                if os.path.exists(mseed_file):
                                                    channel_exists = True
                                                    break
                                            if channel_exists:
                                                IDs.append(code)        
                                                lons.append(lon_station)        
                                                lats.append(lat_station)        
                            create_ini(lat,lon,depth,date,time,mw,stk,dip,rak,output_folder,SETTINGS_FILE,rapids_ini,vm,k,rt,rp,recordings_folder,IDs,lons,lats)
                            lista_output_folders.append(output_folder)
                            lista_ini.append(rapids_ini)


script = os.path.join(parent_folder_simulations,"run_calibration.sh")
with open(script, "w") as f:
    for out_folder, ini_file in zip(lista_output_folders, lista_ini):
        f.write("cd $HOME\n")
        f.write(f"rm -rf '{out_folder}'\n")
        f.write(f"mkdir -p '{out_folder}' \n")
        f.write(f"python -m rapids '{ini_file}' ucsb --run \n")
        f.write(f"cd '{out_folder}/UCSB' \n")
        f.write("bash do_all.sh \n")
        f.write("cd $HOME\n")
        f.write(f"python -m rapids '{ini_file}' ucsbrec --post \n")
        f.write(f"\n")
