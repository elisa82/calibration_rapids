from collections.abc import Iterable
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="obspy.io.stationxml.core")

def get_soil_class(station_pairs: Iterable[tuple[str, str]] | tuple[str, str]):
    def normalize(value: object) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        return text.casefold() if text else None

    requested_pairs = [station_pairs] if isinstance(station_pairs, tuple) else list(station_pairs)
    lookup = {
        (normalize(network), normalize(station)): []
        for network, station in requested_pairs
        if normalize(network) and normalize(station)
    }

    sources = [
        ("/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/Soil_classes/soil_class_carla", "NET", "Code", "EC8", ("Vs30",), "csv", {"sep": r"\s+", "engine": "python", "skiprows": [1]}),
        ("/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/Soil_classes/EC8_PGArecordingStations.xlsx", "network", "station", "ground_type (EC8)", ("vs30",), "excel", {}),
        ("/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/Soil_classes/ESM_stations.xlsx", "Net_code", "Sta_code", "EC8 code", ("Vs30",), "excel", {"header": 1}),
    ]

    def safe_lookup(df, net_val, sta_val, net_col, sta_col, soil_col, vs30_col):
        def lookup_for(nv):
            mask = (df[net_col].astype(str).str.strip().str.upper() == nv.upper()) & \
                   (df[sta_col].astype(str).str.strip().str.upper() == sta_val.upper())
            soil_series = df.loc[mask, soil_col].dropna().astype(str).str.strip()
            vs30_series = df.loc[mask, vs30_col].dropna().astype(str).str.strip()
            soil_val = next((v for v in soil_series if v.lower() != 'unknown'), None)
            vs30_val = next((v for v in vs30_series if v.lower() != 'unknown'), None)
            return soil_val, vs30_val

        soil_val, vs30_val = lookup_for(net_val)
        if soil_val is None and vs30_val is None and net_val.upper() in ("FV", "NI"):
            soil_val, vs30_val = lookup_for("OX")
        return soil_val, vs30_val

    for net_val_orig, sta_val_orig in requested_pairs:
        net_val_norm, sta_val_norm = normalize(net_val_orig), normalize(sta_val_orig)
        soil_text, vs30_text = None, None

        for path, net_col_name, sta_col_name, soil_col_name, vs30_cols_names, file_type, read_kwargs in sources:
            if file_type == "csv":
                df = pd.read_csv(path, dtype=str, **read_kwargs)
            else:
                df = pd.read_excel(path, dtype=str, **read_kwargs)
            df.columns = [str(c).strip() for c in df.columns]
            net_col = next(c for c in df.columns if normalize(c) == normalize(net_col_name))
            sta_col = next(c for c in df.columns if normalize(c) == normalize(sta_col_name))
            soil_col = next(c for c in df.columns if normalize(c) == normalize(soil_col_name))
            vs30_col = next(c for c in df.columns if normalize(c) in {normalize(v) for v in vs30_cols_names})

            soil_text, vs30_text = safe_lookup(df, net_val_orig, sta_val_orig, net_col, sta_col, soil_col, vs30_col)
            if soil_text is not None or vs30_text is not None:
                break  # trovato valore valido, non cercare nelle altre fonti

        lookup[(net_val_norm, sta_val_norm)] = [(soil_text, vs30_text)]

    ec8_list = []
    vs30_list = []
    for net_val_orig, sta_val_orig in requested_pairs:
        key = (normalize(net_val_orig), normalize(sta_val_orig))
        matches = lookup.get(key, [])
        if matches:
            ec8_val, vs30_val = matches[0]
        else:
            ec8_val, vs30_val = None, None
        ec8_list.append(ec8_val)
        vs30_list.append(vs30_val)

    return ec8_list, vs30_list


def create_script_slurm(NUM_JOBS,script_filename, out_folders, ini_files):
    cores_per_job = 20
    memory_per_job = 50  # GB
    total_cores_available = 1000
    time_limit = "1:00:00"  # hh:mm:ss

    max_concurrent_jobs = total_cores_available // cores_per_job
    max_concurrent_jobs = 100
    if max_concurrent_jobs == 0:
        max_concurrent_jobs = 1 

    with open(script_filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=calibration\n")
        f.write(f"#SBATCH --array=1-{NUM_JOBS}%{max_concurrent_jobs}\n")
        f.write(f"#SBATCH --time={time_limit}\n")
        f.write("#SBATCH --ntasks=1\n")
        f.write(f"#SBATCH --cpus-per-task={cores_per_job}\n")
        f.write(f"#SBATCH --mem={memory_per_job}G\n")
        f.write("#SBATCH --output=out_%A_%a.out\n")
        f.write("#SBATCH --account=OGS23_PRACE_IT_1\n")
        f.write("#SBATCH --partition=dcgp_usr_prod\n")
        f.write("#SBATCH --qos=normal\n")

        f.write("\n")
        f.write("module purge\n")
        f.write("module load openmpi/4.1.6--gcc--12.2.0\n")
        f.write("module load gmt/6.4.0--gcc--12.2.0\n")
        f.write("module load python/3.11.6--gcc--12.2.0-nlkgjki\n")

        f.write("\n")
        f.write("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n")
        f.write('echo "Job $SLURM_ARRAY_TASK_ID uses $SLURM_CPUS_PER_TASK cores"\n')

        f.write("\n")
        f.write("source /leonardo/home/userexternal/ezuccolo/urgent_shake_venv/bin/activate\n")
        f.write("source /leonardo/home/userexternal/ezuccolo/UrgentShake/profile.inc\n")

        f.write("\n")
        f.write("unset I_MPI_PMI_LIBRARY\n")
        f.write("export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0\n\n")

        f.write("\n")
        f.write(f"out_folder_list=({' '.join(f'\"{x}\"' for x in out_folders)})\n")
        f.write(f"ini_file_list=({' '.join(f'\"{x}\"' for x in ini_files)})\n")

        f.write("\n")
        f.write("idx=$((SLURM_ARRAY_TASK_ID-1))\n")
        f.write("out_folder=${out_folder_list[$idx]}\n")
        f.write("ini_file=${ini_file_list[$idx]}\n\n")

        f.write("\n")
        f.write("cd $HOME\n")
        f.write('rm -rf "$out_folder"\n')
        f.write('mkdir -p "$out_folder"\n')

        f.write('my_prex_or_die "python -m rapids $ini_file ucsb --run"\n')

        f.write('cd "$out_folder/UCSB" || exit 1\n')
        f.write('my_prex_or_die "bash do_all.sh"\n')

        f.write("cd $HOME\n")
        f.write('my_prex_or_die "python -m rapids $ini_file ucsbrec --post"\n')

        f.write("\n")
    return

def create_ini(lat,lon,depth,date,time,mw,stk,dip,rak,output_folder,SETTINGS_FILE,rapids_ini,vm,k,rt,rp,recordings_folder,IDs,lons,lats,qs_mode):
    gf = f"{vm}_{qs_mode}"
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
qs_mode = {qs_mode}

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
fmin_filter = 0.01
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

    df_ml_md = df_mt[df_mt['Ml'].notna() & df_mt['MD'].notna()]

    X = df_ml_md['MD'].values
    y = df_ml_md['Ml'].values

    slope, intercept = np.polyfit(X, y, 1)  # 1 indica regressione lineare

    df_mt['Ml_new'] = df_mt['Ml']
    mask_md = df_mt['Ml_new'].isna() & df_mt['MD'].notna()
    df_mt.loc[mask_md, 'Ml_new'] = slope * df_mt.loc[mask_md, 'MD'] + intercept

    df_mt['Mw_new'] = df_mt['Mw']
    mask_ml = df_mt['Mw_new'].isna() & df_mt['Ml_new'].notna()
    df_mt.loc[mask_ml, 'Mw_new'] = 0.7028 * df_mt.loc[mask_ml, 'Ml_new'] + 0.6814

    #Gabriele Tarchini, Luca Moratto, Angela Saraò; A Comprehensive Moment Magnitude Catalog for the Northeastern Italy Region. Seismological Research Letters 2025;; 96 (4): 2714–2723. doi: https://doi.org/10.1785/0220240303

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
parent_recordings = '/Volumes/xHD/work/Users/ezuccolo/Git/Concordia/Waveforms'
parent_recordings_RAN = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/RAN_recordings'
parent_folder_simulations = os.path.join(parent_folder, 'Simulations')
os.makedirs(parent_folder_simulations, exist_ok=True)
velocity_models = ['FRIUL7W','NAC_1D','NWSLOVENIA']
kappa = [0,0.025,0.037,0.045]
#1)Parametric spectral inversion of seismic source, path and site parameters: application to northeast Italy
#L Cataldi, V Poggi, G Costa, S Parolai, B Edwards
#Geophysical Journal International 232 (3), 1926-1943
#2) Gentili & Franceschina 2011
#Gentili S., Franceschina G., 2011. High frequency attenuation of shear waves in the southeastern Alps and northern Dinarides, Geophys. J. Int., 185(3), 1393–1416..10.1111/j.1365-246X.2011.05016.x
#3) Malagnini et al., 2002
#Malagnini L., Akinci A., Herrmann R., Pino N., Scognamiglio L., 2002. Characteristics of the ground motion in northeastern Italy, Bull. seism. Soc. Am., 92, 2186–2204..10.1785/0120010219
qs_mode = ['USGS','f1H','f5Hz','f10Hz']
rise_time = ['Somerville1999','GusevChebrov2019']
rad_pattern = ['fixed','randomized']

file_terremoti = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/test_EQ/test_earthquakes_arso_2.txt'
SETTINGS_FILE = f'/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/UrgentShake/settings.ini'
rows = []
with open(file_terremoti, encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            rows.append(parts[:10])

path_lista_events = os.path.join('/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Calibration/calibration_rapids/catalogo_calibrazione_UCSB_Mw3NE.txt')
df = pd.read_csv(
    path_lista_events,
    parse_dates=["datetime"],
    sep=",",
    comment=None,  
    header=0       
)
df["event_id"] = df["datetime"].dt.strftime("%Y%m%d%H%M%S")
df.columns = df.columns.str.strip().str.lstrip("#")
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df_events = pd.DataFrame({
    "event_id": df['event_id'].astype(str),
    "datetime": df['datetime'],
    "date": df['datetime'].dt.date,
    "time": df['datetime'].dt.time,
    "year": df['datetime'].dt.year,
    "month": df['datetime'].dt.month,
    "day": df['datetime'].dt.day,
    "hour": df['datetime'].dt.hour,
    "minute": df['datetime'].dt.minute,
    "second": df['datetime'].dt.second,
    "lat": df['Lat'],
    "lon": df['Lon'],
    "depth": df['Dep'],
    "mw": df['Mw'],
    "stk": df['Str1'],
    "dip": df['Dip1'],
    "rak": df['Rak1'],
})

lista_A_stations = []
lista_ini = []
lista_output_folders = []
num_jobs = 0

channel_groups = [
    ['HHN', 'HHE', 'HHZ'],
    ['EHN', 'EHE', 'EHZ'],
    ['SHN', 'SHE', 'SHZ'],
    ['BHN', 'BHE', 'BHZ'],
    ['HNN', 'HNE', 'HNZ'],
    ['HGN', 'HGE', 'HGZ']
]

soil_cache = {}
for i, row in df_events.iterrows():
    lat = df_events['lat'][i]
    lon = df_events['lon'][i]
    depth = df_events['depth'][i]
    date = df_events['date'][i]
    time = df_events['time'][i]
    mw = df_events['mw'][i]
    stk = df_events['stk'][i]
    dip = df_events['dip'][i]
    rak = df_events['rak'][i]

    stations = []
    IDs = []
    lons = []
    lats = []

    for j in range(2):
        j = 1
        if j==0:
            path_recordings = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'raw')
            path_metadata = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'response')
        else:
            path_recordings = os.path.join(parent_recordings_RAN, f"{row['event_id']}_sac") 
            path_metadata = os.path.join(parent_recordings_RAN, 'StationXML_IT') 
    
        if os.path.exists(path_recordings):

            if i == 0:
                xml_files = glob.glob(os.path.join(path_metadata, "*.xml"))
            else:
                recording_files = glob.glob(os.path.join(path_recordings, "*"))
                stations = []
                for file in recording_files:
                    filename = os.path.basename(file)
                    parts = filename.split('.')
                    if len(parts) >= 3:
                        station_name = parts[2]
                        stations.append(station_name)
                stations = list(set(stations))
                xml_files = []
                for station in stations:
                    path_xml = os.path.join(path_metadata, f"{station}.IT.xml")
                    if os.path.exists(path_xml):
                        xml_files.append(path_xml)

                for file in xml_files:
                    inv = read_inventory(file)
                    for network in inv:
                        for station in network:
                            code = f"{network.code}.{station.code}"
                            lat_station = station.latitude
                            lon_station = station.longitude
                            station_ok = False
                            if code:
                                if lat_station >= 45.3 and lat_station < 47:
                                    if lon_station >= 12.5 and lon_station < 14.5:

                                        dist_m, az, baz = gps2dist_azimuth(lat, lon, lat_station, lon_station)
                                        dist_km = dist_m / 1000

                                        if dist_km < 100:
                                            
                                            if code in soil_cache:
                                                ec8, vs30 = soil_cache[code]
                                            else:
                                                ec8_list, vs30_list = get_soil_class([(network.code, station.code)])
                                                ec8, vs30 = ec8_list[0], vs30_list[0]
                                                soil_cache[code] = (ec8, vs30)


                                            if ec8 == 'A' or (vs30 is not None and float(vs30) > 799.99):
                                                if j == 0:
                                                    for group in channel_groups:
                                                        files_exist = True

                                                        for ch in group:
                                                            mseed_file = os.path.join(path_recordings, f"{code}..{ch}.mseed")
                                                            if not os.path.exists(mseed_file):
                                                                files_exist = False
                                                                break

                                                        if files_exist:
                                                            station_ok = True
                                                            break
                                                else:
                                                    station_ok = True

                            if station_ok:
                                IDs.append(code)
                                lons.append(lon_station)
                                lats.append(lat_station)
                                lista_A_stations.append(code)

            lista_A_stations = list(dict.fromkeys(lista_A_stations))
            
            if len(IDs) > 0:
                for vm in velocity_models:
                    for k in kappa:
                        for rt in rise_time:
                            for rp in rad_pattern:
                                for qs in qs_mode:
                                    modello = f"{vm}_k0_{k}_Tr_{rt}_rp_{rp}_qs_{qs}"
                                    rapids_ini = os.path.join(parent_folder_simulations, f'rapids_{row['event_id']}_{modello}.ini')
                                    output_folder = os.path.join(parent_folder_simulations, row['event_id'], modello)
                                    create_ini(lat,lon,depth,date,time,mw,stk,dip,rak,output_folder,SETTINGS_FILE,rapids_ini,vm,k,rt,rp,path_recordings,IDs,lons,lats,qs)
                                    lista_output_folders.append(output_folder)
                                    lista_ini.append(rapids_ini)
                                    num_jobs = num_jobs + 1

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


script_filename= "run_array.sh"
create_script_slurm(num_jobs,script_filename,lista_output_folders,lista_ini)
