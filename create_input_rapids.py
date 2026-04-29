def create_submit_sequential(scripts):
    """
    Crea uno script bash che sottomette job Slurm in sequenza con dependency.
    """

    output_file="submit_all.sh"
    dependency_type="afterok"
    with open(output_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("prev_job_id=\"\"\n\n")

        for script in scripts:

            f.write(f"echo \"Submitting {script}\"\n")

            f.write("if [ -z \"$prev_job_id\" ]; then\n")
            f.write(f"    job_id=$(sbatch {script} | awk '{{print $4}}')\n")
            f.write("else\n")
            f.write(f"    job_id=$(sbatch --dependency={dependency_type}:$prev_job_id {script} | awk '{{print $4}}')\n")
            f.write("fi\n")

            f.write("echo \" -> Job ID: $job_id\"\n")
            f.write("prev_job_id=$job_id\n\n")

    print(f"Submit script creato: {output_file}")

    return 


from collections.abc import Iterable
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="obspy.io.stationxml.core")

def define_filters(ch, is_sara, current_date):
    from datetime import date
    import pandas as pd

    if ch in ['HH', 'BH']:
        fmin = 0.1
    elif ch in ['EH', 'SH', 'HN', 'HG']:
        fmin = 0.2

    if ch in ['BH']:
        fmax = 5
    else:
        fmax = 19.9

    if is_sara and current_date < pd.Timestamp(2024, 2, 7): # se SARA fmax deve essere 8 fino al 7/02/2024 escluso
        fmax = 8

    return fmin, fmax


def check_station_groups(code, path_recordings,source_rec):
    import os
    import glob
    channel_groups = [
        ['HHN', 'HHE', 'HHZ'],
        ['EHN', 'EHE', 'EHZ'],
        ['SHN', 'SHE', 'SHZ'],
        ['BHN', 'BHE', 'BHZ'],
        ['HNN', 'HNE', 'HNZ'],
        ['HGN', 'HGE', 'HGZ']
    ]

    for idx, group in enumerate(channel_groups):
        all_exist = True
        for ch in group:
            if source_rec == "RAN":
                station_part = code.split('.')[1]
                pattern = os.path.join(path_recordings, f"*.{station_part}.{ch}")
                matching_files = glob.glob(pattern)
                if not matching_files:
                    all_exist = False
                break
            else:
                rec_file = os.path.join(path_recordings, f"{code}..{ch}.mseed")

            if not os.path.exists(rec_file):
                all_exist = False
                break  

        if all_exist:
            sensor_type = "accelerometer" if idx >= len(channel_groups) - 2 else "seismometer"
            return True, group, sensor_type

    return False, None, None


def get_soil_class(net,sta,sensor,parent_folder):
    import pandas as pd
    import os

    soil_class = None

    if net in ['OX', 'FV', 'NI']:
        file_CRS = os.path.join(parent_folder,'calibration_rapids/soil_class_Klin.csv')
        df_crs = pd.read_csv(file_CRS)
        df_crs['Station'] = df_crs['Station'].astype(str).str.strip()
        df_crs['Geomorphological_Scenario'] = df_crs['Geomorphological_Scenario'].astype(str).str.strip()
        df_crs['soil'] = df_crs['Geomorphological_Scenario'].str[0]

        selected_row = df_crs[df_crs['Station'] == sta]
        if not selected_row.empty:
            soil_class = selected_row['soil'].values[0]

    if soil_class is None and net == 'SL':
        file_slovenia = os.path.join(parent_folder,'calibration_rapids/EC8_PGArecordingStations.xlsx')
        df_sl = pd.read_excel(file_slovenia)
        df_sl["station"] = df_sl["station"].astype(str).str.strip()
        df_sl["ground_type (EC8)"] = df_sl["ground_type (EC8)"].astype(str).str.strip()
        df_sl[['station_code', 'sensor_type']] = df_sl['station'].str.split(',', n=1, expand=True)
        df_sl['sensor_type'] = df_sl['sensor_type'].fillna("")  
        selected_row = df_sl[(df_sl['station_code'] == sta) & ((df_sl['sensor_type'] == sensor) | (df_sl['sensor_type'] == ""))]
        if not selected_row.empty:
            soil_class = selected_row['ground_type (EC8)'].values[0]

    if soil_class is None:
        file_ESM = os.path.join(parent_folder,'calibration_rapids/ESM_stations.xlsx')
        df_esm = pd.read_excel(file_ESM)
        df_esm["Net_code"] = df_esm["Net_code"].astype(str).str.strip()
        df_esm["Sta_code"] = df_esm["Sta_code"].astype(str).str.strip()
        df_esm["EC8 code"] = df_esm["EC8 code"].astype(str).str.strip()
        df_esm["Vs30"] = pd.to_numeric(df_esm["Vs30"], errors='coerce')
        net_esm = 'OX' if net in ['FV'] else net
        selected_row = df_esm[(df_esm["Net_code"] == net_esm) & (df_esm["Sta_code"] == sta)]
        if not selected_row.empty:
            soil_class = selected_row["EC8 code"].values[0]

    return soil_class


def create_script_slurm(NUM_JOBS,script_filename, out_folders, ini_files, parent_folder):
    import os
    import math

    out_folders_file = os.path.join(parent_folder,'calibration_rapids','out_folders.txt')
    ini_files_file = os.path.join(parent_folder,'calibration_rapids','ini_files.txt')

    with open(out_folders_file, "w") as f:
        f.write("\n".join(out_folders))

    with open(ini_files_file, "w") as f:
        f.write("\n".join(ini_files))

    cores_per_job = 20
    memory_per_job = 50  # GB
    total_cores_available = 2000
    time_limit = "24:00:00"  # hh:mm:ss

    max_concurrent_jobs = 100
    MAX_ARRAY = 1000

    num_scripts = math.ceil(NUM_JOBS / MAX_ARRAY)
    script_filenames_chunks = []

    for s in range(num_scripts):

        start_job = s * MAX_ARRAY + 1
        end_job = min((s + 1) * MAX_ARRAY, NUM_JOBS)

        script_filename_chunk = f"{script_filename}_part{s+1}.sh"
        script_filenames_chunks.append(script_filename_chunk)

        with open(script_filename_chunk, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=calibration\n")

            # array chunck
            f.write(f"#SBATCH --array={start_job}-{end_job}%{max_concurrent_jobs}\n")

            f.write(f"#SBATCH --time={time_limit}\n")
            f.write(f"#SBATCH --nodes=1\n")
            f.write(f"#SBATCH --cpus-per-task={cores_per_job}\n")
            f.write(f"#SBATCH --mem={memory_per_job}G\n")
            f.write("#SBATCH --output=out_%A_%a.out\n")
            f.write("#SBATCH --account=OGS23_PRACE_IT_1\n")
            f.write("#SBATCH --partition=dcgp_usr_prod\n")
            f.write("#SBATCH --qos=normal\n\n")

            f.write("module purge\n")
            f.write("module load openmpi/4.1.6--gcc--12.2.0\n")
            f.write("module load gmt/6.4.0--gcc--12.2.0\n")
            f.write("module load python/3.11.6--gcc--12.2.0-nlkgjki\n\n")

            f.write("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n")
            f.write('echo "Job $SLURM_ARRAY_TASK_ID uses $SLURM_CPUS_PER_TASK cores"\n\n')

            f.write("source /leonardo/home/userexternal/ezuccolo/rapids_venv/bin/activate\n")
            f.write("source /leonardo/home/userexternal/ezuccolo/UrgentShake/profile.inc\n\n")

            f.write("unset I_MPI_PMI_LIBRARY\n")
            f.write("export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0\n\n")

            f.write("cd $HOME\n\n")

            f.write("idx=$SLURM_ARRAY_TASK_ID\n\n")

            f.write(f"out_folder=$(sed -n \"${{idx}}p\" {out_folders_file})\n")
            f.write(f"ini_file=$(sed -n \"${{idx}}p\" {ini_files_file})\n\n")

            f.write("if [ -z \"$out_folder\" ]; then\n")
            f.write("    exit 0\n")
            f.write("fi\n\n")

            f.write("done_flag=\"$out_folder/.DONE\"\n")
            f.write("if [ -f \"$done_flag\" ]; then\n")
            f.write("    echo \"Task $idx already completed, skipping\"\n")
            f.write("    exit 0\n")
            f.write("fi\n\n")

            f.write("mkdir -p \"$out_folder\"\n\n")
            f.write("my_prex_or_die \"python -m rapids $ini_file ucsb --run\"\n")
            f.write("cd \"$out_folder/UCSB\" || exit 1\n")
            f.write("my_prex_or_die \"bash do_all.sh\"\n")
            f.write("cd $HOME\n")
            f.write("my_prex_or_die \"python -m rapids $ini_file ucsbrec --post\"\n\n")

            f.write("done\n")
    return script_filenames_chunks

def create_ini(lat,lon,depth,date,time,mw,stk,dip,rak,output_folder,SETTINGS_FILE,rapids_ini,vm,k,rt,rp,IDs,lons,lats,qs_mode,fmins,fmaxs,channels,recordings_folder_CRS,recordings_folder_RAN,inventory_folder_CRS,inventory_folder_RAN,gf_folder_root):
    import os
    gf_name = f"{vm}_{qs_mode}"
    gf = os.path.join(gf_folder_root,gf_name)
    IDs_text = ', '.join(IDs)
    lats_text = ', '.join(str(lat) for lat in lats)
    lons_text = ', '.join(str(lon) for lon in lons)
    fmins_text = ', '.join(str(fmin) for fmin in fmins)
    fmaxs_text = ', '.join(str(fmax) for fmax in fmaxs)
    channels_text = ', '.join(str(ch) for ch in channels)

    with open(rapids_ini, "w", encoding="utf-8") as f:
        f.write(f"""output_folder = {output_folder}
settings_file = {SETTINGS_FILE}

fault_geolocation = from_hypo

#type_run: standard o test
type_run = standard

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
dt_ucsb = 0.009765625
npts_ucsb = 8192 
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
fmax_rec = [{fmaxs_text}]
fmin_rec = [{fmins_text}]
channel_rec = [{channels_text}]
time_max_plot = 60
interpolation_map = no
path_rec_CRS = {recordings_folder_CRS}
path_rec_RAN = {recordings_folder_RAN}
path_inv_CRS = {inventory_folder_CRS}
path_inv_RAN = {inventory_folder_RAN}
""")


def retrieve_focal_mechanisms_Mw(datetime_slo,lat_slo,lon_slo,depth_slo,ml_slo,parent_folder):
    from datetime import datetime, timedelta
    import os

    focal_mechanism_catalogue = os.path.join(parent_folder,'calibration_rapids/catalogue1928-2023_Suganal2024.csv')
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


working_path = '/leonardo_scratch/large/userexternal/ezuccolo'
SETTINGS_FILE = f'/leonardo/home/userexternal/ezuccolo/settings.ini'
#SETTINGS_FILE = f'/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/UrgentShake/settings.ini'
#parent_recordings = '/Volumes/xHD/work/Users/ezuccolo/Git/Concordia/Waveforms'
parent_recordings = os.path.join(working_path, 'Waveforms')
gf_folder_root = os.path.join(working_path, 'GF')

parent_recordings_RAN = os.path.join(working_path, 'Calibration', 'RAN_recordings')
parent_folder = os.path.join(working_path, 'Calibration')
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
qs_mode = ['USGS','f1Hz','f5Hz','f10Hz']
rise_time = ['Somerville1999','GusevChebrov2019']
rad_pattern = ['fixed','randomized']

rows = []

path_lista_events = os.path.join(parent_folder,'calibration_rapids/catalogo_calibrazione_UCSB_Mw3NE.txt')
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

    IDs = []
    lons = []
    lats = []
    channels = []
    fmins = []
    fmaxs = []
    stations_seen = set()

    path_rec_CRS = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'raw')
    path_inv_CRS = os.path.join(parent_recordings, 'event_based_dir', row['event_id'], 'response')
    source_rec_CRS = 'web_server_CRS'
    path_rec_RAN = os.path.join(parent_recordings_RAN, f"{row['event_id']}_sac") 
    path_inv_RAN = os.path.join(parent_recordings_RAN, 'StationXML_IT') 
    source_rec_RAN = 'RAN'
    for j in range(2):
        if j==0:
            path_recordings = path_rec_CRS
            path_metadata = path_inv_CRS
            source_rec = source_rec_CRS
        else:
            path_recordings = path_rec_RAN
            path_metadata = path_inv_RAN
            source_rec = source_rec_RAN
            
        if os.path.exists(path_recordings):
            if j == 0:
                xml_files = glob.glob(os.path.join(path_metadata, "*.xml"))
                xml_files = [f for f in xml_files if not os.path.basename(f).startswith("IT.")]
            else:
                recording_files = glob.glob(os.path.join(path_recordings, "*"))
                stations_RAN = []
                for file in recording_files:
                    filename = os.path.basename(file)
                    parts = filename.split('.')
                    if len(parts) >= 3:
                        station_name = parts[2]
                        stations_RAN.append(station_name)
                stations_RAN = list(set(stations_RAN))
                xml_files = []
                for s in stations_RAN:
                    path_xml = os.path.join(path_metadata, f"{s}.IT.xml")
                    if os.path.exists(path_xml):
                        xml_files.append(path_xml)

            inv_all = read_inventory(xml_files[0])
            for file in xml_files[1:]:
                inv_all += read_inventory(file)

            for network in inv_all:
                for station in network:
                    code = f"{network.code}.{station.code}"

                    station_ok, valid_group, sensor_type = check_station_groups(code, path_recordings, source_rec)

                    if not station_ok:
                        continue

                    channel_type = valid_group[0][:2]
                    unique_key = f"{code}.{channel_type}"
                    if unique_key in stations_seen:
                        continue
                    stations_seen.add(unique_key)

                    channel_obj = None
                    for ch in station:
                        if ch.code.startswith(channel_type):
                            channel_obj = ch
                            break
                    if channel_obj is None:
                        continue  

                    is_sara = False
                    sensor_description = ""
                    if network.code != "IT" and channel_obj.sensor is not None:
                        sensor_description = channel_obj.sensor.description or ""
                        is_sara = "SARA" in sensor_description.upper()

                    lat_station = channel_obj.latitude
                    lon_station = channel_obj.longitude
                    if not (45.3 <= lat_station < 47 and 12.5 <= lon_station < 14.5):
                        continue

                    dist_m, az, baz = gps2dist_azimuth(lat, lon, lat_station, lon_station)
                    dist_km = dist_m / 1000
                    max_dist_km = 50 if sensor_type == "accelerometer" else 100

                    if dist_km >= max_dist_km:
                        continue

                    if unique_key in soil_cache:
                        soil = soil_cache[unique_key]
                    else:
                        soil = get_soil_class(network.code, station.code, sensor_type, parent_folder)
                        soil_cache[unique_key] = soil
                        
                    if soil not in ['A','H']:
                        continue

                    IDs.append(code)
                    lons.append(lon_station)
                    lats.append(lat_station)
                    channels.append(valid_group[0][:2])
    
                    event_date = pd.to_datetime(date)
                    fmin_rec, fmax_rec = define_filters(channel_type, is_sara, event_date)
                    fmins.append(fmin_rec)
                    fmaxs.append(fmax_rec)

                    lista_A_stations.append(code)

    lista_A_stations = list(dict.fromkeys(lista_A_stations))

    if len(IDs) > 0:
        for vm in velocity_models:
            for k in kappa:
                for rt in rise_time:
                    for rp in rad_pattern:
                        for qs in qs_mode:
                            modello = f"{vm}_k0_{k}_Tr_{rt}_rp_{rp}_qs_{qs}"
                            rapids_ini = os.path.join(
                                parent_folder_simulations,
                                f"rapids_{row['event_id']}_{modello}.ini"
                            )
                            output_folder = os.path.join(parent_folder_simulations, row['event_id'], modello)
                            create_ini(lat,lon,depth,date,time,mw,stk,dip,rak,output_folder,SETTINGS_FILE,rapids_ini,vm,k,rt,rp,IDs,lons,lats,qs,fmins,fmaxs,channels,path_rec_CRS,path_rec_RAN,path_inv_CRS,path_inv_RAN,gf_folder_root)
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
script_filenames_chunks = create_script_slurm(num_jobs,script_filename,lista_output_folders,lista_ini,parent_folder)
create_submit_sequential(script_filenames_chunks)
