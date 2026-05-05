def response2(f, dt, xi):
    import numpy as np
    from scipy.linalg import expm
    """
    Computes elastic response spectra for SDOF systems.
    
    Parameters:
    -----------
    f  : np.array
        Excitation vector (support acceleration)
    dt : float
        Sampling interval of f
    xi : array-like
        Damping ratios to compute spectra for
    
    Returns:
    --------
    T  : np.array
        Periods (s)
    SA : np.ndarray
        Absolute acceleration spectrum [len(T) x len(xi)]
    """
    
    # Define periods T
    T = np.arange(0, 4+0.025, 0.025)
    T[0] = 0.01
    
    xi = np.atleast_1d(xi)
    
    SA = np.zeros((len(T), len(xi)))
    
    for j, xij in enumerate(xi):
        for i, Ti in enumerate(T):
            w = 2 * np.pi / Ti
            C = 2 * xij * w
            K = w**2
            y = np.zeros((2, len(f)))  # y[0,:] = displacement, y[1,:] = velocity
            A = np.array([[0, 1],
                          [-K, -C]])
            
            Ae = expm(A * dt)
            AeB = np.linalg.solve(A, (Ae - np.eye(2)) @ np.array([[0], [-1]]))[:,0]
            
            for k in range(1, len(f)):
                y[:,k] = Ae @ y[:,k-1] + AeB * f[k]
            # Absolute acceleration
            SA[i,j] = np.max(np.abs(K * y[0,:] + C * y[1,:]))
    
    return T, SA


def compute_derivative(u, dt):
    import numpy as np
    dudt = np.copy(u) 
    npts = len(u)
    for j in range(1, npts - 1):
        dudt[j] = (u[j + 1] - u[j - 1]) / (2 * dt)
    dudt[0] = 0.0
    dudt[npts - 1] = dudt[npts - 2]
    return dudt


def compute_gof_DUR(x, y, tx, ty):
    import numpy as np
    from scipy.special import erfc

    y_interp = np.interp(tx, ty, y)
    energy_integral_rec = np.cumsum(x**2)
    energy_integral_sim = np.cumsum(y_interp**2)

    cumulative_energy_rec = energy_integral_rec[-1]
    cumulative_energy_sim = energy_integral_sim[-1]

    index_5_rec = np.where(energy_integral_rec / cumulative_energy_rec >= 0.05)[0][0]
    index_75_rec = np.where(energy_integral_rec / cumulative_energy_rec >= 0.75)[0][0]
    dur_rec = tx[index_75_rec] - tx[index_5_rec]

    index_5_sim = np.where(energy_integral_sim / cumulative_energy_sim >= 0.05)[0][0]
    index_75_sim = np.where(energy_integral_sim / cumulative_energy_sim >= 0.75)[0][0]
    dur_sim = ty[index_75_sim] - ty[index_5_sim]

    NR = 2 * np.abs(dur_rec - dur_sim) / (dur_rec + dur_sim)
    return 100 * erfc(NR)

def compute_gof_PGV(x, y):
    import numpy as np
    from scipy.special import erfc
    PGV_recorded = np.max(np.abs(x))
    PGV_simulated = np.max(np.abs(y))
    NR = 2 * np.abs(PGV_recorded - PGV_simulated) / (PGV_recorded + PGV_simulated)
    return 100 * erfc(NR)

def compute_gof_PGA(x, y):
    import numpy as np
    from scipy.special import erfc
    PGA_recorded = np.max(np.abs(x))
    PGA_simulated = np.max(np.abs(y))
    NR = 2 * np.abs(PGA_recorded - PGA_simulated) / (PGA_recorded + PGA_simulated)
    return 100 * erfc(NR)

def compute_gof_PGD(x, y):
    import numpy as np
    from scipy.special import erfc
    PGD_recorded = np.max(np.abs(x))
    PGD_simulated = np.max(np.abs(y))
    NR = 2 * np.abs(PGD_recorded - PGD_simulated) / (PGD_recorded + PGD_simulated)
    return 100 * erfc(NR)

def compute_gof_SA(x, y, dtx, dty):
    import numpy as np
    from scipy.special import erfc
    xi = 0.05
    c
    T, SD, PSV, PSA, SV, SA_simulated, ED = response2(y, dtx, dty, xi)
    period_index = np.where((T > 0.001) & (T <= 10.0 + np.finfo(float).eps))[0]

    NR = 2 * np.abs(SA_recorded[period_index] - SA_simulated[period_index]) / \
         (SA_recorded[period_index] + SA_simulated[period_index])
    return 100 * np.mean(erfc(NR))


def compute_gof_FS(x, y, dtx, dty, fmin_val, fmax):
    import numpy as np
    from scipy.special import erfc
    # Calcolo FAS
    FAS_recorded, fr = FR_AMP(x, dtx)
    FAS_simulated, fs = FR_AMP(y, dty)

    # Finestra di smoothing
    time_window = 15
    # intervallo frequenze registrato
    index_fmax = np.max(np.where(fr < fmax + np.finfo(float).eps))
    index_fmin = np.min(np.where(fr > fmin_val - np.finfo(float).eps))
    FAS_recorded_smoothed = pd.Series(FAS_recorded).rolling(window=time_window, center=True).mean().to_numpy()
    FAS_recorded_smoothed = FAS_recorded_smoothed[index_fmin:index_fmax+1]

    # intervallo frequenze simulato
    index_fmax = np.max(np.where(fs < fmax + np.finfo(float).eps))
    index_fmin = np.min(np.where(fs > fmin_val - np.finfo(float).eps))
    FAS_simulated_smoothed = pd.Series(FAS_simulated).rolling(window=time_window, center=True).mean().to_numpy()
    FAS_simulated_smoothed = FAS_simulated_smoothed[index_fmin:index_fmax+1]

    FAS_sim_interp = np.interp(fr_selected, fs_selected, FAS_simulated_smoothed)

    # GOF FS
    NR = 2 * np.abs(FAS_recorded_smoothed - FAS_sim_interp) / \
         (FAS_recorded_smoothed + FAS_sim_interp)
    FS = 100 / NR.size * np.sum(erfc(NR))
    return FS


def compute_gof_ENER(x, y): #cumulative energy
    import numpy as np
    from scipy.special import erfc
    energy_integral_rec = np.cumsum(x**2)
    energy_integral_sim = np.cumsum(y**2)
    cumulative_energy_rec = energy_integral_rec[-1]
    cumulative_energy_sim = energy_integral_sim[-1]

    NR = 2 * np.abs(cumulative_energy_rec - cumulative_energy_sim) / \
         (cumulative_energy_rec + cumulative_energy_sim)
    ENER = 100 * erfc(NR)
    return ENER


from scipy.integrate import cumulative_trapezoid as cumtrapz
from obspy import read
import numpy as np
from obspy.clients.fdsn import Client

parent_folder = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Confronti/Simulations'
parent_recordings = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Confronti/Recordings'
lista_modelli = ['GNDT14', 'GNDT14']
file_terremoti = '/Users/ezuccolo/Library/CloudStorage/Dropbox/Lavoro/Progetti/CONCORDIA/Confronti/test_EQ/test_earthquakes_arso_2.txt'
rows = []
with open(file_terremoti, encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            rows.append(parts[:10])

df_events = pd.DataFrame(rows, columns=[
    "year","month","day","hour","minute","second",
    "lat","lon","depth","Ml"])
df_events = df_events.astype(float)

num_models = len(lista_modelli)

components = ['NS','EW','Z']

GOF_all = {}

for model in range(num_models):
    path_modello = parent_folder
    GOF_all[model] = {}
    event_avgs = []

    for event_idx, event_row in df_events.iterrows():
        event_name = f"EQ_{int(event_row['year'])}_{int(event_row['month'])}_{int(event_row['day'])}_{int(event_row['Ml'])*10}"
        folder_simulation = os.path.join(parent_folder, lista_modelli[model], event_name)
        mseed_file = os.path.join(parent_recordings, event_name, f"{event_name}.mseed")
        GOF_all[model][event_name] = {}
        station_avgs = []

        #le stazioni le prendiamo da stations.ll

        st = read(mseed_file)

        channel_for_component_with_priorities = {
            'NS':  [ 'HHN', 'BHN', 'HNN' ],
            'EW': ['HHE', 'BHE', 'HNE'],
            'Z':['HHZ', 'BHZ', 'HNZ'],
        }

        stations_file = folder_simulation + "/UCSB/stations.ll"
        with open(stations_file, "r") as f:
            nstations = int(f.readline().strip())
            stations_list = []
            lats_list = []
            lons_list = []

            for _ in range(nstations):
                line = f.readline()
                parts = line.strip().split()
                if len(parts) == 3:
                    lon, lat, code = parts
                    stations_list.append(code)
                    lons_list.append(float(lon))
                    lats_list.append(float(lat))
        stations = np.array(stations_list)
        lats = np.array(lats_list)
        lons = np.array(lons_list)

        for station in stations:
            GOF_all[model][event_name][station] = {}
            component_avgs = []

            st_station = st.select(station=station)

            for comp in components:
                if comp == 'NS':
                    comp_str = '000'
                if comp == 'EW': 
                    comp_str = '090'
                if comp == 'Z':
                    comp_str = 'ver'
                isource = 1
                filename = folder_simulation + "/UCSB/HF/" + station  + "." + comp_str + ".gm1D." + str(isource).zfill(3)
                time_series = []
                with open(filename, 'r') as f:
                    content = f.readlines()
                    for x in range(len(content)):
                        if x == 0:
                            val = content[x].split()
                            npts_sim = int(val[0])
                            dt_sim = float(val[1])
                        else:
                            data_sim = content[x].split()
                            for value in data_sim:
                                a = float(value)
                                time_series_sim.append(a)
                time_series_sim = np.asarray(time_series_sim)
                time_sim = np.arange(0, npts_sim) * dt
                sim_vel = time_series_sim * 100
                sim_disp = cumtrapz(sim_vel, time_sim, initial=0)
                sim_acc = compute_derivative(sim_vel, dt_sim)

                channels_priority = channel_for_component_with_priorities[comp]
                selected_tr = None
                for ch in channels_priority:
                    temp = st_station.select(channel=ch)
                    if len(temp) > 0:
                        selected_tr = temp[0]
                        break

                if selected_tr is not None:
                    data_rec = selected_tr.data           
                    dt_rec = selected_tr.stats.delta       
                    npts_rec = selected_tr.stats.npts       
                    start_rec = selected_tr.stats.starttime  

                    time_rec = np.arange(0, npts_rec*dt_rec, dt_rec)
                    if len(time_rec) > npts_rec:  # corregge eventuale overshoot
                        time_rec = time_rec[:npts_rec]

                else:
                    print("Nessun canale disponibile per questa stazione")

                rec_vel = data_rec * 100
                rec_disp = cumtrapz(rec_vel, time_rec, initial=0)
                rec_acc = compute_derivative(rec_vel, dt_rec)

                PGV = compute_gof_PGV(rec_vel,sim_vel)
                PGA = compute_gof_PGV(rec_acc,sim_acc)
                PGD = compute_gof_PGV(rec_disp,sim_disp)
                FS = compute_gof_FS(rec_vel, sim_vel, dt_rec, dt_sim, fmin_val, fmax)
                SA = compute_gof_SA(rec_acc, sim_acc, dt_rec, dt_sim)
                ENER = compute_gof_ENER(rec_vel, sim_vel)
                DUR = compute_gof_DUR(rec_vel, sim_vel, time_rec, time_sim)

                component_values = [FS, PGV, PGA, PGD, SA, ENER, DUR]
                component_avg = np.mean(component_values)
                component_avgs.append(component_avg)

                GOF_all[model][event_name][station][comp] = {
                    'FS': FS,
                    'PGV': PGV,
                    'PGA': PGA,
                    'PGD': PGD,
                    'SA': SA,
                    'ENER': ENER,
                    'DUR': DUR,
                    'COMPONENT_AVG': component_avg
                }

            station_avg = np.mean(component_avgs)
            GOF_all[model][event_name][station]['STATION_AVG'] = station_avg
            station_avgs.append(station_avg)
            print(f"Model {model+1}, Event {event_name}, Station {station+1}, station GOF: {station_avg:.2f}")

        event_avg = np.mean(station_avgs)
        GOF_all[model][event_name]['EVENT_AVG'] = event_avg
        event_avgs.append(event_avg)
        print(f"  Event {event_name} average GOF = {event_avg:.2f}\n")

    model_avg = np.mean(event_avgs)
    GOF_all[model]['MODEL_OVERALL_AVG'] = model_avg
    print(f"Model {model+1} overall GOF = {model_avg:.2f}\n")
