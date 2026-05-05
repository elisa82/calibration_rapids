import os
import glob
import numpy as np

metrics = ["pga", "pgv", "pgd", "dur", "ener", "fft_smoothed", "SA"]
metrics = ["pga", "pgv", "pgd","fft_smoothed", "SA"]
components = ["ns", "ew", "z"]

file_list = "out_folders.txt"

all_results = []

best_folder = None
best_gof_folder = -np.inf

def convert(o):
    """Rende tutto JSON-serializzabile"""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o

def compute_gof(x_rec, x_ucsb, log=False):
    import numpy as np
    from scipy.special import erfc

    x_rec = np.asarray(x_rec)
    x_ucsb = np.asarray(x_ucsb)

    if log:
        eps = 1e-12
        x_rec = np.where(x_rec <= 0, eps, x_rec)
        x_ucsb = np.where(x_ucsb <= 0, eps, x_ucsb)

        NR = np.abs(np.log(x_rec) - np.log(x_ucsb))
        return 100 * np.mean(erfc(NR))

    den = x_rec + x_ucsb
    if np.any(den == 0):
        return 0.0

    NR = 2 * np.abs(x_rec - x_ucsb) / den
    return 100 * erfc(NR)


def get_field(site, metric):
    if metric == "pgd":
        return site["disp"]
    elif metric == "pgv" or metric == "dur" or metric == "ener" or metric == 'fft_smoothed': 
        return site["vel"]
    else:  
        return site["acc"]

with open(file_list, "r") as f:
    folders = [line.strip() for line in f if line.strip()]

def process_folder(folder):

    folder_rec = os.path.join(folder, "REC")

    file_rank_list = glob.glob(os.path.join(folder_rec, "rank_*", "*.npz"))
    local_results = []

    event_global_values = []

    for rank_file in file_rank_list:

        data = np.load(rank_file, allow_pickle=True)

        for key in data.files:
            site = data[key].item()

            site_id = site["ID"]
            lon = site["lon"]
            lat = site["lat"]


            site_gof = {}
            metric_averages = []

            for metric in metrics:

                comp_values = {}
                values = []

                rec_raw = get_field(site, metric)["rec"]
                ucsb_raw = get_field(site, metric)["ucsb"]

                rec = {k.strip(): v for k, v in rec_raw.items()}
                ucsb = {k.strip(): v for k, v in ucsb_raw.items()}

                for comp in components:

                    if metric == "fft_smoothed":
                        key_name = f"{comp}_{metric}"
                        GOF = compute_gof(rec[key_name], ucsb[key_name], log=True)

                    elif metric == "SA":
                        key_name = f"{metric}_{comp}"
                        GOF = compute_gof(rec[key_name], ucsb[key_name], log=True)

                    else:
                        key_name = f"{metric}_{comp}"
                        GOF = compute_gof(rec[key_name], ucsb[key_name])

                    comp_values[comp] = GOF
                    values.append(GOF)
                comp_average = np.mean(values) if values else 0.0
                comp_values["comp_average"] = comp_average

                site_gof[metric] = comp_values
                metric_averages.append(comp_average)

            global_average = np.mean(metric_averages) if metric_averages else 0.0

            site_gof["global_average"] = global_average
            event_global_values.append(global_average)

            local_results.append({
                "folder": folder,
                "ID": site_id,
                "lon": lon,
                "lat": lat,
                "GOF": site_gof
            })

    gof_event = np.mean(event_global_values) if event_global_values else 0.0
    return folder, gof_event, local_results

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
if __name__ == "__main__":
    model_gof = {}
    event_results = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_folder, folder) for folder in folders]

        for future in as_completed(futures):
            folder, gof_event, local_results = future.result()

            modello = os.path.basename(folder)

            model_gof.setdefault(modello, []).append(gof_event)
            event_results.append({
                "folder": folder,
                "model": modello,
                "gof_event": float(gof_event),
                "local_results": local_results  # attenzione: deve essere JSON-serializzabile
            })

    output = {
        "events": event_results,
        "models": {
            modello: {
                "mean_gof": sum(vals) / len(vals),
                "n": len(vals),
                "gof_list": vals
            }
            for modello, vals in model_gof.items()
        }
    }

    with open("gof_results.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)

    all_models_stats = []

    for modello, gofs in model_gof.items():
        mean_gof = np.mean(gofs)
        all_models_stats.append((modello, mean_gof, len(gofs)))

    all_models_stats.sort(key=lambda x: x[1], reverse=True)

    with open("result.txt", "w") as f:

        f.write("CLASSIFICA MODELLI\n")
        f.write("==========================\n\n")

        for modello, mean_gof, n in all_models_stats:
            f.write(f"{modello:80s} GOF: {mean_gof:.3f} (N={n})\n")

        best_model, best_gof_model, _ = all_models_stats[0]

        f.write("\n==========================\n")
        f.write("BEST MODEL\n")
        f.write("==========================\n")
        f.write(f"Best model: {best_model}\n")
        f.write(f"Best GOF:   {best_gof_model:.3f}\n")
