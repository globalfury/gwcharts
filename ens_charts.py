"""Ensemble charts (ECMWF EPS, NOAA GEFS) drawn with the same engine, regions, boundaries and style as charts.py.
charts.py itself is not changed: this script loads the ensemble fields written by fetch_ens.py and swaps in
its own list of charts.

Charts: mslp_pcpn  ensemble-mean MSLP + ensemble-mean 24-h precipitation
        prob10 / prob25 / prob50   probability of more than 10 / 25 / 50 mm in 24 h (+ mean MSLP)

Usage: python ens_charts.py <run_dir> [--regions india,global] [--steps 24,48] [--fields a,b] [--jobs 2]
Output: out/<eps|gefs>/<run>/<field>/<region>/<step:03d>.png + out/<model>/<run>/manifest.json
"""
import os, json, time, argparse, datetime as dt
import numpy as np
from scipy.ndimage import gaussian_filter
import charts
from charts import palette, at, PCP, REGIONS

LABELS = {
    "eps":  ("ECMWF EPS 0.25° (50 members)", "Data: ECMWF open data (CC BY 4.0)"),
    "gefs": ("NOAA GEFS 0.25° (31 members)", "Data: NOAA/NCEP GEFS"),
}

PROB = palette(["#d9ecf7", "#9fd0ee", "#5aaee0", "#2f86c6", "#3fae5a", "#a8d24a", "#f4d03f",
                "#f08a24", "#d93a26", "#9e1c6f"])
PROB_LEVELS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def prob_field(t):
    return dict(
        name=f"Probability of more than {t} mm in 24 h",
        title=f"Probability (%) of more than {t} mm precipitation in the 24 h to valid time, mean MSLP (hPa)",
        shade=at(f"p{t}"), levels=PROB_LEVELS, cmap=PROB, unit=f"Probability of more than {t} mm in 24 h (% of members)",
        extend="neither", contour=at("msl"), cint=4, cint_global=8, cmask=3000, ccolor="#3a3a3a")


ENS_FIELDS = {
    "mslp_pcpn": dict(
        name="Ensemble mean MSLP & 24-h precipitation",
        title="Ensemble mean MSLP (hPa), mean precipitation in the 24 h to valid time (mm)",
        shade=at("rain"), levels=[1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200],
        cmap=palette(PCP, over="#e9c8f5"), unit="Ensemble-mean 24-h precipitation (mm)", extend="max",
        contour=at("msl"), cint=2, cint_global=4, cmask=3000, hl=True),
    "prob10": prob_field(10),
    "prob25": prob_field(25),
    "prob50": prob_field(50),
}


def load(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    D = charts.DATA
    D.update(model=meta["model"], steps=meta["steps"], init=dt.datetime.strptime(meta["run"], "%Y%m%d%H"),
             lat=np.linspace(90, -90, 721), lon=np.arange(-180, 180, 0.25), orog=charts.static_orography())
    D["label"], D["credit"] = LABELS[meta["model"]]
    for f in ("msl", "rain", "p10", "p25", "p50"):
        D[f] = np.load(os.path.join(run_dir, f + ".npy"))
    # same MSLP treatment as charts.py: blend towards a heavily smoothed field over high ground
    w = np.clip((D["orog"] - 500.0) / 1500.0, 0, 1)
    for k in range(D["msl"].shape[0]):
        p = D["msl"][k]
        D["msl"][k] = (1 - w) * gaussian_filter(p, 1.0) + w * gaussian_filter(p, 6.0)
    charts.FIELDS = ENS_FIELDS                      # the drawing code reads its chart list from here
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--regions"); ap.add_argument("--steps"); ap.add_argument("--fields")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    t0 = time.time(); meta = load(a.run_dir); charts.boundaries()
    D = charts.DATA; run = meta["run"]; model = meta["model"]
    regions = a.regions.split(",") if a.regions else list(REGIONS)
    steps = [int(s) for s in a.steps.split(",")] if a.steps else D["steps"]
    fields = a.fields.split(",") if a.fields else list(ENS_FIELDS)
    jobs = [(f, r, s, os.path.join(a.out, model, run, f, r, f"{s:03d}.png")) for f in fields for r in regions for s in steps]
    print(f"{model} {run}: {len(jobs)} charts, data loaded in {time.time() - t0:.0f}s", flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for n, _ in enumerate(pool.imap_unordered(charts._job, jobs, chunksize=4), 1):
            if n % 200 == 0: print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    man = dict(model=model, label=D["label"], run=run, init=D["init"].isoformat() + "Z", steps=D["steps"],
               members=meta.get("members"), ensemble=True,
               fields={f: ENS_FIELDS[f]["name"] for f in ENS_FIELDS}, regions={r: REGIONS[r]["name"] for r in REGIONS})
    with open(os.path.join(a.out, model, run, "manifest.json"), "w") as fh: json.dump(man, fh, indent=1)
    print(f"done {len(jobs)} charts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
