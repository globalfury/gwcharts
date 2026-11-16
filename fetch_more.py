"""Download one run of Environment Canada's GDPS (CMC) or DWD's ICON and put every field we chart
on the common 0.25° grid (lat 90 -> -90, lon -180 -> 179.75), saved as one .npy file per field.

Usage: python fetch_more.py cmc|icon [YYYYMMDDHH]    (default: newest complete 00Z/12Z run)
Writes data/<model>/<run>/<field>.npy (steps x 721 x 1440, float32) + meta.json

Sources: MSC Datamart (Open Government Licence - Canada) and DWD Open Data (attribution).
"""
import os, sys, bz2, json, datetime as dt, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

LAT = np.linspace(90, -90, 721)
LON = np.arange(-180, 180, 0.25)
STEPS = {"cmc": list(range(0, 241, 6)), "icon": list(range(0, 181, 6))}

CMC = "https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/{h}/{s:03d}/{d}T{h}Z_MSC_GDPS_{v}_LatLon0.15_PT{s:03d}H.grib2"
CMC_VARS = {   # our name: (MSC name, scale, offset)
    "msl": ("Pressure_MSL", 0.01, 0), "tp": ("Precip-Accum_Sfc", 1, 0),
    "u10": ("WindU_AGL-10m", 1.94384, 0), "v10": ("WindV_AGL-10m", 1.94384, 0), "mucape": ("CAPE_Sfc", 1, 0),
    "gh850": ("GeopotentialHeight_IsbL-0850", None, 0), "gh500": ("GeopotentialHeight_IsbL-0500", None, 0),
    "gh200": ("GeopotentialHeight_IsbL-0200", None, 0), "t850": ("AirTemp_IsbL-0850", 1, -273.15),
    "u850": ("WindU_IsbL-0850", 1.94384, 0), "v850": ("WindV_IsbL-0850", 1.94384, 0),
    "u500": ("WindU_IsbL-0500", 1.94384, 0), "v500": ("WindV_IsbL-0500", 1.94384, 0),
    "u200": ("WindU_IsbL-0200", 1.94384, 0), "v200": ("WindV_IsbL-0200", 1.94384, 0),
    "vo500": ("RelativeVorticity_IsbL-0500", 1e5, 0),
}
ICON = "https://opendata.dwd.de/weather/nwp/icon/grib/{h}/{low}/icon_global_icosahedral_{kind}_{run}_{s:03d}{lev}_{up}.grib2.bz2"
ICON_VARS = {  # our name: (variable, level or None, scale, offset)
    "msl": ("pmsl", None, 0.01, 0), "tp": ("tot_prec", None, 1, 0), "u10": ("u_10m", None, 1.94384, 0),
    "v10": ("v_10m", None, 1.94384, 0), "mucape": ("cape_ml", None, 1, 0), "tcwv": ("tqv", None, 1, 0),
    "gh850": ("fi", 850, 1 / 9.80665, 0), "gh500": ("fi", 500, 1 / 9.80665, 0), "gh200": ("fi", 200, 1 / 9.80665, 0),
    "t850": ("t", 850, 1, -273.15), "u850": ("u", 850, 1.94384, 0), "v850": ("v", 850, 1.94384, 0),
    "u500": ("u", 500, 1.94384, 0), "v500": ("v", 500, 1.94384, 0), "u200": ("u", 200, 1.94384, 0), "v200": ("v", 200, 1.94384, 0),
}


def get(u, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "gwcharts"}), timeout=120) as r:
                return r.read()
        except Exception:
            if t == tries - 1: raise


def decode(blob):
    """First GRIB message in a byte string -> (values 1-D, keys dict)."""
    import eccodes
    g = eccodes.codes_new_from_message(blob)
    try:
        vals = eccodes.codes_get_values(g)
        keys = {k: eccodes.codes_get(g, k) for k in ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
                                                      "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees", "units")
                if eccodes.codes_is_defined(g, k)}
    finally:
        eccodes.codes_release(g)
    return vals, keys


# ------------------------------------------------------------------ regridding to 0.25°
def regular_to_quarter(vals, k):
    """Bilinear interpolation from a regular lat-lon grid (e.g. CMC 0.15°) to our 0.25° grid."""
    from scipy.ndimage import map_coordinates
    ni, nj = int(k["Ni"]), int(k["Nj"])
    a = vals.reshape(nj, ni)
    lat0, lon0 = k["latitudeOfFirstGridPointInDegrees"], k["longitudeOfFirstGridPointInDegrees"]
    di, dj = k["iDirectionIncrementInDegrees"], k["jDirectionIncrementInDegrees"]
    if lat0 < 0: dj = abs(dj)                                         # rows run south -> north
    else: dj = -abs(dj)
    LA, LO = np.meshgrid(LAT, LON, indexing="ij")
    jj = (LA - lat0) / dj
    ii = ((LO - lon0) % 360) / di
    a = np.concatenate([a, a[:, :1]], axis=1)                         # wrap round in longitude
    return map_coordinates(a, [jj, ii], order=1, mode="nearest").astype(np.float32)


_ICON_IDX = {}


def icon_index(run):
    """Nearest ICON cell for every 0.25° grid point, from the run's cell-centre coordinates."""
    if "idx" in _ICON_IDX: return _ICON_IDX["idx"]
    from scipy.spatial import cKDTree
    h = run.strftime("%H"); r = run.strftime("%Y%m%d%H")
    clat, _ = decode(bz2.decompress(get(f"https://opendata.dwd.de/weather/nwp/icon/grib/{h}/clat/icon_global_icosahedral_time-invariant_{r}_CLAT.grib2.bz2")))
    clon, _ = decode(bz2.decompress(get(f"https://opendata.dwd.de/weather/nwp/icon/grib/{h}/clon/icon_global_icosahedral_time-invariant_{r}_CLON.grib2.bz2")))
    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    tree = cKDTree(xyz(clat, clon))
    LA, LO = np.meshgrid(LAT, LON, indexing="ij")
    _, idx = tree.query(xyz(LA.ravel(), LO.ravel()), workers=-1)
    _ICON_IDX["idx"] = idx.reshape(LA.shape).astype(np.int32)
    return _ICON_IDX["idx"]


# ------------------------------------------------------------------ runs
def latest_run(model):
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    run = now.replace(hour=now.hour // 12 * 12)
    last = STEPS[model][-1]
    for _ in range(6):
        try:
            u = file_url(model, run, "msl", last); urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers={"User-Agent": "gwcharts"}), timeout=30)
            return run
        except Exception:
            run -= dt.timedelta(hours=12)
    raise SystemExit(f"no complete {model.upper()} run found")


def file_url(model, run, name, step):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    if model == "cmc":
        return CMC.format(d=d, h=h, s=step, v=CMC_VARS[name][0])
    var, lev = ICON_VARS[name][:2]
    return ICON.format(h=h, low=var, run=run.strftime("%Y%m%d%H"), s=step, kind="pressure-level" if lev else "single-level",
                       lev=f"_{lev}" if lev else "", up=var.upper())


def fetch(model, run=None, root=None):
    run = dt.datetime.strptime(run, "%Y%m%d%H") if run else latest_run(model)
    out = os.path.join(root or f"data/{model}", run.strftime("%Y%m%d%H")); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "meta.json")): return out, run
    steps, names = STEPS[model], list(CMC_VARS if model == "cmc" else ICON_VARS)
    idx = icon_index(run) if model == "icon" else None
    arrays = {n: np.lib.format.open_memmap(os.path.join(out, f"{n}.npy.part"), mode="w+", dtype=np.float32, shape=(len(steps), 721, 1440)) for n in names}

    def one(job):
        k, step, name = job
        if name == "tp" and step == 0:
            arrays[name][k] = 0; return
        blob = get(file_url(model, run, name, step))
        if model == "icon": blob = bz2.decompress(blob)
        vals, keys = decode(blob)
        if model == "cmc":
            scale, off = CMC_VARS[name][1:]
            if scale is None:                                         # height comes in dam or m depending on product
                scale = 10.0 if str(keys.get("units", "")).lower() in ("dam", "dm") else 1.0
            grid = regular_to_quarter(vals, keys)
        else:
            scale, off = ICON_VARS[name][2:]
            grid = vals[idx].astype(np.float32)
        arrays[name][k] = grid * scale + off
    jobs = [(k, s, n) for k, s in enumerate(steps) for n in names]
    with ThreadPoolExecutor(12) as ex:
        for n, _ in enumerate(ex.map(one, jobs), 1):
            if n % 100 == 0: print(f"  {model}: {n}/{len(jobs)} files", flush=True)
    for n, a in arrays.items():
        a.flush(); del a
        os.replace(os.path.join(out, f"{n}.npy.part"), os.path.join(out, f"{n}.npy"))
    json.dump(dict(model=model, run=run.strftime("%Y%m%d%H"), steps=steps), open(os.path.join(out, "meta.json"), "w"))
    return out, run


if __name__ == "__main__":
    a = sys.argv[1:]
    print(*fetch(a[0], a[1] if len(a) > 1 else None))
