"""Download MSLP and 10 m wind of one run of each open global model and put them on one global 0.25° grid
(lat 50 S..50 N, lon 0..359.75 E). Used by cyclone_mme.py (the MME tab of the Cyclones page).

Models (all open data, 00 and 12 UTC runs, every 6 h):
  ifs   ECMWF IFS HRES 0.25°  (ECMWF open data, CC BY 4.0)               0-240 h
  aifs  ECMWF AIFS 0.25°      (ECMWF open data, CC BY 4.0)               0-240 h
  gfs   NCEP GFS 0.25°        (NOAA, public domain; AWS Open Data copy)  0-240 h
  aigfs NOAA AIGFS 0.25°      (NOAA, public domain; AWS Open Data copy)  0-240 h
  cmc   CMC GDPS 15 km        (ECCC MSC Datamart, OGL - Canada)          0-240 h
  icon  DWD ICON 13 km        (DWD Open Data)                            0-180 h
  ukmo  Met Office Global 10 km (Met Office on AWS, CC BY-SA 4.0)        0-168 h

fetch(model, run) -> dict(steps=[...], msl=[nt,ny,nx] hPa, u=[...] m/s, v=[...] m/s)  or None if the run is not complete yet.
"""
import os, io, bz2, tempfile, datetime as dt, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np

LAT = np.arange(-50, 50.001, 0.25)
LON = np.arange(0, 360, 0.25)
UA = {"User-Agent": "gwcharts"}

STEPS = {"ifs": range(0, 241, 6), "aifs": range(0, 241, 6), "gfs": range(0, 241, 6), "aigfs": range(0, 241, 6),
         "cmc": range(0, 241, 6), "icon": range(0, 181, 6), "ukmo": range(0, 169, 6)}

ECMWF = "https://data.ecmwf.int/forecasts/{d}/{h}z/{m}/0p25/oper/{d}{h}0000-{s}h-oper-fc"
GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{s:03d}"
AIGFS = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com/aigfs.{d}/{h}/model/atmos/grib2/aigfs.t{h}z.sfc.f{s:03d}.grib2"
CMC = "https://dd.weather.gc.ca/{d}/WXO-DD/model_gdps/15km/{h}/{s:03d}/{d}T{h}Z_MSC_GDPS_{v}_LatLon0.15_PT{s:03d}H.grib2"
ICON = "https://opendata.dwd.de/weather/nwp/icon/grib/{h}/{low}/icon_global_icosahedral_single-level_{r}_{s:03d}_{up}.grib2.bz2"
ICON_C = "https://opendata.dwd.de/weather/nwp/icon/grib/{h}/{low}/icon_global_icosahedral_time-invariant_{r}_{up}.grib2.bz2"
UKMO = ("https://met-office-atmospheric-model-data.s3.eu-west-2.amazonaws.com/global-deterministic-10km/"
        "{r}/{v}-PT{s:04d}H00M-{f}.nc")


# ---------------------------------------------------------------- http
def get(u, rng=None, tries=8):
    for t in range(tries):
        try:
            h = dict(UA)
            if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=120) as r:
                return r.read()
        except Exception:
            if t == tries - 1: raise
            import time; time.sleep(3 * (t + 1))


def exists(u, tries=3):
    import time
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers=UA), timeout=30) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code in (403, 404): return False
        except Exception:
            pass
        time.sleep(3)
    return False


# ---------------------------------------------------------------- regridding
def to_grid(a, lat, lon):
    """Bilinear interpolation of a global regular lat/lon field a[lat, lon] (any order, 0..360 or -180..180) to LAT x LON."""
    from scipy.interpolate import RegularGridInterpolator
    lon = np.asarray(lon, float) % 360
    lat = np.asarray(lat, float)
    jj = np.where((lat >= LAT[0] - 1) & (lat <= LAT[-1] + 1))[0]
    sub = a[jj]; la = lat[jj]
    oj, oi = np.argsort(la), np.argsort(lon)
    la, lo, sub = la[oj], lon[oi], sub[np.ix_(oj, oi)]
    lo = np.concatenate([lo[-1:] - 360, lo, lo[:1] + 360])               # wrap around 0/360
    sub = np.concatenate([sub[:, -1:], sub, sub[:, :1]], axis=1)
    f = RegularGridInterpolator((la, lo), sub, bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(LAT, LON, indexing="ij")
    return f((LA, LO)).astype(np.float32)


def grib_messages(blob):
    """Every GRIB message in a byte string -> list of (shortName, step, field on our grid)."""
    import eccodes
    out = []
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tf:
        tf.write(blob); tf.flush()
        with open(tf.name, "rb") as f:
            while True:
                g = eccodes.codes_grib_new_from_file(f)
                if g is None: break
                try:
                    ni, nj = eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")
                    la0, la1 = eccodes.codes_get(g, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(g, "latitudeOfLastGridPointInDegrees")
                    lo0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
                    di = eccodes.codes_get(g, "iDirectionIncrementInDegrees")
                    vals = eccodes.codes_get_values(g).reshape(nj, ni)
                    lat = np.linspace(la0, la1, nj); lon = lo0 + di * np.arange(ni)
                    name = eccodes.codes_get(g, "shortName")
                    step = int(eccodes.codes_get(g, "endStep")) if eccodes.codes_is_defined(g, "endStep") else 0
                    out.append((name, step, to_grid(vals, lat, lon)))
                finally:
                    eccodes.codes_release(g)
    return out


# ---------------------------------------------------------------- availability
def ready(model, run):
    d, h, r = run.strftime("%Y%m%d"), run.strftime("%H"), run.strftime("%Y%m%d%H")
    s = max(STEPS[model])
    if model in ("ifs", "aifs"):
        return exists(ECMWF.format(d=d, h=h, m="ifs" if model == "ifs" else "aifs-single", s=s) + ".index")
    if model == "gfs": return exists(GFS.format(d=d, h=h, s=s) + ".idx")
    if model == "aigfs": return exists(AIGFS.format(d=d, h=h, s=s) + ".idx")
    if model == "cmc": return exists(CMC.format(d=d, h=h, s=s, v="Pressure_MSL"))
    if model == "icon": return exists(ICON.format(h=h, low="pmsl", r=r, s=s, up="PMSL"))
    if model == "ukmo":
        v = (run + dt.timedelta(hours=s)).strftime("%Y%m%dT%H%MZ")
        return exists(UKMO.format(r=run.strftime("%Y%m%dT%H%MZ"), v=v, s=s, f="pressure_at_mean_sea_level"))
    raise ValueError(model)


# ---------------------------------------------------------------- per model
def _assemble(steps, got):
    """got: {(step, 'msl'|'u'|'v'): field} -> dict of arrays (steps with any field missing are dropped)."""
    ok = [s for s in steps if all((s, k) in got for k in ("msl", "u", "v"))]
    if len(ok) < len(steps) * 0.8: raise RuntimeError(f"only {len(ok)} of {len(steps)} steps complete")
    return dict(steps=ok, msl=np.stack([got[(s, "msl")] for s in ok]),
                u=np.stack([got[(s, "u")] for s in ok]), v=np.stack([got[(s, "v")] for s in ok]))


def _ecmwf(model, run):
    """Byte ranges from the ECMWF open-data .index files (JSON lines), so only msl/10u/10v are transferred."""
    import json
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    m = "ifs" if model == "ifs" else "aifs-single"
    want = {"msl": "msl", "10u": "u", "10v": "v"}

    def one(s):
        base = ECMWF.format(d=d, h=h, m=m, s=s)
        res = {}
        for line in get(base + ".index").decode().splitlines():
            e = json.loads(line)
            if e.get("param") in want and e.get("levtype", "sfc") == "sfc":
                blob = get(base + ".grib2", (e["_offset"], e["_offset"] + e["_length"] - 1))
                for name, _, f in grib_messages(blob):
                    k = want[e["param"]]; res[(s, k)] = f / 100 if k == "msl" else f
        return res
    got = {}
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, STEPS[model]): got.update(r)
    return _assemble(list(STEPS[model]), got)


def _noaa(model, run):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    pat = {"msl": ":PRMSL:mean sea level:", "u": ":UGRD:10 m above ground:", "v": ":VGRD:10 m above ground:"}
    tmpl = GFS if model == "gfs" else AIGFS

    def one(s):
        u = tmpl.format(d=d, h=h, s=s)
        lines = get(u + ".idx").decode().splitlines(); offs = [int(l.split(":")[1]) for l in lines]
        res = {}
        for n, l in enumerate(lines):
            for k, p in pat.items():
                if p in l and (s, k) not in res:
                    end = offs[n + 1] - 1 if n + 1 < len(lines) else ""
                    f = grib_messages(get(u, (offs[n], end)))[0][2]
                    res[(s, k)] = f / 100 if k == "msl" else f
        return res
    got = {}
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, STEPS[model]): got.update(r)
    return _assemble(list(STEPS[model]), got)


def _cmc(run):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    names = {"msl": "Pressure_MSL", "u": "WindU_AGL-10m", "v": "WindV_AGL-10m"}

    def one(job):
        s, k = job
        f = grib_messages(get(CMC.format(d=d, h=h, s=s, v=names[k])))[0][2]
        return (s, k), (f / 100 if k == "msl" else f)
    jobs = [(s, k) for s in STEPS["cmc"] for k in names]
    with ThreadPoolExecutor(4) as ex:
        got = dict(ex.map(one, jobs))
    return _assemble(list(STEPS["cmc"]), got)


def _icon(run):
    import eccodes
    from scipy.spatial import cKDTree
    h, r = run.strftime("%H"), run.strftime("%Y%m%d%H")

    def raw(blob):
        g = eccodes.codes_new_from_message(bz2.decompress(blob))
        try: return eccodes.codes_get_values(g)
        finally: eccodes.codes_release(g)
    clat = raw(get(ICON_C.format(h=h, low="clat", r=r, up="CLAT")))
    clon = raw(get(ICON_C.format(h=h, low="clon", r=r, up="CLON")))
    keep = np.where((clat > LAT[0] - 2) & (clat < LAT[-1] + 2))[0]

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    LA, LO = np.meshgrid(LAT, LON, indexing="ij")
    _, nn = cKDTree(xyz(clat[keep], clon[keep])).query(xyz(LA.ravel(), LO.ravel()))
    idx = keep[nn].reshape(LA.shape)
    names = {"msl": ("pmsl", "PMSL"), "u": ("u_10m", "U_10M"), "v": ("v_10m", "V_10M")}

    def one(job):
        s, k = job
        vals = raw(get(ICON.format(h=h, low=names[k][0], r=r, s=s, up=names[k][1])))
        f = vals[idx].astype(np.float32)
        return (s, k), (f / 100 if k == "msl" else f)
    jobs = [(s, k) for s in STEPS["icon"] for k in names]
    with ThreadPoolExecutor(10) as ex:
        got = dict(ex.map(one, jobs))
    return _assemble(list(STEPS["icon"]), got)


def _ukmo(run):
    import h5py
    rr = run.strftime("%Y%m%dT%H%MZ")
    files = {"msl": "pressure_at_mean_sea_level", "ws": "wind_speed_at_10m", "wd": "wind_direction_at_10m"}

    def read(blob):
        with h5py.File(io.BytesIO(blob), "r") as f:
            var = [k for k in f.keys() if f[k].ndim == 2 and "bnds" not in k][0]
            ds = f[var]; a = ds[()].astype(np.float64)
            fill = ds.attrs.get("_FillValue")
            if fill is not None: a[a == np.asarray(fill).ravel()[0]] = np.nan
            sc = ds.attrs.get("scale_factor"); of = ds.attrs.get("add_offset")
            if sc is not None: a = a * float(np.asarray(sc).ravel()[0])
            if of is not None: a = a + float(np.asarray(of).ravel()[0])
            return to_grid(a, f["latitude"][()], f["longitude"][()])

    def one(s):
        v = (run + dt.timedelta(hours=s)).strftime("%Y%m%dT%H%MZ")
        x = {k: read(get(UKMO.format(r=rr, v=v, s=s, f=fn))) for k, fn in files.items()}
        d = np.radians(x["wd"])                                    # direction the wind blows FROM
        return {(s, "msl"): x["msl"] / 100, (s, "u"): -x["ws"] * np.sin(d), (s, "v"): -x["ws"] * np.cos(d)}
    got = {}
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(one, STEPS["ukmo"]): got.update(r)
    return _assemble(list(STEPS["ukmo"]), got)


def fetch(model, run):
    if not ready(model, run): return None
    if model in ("ifs", "aifs"): return _ecmwf(model, run)
    if model in ("gfs", "aigfs"): return _noaa(model, run)
    if model == "cmc": return _cmc(run)
    if model == "icon": return _icon(run)
    if model == "ukmo": return _ukmo(run)
    raise ValueError(model)


if __name__ == "__main__":
    import sys, time
    m, r = sys.argv[1], dt.datetime.strptime(sys.argv[2], "%Y%m%d%H")
    t0 = time.time(); x = fetch(m, r)
    if x is None: print("not ready")
    else:
        os.makedirs("mme_cache", exist_ok=True)
        np.savez_compressed(f"mme_cache/{m}_{sys.argv[2]}.npz", **x)
        print(m, len(x["steps"]), "steps", f"{time.time() - t0:.0f} s", "msl range", np.nanmin(x["msl"]), np.nanmax(x["msl"]),
              "max wind", np.nanmax(np.hypot(x["u"], x["v"])))
