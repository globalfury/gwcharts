"""Download one run of an ensemble — ECMWF EPS (50 perturbed members, 0.25°, ECMWF open data, CC BY 4.0)
or NOAA GEFS (31 members: control + 30, 0.25°, public domain) — and reduce it to the fields we chart:

  msl.npy    ensemble-mean MSLP (hPa)
  rain.npy   ensemble-mean precipitation in the 24 h ending at each step (mm)
  p10.npy, p25.npy, p50.npy   probability (%) that the 24-h precipitation is more than 10 / 25 / 50 mm

Steps 24..240 h every 6 h, 00 and 12 UTC runs only. Only the GRIB messages we need are transferred
(byte ranges from the .index / .idx files), one forecast step at a time, so memory stays near 1.5 GB.

Usage: python fetch_ens.py eps|gefs [YYYYMMDDHH]      (default: newest complete 00Z/12Z run)
Writes data/<model>/<run>/<field>.npy (steps x 721 x 1440) + meta.json
"""
import os, sys, json, time, datetime as dt, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

STEPS = list(range(24, 241, 6))            # chart steps (24-h windows need 24 h of forecast)
NEED = list(range(6, 241, 6))              # steps downloaded
HOURS = (0, 12)
THRESH = (10, 25, 50)                       # mm in 24 h
NLAT, NLON = 721, 1440

EPS_SOURCES = ["https://data.ecmwf.int/forecasts", "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]
EPS_MEMBERS = 50
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]


def get(u, rng=None, tries=5):
    for t in range(tries):
        try:
            h = {"User-Agent": "gwcharts"}
            if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=90) as r:
                return r.read()
        except Exception:
            if t == tries - 1: raise
            time.sleep(2 * (t + 1))


def decode(blob):
    import eccodes
    g = eccodes.codes_new_from_message(blob)
    try:
        v = eccodes.codes_get_values(g).astype(np.float32).reshape(NLAT, NLON)
        lon0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
    finally:
        eccodes.codes_release(g)
    if abs(lon0) < 1e-6:                    # grid starts at 0°E -> move to -180..179.75
        v = np.roll(v, NLON // 2, axis=1)
    return v


# ------------------------------------------------------------------ ECMWF EPS
def eps_base(src, run, step):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    return f"{src}/{d}/{h}z/ifs/0p25/enfo/{d}{h}0000-{step}h-enfo-ef"


def eps_index(run, step):
    last = None
    for src in EPS_SOURCES:                 # ECMWF portal first, AWS mirror as fallback
        try:
            lines = get(eps_base(src, run, step) + ".index").decode().splitlines()
            return src, [json.loads(l) for l in lines]
        except Exception as e:
            last = e
    raise last


def eps_step(run, step, want_msl):
    src, idx = eps_index(run, step)
    params = ("msl", "tp") if want_msl else ("tp",)
    pick = {(j["param"], int(j["number"])): (j["_offset"], j["_offset"] + j["_length"] - 1)
            for j in idx if j.get("type") == "pf" and j["param"] in params}
    members = list(range(1, EPS_MEMBERS + 1))
    missing = [(p, m) for p in params for m in members if (p, m) not in pick]
    if missing: raise RuntimeError(f"step {step}: {len(missing)} messages missing")
    url = eps_base(src, run, step) + ".grib2"
    keys = [(p, m) for p in params for m in members]
    with ThreadPoolExecutor(12) as ex:
        blobs = list(ex.map(lambda k: get(url, pick[k]), keys))
    out = {p: np.empty((len(members), NLAT, NLON), np.float32) for p in params}
    for (p, m), b in zip(keys, blobs):
        v = decode(b)
        out[p][m - 1] = v / 100.0 if p == "msl" else v * 1000.0          # Pa -> hPa, m -> mm
    return out.get("msl"), out["tp"]                                     # tp = total since start


def eps_ready(run):
    try:
        _, idx = eps_index(run, 240)
    except Exception:
        return False
    return sum(1 for j in idx if j.get("type") == "pf" and j["param"] in ("msl", "tp")) >= 2 * EPS_MEMBERS


# ------------------------------------------------------------------ NOAA GEFS
def gefs_url(run, mem, step):
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    return f"{GEFS}/gefs.{d}/{h}/atmos/pgrb2sp25/{mem}.t{h}z.pgrb2s.0p25.f{step:03d}"


def gefs_member(run, mem, step, want_msl):
    u = gefs_url(run, mem, step)
    lines = get(u + ".idx").decode().splitlines()
    offs = [int(l.split(":")[1]) for l in lines]
    found = {}
    for n, l in enumerate(lines):
        end = offs[n + 1] - 1 if n + 1 < len(lines) else ""
        if ":APCP:surface:" in l and f"{step - 6}-{step} hour acc" in l: found["tp"] = (offs[n], end)
        if want_msl and ":PRMSL:mean sea level:" in l: found["msl"] = (offs[n], end)
    need = ("msl", "tp") if want_msl else ("tp",)
    if any(k not in found for k in need): raise RuntimeError(f"{mem} f{step:03d}: fields missing")
    return {k: get(u, found[k]) for k in need}


def gefs_step(run, step, want_msl):
    with ThreadPoolExecutor(12) as ex:
        blobs = list(ex.map(lambda m: gefs_member(run, m, step, want_msl), GEFS_MEMBERS))
    n = len(GEFS_MEMBERS)
    tp = np.empty((n, NLAT, NLON), np.float32); msl = np.empty_like(tp) if want_msl else None
    for i, b in enumerate(blobs):
        tp[i] = decode(b["tp"])                                         # kg m-2 = mm, 6-h bucket
        if want_msl: msl[i] = decode(b["msl"]) / 100.0
    return msl, tp


def gefs_ready(run):
    try:
        for m in (GEFS_MEMBERS[0], GEFS_MEMBERS[-1]):
            get(gefs_url(run, m, 240) + ".idx")
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ common
def latest_run(model):
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    run = now.replace(hour=now.hour // 12 * 12)
    ready = eps_ready if model == "eps" else gefs_ready
    for _ in range(6):
        if ready(run): return run
        run -= dt.timedelta(hours=12)
    raise SystemExit(f"no complete {model.upper()} run found")


def fetch(model, run=None, root=None):
    run = dt.datetime.strptime(run, "%Y%m%d%H") if run else latest_run(model)
    if run.hour not in HOURS: raise SystemExit("only 00Z and 12Z runs are charted")
    root = root or f"data/{model}"
    d = os.path.join(root, run.strftime("%Y%m%d%H")); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "meta.json")): return d, run
    one = eps_step if model == "eps" else gefs_step
    nk = len(STEPS)
    msl = np.zeros((nk, NLAT, NLON), np.float32); rain = np.zeros_like(msl)
    prob = {t: np.zeros((nk, NLAT, NLON), np.uint8) for t in THRESH}
    hist = {}                      # EPS: total precipitation per member; GEFS: 6-h buckets per member
    t0 = time.time(); members = None
    for s in NEED:
        m, p = one(run, s, s >= 24)
        members = p.shape[0]
        hist[s] = p
        for old in [k for k in hist if k < s - 24]: del hist[old]
        if s < 24: continue
        if model == "eps":
            r24 = p - hist[s - 24] if s - 24 in hist else p.copy()
        else:
            r24 = hist[s] + hist[s - 6] + hist[s - 12] + hist[s - 18]
        np.maximum(r24, 0, out=r24)
        k = STEPS.index(s)
        msl[k] = m.mean(axis=0); rain[k] = r24.mean(axis=0)
        for t in THRESH:
            prob[t][k] = np.rint(100.0 * (r24 > t).sum(axis=0) / members).astype(np.uint8)
        print(f"  {model} +{s} h done ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(d, "msl.npy"), msl); np.save(os.path.join(d, "rain.npy"), rain)
    for t in THRESH: np.save(os.path.join(d, f"p{t}.npy"), prob[t])
    json.dump(dict(model=model, run=run.strftime("%Y%m%d%H"), steps=STEPS, members=members, ensemble=True),
              open(os.path.join(d, "meta.json"), "w"))
    return d, run


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("eps", "gefs"): raise SystemExit(__doc__)
    print(*fetch(a[0], a[1] if len(a) > 1 else None))
