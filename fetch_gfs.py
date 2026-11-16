"""Download the fields we chart from one NOAA run (public domain) — GFS 0.25° or AIGFS (NOAA's AI model) —
using byte ranges from the AWS Open Data copies, so only the needed GRIB messages are transferred.
Usage: python fetch_gfs.py [YYYYMMDDHH] [gfs|aigfs]   (default: newest complete GFS run)
Writes data/<model>/<run>/gfs.grib2 + index.json (one entry per message, in file order)."""
import os, sys, json, datetime as dt, urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
# our name -> exact ":VAR:LEVEL:" text in the .idx line
WANT = {
    "msl": ":PRMSL:mean sea level:", "u10": ":UGRD:10 m above ground:", "v10": ":VGRD:10 m above ground:",
    "mucape": ":CAPE:255-0 mb above ground:", "tcwv": ":PWAT:entire atmosphere (considered as a single layer):",
    "gh850": ":HGT:850 mb:", "gh500": ":HGT:500 mb:", "gh200": ":HGT:200 mb:", "t850": ":TMP:850 mb:",
    "u850": ":UGRD:850 mb:", "v850": ":VGRD:850 mb:", "u500": ":UGRD:500 mb:", "v500": ":VGRD:500 mb:",
    "u200": ":UGRD:200 mb:", "v200": ":VGRD:200 mb:", "absv500": ":ABSV:500 mb:",
}
STEPS = list(range(0, 241, 6))
AI_BUCKET = "https://noaa-nws-graphcastgfs-pds.s3.amazonaws.com"
MODEL = "gfs"


def urls(run, step):
    """GRIB files that hold the fields for one step (AIGFS splits surface and pressure-level fields)."""
    d, h = run.strftime("%Y%m%d"), run.strftime("%H")
    if MODEL == "aigfs":
        base = f"{AI_BUCKET}/aigfs.{d}/{h}/model/atmos/grib2/aigfs.t{h}z"
        return [f"{base}.sfc.f{step:03d}.grib2", f"{base}.pres.f{step:03d}.grib2"]
    return [f"{BUCKET}/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{step:03d}"]


def url(run, step, idx=False):
    return urls(run, step)[-1] + (".idx" if idx else "")


def get(u, rng=None, tries=4):
    for t in range(tries):
        try:
            req = urllib.request.Request(u, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if t == tries - 1: raise


def latest_run(model=None, hours=(0, 6, 12, 18)):
    global MODEL
    if model: MODEL = model
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    run = now.replace(hour=now.hour // 6 * 6)
    for _ in range(8):
        if run.hour not in hours:
            run -= dt.timedelta(hours=6); continue
        try:
            get(url(run, STEPS[-1], idx=True)); return run
        except Exception:
            run -= dt.timedelta(hours=6)
    raise SystemExit(f"no complete {MODEL.upper()} run found")


def step_messages(run, step):
    picks, seen = [], set()
    for u in urls(run, step):
        lines = get(u + ".idx").decode().splitlines()
        offs = [int(l.split(":")[1]) for l in lines]
        for n, l in enumerate(lines):
            end = offs[n + 1] - 1 if n + 1 < len(lines) else ""
            for key, pat in WANT.items():
                if pat in l and key not in seen: picks.append((key, u, offs[n], end)); seen.add(key)
            if step == 0 and ":HGT:surface:" in l and "orog" not in seen: picks.append(("orog", u, offs[n], end)); seen.add("orog")
            tot = f":APCP:surface:0-{step // 24} day acc" if step % 24 == 0 else f":APCP:surface:0-{step} hour acc"
            if step > 0 and tot in l and "tp" not in seen: picks.append(("tp", u, offs[n], end)); seen.add("tp")
    return step, [(key, get(u, (a, b))) for key, u, a, b in picks]


def fetch(run=None, root=None, model="gfs"):
    global MODEL
    MODEL = model; root = root or f"data/{model}"
    run = dt.datetime.strptime(run, "%Y%m%d%H") if run else latest_run()
    d = os.path.join(root, run.strftime("%Y%m%d%H")); os.makedirs(d, exist_ok=True)
    gpath = os.path.join(d, "gfs.grib2")
    if os.path.exists(os.path.join(d, "index.json")): return d, run
    with ThreadPoolExecutor(8) as ex:
        results = sorted(ex.map(lambda s: step_messages(run, s), STEPS))
    index = []
    with open(gpath + ".part", "wb") as fh:
        for step, msgs in results:
            for key, blob in msgs:
                fh.write(blob); index.append([step, key])
    os.replace(gpath + ".part", gpath)
    json.dump(dict(model=model, run=run.strftime("%Y%m%d%H"), messages=index), open(os.path.join(d, "index.json"), "w"))
    return d, run


if __name__ == "__main__":
    a = sys.argv[1:]
    print(*fetch(a[0] if a and a[0].isdigit() else None, model=a[-1] if a and not a[-1].isdigit() else "gfs"))
