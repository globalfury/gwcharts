"""
map_publish.py — data for the Live Map (globalweathercharts.com/map), Windy-style animated wind.

Models:  gfs = NOAA GFS 0.25° (US public domain), AWS Open Data copy, byte ranges from the .idx files
         ifs = ECMWF IFS 0.25° open data (CC BY 4.0, attribution "ECMWF"), via ecmwf-opendata
For each forecast hour 0–240 h every 6 h and each level (10 m, 925 … 10 hPa) it stores one small
PNG on a 0.5° global grid (720 × 361, row 0 = 90°N, column 0 = 0°E):
    red   = u wind   green = v wind    (0…255 ↔ −LIM…+LIM m/s, LIM per level, in meta.json)
    blue  = temperature (0…255 ↔ −90…+50 °C; 2 m temperature on the 10 m level)

R2 (bucket gwcharts):
    map/<model>/<run>/<level>/<hhh>.png     e.g. map/gfs/2026112400/850/024.png
    map/<model>/<run>/meta.json             levels, limits, grid, steps (written after all images)
    map/<model>/latest.json                 {"run": "2026112400", ...}   (written last)
    map/places.json                         city list for the temperature labels
Keeps the newest 2 runs per model.

    python map_publish.py gfs                    # newest complete 00/12 UTC run, skip if already published
    python map_publish.py ifs 2026112400         # a given run
    python map_publish.py gfs --local out/       # test: write files to a folder, no upload
Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY for uploads (same secrets as the chart jobs).
"""
import argparse, datetime as dt, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import eccodes
from PIL import Image

import fetch_gfs as F

BUCKET = os.environ.get("R2_BUCKET", "gwcharts")
ROOT = "map/"
PREFIX = ROOT              # set to map/<model>/ in main()
RUN_HOURS = tuple(int(h) for h in os.environ.get("MAP_RUN_HOURS", "0,12").split(","))
STEPS = list(range(0, 241, 6))
# level id, label, approx. altitude, |u|,|v| limit (m/s)
LEVELS = [
    ("10m", "Surface (10 m)", "", 40),
    ("925", "925 hPa", "750 m", 60),
    ("850", "850 hPa", "1.5 km", 60),
    ("700", "700 hPa", "3 km · FL100", 80),
    ("500", "500 hPa", "5.5 km · FL180", 100),
    ("300", "300 hPa", "9 km · FL300", 128),
    ("250", "250 hPa", "10 km · FL340", 128),
    ("200", "200 hPa", "12 km · FL390", 128),
    ("150", "150 hPa", "13.5 km · FL450", 128),
    ("100", "100 hPa", "16 km · FL530", 128),
    ("70", "70 hPa", "18.5 km", 128),
    ("50", "50 hPa", "20.5 km", 128),
    ("30", "30 hPa", "24 km", 128),
    ("10", "10 hPa", "31 km", 128),
]
IFS_LEVELS = [l for l in LEVELS if l[0] not in ("70", "30")]   # ECMWF open data has no 70/30 hPa
T_MIN, T_MAX = -90.0, 50.0


def idx_pattern(level, var):
    if level == "10m":
        return ":TMP:2 m above ground:" if var == "TMP" else f":{var}:10 m above ground:"
    return f":{var}:{level} mb:"


def latest_run():
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    run = now.replace(hour=now.hour // 6 * 6)
    for _ in range(12):
        if run.hour in RUN_HOURS:
            try:
                F.get(F.url(run, STEPS[-1], idx=True))
                return run
            except Exception:
                pass
        run -= dt.timedelta(hours=6)
    raise SystemExit("no complete GFS run found")


def smooth_half(a):
    """1-2-1 filter in both directions (wraps in longitude), then every second point: no grainy aliasing."""
    a = (np.roll(a, 1, 1) + 2 * a + np.roll(a, -1, 1)) / 4.0
    up = np.vstack([a[:1], a[:-1]]); dn = np.vstack([a[1:], a[-1:]])
    a = (up + 2 * a + dn) / 4.0
    return a[::2, ::2]


def q(x, lo, hi):
    return np.clip(np.round((x - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG", optimize=True)
    return b.getvalue()


def do_step(run, step):
    """Download u, v, T for all levels at one step → {level: png bytes}."""
    lines = F.get(F.url(run, step, idx=True)).decode().splitlines()
    offs = [int(l.split(":")[1]) for l in lines]
    out = {}
    for lev, _, _, lim in LEVELS:
        f = {}
        for var in ("UGRD", "VGRD", "TMP"):
            pat = idx_pattern(lev, var)
            n = next(i for i, l in enumerate(lines) if pat in l)
            blob = F.get(F.url(run, step), (offs[n], offs[n + 1] - 1))
            g = eccodes.codes_new_from_message(blob)
            try:
                a = eccodes.codes_get_values(g).reshape(721, 1440)
            finally:
                eccodes.codes_release(g)
            f[var] = smooth_half(a)                    # 0.25° → 0.5°  (361 × 720), 1-2-1 smoothed
        img = np.dstack([q(f["UGRD"], -lim, lim), q(f["VGRD"], -lim, lim), q(f["TMP"] - 273.15, T_MIN, T_MAX)])
        out[lev] = png(img)
    return step, out


def ifs_step(run, step, client):
    """ECMWF open data: u, v, T at the IFS levels for one step → {level: png bytes}."""
    import tempfile
    tmp = tempfile.mkdtemp()
    fs, fp = os.path.join(tmp, "s.grib2"), os.path.join(tmp, "p.grib2")
    d, h = run.strftime("%Y%m%d"), run.hour
    levs = [int(l[0]) for l in IFS_LEVELS if l[0] != "10m"]
    for k in range(4):
        try:
            client.retrieve(date=d, time=h, type="fc", step=step, param=["10u", "10v", "2t"], target=fs)
            client.retrieve(date=d, time=h, type="fc", step=step, levtype="pl", levelist=levs, param=["u", "v", "t"], target=fp)
            break
        except Exception as e:
            if k == 3: raise
            time.sleep(10 * (k + 1))
    f = {}
    for path in (fs, fp):
        with open(path, "rb") as fh:
            while True:
                g = eccodes.codes_grib_new_from_file(fh)
                if g is None: break
                try:
                    sn = eccodes.codes_get(g, "shortName"); lt = eccodes.codes_get(g, "typeOfLevel")
                    lev = "10m" if lt != "isobaricInhPa" else str(eccodes.codes_get(g, "level"))
                    var = {"10u": "UGRD", "10v": "VGRD", "2t": "TMP", "u": "UGRD", "v": "VGRD", "t": "TMP"}.get(sn)
                    if var:
                        a = eccodes.codes_get_values(g).reshape(721, 1440)
                        a = np.roll(a, -720, axis=1)          # grid starts at 180°: move it to start at 0°E like GFS
                        f[(lev, var)] = smooth_half(a)
                finally:
                    eccodes.codes_release(g)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    out = {}
    for lev, _, _, lim in IFS_LEVELS:
        img = np.dstack([q(f[(lev, "UGRD")], -lim, lim), q(f[(lev, "VGRD")], -lim, lim), q(f[(lev, "TMP")] - 273.15, T_MIN, T_MAX)])
        out[lev] = png(img)
    return step, out


def ifs_latest(client):
    for _ in range(6):
        try:
            r = client.latest(type="fc", step=STEPS[-1], param="10u")
        except Exception:
            time.sleep(5); continue
        while r.hour not in RUN_HOURS:
            r -= dt.timedelta(hours=6)
        return r
    raise SystemExit("ECMWF open data not reachable")


def r2():
    import boto3
    from botocore.config import Config
    e = lambda k: os.environ[k].strip()
    return boto3.client("s3", endpoint_url=f"https://{e('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
                        aws_access_key_id=e("R2_ACCESS_KEY_ID"), aws_secret_access_key=e("R2_SECRET_ACCESS_KEY"),
                        region_name="auto", config=Config(retries={"max_attempts": 8, "mode": "adaptive"},
                                                          max_pool_connections=32))


class Sink:
    """Writes either to R2 or to a local folder (for tests)."""
    def __init__(self, local=None):
        self.local, self.s3 = local, (None if local else r2())

    def put(self, key, body, ctype, cache):
        if self.local:
            p = os.path.join(self.local, key); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(body); return
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ctype, CacheControl=cache)

    def get_json(self, key):
        try:
            if self.local:
                return json.load(open(os.path.join(self.local, key)))
            return json.loads(self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        except Exception:
            return None

    def runs(self):
        if self.local:
            d = os.path.join(self.local, PREFIX)
            return sorted(x for x in os.listdir(d) if x.isdigit()) if os.path.isdir(d) else []
        out, tok = set(), None
        while True:
            kw = dict(Bucket=BUCKET, Prefix=PREFIX, Delimiter="/")
            if tok: kw["ContinuationToken"] = tok
            r = self.s3.list_objects_v2(**kw)
            out |= {p["Prefix"][len(PREFIX):-1] for p in r.get("CommonPrefixes", []) if p["Prefix"][len(PREFIX):-1].isdigit()}
            if not r.get("IsTruncated"): break
            tok = r["NextContinuationToken"]
        return sorted(out)

    def delete_run(self, run):
        if self.local:
            import shutil; shutil.rmtree(os.path.join(self.local, PREFIX, run), ignore_errors=True); return
        keys, tok = [], None
        while True:
            kw = dict(Bucket=BUCKET, Prefix=f"{PREFIX}{run}/")
            if tok: kw["ContinuationToken"] = tok
            r = self.s3.list_objects_v2(**kw)
            keys += [{"Key": o["Key"]} for o in r.get("Contents", [])]
            if not r.get("IsTruncated"): break
            tok = r["NextContinuationToken"]
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys[i:i + 1000], "Quiet": True})


def main():
    global PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["gfs", "ifs"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--local")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--steps", help="test only: comma list of steps")
    a = ap.parse_args()
    PREFIX = f"{ROOT}{a.model}/"
    F.MODEL = "gfs"
    client = None
    if a.model == "ifs":
        from ecmwf.opendata import Client
        src = os.environ.get("ECMWF_SOURCE", "ecmwf")
        client = Client(source=src, model="ifs", resol="0p25")
        run = dt.datetime.strptime(a.run, "%Y%m%d%H") if a.run else ifs_latest(client)
        levels = IFS_LEVELS
    else:
        run = dt.datetime.strptime(a.run, "%Y%m%d%H") if a.run else latest_run()
        levels = LEVELS
    rs = run.strftime("%Y%m%d%H")
    sink = Sink(a.local)
    latest = sink.get_json(PREFIX + "latest.json") or {}
    if latest.get("run") == rs and not a.force:
        print(a.model, "run", rs, "already published"); return
    steps = [int(x) for x in a.steps.split(",")] if a.steps else STEPS
    print(a.model.upper(), "run", rs, "—", len(steps), "steps ×", len(levels), "levels")
    t0 = time.time()

    def work(step):
        s_, imgs = ifs_step(run, step, client) if a.model == "ifs" else do_step(run, step)
        for lev, body in imgs.items():
            sink.put(f"{PREFIX}{rs}/{lev}/{s_:03d}.png", body, "image/png", "public, max-age=604800, immutable")
        return s_, sum(len(b) for b in imgs.values())

    with ThreadPoolExecutor(4 if a.model == "ifs" else 6) as ex:
        done = sorted(ex.map(work, steps))
    mb = sum(b for _, b in done) / 1e6
    print(f"{len(done) * len(levels)} images, {mb:.0f} MB, {time.time() - t0:.0f} s")

    name = "ECMWF IFS 0.25° (shown at 0.5°)" if a.model == "ifs" else "GFS 0.25° (shown at 0.5°)"
    src = "ECMWF open data (CC BY 4.0)" if a.model == "ifs" else "NOAA/NCEP GFS (US public domain)"
    meta = dict(model=name, short="ECMWF" if a.model == "ifs" else "GFS", source=src, run=rs,
                steps=steps, grid=dict(nx=720, ny=361, lon0=0.0, lat0=90.0, dx=0.5),
                temp=[T_MIN, T_MAX],
                levels=[dict(id=l, name=n, alt=h, lim=m) for l, n, h, m in levels],
                made=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"))
    sink.put(f"{PREFIX}{rs}/meta.json", json.dumps(meta).encode(), "application/json", "public, max-age=604800")
    # city list for the temperature labels (from places.json in this repo)
    here = os.path.dirname(os.path.abspath(__file__))
    pl = json.load(open(os.path.join(here, "places.json"), encoding="utf-8"))
    slim = {k: [[x[0], 0, round(x[2], 2), round(x[3], 2)] for x in v] for k, v in pl.items()}
    sink.put(ROOT + "places.json", json.dumps(slim, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
             "application/json", "public, max-age=86400")
    sink.put(PREFIX + "latest.json", json.dumps(dict(run=rs, made=meta["made"])).encode(),
             "application/json", "public, max-age=120")
    for old in sink.runs()[:-2]:                    # keep the newest 2 runs of this model
        if old != rs:
            sink.delete_run(old); print("deleted old run", old)
    print("published", a.model, rs)


if __name__ == "__main__":
    main()
