"""
waves_publish.py — "ECMWF MJO-Waves" tab (globalweathercharts.com/waves).

NCICS-style maps of 5-day OLR anomalies with the MJO, Kelvin waves, equatorial Rossby (ER) waves and the
low-frequency background picked out by space-time (wavenumber-frequency) filtering, as in
Wheeler & Weickmann (2001) and NCICS (Schreck), but with ECMWF ENS forecasts in place of CFS.

Data
  Observed : NOAA CPC Blended OLR (CBO V1), 1° daily (00–00 UTC) and its 1991-2020 daily climatology
             https://ftp.cpc.ncep.noaa.gov/precip/CBO_V1/1deg-DLY/   (US Government, public domain)
             It arrives about 2 days late.
  Forecast : ECMWF ENS 00 UTC, 50 perturbed members, top net thermal radiation (ttr, OLR = −ttr),
             daily means for days 1–15, ensemble mean (ECMWF open data, CC BY 4.0).
  Days between the last observed day and the run date are filled with the day-1 ENS forecast of the
  run started on that day (kept in R2 waves/state/day1/).
  Bias: ECMWF OLR minus CBO OLR on day 1 is tracked as a running mean (about 30 days, smoothed in space)
  and taken off every forecast day, so forecast anomalies are measured against the same climatology.

Method
  Last ~720 days of anomalies (observed + gap days + 15 forecast days), then zeros (= climatology) as in
  Wheeler & Weickmann. 2-D FFT in time and longitude at every latitude 39.5°S–39.5°N, keep the band of
  each mode, transform back. Bands (NCICS / Kiladis et al.):
     MJO     eastward wavenumbers 0–9, periods 30–96 days
     Kelvin  eastward 1–14, 2.5–20 days, equivalent depth 8–90 m       (plotted ×2, as NCICS)
     ER      westward 1–10, 9.7–48 days, n = 1 equivalent depth 8–90 m
     Low     all wavenumbers, periods longer than 120 days
  The filtered signal continues a few days past the last forecast day ("wave projection").

Output (R2 bucket gwcharts)
  waves/<YYYYMMDD>/map_<region>.png, hov_<band>.png, meta.json
  waves/index.json          newest first, written last
  waves/state/day1/<YYYYMMDD>.npy, waves/state/bias.npz      (working state)
Keeps 15 days of products.

  python waves_publish.py                       # newest complete 00 UTC ENS run, skip if already published
  python waves_publish.py --run 20261126 --force
  python waves_publish.py --local out/          # test: write everything to a folder, no upload
Environment: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (same secrets as the chart jobs);
WAVES_CACHE (default ./waves_cache) keeps the CBO files between runs (GitHub actions/cache).
"""
import argparse, datetime as dt, io, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BUCKET = os.environ.get("R2_BUCKET", "gwcharts")
BRAND = os.environ.get("SITE_NAME", "globalweathercharts.com")
KEEP_DAYS = 15
CACHE = os.environ.get("WAVES_CACHE", "waves_cache")

CBO = "https://ftp.cpc.ncep.noaa.gov/precip/CBO_V1/1deg-DLY"
EC_SOURCES = ["https://data.ecmwf.int/forecasts",                 # ECMWF portal (last ~4 days)
              "https://storage.googleapis.com/ecmwf-open-data",   # Google Cloud mirror (long history)
              "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]   # AWS mirror
MEMBERS = 50
FC_DAYS = 15
HIST_DAYS = 720                      # length of the record that is filtered
PAD_DAYS = 300                       # zeros after the last forecast day
LAT = np.arange(-39.5, 40.0, 1.0)    # 80 rows (CBO grid centres)
LON = np.arange(0.5, 360.0, 1.0)     # 360 columns
CBO_ROWS = slice(50, 130)            # CBO lat −89.5…89.5 → these rows are −39.5…39.5

# mode, kMin, kMax (eastward positive), tMin, tMax (days), hMin, hMax (m), kind
MODES = {
    "mjo":    (0, 9, 30.0, 96.0, None, None, None),
    "kelvin": (1, 14, 2.5, 20.0, 8.0, 90.0, "kelvin"),
    "er":     (-10, -1, 9.7, 48.0, 8.0, 90.0, "er"),
    "low":    (-999, 999, 120.0, np.inf, None, None, None),
}

REGIONS = {   # key: (title, lat range, lon range); lon may run past 360 or below 0
    "sasia": ("South Asia & surroundings", (10, 30), (40, 130)),
    "iomc":  ("Indian Ocean & Maritime Continent", (-20, 20), (40, 160)),
    "wpac":  ("West Pacific", (-20, 30), (100, 180)),
    "afr":   ("Africa & Arabian Sea", (-20, 30), (-20, 80)),
    "amer":  ("Americas & Atlantic", (-20, 30), (-130, -10)),
    "trop":  ("Global tropics", (-30, 30), (0, 360)),
}
HOVS = {      # key: (title, lat band)
    "eq":  ("Equator (5°S–5°N)", (-5, 5)),
    "ntr": ("Northern tropics (10°N–20°N)", (10, 20)),
}


# ============================================================ small helpers
def get(u, rng=None, tries=5, timeout=90):
    for t in range(tries):
        try:
            h = {"User-Agent": "gwcharts"}
            if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) or t == tries - 1: raise
            time.sleep(4 * (t + 1))                   # 429/503 "slow down" and other server errors
        except Exception:
            if t == tries - 1: raise
            time.sleep(3 * (t + 1))


def ymd(d): return d.strftime("%Y%m%d")


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


class Store:
    """R2 (or a local folder for tests) for products and working state."""
    def __init__(self, local=None):
        self.local = local
        if not local:
            import boto3
            from botocore.config import Config
            e = lambda n: os.environ[n].strip()
            self.s3 = boto3.client(
                "s3", endpoint_url=os.environ.get("R2_ENDPOINT") or f"https://{e('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
                aws_access_key_id=e("R2_ACCESS_KEY_ID"), aws_secret_access_key=e("R2_SECRET_ACCESS_KEY"),
                region_name="auto", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))

    def read(self, key):
        try:
            if self.local:
                return open(os.path.join(self.local, key), "rb").read()
            return self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        except Exception:
            return None

    def write(self, key, body, ctype="application/octet-stream", cache="no-cache"):
        if self.local:
            p = os.path.join(self.local, key); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(body); return
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ctype, CacheControl=cache)

    def list(self, prefix):
        if self.local:
            out = []
            for r, _, fs in os.walk(os.path.join(self.local, prefix.rstrip("/"))):
                out += [os.path.relpath(os.path.join(r, f), self.local) for f in fs]
            return out
        keys, tok = [], None
        while True:
            kw = dict(Bucket=BUCKET, Prefix=prefix)
            if tok: kw["ContinuationToken"] = tok
            r = self.s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"): return keys
            tok = r["NextContinuationToken"]

    def delete(self, keys):
        if self.local:
            for k in keys:
                try: os.remove(os.path.join(self.local, k))
                except OSError: pass
            return
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]})

    def npy(self, key):
        b = self.read(key)
        return None if b is None else np.load(io.BytesIO(b), allow_pickle=False)

    def put_npy(self, key, a):
        bio = io.BytesIO(); np.save(bio, a.astype(np.float32)); self.write(key, bio.getvalue())


# ============================================================ observed OLR (CPC Blended OLR)
def clim_tropics():
    """366-day CBO climatology (1992 calendar, 29 Feb = index 59) for the 80 tropical rows."""
    p = os.path.join(CACHE, "cbo_clim_tropics.npy")
    if os.path.exists(p):
        return np.load(p)
    log("downloading CBO climatology (95 MB, once)")
    raw = get(f"{CBO}/CLIM/CBO_V1_1deg-DLY_CLIM_1991-2020_366.lnx", timeout=600)
    c = np.frombuffer(raw, "<f4").reshape(366, 180, 360)[:, CBO_ROWS].copy()
    os.makedirs(CACHE, exist_ok=True); np.save(p, c)
    return c


def clim_index(d):
    return (dt.date(1992, d.month, d.day) - dt.date(1992, 1, 1)).days


def cbo_available(years):
    names = set()
    for y in years:
        try:
            html = get(f"{CBO}/{y}/").decode("latin-1")
            names |= set(re.findall(r"CBO_V1_1deg-DLY\.lnx\.(\d{8})", html))
        except Exception as e:
            log("CBO listing failed for", y, e)
    return sorted(names)


def fill_holes(a):
    """A few missing pixels in a day: take the mean of the neighbouring pixels (or 0 = climatology)."""
    bad = ~np.isfinite(a)
    if not bad.any() or bad.mean() > 0.01: return a
    a = a.copy()
    for y, x in zip(*np.where(bad)):
        nb = a[max(y - 1, 0):y + 2, [(x - 1) % 360, x, (x + 1) % 360]]
        a[y, x] = np.nanmean(nb) if np.isfinite(nb).any() else 0.0
    return a


def observed(start, end):
    """Anomalies (days, 80, 360) for start..end from the CBO daily files; missing days = NaN.
    Files from the last 150 days are fetched again when the cached copy is over 20 days old,
    because CPC replaces its near-real-time values with the final version 3–4 months later."""
    os.makedirs(os.path.join(CACHE, "cbo"), exist_ok=True)
    avail = set(cbo_available(range(start.year, end.year + 1)))
    days = [start + dt.timedelta(n) for n in range((end - start).days + 1)]
    now = time.time()

    def need(d):
        p = os.path.join(CACHE, "cbo", ymd(d) + ".npy")
        if ymd(d) not in avail: return None
        if os.path.exists(p):
            age = (dt.date.today() - d).days
            if age > 150 or now - os.path.getmtime(p) < 20 * 86400: return None
        return d

    todo = [d for d in days if need(d)]
    log(f"CBO: {len(avail)} files listed, {len(todo)} to download")

    def fetch(d):
        raw = get(f"{CBO}/{d.year}/CBO_V1_1deg-DLY.lnx.{ymd(d)}")
        a = np.frombuffer(raw, "<f4").reshape(2, 180, 360)[1, CBO_ROWS].copy()
        a[np.abs(a) > 900] = np.nan
        np.save(os.path.join(CACHE, "cbo", ymd(d) + ".npy"), a.astype(np.float32))

    with ThreadPoolExecutor(6) as ex:
        for _ in ex.map(fetch, todo): pass
    out = np.full((len(days), LAT.size, LON.size), np.nan, np.float32)
    for i, d in enumerate(days):
        p = os.path.join(CACHE, "cbo", ymd(d) + ".npy")
        if os.path.exists(p): out[i] = fill_holes(np.load(p))
    have = [d for i, d in enumerate(days) if np.isfinite(out[i]).any()]
    return out, (max(have) if have else None)


# ============================================================ ECMWF ENS OLR
_W5 = np.array([1, 2, 2, 2, 1], np.float64) / 8.0
_LONI = (4 * np.arange(360)[:, None] + np.arange(5)[None, :]) % 1440
_LATI = 200 + 4 * np.arange(80)[:, None] + np.arange(5)[None, :]     # rows of a −90…90 0.25° grid


def to1deg(v):
    """0.25° global field (lat 90→−90, lon −180…179.75) → mean over 1° boxes on the CBO grid (80 × 360)."""
    v = np.roll(v, 720, axis=1)[::-1].astype(np.float64)             # lon 0…359.75, lat −90…90
    vx = (v[:, _LONI] * _W5).sum(-1)
    return (vx[_LATI] * _W5[None, :, None]).sum(1).astype(np.float32)


def decode(blob):
    import eccodes
    g = eccodes.codes_new_from_message(blob)
    try:
        v = eccodes.codes_get_values(g).reshape(721, 1440)
        lon0 = eccodes.codes_get(g, "longitudeOfFirstGridPointInDegrees")
    finally:
        eccodes.codes_release(g)
    if abs(lon0) < 1e-6:
        v = np.roll(v, 720, axis=1)
    return v


def ens_index(run, step):
    last = None
    d = ymd(run)
    for src in EC_SOURCES:
        base = f"{src}/{d}/00z/ifs/0p25/enfo/{d}000000-{step}h-enfo-ef"
        try:
            idx = [json.loads(l) for l in get(base + ".index", tries=3).decode().splitlines()]
            if any(j.get("param") == "ttr" for j in idx):
                return base + ".grib2", idx
        except Exception as e:
            last = e
    raise last or RuntimeError(f"{d} +{step} h: no ttr in any source")


def ens_ttr_mean(run, step, members=MEMBERS):
    """Ensemble-mean accumulated ttr (J m-2) at one step, on the 1° tropical grid."""
    url, idx = ens_index(run, step)
    pick = {int(j["number"]): (j["_offset"], j["_offset"] + j["_length"] - 1)
            for j in idx if j.get("type") == "pf" and j["param"] == "ttr"}
    mem = [m for m in range(1, members + 1) if m in pick]
    if len(mem) < min(members, 10):
        raise RuntimeError(f"{ymd(run)} +{step} h: only {len(mem)} ttr members")
    with ThreadPoolExecutor(12) as ex:
        fields = list(ex.map(lambda m: to1deg(decode(get(url, pick[m]))), mem))
    return np.mean(fields, axis=0)


def ens_ready(run):
    try:
        _, idx = ens_index(run, FC_DAYS * 24)
    except Exception:
        return False
    return sum(1 for j in idx if j.get("type") == "pf" and j["param"] == "ttr") >= MEMBERS


def ens_daily_olr(run):
    """Daily-mean OLR (W m-2) for days 1…15 of the 00 UTC run: (15, 80, 360)."""
    prev = np.zeros((LAT.size, LON.size), np.float32)
    out = []
    for d in range(1, FC_DAYS + 1):
        acc = ens_ttr_mean(run, 24 * d)
        out.append(-(acc - prev) / 86400.0)
        prev = acc
        log(f"  ENS day {d} done")
    return np.array(out, np.float32)


def day1(run, store, members=MEMBERS):
    """Day-1 ENS-mean OLR of the run started on date `run` (cached in R2)."""
    key = f"waves/state/day1/{ymd(run)}.npy"
    a = store.npy(key)
    if a is not None: return a
    a = -ens_ttr_mean(run, 24, members) / 86400.0
    store.put_npy(key, a)
    return a


# ============================================================ filtering
EARTH_R, OMEGA, G = 6.371e6, 7.292e-5, 9.81


def band_mask(nt, nx, mode):
    kMin, kMax, tMin, tMax, hMin, hMax, kind = MODES[mode]
    f = np.fft.fftfreq(nt)[:, None]                     # cycles per day
    s = np.fft.fftfreq(nx, 1.0 / nx)[None, :]           # zonal index
    k = np.where(f > 0, -s, s)                          # eastward-positive wavenumber of each coefficient
    k = np.where(f == 0, np.abs(s), k)
    fa = np.abs(f)
    fmin = 0.0 if np.isinf(tMax) else 1.0 / tMax
    m = (fa >= fmin) & (fa <= 1.0 / tMin) & (k >= kMin) & (k <= kMax)
    if kind:
        km = k / EARTH_R                                # 1/m
        def freq(h):
            c = np.sqrt(G * h)
            if kind == "kelvin":
                w = c * km
            else:                                       # n = 1 equatorial Rossby
                beta = 2 * OMEGA / EARTH_R
                w = -beta * km / (km ** 2 + 3 * beta / c)
            return w * 86400 / (2 * np.pi)
        lo, hi = np.minimum(freq(hMin), freq(hMax)), np.maximum(freq(hMin), freq(hMax))
        m &= (fa >= lo) & (fa <= hi)
    return m


def filter_modes(series):
    """series (nt, 80, 360) anomalies, already zero-padded. Returns {mode: filtered array}."""
    nt, ny, nx = series.shape
    x = np.nan_to_num(series).astype(np.float64)
    taper = np.ones(nt)
    n0 = 60                                             # cosine taper over the first 60 days
    taper[:n0] = 0.5 * (1 - np.cos(np.pi * np.arange(n0) / n0))
    x *= taper[:, None, None]
    X = np.fft.fft2(x, axes=(0, 2))
    out = {}
    for mode in MODES:
        m = band_mask(nt, nx, mode)
        out[mode] = np.real(np.fft.ifft2(X * m[:, None, :], axes=(0, 2))).astype(np.float32)
    return out


def smooth(a, n=3):
    for _ in range(n):
        a = (np.roll(a, 1, 1) + 2 * a + np.roll(a, -1, 1)) / 4
        a = np.concatenate([a[:1], (a[:-2] + 2 * a[1:-1] + a[2:]) / 4, a[-1:]], 0)
    return a


# ============================================================ drawing
def _colours():
    from matplotlib.colors import ListedColormap, BoundaryNorm
    lev = [-54, -42, -30, -18, -6, 6, 18, 30, 42, 54]
    cols = ["#0f4a45", "#2b7a70", "#5aa89c", "#a7d6cc", "#ffffff", "#f1e0bd", "#dcbb82", "#bd8f4f", "#8f6026"]
    cm = ListedColormap(cols); cm.set_under("#062b28"); cm.set_over("#5c3a10")
    return lev, cm, BoundaryNorm(lev, cm.N)


LINES = [("low", "#9b30d0", "--", 1.0, "Low (>120 d)"), ("mjo", "#000000", "-", 1.0, "MJO"),
         ("kelvin", "#1f3fff", "-", 2.0, "Kelvin ×2"), ("er", "#e01b1b", "-", 1.0, "ER")]

_SEGS = None


def segments():
    global _SEGS
    if _SEGS is None:
        raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundaries_india_view.json")))
        segs = [np.array(l) for k in ("coast_coarse", "borders_coarse") for l in raw[k]]
        _SEGS = [s for s in segs if len(s) > 1 and np.abs(s[:, 1]).min() < 45]
    return _SEGS


def ext(a, x0, x1):
    """Repeat the 0.5…359.5 longitude axis so any range x0…x1 (degrees, may be <0 or >360) is covered."""
    lon = np.concatenate([LON - 360, LON, LON + 360])
    a = np.concatenate([a, a, a], axis=-1)
    i = (lon >= x0 - 1.5) & (lon <= x1 + 1.5)
    return lon[i], a[..., i]


def draw_map(ax, shade, lines, lat_rng, lon_rng, levs, cmap, norm):
    from matplotlib.collections import LineCollection
    (y0, y1), (x0, x1) = lat_rng, lon_rng
    j = (LAT >= y0 - 1.5) & (LAT <= y1 + 1.5)
    lo, s = ext(shade[j], x0, x1)
    ax.contourf(lo, LAT[j], s, levels=levs, cmap=cmap, norm=norm, extend="both")
    for key, col, ls, mult, _ in LINES:
        lo, f = ext(lines[key][j] * mult, x0, x1)
        if np.nanmin(f) < -12:
            ax.contour(lo, LAT[j], f, levels=[-36, -12], colors=col, linestyles=ls, linewidths=1.25)
    segs = []
    for sh in (-360, 0, 360):
        segs += [sg + [sh, 0] for sg in segments() if sg[:, 0].max() + sh >= x0 - 5 and sg[:, 0].min() + sh <= x1 + 5]
    ax.add_collection(LineCollection(segs, colors="#6b6b6b", linewidths=0.6, zorder=3))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.tick_params(labelsize=8.5, length=2.5)


def lonlab(x):
    x = (x + 180) % 360 - 180
    return "0°" if x == 0 else ("180°" if abs(x) == 180 else f"{abs(x):g}°{'E' if x > 0 else 'W'}")


def latlab(y):
    return "Eq" if y == 0 else f"{abs(y):g}°{'N' if y > 0 else 'S'}"


def ticks(ax, lat_rng, lon_rng):
    (y0, y1), (x0, x1) = lat_rng, lon_rng
    span = x1 - x0
    st = 60 if span > 200 else (30 if span > 90 else 20)
    xs = np.arange(np.ceil(x0 / st) * st, x1 + 0.1, st)
    ys = np.arange(np.ceil(y0 / 10) * 10, y1 + 0.1, 10)
    ax.set_xticks(xs); ax.set_xticklabels([lonlab(x) for x in xs])
    ax.set_yticks(ys); ax.set_yticklabels([latlab(y) for y in ys])


def footer(fig, run, y, legend_x=0.60, kelvin2=True):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, ls=ls, lw=1.6) for _, c, ls, _, _ in LINES]
    labs = [l if kelvin2 else l.replace(" ×2", "") for *_, l in LINES]
    fig.legend(hs, labs, loc="upper left", bbox_to_anchor=(legend_x, y), ncol=4,
               frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=1.2)
    fig.text(legend_x, y - 0.035, "Contours: filtered OLR at −12 and −36 W m⁻² (enhanced convection)",
             fontsize=8.5, color="#333", va="top")


def plot_map(key, P, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title, lat_rng, lon_rng = REGIONS[key]
    levs, cmap, norm = _colours()
    (y0, y1), (x0, x1) = lat_rng, lon_rng
    asp = (y1 - y0) / (x1 - x0)
    W = 15.0 if x1 - x0 > 200 else 12.5
    pw = W * 0.43; ph = pw * asp
    top, bot = 0.95, 1.75                               # inches above / below the panel grid
    H = top + 4 * (ph + 0.42) + bot
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.text(0.5, 1 - 0.22 / H, f"ECMWF MJO-Waves · 5-day OLR anomaly (W m⁻²) with filtered waves · {title}",
             ha="center", va="top", fontsize=13.5, weight="bold", color="#0f2240")
    fig.text(0.5, 1 - 0.55 / H, f"Left: observed (NOAA CPC Blended OLR)    Right: ECMWF ENS mean, 00 UTC {P['run'].strftime('%d %b %Y')}, "
             f"then filtered wave projection", ha="center", va="top", fontsize=10, color="#444")
    cols = [P["obs_pentads"], P["fc_pentads"]]
    lx = [0.055, 0.535]
    for c in range(2):
        for r in range(4):
            pent = cols[c][r]
            yb = (bot + (3 - r) * (ph + 0.42)) / H
            ax = fig.add_axes([lx[c], yb, 0.43, ph / H])
            draw_map(ax, pent["shade"], pent["lines"], lat_rng, lon_rng, levs, cmap, norm)
            ticks(ax, lat_rng, lon_rng)
            if r < 3: ax.set_xticklabels([])
            if c == 1: ax.set_yticklabels([])
            ax.set_title(pent["label"], loc="left", fontsize=11, weight="bold", pad=3)
            ax.set_title(pent["kind"], loc="right", fontsize=10.5, pad=3,
                         color={"Observed": "#111", "ECMWF ENS": "#0b5cad"}.get(pent["kind"], "#7a2ea0"))
    cax = fig.add_axes([0.07, 0.95 / H, 0.42, 0.17 / H])
    import matplotlib.cm as mcm
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="both", ticks=levs)
    cb.ax.tick_params(labelsize=9); cb.set_label("OLR anomaly (W m⁻²) · negative (green) = more cloud and rain", fontsize=9)
    footer(fig, P["run"], 1.30 / H, legend_x=0.535)
    fig.text(0.99, 0.12 / H, f"{BRAND}  ·  Data: NOAA CPC Blended OLR; ECMWF open data (CC BY 4.0)  ·  "
             f"Wave filtering after Wheeler & Weickmann (2001)", ha="right", va="bottom", fontsize=8.5, color="#555")
    fig.savefig(out, dpi=100); plt.close(fig)


def plot_hov(key, P, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    title, (b0, b1) = HOVS[key]
    levs, cmap, norm = _colours()
    j = (LAT >= b0) & (LAT <= b1)
    days, shade, lines = P["hov_days"], P["hov_shade"][:, j].mean(1), {k: v[:, j].mean(1) for k, v in P["hov_lines"].items()}
    t = mdates.date2num(days)
    W, H = 11.5, 13.0
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0.10, 0.12, 0.86, 0.80])
    lo, s = ext(shade, 0, 360)
    ax.contourf(lo, t, s, levels=levs, cmap=cmap, norm=norm, extend="both")
    for k, col, ls, mult, _ in LINES:                   # daily values: Kelvin drawn as is (not ×2)
        lo, f = ext(lines[k], 0, 360)
        ax.contour(lo, t, f, levels=[-36, -12], colors=col, linestyles=ls, linewidths=1.3)
    for x in (P["f0"], P["proj0"]):
        ax.axhline(mdates.date2num(x) - 0.5, color="#0b5cad" if x == P["f0"] else "#7a2ea0", lw=1.6, ls="-")
    ax.set_xlim(0, 360); ax.set_ylim(t[-1] + 0.5, t[0] - 0.5)
    xs = np.arange(0, 361, 60); ax.set_xticks(xs); ax.set_xticklabels([lonlab(x) for x in xs])
    ax.yaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 11, 21]))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(labelsize=9.5)
    box = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
    ax.text(3, mdates.date2num(P["f0"]) + 0.2, "ECMWF ENS forecast", color="#0b5cad", va="top", fontsize=9.5, weight="bold", bbox=box)
    ax.text(3, mdates.date2num(P["proj0"]) + 0.2, "Wave projection", color="#7a2ea0", va="top", fontsize=9.5, weight="bold", bbox=box)
    ax.grid(axis="x", color="#999", lw=0.4, ls=":")
    fig.text(0.5, 0.975, f"ECMWF MJO-Waves · OLR anomaly (W m⁻²) Hovmöller, {title}", ha="center", va="top",
             fontsize=13.5, weight="bold", color="#0f2240")
    fig.text(0.5, 0.952, f"Observed (NOAA CPC Blended OLR) to {P['obs_last'].strftime('%d %b')}, then ECMWF ENS mean "
             f"00 UTC {P['run'].strftime('%d %b %Y')}, then filtered wave projection", ha="center", va="top", fontsize=10, color="#444")
    cax = fig.add_axes([0.10, 0.06, 0.40, 0.013])
    import matplotlib.cm as mcm
    cb = fig.colorbar(mcm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both", ticks=levs)
    cb.ax.tick_params(labelsize=9); cb.set_label("OLR anomaly (W m⁻²) · negative = more cloud and rain", fontsize=9)
    footer(fig, P["run"], 0.085, legend_x=0.54, kelvin2=False)
    fig.text(0.99, 0.008, f"{BRAND}  ·  Data: NOAA CPC Blended OLR; ECMWF open data (CC BY 4.0)  ·  "
             f"Wave filtering after Wheeler & Weickmann (2001)", ha="right", va="bottom", fontsize=8.5, color="#555")
    fig.savefig(out, dpi=100); plt.close(fig)


def quantize(path):
    from PIL import Image
    im = Image.open(path).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(path, optimize=True)


# ============================================================ main job
def newest_run():
    today = dt.datetime.now(dt.timezone.utc).date()
    for back in range(0, 3):
        d = today - dt.timedelta(back)
        if ens_ready(d): return d
    raise SystemExit("no complete ECMWF ENS 00 UTC run found")


def update_bias(store, obs, days, run):
    """Running mean of (ECMWF day-1 OLR − CBO OLR), updated with every day that has both."""
    st = store.read("waves/state/bias.npz")
    clim = clim_tropics()
    obs_tot = {d: obs[i] + clim[clim_index(d)] for i, d in enumerate(days) if np.isfinite(obs[i]).all()}
    if st is None:
        # first run: start from the last ~30 days with observations (10 members to keep it quick)
        cand = sorted(obs_tot)[-30:]
        log(f"bias: first run, using {len(cand)} past day-1 forecasts")
        diffs = []
        for d in cand:
            try:
                diffs.append(day1(d, store, members=10) - obs_tot[d])
            except Exception as e:
                log("  no day-1 for", d, e)
        if not diffs: raise SystemExit("could not build the bias field")
        bias, last, n = np.mean(diffs, 0), max(cand), len(diffs)
    else:
        z = np.load(io.BytesIO(st))
        bias, last, n = z["bias"], dt.date.fromisoformat(str(z["last"])), int(z["n"])
        for d in sorted(obs_tot):
            if d <= last: continue
            a = store.npy(f"waves/state/day1/{ymd(d)}.npy")
            if a is None: continue
            alpha = max(1.0 / 30, 1.0 / (n + 1))
            bias = (1 - alpha) * bias + alpha * (a - obs_tot[d]); last, n = d, n + 1
    bio = io.BytesIO(); np.savez(bio, bias=bias.astype(np.float32), last=last.isoformat(), n=n)
    store.write("waves/state/bias.npz", bio.getvalue())
    log(f"bias: {n} days, up to {last}, tropical mean {np.nanmean(bias):+.1f} W m-2")
    return smooth(bias.astype(np.float64), 4).astype(np.float32)


def build(run, store):
    clim = clim_tropics()
    start = run - dt.timedelta(HIST_DAYS)
    obs, obs_last = observed(start, run - dt.timedelta(1))
    if obs_last is None: raise SystemExit("no CBO observations")
    log(f"observed to {obs_last}")
    hist_days = [start + dt.timedelta(n) for n in range(HIST_DAYS)]
    bias = update_bias(store, obs, hist_days, run)

    log("ECMWF ENS daily OLR, days 1–15")
    fc = ens_daily_olr(run)
    store.put_npy(f"waves/state/day1/{ymd(run)}.npy", fc[0])

    f0 = obs_last + dt.timedelta(1)
    gap = [f0 + dt.timedelta(n) for n in range((run - f0).days)]
    series_days = [start + dt.timedelta(n) for n in range((obs_last - start).days + 1)] + gap + \
                  [run + dt.timedelta(n) for n in range(FC_DAYS)]
    anom = list(obs[: (obs_last - start).days + 1])
    for g in gap:
        try:
            a = day1(g, store)
            anom.append(a - bias - clim[clim_index(g)]); log("gap day", g, "from its day-1 forecast")
        except Exception as e:
            log("gap day", g, "missing:", e); anom.append(np.full_like(fc[0], np.nan))
    for i, d in enumerate(series_days[-FC_DAYS:]):
        anom.append(fc[i] - bias - clim[clim_index(d)])
    anom = np.array(anom, np.float32)
    # fill any missing day by linear interpolation in time
    bad = ~np.isfinite(anom).all(axis=(1, 2))
    if bad.any():
        good = np.where(~bad)[0]
        for i in np.where(bad)[0]:
            p, q = good[good < i], good[good > i]
            if len(p) and len(q):
                p, q = p[-1], q[0]; w = (i - p) / (q - p)
                anom[i] = (1 - w) * np.nan_to_num(anom[p]) + w * np.nan_to_num(anom[q])
            else:
                anom[i] = np.nan_to_num(anom[(p if len(p) else q)[-1 if len(p) else 0]])
        log(f"{int(bad.sum())} missing days filled by interpolation")
    anom = np.nan_to_num(anom)
    n_real = len(anom)
    series = np.concatenate([anom, np.zeros((PAD_DAYS,) + anom.shape[1:], np.float32)])
    log(f"filtering {series.shape}")
    F = filter_modes(series)
    allday = series_days + [series_days[-1] + dt.timedelta(n + 1) for n in range(PAD_DAYS)]
    idx = {d: i for i, d in enumerate(allday)}
    total = sum(F[m] for m in ("mjo", "kelvin", "er", "low"))

    def pent(d0, kind):
        ii = [idx[d0 + dt.timedelta(n)] for n in range(5)]
        shade = total[ii].mean(0) if kind == "Wave projection" else anom[ii].mean(0)
        lab = f"{(d0).strftime('%-d-%b')} to {(d0 + dt.timedelta(4)).strftime('%-d-%b')}"
        return {"label": lab, "kind": kind, "shade": shade, "lines": {m: F[m][ii].mean(0) for m in MODES}}

    obs_p = [pent(f0 - dt.timedelta(5 * (4 - r)), "Observed") for r in range(4)]
    last_fc = run + dt.timedelta(FC_DAYS - 1)
    fc_p = []
    for r in range(4):
        d0 = f0 + dt.timedelta(5 * r)
        fc_p.append(pent(d0, "ECMWF ENS" if d0 + dt.timedelta(4) <= last_fc else "Wave projection"))
    proj0 = last_fc + dt.timedelta(1)                     # first day with no forecast
    h0 = f0 - dt.timedelta(90)
    hdays = [h0 + dt.timedelta(n) for n in range((f0 + dt.timedelta(20) - h0).days)]
    hi = [idx[d] for d in hdays]
    hshade = np.where(np.array([d < proj0 for d in hdays])[:, None, None], series[hi], total[hi])
    return {"run": run, "obs_last": obs_last, "f0": f0, "proj0": proj0, "last_fc": last_fc,
            "obs_pentads": obs_p, "fc_pentads": fc_p, "hov_days": hdays, "hov_shade": hshade,
            "hov_lines": {m: F[m][hi] for m in MODES}, "gap": gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="YYYYMMDD (00 UTC ENS run)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--local", help="write to this folder instead of R2")
    a = ap.parse_args()
    store = Store(a.local)
    run = dt.datetime.strptime(a.run, "%Y%m%d").date() if a.run else newest_run()
    key = ymd(run)
    if store.read(f"waves/{key}/meta.json") and not a.force:
        log(f"{key} already published"); return
    if not ens_ready(run): raise SystemExit(f"ENS {key} 00 UTC not complete yet")
    log(f"run {key}")
    P = build(run, store)
    tmp = os.path.join(CACHE, "png"); os.makedirs(tmp, exist_ok=True)
    files = []
    for k in REGIONS:
        p = os.path.join(tmp, f"map_{k}.png"); plot_map(k, P, p); quantize(p); files.append(f"map_{k}.png")
    for k in HOVS:
        p = os.path.join(tmp, f"hov_{k}.png"); plot_hov(k, P, p); quantize(p); files.append(f"hov_{k}.png")
    log("charts drawn")
    for f in files:
        store.write(f"waves/{key}/{f}", open(os.path.join(tmp, f), "rb").read(), "image/png", "public, max-age=86400")
    meta = {"run": key, "init": f"{key}00", "obs_last": P["obs_last"].isoformat(), "fc_start": P["f0"].isoformat(),
            "fc_last": P["last_fc"].isoformat(), "proj_start": P["proj0"].isoformat(),
            "gap_days": [g.isoformat() for g in P["gap"]],
            "maps": [{"id": k, "title": v[0]} for k, v in REGIONS.items()],
            "hovs": [{"id": k, "title": v[0]} for k, v in HOVS.items()],
            "made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}
    store.write(f"waves/{key}/meta.json", json.dumps(meta).encode(), "application/json", "public, max-age=300")
    # run list + clean-up
    keys = store.list("waves/")
    runs = sorted({k.split("/")[1] for k in keys if re.match(r"waves/\d{8}/meta\.json$", k)}, reverse=True)
    cut = ymd(run - dt.timedelta(KEEP_DAYS))
    old = [k for k in keys if re.match(r"waves/\d{8}/", k) and k.split("/")[1] < cut]
    old += [k for k in keys if k.startswith("waves/state/day1/") and k[-12:-4] < ymd(run - dt.timedelta(40))]
    if old: store.delete(old); log(f"deleted {len(old)} old objects")
    runs = [r for r in runs if r >= cut]
    store.write("waves/index.json", json.dumps({"runs": runs, "latest": runs[0]}).encode(), "application/json", "no-cache")
    log(f"published {key}; runs kept: {len(runs)}")


if __name__ == "__main__":
    main()
