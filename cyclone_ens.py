"""ECMWF ENS tropical-cyclone products for every basin -> R2 bucket (cyclone/ens/...).

For the newest 00/12 UTC run it downloads ECMWF's own cyclone tracks (open data, BUFR type "tf") for the
51-member ensemble and HRES, finds every system that enough members track, and for each one makes:
  landfall summary, spaghetti tracks, strike probability (120 km), MSW plume, MSLP plume,
  run-comparison table (last 6 runs of the same system), package zip, mean / HRES tracks, member landfalls.

Usage: python cyclone_ens.py [--run YYYYMMDDHH] [--force] [--local DIR]
  --local DIR  write to DIR instead of R2 (for tests)
Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY for R2.

Bucket layout:
  cyclone/ens/index.json                    runs (newest first) and their systems; what the website reads
  cyclone/ens/<run>/<system>/<file>          figures and files
"""
import os, re, io, sys, json, time, zipfile, argparse, textwrap, datetime as dt, urllib.request
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import eccodes
import cartopy.crs as ccrs
from cartopy.io import shapereader
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "https://data.ecmwf.int"
KEEP_DAYS = float(os.environ.get("CYCLONE_KEEP_DAYS", "15"))
SITE = os.environ.get("SITE_NAME", "globalweathercharts.com")
REF_RADIUS_DEG, REF_HOURS, MIN_MEMBERS, STRIKE_KM = 4.0, 72, 10, 120

BASINS = {"B": "Bay of Bengal", "A": "Arabian Sea", "W": "West Pacific", "E": "East Pacific", "C": "Central Pacific",
          "L": "North Atlantic", "S": "South-West Indian Ocean", "U": "Australian region", "P": "South Pacific"}
ORDER = "BAWECLSUP"
SHIFT = set("WCPU")                         # basins drawn with 0..360 longitudes (they can cross 180°)
NIO = set("AB")                              # IMD categories and the fixed BoB / Arabian Sea frames
BASIN_FRAME = {"B": [77, 99, 8, 27], "A": [52, 78, 8, 28]}
BASIN_LIMIT = {"B": (72, 101), "A": (40, 82)}

COAST_NIO = {"Chennai": (13.08, 80.27), "Nellore": (14.44, 79.99), "Ongole": (15.50, 80.05), "Bapatla": (15.90, 80.47),
    "Machilipatnam": (16.17, 81.13), "Narsapur": (16.43, 81.70), "Yanam": (16.73, 82.21), "Kakinada": (16.93, 82.22),
    "Visakhapatnam": (17.69, 83.22), "Bhimunipatnam": (17.89, 83.45), "Kalingapatnam": (18.34, 84.12),
    "Gopalpur": (19.26, 84.90), "Puri": (19.81, 85.83), "Paradip": (20.26, 86.67), "Balasore": (21.49, 86.93),
    "Digha": (21.62, 87.51), "Sagar Island": (21.65, 88.05), "Khepupara": (21.99, 90.23), "Chittagong": (22.34, 91.83),
    "Cox's Bazar": (21.43, 92.00), "Sittwe": (20.15, 92.90), "Karaikal": (10.92, 79.84), "Puducherry": (11.93, 79.83),
    "Thoothukudi": (8.80, 78.13), "Kanniyakumari": (8.08, 77.55), "Thiruvananthapuram": (8.49, 76.95), "Kochi": (9.97, 76.27),
    "Kozhikode": (11.25, 75.78), "Mangaluru": (12.87, 74.84), "Karwar": (14.81, 74.13), "Panaji": (15.49, 73.83),
    "Ratnagiri": (16.99, 73.30), "Mumbai": (18.94, 72.83), "Dahanu": (19.97, 72.71), "Veraval": (20.90, 70.37),
    "Diu": (20.71, 70.98), "Porbandar": (21.64, 69.60), "Dwarka": (22.24, 68.97), "Jakhau": (23.22, 68.72),
    "Karachi": (24.86, 67.01), "Ormara": (25.21, 64.64), "Gwadar": (25.12, 62.33), "Chabahar": (25.29, 60.64),
    "Muscat": (23.59, 58.41), "Sur": (22.57, 59.53), "Duqm": (19.67, 57.70), "Salalah": (17.02, 54.09),
    "Al Mukalla": (14.54, 49.13), "Aden": (12.79, 45.03), "Bosaso": (11.28, 49.18), "Trincomalee": (8.57, 81.23),
    "Yangon": (16.80, 96.15), "Gwa": (17.60, 94.60)}
CITIES_NIO = {"Kolkata": (88.36, 22.57), "Bhubaneswar": (85.82, 20.30), "Visakhapatnam": (83.22, 17.69),
    "Chennai": (80.27, 13.08), "Machilipatnam": (81.13, 16.17), "Gopalpur": (84.90, 19.26), "Mumbai": (72.83, 18.94),
    "Goa": (73.83, 15.49), "Kochi": (76.27, 9.97), "Veraval": (70.37, 20.90), "Karachi": (67.01, 24.86),
    "Muscat": (58.41, 23.59), "Salalah": (54.09, 17.02), "Aden": (45.03, 12.79), "Colombo": (79.86, 6.93),
    "Yangon": (96.15, 16.80), "Dhaka": (90.41, 23.81)}
PLACES = json.load(open(os.path.join(HERE, "places.json"), encoding="utf-8"))

# site colour scheme (Global Weather Charts)
C_MEM, C_MEAN, C_HRES, C_CTRL, C_LAND, C_SEA, C_HEAD, C_HIL = "#2a9d8f", "#14213d", "#e85d04", "#7b2cbf", "#ecebe6", "#f2f7fb", "#0f2240", "#dcecfb"
BANDS = ["#eef7f5", "#e3f1f8", "#e8e8f8", "#f3e8f5", "#fbe9e4", "#fdf1dc", "#fdf8d5"]
CATS_NIO = [(17, 27, "D", BANDS[0]), (28, 33, "DD", BANDS[1]), (34, 47, "CS", BANDS[2]), (48, 63, "SCS", BANDS[3]),
            (64, 89, "VSCS", BANDS[4]), (90, 119, "ESCS", BANDS[5]), (120, 250, "SuCS", BANDS[6])]
CATS_SS = [(20, 33, "TD", BANDS[0]), (34, 63, "TS", BANDS[1]), (64, 82, "Cat 1", BANDS[2]), (83, 95, "Cat 2", BANDS[3]),
           (96, 112, "Cat 3", BANDS[4]), (113, 136, "Cat 4", BANDS[5]), (137, 250, "Cat 5", BANDS[6])]
DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def cat(w, basin):
    if w is None or np.isnan(w): return "–"
    w = round(w)
    if basin in NIO:
        for lo, hi, n in [(0, 16, "L"), (17, 27, "D"), (28, 33, "DD"), (34, 47, "CS"), (48, 63, "SCS"),
                          (64, 89, "VSCS"), (90, 119, "ESCS"), (120, 999, "SuCS")]:
            if lo <= w <= hi: return n
    for lo, hi, n in [(0, 33, "TD"), (34, 63, "TS"), (64, 82, "Cat 1"), (83, 95, "Cat 2"), (96, 112, "Cat 3"),
                      (113, 136, "Cat 4"), (137, 999, "Cat 5")]:
        if lo <= w <= hi: return n


def fmt_lat(v): return f"{abs(v):.1f}°{'N' if v >= 0 else 'S'}"
def fmt_lon(v):
    v = (v + 180) % 360 - 180
    return f"{abs(v):.1f}°{'E' if v >= 0 else 'W'}"


# ---------------------------------------------------------------- downloads
def run_dir(run, stream):
    return f"{ROOT}/forecasts/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{stream}/"


def tf_url(run, stream):
    try:
        html = urllib.request.urlopen(run_dir(run, stream), timeout=60).read().decode()
    except Exception:
        return None
    m = re.findall(r'href="([^"]*-tf\.bufr)"', html)
    return ROOT + m[0] if m else None


def latest_run():
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    t = now.replace(hour=12 if now.hour >= 12 else 0, minute=0, second=0, microsecond=0)
    for _ in range(6):
        if tf_url(t, "enfo"): return t
        t -= dt.timedelta(hours=12)
    return None


def download(url, out):
    for k in range(4):
        try:
            urllib.request.urlretrieve(url, out); return out
        except Exception as e:
            print(f"  retry {k + 1}: {e}"); time.sleep(20)
    raise RuntimeError(f"cannot download {url}")


# ---------------------------------------------------------------- BUFR decoding (all basins)
MISS = 1e90
def _arr(h, key, nsub, comp):
    if comp or nsub == 1:
        a = np.atleast_1d(eccodes.codes_get_array(h, key)).astype(float)
        return np.repeat(a, nsub) if len(a) == 1 else a
    return np.array([eccodes.codes_get(h, f"/subsetNumber={s}/{key}") for s in range(1, nsub + 1)], float)


def decode(path, label):
    """List of tracks: dict(sid, basin, member, kind, df[t, lat, lon, w (kt), p (hPa)])."""
    tracks = []
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_bufr_new_from_file(f)
            if h is None: break
            eccodes.codes_set(h, "unpack", 1)
            sid = eccodes.codes_get(h, "stormIdentifier").strip()
            if sid[-1:] not in BASINS or not eccodes.codes_is_defined(h, "timePeriod"):
                eccodes.codes_release(h); continue
            nsub = eccodes.codes_get(h, "numberOfSubsets")
            comp = eccodes.codes_get(h, "compressedData") == 1
            t0 = pd.Timestamp(year=eccodes.codes_get(h, "year"), month=eccodes.codes_get(h, "month"),
                              day=eccodes.codes_get(h, "day"), hour=eccodes.codes_get(h, "hour"))
            tp = [0] + list(np.atleast_1d(eccodes.codes_get_array(h, "timePeriod")).astype(int))
            nt = len(tp)
            mem = _arr(h, "ensembleMemberNumber", nsub, comp) if label == "ENS" else np.zeros(nsub)
            lat = np.array([_arr(h, f"#{2*(k+1)}#latitude", nsub, comp) for k in range(nt)])
            lon = np.array([_arr(h, f"#{2*(k+1)}#longitude", nsub, comp) for k in range(nt)])
            p = np.array([_arr(h, f"#{k+1}#pressureReducedToMeanSeaLevel", nsub, comp) for k in range(nt)])
            w = np.array([_arr(h, f"#{k+1}#windSpeedAt10M", nsub, comp) for k in range(nt)])
            eccodes.codes_release(h)
            b = sid[-1]
            for s in range(nsub):
                df = pd.DataFrame(dict(t=[t0 + pd.Timedelta(hours=x) for x in tp],
                                       lat=lat[:, s], lon=lon[:, s], w=w[:, s] * 1.94384, p=p[:, s] / 100))
                df.loc[(df.lat.abs() > MISS) | (df.lon.abs() > MISS), ["lat", "lon"]] = np.nan
                df.loc[df.w.abs() > MISS, "w"] = np.nan; df.loc[df.p.abs() > MISS, "p"] = np.nan
                df = df.dropna(subset=["lat"]).reset_index(drop=True)
                if df.empty: continue
                df["lon"] = df.lon % 360 if b in SHIFT else (df.lon + 180) % 360 - 180
                m = int(mem[s])
                kind = "HRES" if label == "HRES" else ("Control" if m == 51 else "Member")
                tracks.append(dict(sid=sid, basin=b, member=m, kind=kind, df=df))
    return tracks


def dist_deg(lat, lon, la0, lo0):
    return np.hypot(lat - la0, (lon - lo0) * np.cos(np.radians(la0)))


def _early(tr):
    """Positions in the first REF_HOURS (+ the rest of the first day) of a track, cached."""
    if "early" not in tr:
        d = tr["df"]; d = d[d.t <= d.t.iloc[0].normalize() + pd.Timedelta(hours=REF_HOURS + 24)]
        tr["early"] = (d.lat.to_numpy(float), d.lon.to_numpy(float))
    return tr["early"]


def pick(pool, basin, la0, lo0):
    """For each member keep the track of this basin that comes closest to (la0, lo0) in its first REF_HOURS
    (ECMWF numbers new storms 70B, 71B... independently in each member)."""
    best = {}
    for tr in pool:
        if tr["basin"] != basin: continue
        la, lo = _early(tr)
        if not len(la): continue
        dd = float(np.min(np.hypot(la - la0, (lo - lo0) * np.cos(np.radians(la0)))))
        if dd <= REF_RADIUS_DEG and (tr["member"] not in best or dd < best[tr["member"]][0]):
            best[tr["member"]] = (dd, tr)
    return [v[1] for k, v in sorted(best.items())]


def mean_peak(mem):
    """Peak of the ensemble-mean wind (times when at least MIN_MEMBERS members track the system)."""
    a = pd.concat([m["df"][["t", "w"]] for m in mem]).groupby("t").w.agg(["mean", "size"])
    a = a[a["size"] >= MIN_MEMBERS]
    return float(a["mean"].max()) if len(a) else 0.0


def strong_enough(basin, mem, hres=None):
    """Outside the North Indian Ocean only storms (ensemble mean or HRES at least 34 kt); in the NIO depressions too."""
    need = 17 if basin in NIO else 34
    return mean_peak(mem) >= need or (hres is not None and np.nanmax(hres["df"].w.fillna(0)) >= need + 6)


def mean_track(mem):
    a = pd.concat([m["df"][["t", "lat", "lon"]] for m in mem]).groupby("t").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"))
    return a[a.n >= min(MIN_MEMBERS, a.n.max())]


def duplicate(basin, mem, systems):
    """True if this group follows the same storm as a system already found (ECMWF sometimes gives one storm
    two identifiers, e.g. 04P and 70P)."""
    a = mean_track(mem)
    for s in systems:
        if s["basin"] != basin: continue
        b = s["mt"]; common = a.index.intersection(b.index)
        if len(common) and (dist_deg(a.lat[common], a.lon[common], b.lat[common], b.lon[common]) < 2.5).mean() >= 0.5:
            return True
    return False


def find_systems(ens, hres):
    """Systems = HRES tracks that at least MIN_MEMBERS members follow, then storms only the ensemble has
    (at least 20 members). Each member track is used by one system only."""
    pool = list(ens); systems = []
    seeds = sorted(hres, key=lambda t: -np.nanmax(t["df"].w.fillna(0)))
    for h in seeds:
        r0 = h["df"].iloc[0]
        if any(s["basin"] == h["basin"] and dist_deg(r0.lat, r0.lon, s["la0"], s["lo0"]) < 3 for s in systems):
            continue
        mem = pick(pool, h["basin"], r0.lat, r0.lon)
        if len(mem) < MIN_MEMBERS: continue
        if not strong_enough(h["basin"], mem, h) or abs(r0.lat) > 40 or duplicate(h["basin"], mem, systems): continue
        systems.append(dict(basin=h["basin"], sid=h["sid"], la0=r0.lat, lo0=r0.lon, ens=mem, hres=h, mt=mean_track(mem)))
        ids = {id(m) for m in mem}; pool = [t for t in pool if id(t) not in ids]
    while True:                                                          # storms only the ensemble has:
        seeds = {}                                                       # biggest group of member tracks first
        for t in pool:
            r0 = t["df"].iloc[0]; seeds.setdefault((t["basin"], round(r0.lat / 2), round(r0.lon / 2)), (r0.lat, r0.lon))
        best = None
        for (b, _, _), (la, lo) in seeds.items():
            mem = pick(pool, b, la, lo)
            if len(mem) >= 20 and (best is None or len(mem) > len(best[3])): best = (b, la, lo, mem)
        if best is None: break
        b, la, lo, mem = best
        ids = {id(m) for m in mem}; pool = [x for x in pool if id(x) not in ids]      # used up either way
        if strong_enough(b, mem) and abs(la) <= 40 and not duplicate(b, mem, systems):
            systems.append(dict(basin=b, sid=mem[0]["sid"], la0=la, lo0=lo, ens=mem, hres=None, mt=mean_track(mem)))
    systems.sort(key=lambda s: (ORDER.index(s["basin"]), s["lo0"]))
    n = {}
    for s in systems:
        n[s["basin"]] = n.get(s["basin"], 0) + 1
        s["id"] = f"{s['basin'].lower()}{n[s['basin']]}"
    for s in systems:
        same = [x for x in systems if x["basin"] == s["basin"]]
        s["name"] = BASINS[s["basin"]] + (f" · system {same.index(s) + 1}" if len(same) > 1 else "")
    return systems


# ---------------------------------------------------------------- map helpers
_LAND = None
def land_geoms():
    global _LAND
    if _LAND is None:
        _LAND = list(shapereader.Reader(shapereader.natural_earth("10m", "physical", "land")).geometries())
    return _LAND


def shifted(geoms, bb, shift):
    out = []
    for g in geoms:
        for off in ((0, 360) if shift else (0,)):
            gg = affinity.translate(g, xoff=off) if off else g
            if gg.intersects(bb): out.append(gg.intersection(bb))
    return out


def lines_only(g):
    if g.is_empty: return []
    if g.geom_type in ("LineString", "MultiLineString"): return [g]
    return [x for x in getattr(g, "geoms", []) if x.geom_type in ("LineString", "MultiLineString")]


_BND = None
def boundaries():
    global _BND
    if _BND is None:
        sys.path.insert(0, HERE); import charts
        _BND = charts.boundaries()
    return _BND


def make_axes(ext, shift, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180 if shift else 0))
    ax.set_extent(ext, ccrs.PlateCarree())
    return fig, ax


def basemap(ax, ext, basin):
    shift = basin in SHIFT; pc = ccrs.PlateCarree()
    bb = box(ext[0] - 3, ext[2] - 3, ext[1] + 3, ext[3] + 3)
    ax.set_facecolor(C_SEA)
    ax.add_geometries(shifted(land_geoms(), bb, shift), pc, fc=C_LAND, ec="none", zorder=0)
    try:                                       # state / province lines (none north of 30.5°N between 60 and 100°E)
        st = shapereader.Reader(shapereader.natural_earth("10m", "cultural", "admin_1_states_provinces_lines")).geometries()
        hole = box(60, 30.5, 100, 45)
        sl = [x for g in shifted(list(st), bb, shift) for x in lines_only(g.difference(hole))]
        ax.add_geometries(sl, pc, fc="none", ec="#aaaaaa", lw=0.4, zorder=1)
    except Exception as e:
        print("  (no state lines:", e, ")")
    b = boundaries()
    for kind, lw, col in (("coast", 0.7, "#333"), ("borders", 0.8, "#444")):
        parts = [x for g in shifted([b[f"{kind}_fine"]], bb, shift) for x in lines_only(g)]
        ax.add_geometries(parts, pc, fc="none", ec=col, lw=lw, zorder=1)
    import matplotlib.ticker as mticker
    step = 5 if ext[1] - ext[0] <= 40 else 10
    gl = ax.gridlines(draw_labels=True, lw=0.4, color="gray", alpha=0.5, ls="--", crs=pc)
    gl.xlocator = mticker.FixedLocator([((v + 180) % 360) - 180 for v in np.arange(-180, 540, step)])
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 5))
    gl.top_labels = gl.right_labels = False
    if basin in NIO:
        cities = {k: v for k, v in CITIES_NIO.items()}
    else:
        cities = {}
        for name, ctry, la, lo, sr in sorted(PLACES["big"] + PLACES["coast"], key=lambda x: x[4]):
            lo = lo % 360 if shift else lo
            if ext[0] < lo < ext[1] and ext[2] < la < ext[3] and len(cities) < 16 and sr <= 4:
                if all(np.hypot(lo - x, la - y) > 2.0 for x, y in cities.values()): cities[name] = (lo, la)
    for c, (x, y) in cities.items():
        x = x % 360 if shift else x
        if ext[0] < x < ext[1] and ext[2] < y < ext[3]:
            ax.plot(x, y, "ks", ms=3, zorder=6, transform=pc)
            ax.text(x + 0.15, y + 0.1, c, fontsize=7.5, zorder=6, transform=pc)


def frame(basin, allm):
    q = allm[(allm.lat.between(allm.lat.quantile(.01), allm.lat.quantile(.99))) &
             (allm.lon.between(allm.lon.quantile(.01), allm.lon.quantile(.99)))]
    if basin in BASIN_FRAME:
        f = list(BASIN_FRAME[basin]); lo_min, lo_max = BASIN_LIMIT[basin]
        if len(q):
            f[0] = max(lo_min, min(f[0], float(q.lon.min()) - 2)); f[1] = min(lo_max, max(f[1], float(q.lon.max()) + 2))
            f[2] = max(-5, min(f[2], float(q.lat.min()) - 1.5)); f[3] = min(40, max(f[3], float(q.lat.max()) + 1.5))
        return f
    x0, x1 = float(q.lon.min()) - 3, float(q.lon.max()) + 3
    y0, y1 = float(q.lat.min()) - 2.5, float(q.lat.max()) + 2.5
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max(x1 - x0, 22.0); h = max(y1 - y0, 18.0)
    w, h = max(w, h * 1.1), max(h, w / 1.4)                              # not too thin in either direction
    y0, y1 = max(-70, cy - h / 2), min(70, cy + h / 2)
    return [cx - w / 2, cx + w / 2, y0, y1]


def tfmt(a):
    a.xaxis.set_major_locator(mdates.HourLocator(byhour=[0])); a.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n00Z")); a.grid(alpha=0.4)


# ---------------------------------------------------------------- landfall
def coast_names(basin, ext):
    if basin in NIO: return {k: (v[0], v[1]) for k, v in COAST_NIO.items()}
    shift = basin in SHIFT; out = {}
    for name, ctry, la, lo, sr in PLACES["coast"]:
        lo = lo % 360 if shift else lo
        if ext[0] - 8 < lo < ext[1] + 8 and ext[2] - 8 < la < ext[3] + 8:
            out[f"{name} ({ctry})" if ctry else name] = (la, lo)
    return out


def near(la, lo, names):
    if not names: return f"{fmt_lat(la)} {fmt_lon(lo)}"
    k = min(names, key=lambda c: dist_deg(names[c][0], names[c][1], la, lo))
    dy = (la - names[k][0]) * 111.2; dx = (lo - names[k][1]) * 111.2 * np.cos(np.radians(la)); d = np.hypot(dx, dy)
    return f"near {k}" if d < 10 else f"{d:.0f} km {DIRS[int((np.degrees(np.arctan2(dx, dy)) % 360 + 11.25) // 22.5) % 16]} of {k}"


def landfall(d, LAND):
    d = d.sort_values("t").reset_index(drop=True); on = [LAND.contains(Point(x.lon, x.lat)) for x in d.itertuples()]
    for i in range(1, len(d)):
        if on[i] and not on[i - 1]:
            a, b = d.iloc[i - 1], d.iloc[i]; lo_, hi_ = 0.0, 1.0
            for _ in range(25):
                f = (lo_ + hi_) / 2
                if LAND.contains(Point(a.lon + f * (b.lon - a.lon), a.lat + f * (b.lat - a.lat))): hi_ = f
                else: lo_ = f
            f = hi_; ip = lambda x, y: x + f * (y - x)
            return dict(t=(a.t + (b.t - a.t) * f).round("h"), lat=ip(a.lat, b.lat), lon=ip(a.lon, b.lon), w=ip(a.w, b.w), p=ip(a.p, b.p))
    return None


# ---------------------------------------------------------------- products for one system
def products(s, run, out):
    os.makedirs(out, exist_ok=True)
    b = s["basin"]; IC = f"{run:%H} UTC {run:%d %b %Y}"; BN = s["name"]; pc = ccrs.PlateCarree()
    ENS = s["ens"]; HRES = s["hres"]
    CTRL = next((t for t in ENS if t["kind"] == "Control"), None); CTRL_RAW = CTRL; HLAB = "HRES"
    if HRES and CTRL and len(HRES["df"]) == len(CTRL["df"]) and np.allclose(HRES["df"][["lat", "lon"]], CTRL["df"][["lat", "lon"]]):
        CTRL = None; HLAB = "HRES (= ENS control)"
    allm = pd.concat([t["df"].assign(member=t["member"]) for t in ENS])
    g = allm.groupby("t")
    stat = pd.DataFrame(dict(n=g.size(), lat=g.lat.mean(), lon=g.lon.mean(), w=g.w.mean(), w10=g.w.quantile(.1),
                             w90=g.w.quantile(.9), wmax=g.w.max(), p=g.p.mean(), p10=g.p.quantile(.1),
                             p90=g.p.quantile(.9), pmin=g.p.min())).reset_index()
    mean = stat[stat.n >= MIN_MEMBERS]
    if mean.empty: mean = stat[stat.n == stat.n.max()]
    ext = frame(b, allm); shift = b in SHIFT
    nmem = sum(t["kind"] == "Member" for t in ENS)
    title_b = ""
    src_txt = f"Source: ECMWF open data (CC BY 4.0) · {SITE}"
    def src(ax):
        ax.text(0.005, 0.005, src_txt, transform=ax.transAxes, fontsize=7.5, color="#333", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    def write_track(df, fn, note):
        with open(fn, "w") as f:
            f.write(f"# {note}\n# columns: 1 YYYYMMDDHH lat lon(°E{', 0-360' if shift else ''}) MSW(kt) MSLP(hPa)\n")
            for r in df.itertuples():
                f.write(f"1 {r.t:%Y%m%d%H} {r.lat:.1f} {r.lon:.1f} {r.w:.0f} {r.p:.0f}\n")
    files = []
    write_track(mean, f"{out}/ECMWF_ENSMEAN_track.txt", f"ECMWF ENS mean, {BN}, IC {run:%Y%m%d %H}Z, positions where >= {MIN_MEMBERS} members track the system")
    files.append(("Ensemble-mean track", "ECMWF_ENSMEAN_track.txt"))
    href = HRES or CTRL_RAW
    if href:
        write_track(href["df"], f"{out}/ECMWF_HRES_track.txt",
                    f"ECMWF {HLAB if HRES else 'ENS CONTROL (HRES has no track for this system)'}, {BN}, IC {run:%Y%m%d %H}Z")
        files.append(("HRES track", "ECMWF_HRES_track.txt"))
    allm.to_csv(f"{out}/ECMWF_ENS_all_members.csv", index=False)

    # E1 spaghetti
    fig, ax = make_axes(ext, shift); basemap(ax, ext, b)
    for t in ENS:
        if t["kind"] == "Member": ax.plot(t["df"].lon, t["df"].lat, "-", color=C_MEM, lw=0.7, alpha=0.5, zorder=2, transform=pc)
    ax.plot([], [], "-", color=C_MEM, lw=1, label=f"ENS members ({nmem}/50)")
    if CTRL: ax.plot(CTRL["df"].lon, CTRL["df"].lat, "--", color=C_CTRL, lw=2, label="Control", zorder=3, transform=pc)
    if HRES: ax.plot(HRES["df"].lon, HRES["df"].lat, "-", color=C_HRES, lw=2.2, label=HLAB, zorder=3, transform=pc)
    ax.plot(mean.lon, mean.lat, "-", color=C_MEAN, lw=3, label=f"Ensemble mean (≥{MIN_MEMBERS} members)", zorder=4, transform=pc)
    for r in mean[mean.t.dt.hour == 0].itertuples():
        ax.plot(r.lon, r.lat, "o", mfc="white", mec=C_MEAN, mew=2, ms=7, zorder=5, transform=pc)
        ax.annotate(f"{r.t:%d/%H}", (r.lon, r.lat), xycoords=pc._as_mpl_transform(ax), xytext=(6, -10), textcoords="offset points",
                    fontsize=8, fontweight="bold", zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"ECMWF ENS Tropical Cyclone Tracks — IC: {IC}\n{BN}", fontsize=12.5); src(ax)
    plt.savefig(f"{out}/E1_ENS_spaghetti.png", dpi=150, bbox_inches="tight"); plt.close()

    # E2 strike probability
    lons = np.arange(ext[0], ext[1] + 0.01, 0.25); lats = np.arange(ext[2], ext[3] + 0.01, 0.25)
    LO, LA = np.meshgrid(lons, lats); hit = np.zeros(LO.shape)
    for t in ENS:
        d = t["df"].set_index("t")[["lat", "lon"]].resample("1h").interpolate().dropna()
        m = np.zeros(LO.shape, bool)
        for la, lo in zip(d.lat, d.lon):
            m |= np.hypot((LA - la) * 111.2, (LO - lo) * 111.2 * np.cos(np.radians(la))) <= STRIKE_KM
        hit += m
    prob = 100 * hit / 51
    fig, ax = make_axes(ext, shift); basemap(ax, ext, b)
    lev = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    if prob.max() >= 5:
        cs = ax.contourf(LO, LA, prob, levels=lev, cmap="RdPu", alpha=0.85, zorder=2, transform=pc)
        ax.contour(LO, LA, prob, levels=[10, 50, 90], colors="k", linewidths=0.6, zorder=3, transform=pc)
        plt.colorbar(cs, ax=ax, shrink=0.7, label=f"Probability (%) of centre passing within {STRIKE_KM} km")
    ax.plot(mean.lon, mean.lat, "-", color=C_MEAN, lw=2.2, zorder=4, label="Ensemble mean", transform=pc)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"ECMWF ENS Strike Probability (whole forecast) — IC: {IC}\n{BN}", fontsize=12.5); src(ax)
    plt.savefig(f"{out}/E2_ENS_strike_probability.png", dpi=150, bbox_inches="tight"); plt.close()

    # E3 / E4 plumes
    cats = CATS_NIO if b in NIO else CATS_SS
    def plume(var, fn, ylab, title):
        fig, (ax, an) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[5, 1], hspace=0.08))
        if var == "w":
            top = np.nanmax(stat.wmax) + 8; ax.set_ylim(10, top)
            for lo, hi, n, c in cats:
                if lo <= top:
                    ax.axhspan(lo, min(hi + 1, top), color=c, zorder=0)
                    ax.text(1.005, (lo + min(hi + 1, top)) / 2, n, transform=ax.get_yaxis_transform(), va="center", fontsize=9, fontweight="bold")
        for t in ENS:
            if t["kind"] == "Member": ax.plot(t["df"].t, t["df"][var], "-", color=C_MEM, lw=0.6, alpha=0.4)
        ax.fill_between(mean.t, mean[var + "10"], mean[var + "90"], color=C_MEM, alpha=0.22, label="10–90 percentile")
        ax.plot(mean.t, mean[var], "-", color=C_MEAN, lw=2.8, label="Ensemble mean")
        if CTRL: ax.plot(CTRL["df"].t, CTRL["df"][var], "--", color=C_CTRL, lw=2, label="Control")
        if HRES: ax.plot(HRES["df"].t, HRES["df"][var], "-", color=C_HRES, lw=2.2, label=HLAB)
        if var == "p": ax.invert_yaxis()
        ax.set_ylabel(ylab); ax.legend(loc="upper left", fontsize=9, ncol=2); tfmt(ax)
        ax.set_title(f"{title} — IC: {IC}\n{BN}", fontsize=13)
        an.bar(stat.t, stat.n, width=0.2, color=C_MEM, alpha=0.7); an.set_ylabel("Members"); an.set_ylim(0, 52); tfmt(an)
        src(an)
        plt.savefig(f"{out}/{fn}", dpi=150, bbox_inches="tight"); plt.close()
    plume("w", "E3_ENS_MSW_plume.png", "Maximum sustained wind (kt)", "ECMWF ENS Maximum Sustained Wind")
    plume("p", "E4_ENS_MSLP_plume.png", "Central MSLP (hPa)", "ECMWF ENS Central Pressure")

    # landfall statistics
    lo_all = (min(allm.lon.min(), ext[0]) - 5, max(allm.lon.max(), ext[1]) + 5)
    bb = box(lo_all[0], max(-80, allm.lat.min() - 5), lo_all[1], min(80, allm.lat.max() + 5))
    LAND = prep(unary_union(shifted(land_geoms(), bb, shift)))
    names = coast_names(b, ext)
    mean_lf = landfall(mean, LAND); h_lf = landfall(href["df"], LAND) if href else None
    L = pd.DataFrame([dict(member=k, **l) for k, d in allm.groupby("member") if (l := landfall(d, LAND))])
    n_all = allm.member.nunique()
    pk = mean.loc[mean.w.idxmax()] if mean.w.notna().any() else None
    hp = href["df"].loc[href["df"].w.idxmax()] if href is not None and href["df"].w.notna().any() else None
    hname = "HRES" if HRES else "Control"
    rows = [["Ensemble-mean peak", f"{pk.w:.0f} kt ({cat(pk.w, b)}), {pk.t:%HZ %d %b}, {pk.p:.0f} hPa" if pk is not None else "–"],
            ["Ensemble-mean landfall", f"{mean_lf['t']:%HZ %d %b}, {near(mean_lf['lat'], mean_lf['lon'], names)}" if mean_lf else "no landfall"],
            [f"{hname} peak", f"{hp.w:.0f} kt ({cat(hp.w, b)}), {hp.t:%HZ %d %b}, {hp.p:.0f} hPa" if hp is not None else "–"],
            [f"{hname} landfall", f"{h_lf['t']:%HZ %d %b}, {near(h_lf['lat'], h_lf['lon'], names)}, {h_lf['w']:.0f} kt" if h_lf else "no landfall"],
            ["Members making landfall", f"{len(L)} / {n_all}"]]
    bins = None
    if len(L):
        q10, q50, q90 = L.w.quantile([.1, .5, .9]).round().astype(int)
        if b in NIO:
            bins = pd.cut(L.lat, [0, 15, 16, 17, 18, 19, 20, 30]).value_counts().sort_index()
            bl = lambda iv: f"{iv.left:.0f}–{iv.right:.0f}°N"
        else:
            e = np.arange(np.floor(L.lat.min()), np.ceil(L.lat.max()) + 1.001, 1.0)
            if len(e) < 2: e = np.array([np.floor(L.lat.min()), np.floor(L.lat.min()) + 1])
            bins = pd.cut(L.lat, e, include_lowest=True).value_counts().sort_index()
            bl = lambda iv: (f"{iv.left:.0f}–{iv.right:.0f}°N" if iv.left >= 0 else f"{abs(iv.right):.0f}–{abs(iv.left):.0f}°S")
        top = bins.idxmax()
        rows += [["Median landfall latitude", f"{fmt_lat(L.lat.median())} (50 % between {L.lat.quantile(.25):.1f} and {fmt_lat(L.lat.quantile(.75))})"],
                 ["Median landfall time", f"{L.t.median():%HZ %d %b} (50 % between {L.t.quantile(.25):%HZ %d} and {L.t.quantile(.75):%HZ %d})"],
                 ["Median MSW at landfall", f"{q50} kt ({cat(q50, b)}); 10–90 %: {q10}–{q90} kt"],
                 ["Most likely 1° latitude band", f"{bl(top)}: {int(bins.max())} members"]]
        L.assign(lon=L.lon.map(lambda v: round((v + 180) % 360 - 180, 2))).to_csv(f"{out}/ECMWF_ENS_member_landfalls.csv", index=False)
        files.append(("Member landfalls (csv)", "ECMWF_ENS_member_landfalls.csv"))

    fig = plt.figure(figsize=(7.4, 7.6), dpi=150)
    fig.text(0.5, 0.955, "ECMWF Ensemble — landfall summary", ha="center", fontsize=16, fontweight="bold", color=C_HEAD)
    fig.text(0.5, 0.92, f"{BN} · IC {IC} · {n_all} members", ha="center", fontsize=11)
    ax = fig.add_axes([0.03, 0.20, 0.94, 0.68]); ax.axis("off")
    cells = [[k, v] for k, v in rows]
    if mean_lf: cells[1][1] += f"\n{fmt_lat(mean_lf['lat'])} {fmt_lon(mean_lf['lon'])}, {mean_lf['w']:.0f} kt"
    tb = ax.table(cellText=cells, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[0.34, 0.66])
    tb.auto_set_font_size(False); tb.set_fontsize(9.4)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#bbb"); c.set_facecolor("#eef1f5" if j == 0 else C_HIL)
        if j == 0: c.get_text().set_fontweight("bold")
    if bins is not None:
        lat_bins = "  ".join(f"{bl(iv)}: {n}" for iv, n in bins.items() if n)
        fig.text(0.5, 0.13, "Members by landfall latitude:\n" + "\n".join(textwrap.wrap(lat_bins, 95)), ha="center", fontsize=8.8, color="#333")
    fig.text(0.5, 0.05, f"Source: ECMWF open data (CC BY 4.0); own analysis · {SITE}", ha="center", fontsize=8.5, style="italic", color="#555")
    plt.savefig(f"{out}/ECMWF_ENS_landfall_summary.png", dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white"); plt.close()

    items = [("Landfall summary", "ECMWF_ENS_landfall_summary.png"), ("Spaghetti tracks", "E1_ENS_spaghetti.png"),
             ("Strike probability", "E2_ENS_strike_probability.png"), ("MSW plume", "E3_ENS_MSW_plume.png"),
             ("MSLP plume", "E4_ENS_MSLP_plume.png")]
    pkg = f"ECMWF_ENS_{s['id']}_{run:%Y%m%d%H}.zip"
    with zipfile.ZipFile(f"{out}/{pkg}", "w", zipfile.ZIP_DEFLATED) as z:
        for _, f in items + files: z.write(f"{out}/{f}", f)
        z.write(f"{out}/ECMWF_ENS_all_members.csv", "ECMWF_ENS_all_members.csv")
    os.remove(f"{out}/ECMWF_ENS_all_members.csv")
    files.insert(0, ("Package (zip)", pkg))
    track = [[f"{r.t:%Y%m%d%H}", round(r.lat, 2), round(r.lon, 2)] for r in mean.itertuples()]
    print(f"  {BN}: {nmem} members{' + control' if CTRL_RAW else ''}, HRES {'yes' if HRES else 'no'}, landfalls {len(L)}/{n_all}")
    return dict(id=s["id"], b=b, name=BN, sid=s["sid"], lat=round(float(s["la0"]), 2), lon=round(float(s["lo0"]), 2),
                rows=rows, items=[{"t": t, "f": f} for t, f in items], files=[{"t": t, "f": f} for t, f in files], track=track)


# ---------------------------------------------------------------- run comparison
def match_prev(sysd, prev_run):
    """The system of an earlier run that is the same storm: same basin, mean tracks within 4° at common times."""
    best = None
    cur = {t: (la, lo) for t, la, lo in sysd["track"]}
    for o in prev_run.get("systems", []):
        if o["b"] != sysd["b"]: continue
        d = [dist_deg(la, lo, *cur[t]) for t, la, lo in o.get("track", []) if t in cur]
        if not d:                                       # no common time: compare this run's start with their track
            pts = [(la, lo) for t, la, lo in o.get("track", [])]
            d = [min(dist_deg(la, lo, sysd["lat"], sysd["lon"]) for la, lo in pts)] if pts else []
        if d and min(d) < 4 and (best is None or min(d) < best[0]): best = (min(d), o["id"])
    return best[1] if best else None


def compare_png(cols, out, name, updated):
    """WhatsApp-size table of the last runs of one system (oldest -> newest)."""
    keys = []
    for c in cols:
        for k, v in c["rows"]:
            if k not in keys: keys.append(k)
    wrap = lambda v: "\n".join(textwrap.wrap(v, 18)) if v else "–"
    rows = [[k] + [wrap(dict(c["rows"]).get(k, "")) for c in cols] for k in keys]
    hdr = [""] + [c["label"].rsplit(" ", 1)[0].replace(" UTC ", " UTC\n") for c in cols]
    w = 2.6 + 2.0 * len(cols)
    fig = plt.figure(figsize=(w, 1.05 * len(rows) + 2.2), dpi=150)
    fig.text(0.5, 0.97, f"ECMWF Ensemble — run comparison ({name})", ha="center", va="top", fontsize=16, fontweight="bold", color=C_HEAD)
    fig.text(0.5, 0.925, f"Last {len(cols)} runs (oldest → newest)  ·  {updated}", ha="center", va="top", fontsize=10)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.82]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=hdr, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[2.6 / w] + [2.0 / w] * len(cols))
    tb.auto_set_font_size(False); tb.set_fontsize(8.6)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#bbb")
        if i == 0: c.set_facecolor(C_HEAD if j else "white"); c.get_text().set_color("white"); c.get_text().set_fontweight("bold")
        elif j == 0: c.set_facecolor("#eef1f5"); c.get_text().set_fontweight("bold")
        elif j == len(cols): c.set_facecolor(C_HIL); c.get_text().set_fontweight("bold")
    fig.text(0.5, 0.02, f"Source: ECMWF open data (CC BY 4.0); own analysis · {SITE}", ha="center", fontsize=8, style="italic", color="#555")
    plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.2); plt.close()


# ---------------------------------------------------------------- storage (R2 or a local folder)
class Store:
    def __init__(self, local=None):
        self.local = local
        if not local:
            sys.path.insert(0, HERE); import publish
            self.p = publish; self.s3 = publish.r2()
    def get_json(self, key, default):
        if self.local:
            f = os.path.join(self.local, key)
            return json.load(open(f)) if os.path.exists(f) else default
        return self.p.get_json(self.s3, key, default)
    def put(self, key, body, ctype, cache):
        if self.local:
            f = os.path.join(self.local, key); os.makedirs(os.path.dirname(f), exist_ok=True)
            open(f, "wb").write(body if isinstance(body, bytes) else body.encode()); return
        self.p.put(self.s3, key, body, ctype, cache)
    def delete_prefix(self, prefix):
        if self.local:
            import shutil; shutil.rmtree(os.path.join(self.local, prefix), ignore_errors=True); return
        pg = self.s3.get_paginator("list_objects_v2")
        for page in pg.paginate(Bucket=self.p.BUCKET, Prefix=prefix):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys: self.s3.delete_objects(Bucket=self.p.BUCKET, Delete={"Objects": keys, "Quiet": True})


CTYPE = {".png": "image/png", ".txt": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".zip": "application/zip"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--force", action="store_true"); ap.add_argument("--local")
    ap.add_argument("--work", default="work_cyclone")
    a = ap.parse_args()
    run = dt.datetime.strptime(a.run, "%Y%m%d%H") if a.run else latest_run()
    if run is None: print("no ECMWF cyclone-track file in the last 3 days"); return
    st = Store(a.local)
    index = st.get_json("cyclone/ens/index.json", {"runs": []})
    todo = [run]
    if not index["runs"] and not a.run:                     # first time: also the earlier runs ECMWF still has
        todo = [run - dt.timedelta(hours=12 * k) for k in (3, 2, 1, 0)]
    for r in todo:
        try:
            publish_run(r, st, a)
        except Exception as e:
            print(f"run {r:%Y%m%d%H} failed: {e}")
            if r == run: raise


def publish_run(run, st, a):
    ic = f"{run:%Y%m%d%H}"
    index = st.get_json("cyclone/ens/index.json", {"runs": []})
    if any(r["ic"] == ic for r in index["runs"]) and not a.force:
        print(f"ECMWF ENS {ic} already published"); return
    os.makedirs(a.work, exist_ok=True)
    eu = tf_url(run, "enfo"); hu = tf_url(run, "oper")
    if not eu: print(f"ECMWF ENS tracks for {ic} not out yet"); return
    ef = download(eu, f"{a.work}/enfo_{ic}.bufr")
    hf = download(hu, f"{a.work}/oper_{ic}.bufr") if hu else None
    ens = decode(ef, "ENS"); hres = decode(hf, "HRES") if hf else []
    systems = find_systems(ens, hres)
    print(f"run {ic}: {len(systems)} system(s): " + ", ".join(f"{s['name']} ({len(s['ens'])} members)" for s in systems))
    label = f"{run:%H} UTC {run.day} {run:%b %Y}"
    ist = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=5, minutes=30)
    updated = f"{ist:%d %b %Y %H:%M} IST"
    entry = {"ic": ic, "label": label, "systems": []}
    for s in systems:
        out = os.path.join(a.work, ic, s["id"])
        try:
            entry["systems"].append(products(s, run, out))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {s['name']}: FAILED {e}")
    runs = [r for r in index["runs"] if r["ic"] != ic] + [entry]
    runs.sort(key=lambda r: r["ic"], reverse=True)
    cut = (run - dt.timedelta(days=KEEP_DAYS)).strftime("%Y%m%d%H")
    old = [r for r in runs if r["ic"] < cut]; runs = [r for r in runs if r["ic"] >= cut]
    # link each system to the same storm in the previous run, and make the comparison PNG (last 6 runs)
    pos = [r["ic"] for r in runs].index(ic)
    byic = {r["ic"]: r for r in runs}
    for sd in entry["systems"]:
        sd["prev"] = None
        for r in runs[pos + 1:pos + 4]:                     # the same storm in one of the 3 previous runs
            m = match_prev(sd, r)
            if m: sd["prev"] = {"ic": r["ic"], "id": m}; break
        cols = [dict(label=label, rows=sd["rows"])]; p = sd["prev"]
        while p and len(cols) < 6 and p["ic"] in byic:
            o = next((x for x in byic[p["ic"]]["systems"] if x["id"] == p["id"]), None)
            if not o: break
            cols.insert(0, dict(label=byic[p["ic"]]["label"], rows=o["rows"])); p = o.get("prev")
            if isinstance(p, str): p = None                  # (older index format)
        if len(cols) >= 2:
            fn = f"ECMWF_ENS_comparison_{sd['id']}.png"
            compare_png(cols, os.path.join(a.work, ic, sd["id"], fn), sd["name"], updated); sd["cmp"] = fn
    # upload: files first, the index last
    n = 0
    for sd in entry["systems"]:
        d = os.path.join(a.work, ic, sd["id"])
        for f in os.listdir(d):
            ext = os.path.splitext(f)[1]
            st.put(f"cyclone/ens/{ic}/{sd['id']}/{f}", open(os.path.join(d, f), "rb").read(),
                   CTYPE.get(ext, "application/octet-stream"), "public, max-age=86400"); n += 1
    index = {"updated": updated, "keep_days": KEEP_DAYS, "runs": runs}
    st.put("cyclone/ens/index.json", json.dumps(index, separators=(",", ":")), "application/json", "public, max-age=120")
    for r in old: st.delete_prefix(f"cyclone/ens/{r['ic']}/")
    print(f"published ECMWF ENS {ic}: {len(entry['systems'])} system(s), {n} files")


if __name__ == "__main__":
    main()
