"""Chart engine: draws every field x region x forecast hour for one ECMWF IFS run.
Usage: python charts.py <run_dir> [--fields a,b] [--regions india,global] [--steps 0,24] [--jobs 2]
Output: out/ifs/<run>/<field>/<region>/<step:03d>.png  + out/ifs/<run>/manifest.json
"""
import os, json, time, argparse, datetime as dt
import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs, cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter, uniform_filter
# ================================================================ regions (lon may pass 180 for Pacific views)
REGIONS = {
    "global":  dict(name="Global",                         lon=(-180, 180), lat=(-80, 85), proj="rob"),
    "india":   dict(name="India",                          lon=(64, 100),   lat=(4, 38)),
    "nio":     dict(name="North Indian Ocean & S. Asia",   lon=(40, 110),   lat=(-15, 35)),
    "wd":      dict(name="Mediterranean to India (WD track)", lon=(10, 100), lat=(10, 55)),
    "easia":   dict(name="East Asia",                      lon=(95, 150),   lat=(10, 55)),
    "wpac":    dict(name="West Pacific",                   lon=(100, 180),  lat=(-5, 45)),
    "aus":     dict(name="Australia & SW Pacific",         lon=(105, 185),  lat=(-48, 2)),
    "europe":  dict(name="Europe",                         lon=(-30, 45),   lat=(30, 72)),
    "africa":  dict(name="Africa & Middle East",           lon=(-25, 65),   lat=(-38, 40)),
    "namer":   dict(name="North America",                  lon=(-135, -55), lat=(15, 62)),
    "atl":     dict(name="Tropical Atlantic",              lon=(-100, -10), lat=(0, 45)),
    "samer":   dict(name="South America",                  lon=(-95, -28),  lat=(-58, 15)),
}


BRAND = os.environ.get("SITE_NAME", "globalweathercharts.com")
KT = 1.94384
DATA = {}   # filled once, then shared with forked workers

# ================================================================ data
def _open(path, short, level=None):
    keys = {"shortName": short}
    if level is not None: keys["level"] = level
    ds = xr.open_dataset(path, engine="cfgrib", indexpath="", backend_kwargs={"filter_by_keys": keys})
    da = ds[list(ds.data_vars)[0]]
    return da


def load_run(run_dir):
    if os.path.exists(os.path.join(run_dir, "index.json")):
        load_gfs(run_dir)
    else:
        load_ifs(run_dir)
    postprocess()
    return DATA


def load_gfs(run_dir):
    """NOAA GFS 0.25°: messages listed in index.json (written by fetch_gfs.py)."""
    import eccodes
    meta = json.load(open(os.path.join(run_dir, "index.json")))
    steps = sorted({s for s, _ in meta["messages"]})
    k_of = {s: i for i, s in enumerate(steps)}
    nlat, nlon = 721, 1440
    store = {}
    with open(os.path.join(run_dir, "gfs.grib2"), "rb") as fh:
        for step, key in meta["messages"]:
            g = eccodes.codes_grib_new_from_file(fh)
            vals = eccodes.codes_get_values(g).reshape(nlat, nlon).astype(np.float32)
            eccodes.codes_release(g)
            vals = np.roll(vals, nlon // 2, axis=1)                    # 0..360 -> -180..180
            if key == "orog": store["orog"] = vals; continue
            arr = store.setdefault(key, np.zeros((len(steps), nlat, nlon), np.float32))
            arr[k_of[step]] = vals
    lat = np.linspace(90, -90, nlat); lon = np.linspace(-180, 179.75, nlon)
    f = (2 * 7.2921e-5 * np.sin(np.radians(lat)))[:, None].astype(np.float32)
    DATA.update(model="gfs", label="NOAA GFS 0.25°", credit="Data: NOAA/NCEP GFS",
                steps=steps, init=dt.datetime.strptime(meta["run"], "%Y%m%d%H"), lat=lat, lon=lon,
                orog=store["orog"], msl=store["msl"] / 100.0,
                tp=store.get("tp", np.zeros_like(store["msl"])), u10=store["u10"] * KT, v10=store["v10"] * KT,
                mucape=store["mucape"], tcwv=store["tcwv"], vo500=(store["absv500"] - f) * 1e5,
                t850=store["t850"] - 273.15)
    for L in (850, 500, 200):
        DATA[f"gh{L}"] = store[f"gh{L}"]; DATA[f"u{L}"] = store[f"u{L}"] * KT; DATA[f"v{L}"] = store[f"v{L}"] * KT


def load_ifs(run_dir):
    sfc, pl, const = (os.path.join(run_dir, f) for f in ("sfc.grib2", "pl.grib2", "const.grib2"))
    msl = _open(sfc, "msl")
    DATA["steps"] = [int(s / np.timedelta64(1, "h")) for s in msl.step.values]
    DATA["init"] = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=int(msl.time.values.astype("datetime64[s]").astype(int)))
    DATA["lat"], DATA["lon"] = msl.latitude.values, msl.longitude.values
    z = _open(const, "z").values / 9.80665
    DATA["orog"] = z
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    DATA["msl"] = f32(msl.values / 100.0)
    DATA["tp"] = f32(_open(sfc, "tp").values * 1000.0)
    DATA["u10"] = f32(_open(sfc, "10u").values * KT); DATA["v10"] = f32(_open(sfc, "10v").values * KT)
    DATA["mucape"] = f32(_open(sfc, "mucape").values); DATA["tcwv"] = f32(_open(sfc, "tcwv").values)
    for short, levs in (("gh", (850, 500, 200)), ("u", (850, 500, 200)), ("v", (850, 500, 200)), ("t", (850,)), ("vo", (500,))):
        da = _open(pl, short)
        for L in levs:
            a = da.sel(isobaricInhPa=L).values
            if short in ("u", "v"): a = a * KT
            if short == "t": a = a - 273.15
            if short == "vo": a = a * 1e5
            DATA[f"{short}{L}"] = f32(a)
    DATA.update(model="ifs", label="ECMWF IFS 0.25°", credit="Data: ECMWF open data (CC BY 4.0)")


def postprocess():
    z = DATA["orog"]
    # MSLP is fictitious over high ground: blend towards a heavily smoothed field there
    w = np.clip((z - 500.0) / 1500.0, 0, 1)
    for k in range(DATA["msl"].shape[0]):
        p = DATA["msl"][k]
        DATA["msl"][k] = (1 - w) * gaussian_filter(p, 1.0) + w * gaussian_filter(p, 6.0)
    # vorticity: light smoothing (raw 0.25° vorticity is very noisy) and sign flip in the
    # southern hemisphere so that "cyclonic" is positive everywhere
    sgn = np.where(DATA["lat"] < 0, -1.0, 1.0)[:, None]
    DATA["vo500"] = np.stack([gaussian_filter(a, 1.5) * sgn for a in DATA["vo500"]]).astype(np.float32)


def region_view(lon_rng, lat_rng, stride=1):
    lat, lon = DATA["lat"], DATA["lon"]
    lonx = lon % 360 if lon_rng[1] > 180 else lon
    ii = np.where((lonx >= lon_rng[0] - 1) & (lonx <= lon_rng[1] + 1))[0]
    ii = ii[np.argsort(lonx[ii])][::stride]
    jj = np.where((lat >= lat_rng[0] - 1) & (lat <= lat_rng[1] + 1))[0][::stride]
    lx = lonx[ii]
    if lon_rng == (-180, 180):                      # close the seam at the dateline
        ii = np.append(ii, ii[0]); lx = np.append(lx, lx[0] + 360)
    return lx, lat[jj], (lambda a: a[..., jj, :][..., ii])


# ================================================================ derived quantities (k = step index)
def at(name):
    return lambda k: DATA[name][k]


def precip6(k):
    s = DATA["steps"][k]
    if s < 6: return None
    return np.maximum(DATA["tp"][k] - DATA["tp"][DATA["steps"].index(s - 6)], 0)


def precip_total(k):
    return None if DATA["steps"][k] == 0 else np.maximum(DATA["tp"][k], 0)


def speed(lev):
    return lambda k: np.hypot(DATA[f"u{lev}"][k], DATA[f"v{lev}"][k])


# ================================================================ colour scales
def palette(colors, over=None, under=None):
    c = ListedColormap(colors)
    if over: c.set_over(over)
    if under: c.set_under(under)
    return c


def ramp(name, n, lo=0.0, hi=1.0):
    cm = plt.get_cmap(name)
    return [cm(lo + (hi - lo) * i / max(1, n - 1)) for i in range(n)]


PCP = ["#c6ecc0", "#8fd68a", "#4fb85a", "#1f8f3a", "#0e6b2a", "#f2ea5a", "#f5c12e", "#f08a24",
       "#e2401f", "#b3171c", "#d24bd0", "#8a2be2"]

FIELDS = {
 "mslp_pcpn": dict(
    name="MSLP, 10 m wind & 6-h precipitation",
    title="MSLP (hPa), 10 m wind (kt), 6-h precipitation (mm)",
    shade=precip6, levels=[0.5, 1, 2, 4, 6, 10, 15, 20, 30, 40, 60, 80, 100], cmap=palette(PCP, over="#e9c8f5"),
    unit="6-h precipitation (mm)", extend="max",
    contour=at("msl"), cint=2, cint_global=4, cmask=3000, hl=True, barbs=("u10", "v10")),
 "apcp": dict(
    name="Total precipitation since start",
    title="Total precipitation since start of run (mm), MSLP (hPa)",
    shade=precip_total, levels=[1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500],
    cmap=palette(PCP, over="#e9c8f5"), unit="Accumulated precipitation (mm)", extend="max",
    contour=at("msl"), cint=4, cint_global=8, cmask=3000),
 "z500_vort": dict(
    name="500 hPa height, wind & vorticity",
    title="500 hPa geopotential height (dam), wind (kt), cyclonic relative vorticity (×10⁻⁵ s⁻¹)",
    shade=at("vo500"), levels=[4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50],
    cmap=palette(ramp("YlOrRd", 10, 0.08, 1.0), over="#5a0a3a"), unit="Cyclonic relative vorticity (×10⁻⁵ s⁻¹)", extend="max",
    contour=at("gh500"), cint=60, cint_global=60, cdiv=10, barbs=("u500", "v500"), ccolor="#1a1a1a"),
 "t850": dict(
    name="850 hPa temperature, height & wind",
    title="850 hPa temperature (°C), geopotential height (dam), wind (kt)",
    shade=at("t850"), levels=list(range(-30, 35, 2)),
    cmap=palette(ramp("turbo", 32, 0.05, 0.95), over="#5a0a0a", under="#2a0a4a"), unit="850 hPa temperature (°C)", extend="both",
    contour=at("gh850"), cint=30, cint_global=60, cdiv=10, cmask=1300, barbs=("u850", "v850")),
 "jet200": dict(
    name="200 hPa jet stream & height",
    title="200 hPa wind speed (kt), geopotential height (dam), wind (kt)",
    shade=speed(200), levels=[50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 190],
    cmap=palette(["#d6eaf8", "#a5d3f0", "#6ab6e6", "#3a8fd0", "#52c08a", "#b9d64a", "#f4d03f", "#f0932b",
                  "#e0562b", "#c0283c", "#9b2d9b", "#6c1d82"], over="#f2c9f2"),
    unit="200 hPa wind speed (kt)", extend="max",
    contour=at("gh200"), cint=120, cint_global=120, cdiv=10, barbs=("u200", "v200")),
 "cape": dict(
    name="CAPE & 10 m wind",
    title="Most-unstable CAPE (J/kg), 10 m wind (kt)",
    shade=at("mucape"), levels=[100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
    cmap=palette(["#e3f2d6", "#b8dfa0", "#7fc77a", "#f3e96a", "#f6bf3c", "#ef8427", "#de4321", "#b5182a", "#7d1160", "#b84fd0"],
                 over="#f0c8f8"), unit="Most-unstable CAPE (J/kg)", extend="max", barbs=("u10", "v10")),
 "pwat": dict(
    name="Precipitable water & MSLP",
    title="Total column water vapour / precipitable water (mm), MSLP (hPa)",
    shade=at("tcwv"), levels=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    cmap=palette(["#8a5a2b", "#b98a4d", "#d9b77c", "#efe0b8", "#d4ecd0", "#a9dcb0", "#72c5a8", "#3fa8b0",
                  "#2b86bd", "#2560a8", "#2a3f95", "#4b2a8c", "#7a2ea0"], over="#c060c0", under="#6a4020"),
    unit="Precipitable water (mm)", extend="both",
    contour=at("msl"), cint=4, cint_global=8, cmask=3000),
}


# ================================================================ coastlines & country boundaries
# boundaries_india_view.json: Natural Earth 10m "India point of view" countries (Jammu & Kashmir incl.
# Gilgit-Baltistan, Aksai Chin and Arunachal Pradesh shown as part of India), split into coastlines and
# land borders and simplified to two detail levels.
_BND = {}


def boundaries():
    if not _BND:
        from shapely.geometry import MultiLineString
        raw = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "boundaries_india_view.json")))
        _BND.update({k: MultiLineString(v) for k, v in raw.items()})
    return _BND


def draw_boundaries(ax, reg, glob):
    from shapely.geometry import box
    from shapely import affinity
    b = boundaries(); pc = ccrs.PlateCarree()
    for kind, lw, col in (("coast", 0.55 if glob else 0.85, "#1a1a1a"), ("borders", 0.4 if glob else 0.7, "#3a3a3a")):
        g = b[f"{kind}_{'coarse' if glob else 'fine'}"]
        if glob:
            parts = [g]
        else:
            (x0, x1), (y0, y1) = reg["lon"], reg["lat"]
            bx = box(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
            parts = [g.intersection(bx)]
            if x1 > 180: parts.append(affinity.translate(g, xoff=360).intersection(bx))
        parts = [p for p in parts if not p.is_empty]
        if parts:
            ax.add_geometries(parts, pc, facecolor="none", edgecolor=col, linewidth=lw, zorder=3)


# ================================================================ drawing
def figure_for(reg):
    if reg.get("proj") == "rob":
        fig = plt.figure(figsize=(13, 7.6), dpi=100)
        ax = fig.add_axes([0.02, 0.12, 0.96, 0.76], projection=ccrs.Robinson(central_longitude=80))
        ax.set_global()
        return fig, ax, 0.12
    (x0, x1), (y0, y1) = reg["lon"], reg["lat"]
    asp = (y1 - y0) / ((x1 - x0) * np.cos(np.radians((y0 + y1) / 2)))
    W = 11.0; maph = min(13.0, max(4.5, W * 0.90 * asp)); H = maph + 1.9
    fig = plt.figure(figsize=(W, H), dpi=105)
    ax = fig.add_axes([0.06, 1.15 / H, 0.90, maph / H],
                      projection=ccrs.PlateCarree(central_longitude=180 if x1 > 180 else 0))
    ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
    return fig, ax, 1.15 / H


def draw(field_key, region_key, step, out_path):
    F, reg = FIELDS[field_key], REGIONS[region_key]
    k = DATA["steps"].index(step)
    glob = reg.get("proj") == "rob"
    lons, lats, cut = region_view(*(((-180, 180), (-90, 90)) if glob else (reg["lon"], reg["lat"])), stride=2 if glob else 1)
    pc = ccrs.PlateCarree()
    fig, ax, bottom = figure_for(reg)
    orog = cut(DATA["orog"])

    sh = F["shade"](k)
    cf = None
    if sh is not None:
        cmap = F["cmap"]; norm = BoundaryNorm(F["levels"], cmap.N)
        cf = ax.contourf(lons, lats, cut(sh), levels=F["levels"], cmap=cmap, norm=norm, extend=F["extend"], transform=pc)
    draw_boundaries(ax, reg, glob)
    if not glob:
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none", edgecolor="#555", linewidth=0.4)

    if "contour" in F:
        c = cut(F["contour"](k)).astype(float)
        if F.get("cmask"): c = np.where(orog > F["cmask"], np.nan, c)
        div = F.get("cdiv", 1); ci = F["cint_global"] if glob else F["cint"]
        lev = np.arange(np.floor(np.nanmin(c) / ci) * ci, np.nanmax(c) + ci, ci)
        cs = ax.contour(lons, lats, c, levels=lev, colors=F.get("ccolor", "#111"), linewidths=0.6 if glob else 0.9, transform=pc)
        ax.clabel(cs, cs.levels[::2] if div == 1 else cs.levels, fmt=lambda v: f"{v / div:.0f}", fontsize=7 if glob else 8, inline_spacing=2)

    if F.get("hl"):
        p = cut(F["contour"](k)).astype(float)
        win = max(9, int(len(lons) / (10 if glob else 7)))
        for filt, lab, col, sign in ((maximum_filter, "H", "#0b3d91", 1), (minimum_filter, "L", "#b00020", -1)):
            ps = gaussian_filter(p, 2 if glob else 3)
            ext = (ps == filt(ps, size=win, mode="nearest")) & (orog < 1000)
            contrast = sign * (ps - uniform_filter(ps, size=win, mode="nearest"))
            for j, i in zip(*np.where(ext & (contrast > 0.8))):
                if 2 < j < p.shape[0] - 3 and 2 < i < p.shape[1] - 3 and abs(lats[j]) < 72:
                    ax.text(lons[i], lats[j], lab, color=col, fontsize=13 if glob else 18, fontweight="bold",
                            ha="center", va="center", transform=pc, clip_on=True)
                    ax.text(lons[i], lats[j], f"\n\n{p[j, i]:.0f}", color=col, fontsize=7 if glob else 8,
                            ha="center", va="center", transform=pc, clip_on=True)

    if F.get("barbs") and not glob:
        U, V = cut(DATA[F["barbs"][0]][k]), cut(DATA[F["barbs"][1]][k])
        sk = max(1, int(round((reg["lon"][1] - reg["lon"][0]) / 0.25 / 26)))
        LO, LA = np.meshgrid(lons[::sk], lats[::sk])
        Us, Vs = U[::sk, ::sk], V[::sk, ::sk]
        if F["barbs"][0] == "u850":                                  # 850 hPa is underground on high plateaus
            m = orog[::sk, ::sk] > 1300; Us = np.where(m, np.nan, Us); Vs = np.where(m, np.nan, Vs)
        ax.barbs(LO, LA, Us, Vs, length=5.2, linewidth=0.6, color="#111", transform=pc, flip_barb=LA < 0)

    if not glob:
        span = reg["lon"][1] - reg["lon"][0]; gs = 5 if span <= 40 else 10 if span <= 90 else 20
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#888", alpha=0.6, linestyle="--",
                          xlocs=mticker.MultipleLocator(gs), ylocs=mticker.MultipleLocator(gs))
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 9}

    init = DATA["init"]; valid = init + dt.timedelta(hours=step)
    H = fig.get_figheight(); top = 1 - 0.32 / H
    title = F["title"] if not glob else F["title"].replace(", 10 m wind (kt)", "").replace(", wind (kt)", "")
    fig.text(0.04, top, DATA["label"] + " · " + title, fontsize=12.5 if len(title) < 70 else 11, fontweight="bold", va="top")
    fig.text(0.04, top - 0.30 / H, f"Init: {init:%HZ %a %d %b %Y}   Forecast hour: {step}   Valid: {valid:%HZ %a %d %b %Y}",
             fontsize=10.5, va="top")
    fig.text(0.97, top - 0.30 / H, f"{BRAND}  ·  {DATA['credit']}", fontsize=9, ha="right", va="top", color="#555")
    cax = fig.add_axes([0.15, 0.075, 0.70, 0.022] if glob else [0.15, bottom - 0.62 / H, 0.70, 0.17 / H])
    if cf is not None:
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=F["levels"])
        cb.ax.tick_params(labelsize=8.5); cb.set_label(F["unit"], fontsize=9)
    else:
        cax.axis("off"); cax.text(0.5, 0.5, "Not available at the start of the run (0 h)", ha="center", va="center", fontsize=9)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import io
    from PIL import Image
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    Image.open(buf).convert("RGB").quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(out_path, "PNG", optimize=True)


def _job(args):
    f, r, s, path = args
    draw(f, r, s, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--regions"); ap.add_argument("--steps"); ap.add_argument("--fields")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    t0 = time.time(); load_run(a.run_dir); boundaries()
    run = DATA["init"].strftime("%Y%m%d%H")
    regions = a.regions.split(",") if a.regions else list(REGIONS)
    steps = [int(s) for s in a.steps.split(",")] if a.steps else DATA["steps"]
    fields = a.fields.split(",") if a.fields else list(FIELDS)
    jobs = [(f, r, s, os.path.join(a.out, DATA["model"], run, f, r, f"{s:03d}.png")) for f in fields for r in regions for s in steps]
    print(f"run {run}: {len(jobs)} charts, data loaded in {time.time() - t0:.0f}s", flush=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for n, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), 1):
            if n % 200 == 0: print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    man = dict(model=DATA["model"], label=DATA["label"], run=run, init=DATA["init"].isoformat() + "Z", steps=DATA["steps"],
               fields={f: FIELDS[f]["name"] for f in FIELDS}, regions={r: REGIONS[r]["name"] for r in REGIONS})
    with open(os.path.join(a.out, DATA["model"], run, "manifest.json"), "w") as fh: json.dump(man, fh, indent=1)
    print(f"done {len(jobs)} charts in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
