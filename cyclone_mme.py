"""Multi-model ensemble (MME) of open global models for the Bay of Bengal and the Arabian Sea -> R2 (cyclone/mme/...).

For each 00/12 UTC run it takes MSLP and 10 m wind from every open model as soon as that model's run is complete
(mme_fields.py: ECMWF HRES and AIFS, NCEP GFS, NOAA AIGFS, CMC GDPS, DWD ICON, UKMO Global), tracks every
depression / cyclone with one and the same tracker (mme_track.py), puts the tracks of the same storm from the
different models together, and makes, for each system:
  landfall summary (consensus + every model), tracks map, MSW plume, MSLP plume, track spread,
  run-comparison table (same storm in up to 6 runs), package zip, consensus track, all model tracks (csv).
The consensus is the plain (equal-weight) mean of the models tracking the system at that time (at least 3).
Storm numbers and names (e.g. 05B AKVAN) and the observed track come from the JTWC best
track as relayed by NCAR/RAL TCGP (public).

Each model's tracks are stored as soon as they are made (cyclone/mme/<run>/tracks.json), so a later invocation
only fetches the models that were not ready yet and re-draws the products with the larger set.

Usage: python cyclone_mme.py [--run YYYYMMDDHH] [--force] [--redraw] [--models a,b] [--local DIR] [--cache DIR]
  --local DIR   write to DIR instead of R2 (tests)     --cache DIR  reuse/keep model fields (.npz) there (tests)
Bucket layout:
  cyclone/mme/index.json                runs (newest first), the models in each run, and their systems
  cyclone/mme/<run>/tracks.json         every model's tracks for that run
  cyclone/mme/<run>/<system>/<file>     figures and files
"""
import os, re, sys, json, time, zipfile, argparse, textwrap, datetime as dt, urllib.request
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cyclone_ens as E                        # map, landfall and storage helpers shared with the ECMWF ENS tab
import mme_fields, mme_track

SITE = E.SITE
KEEP_DAYS = float(os.environ.get("CYCLONE_KEEP_DAYS", "15"))
PREFIX = "cyclone/mme"
# key: (long name, short name, colour, data credit)
MODELS = {
    "ifs":   ("ECMWF HRES", "HRES", "#e85d04", "ECMWF (CC BY 4.0)"),
    "aifs":  ("ECMWF AIFS", "AIFS", "#c9184a", "ECMWF (CC BY 4.0)"),
    "gfs":   ("NCEP GFS", "GFS", "#1d6fd1", "NOAA/NCEP"),
    "aigfs": ("NOAA AIGFS", "AIGFS", "#4cc9f0", "NOAA/NCEP"),
    "cmc":   ("CMC GDPS", "CMC", "#2b9348", "ECCC (OGL-Canada)"),
    "icon":  ("DWD ICON", "ICON", "#7b2cbf", "DWD"),
    "ukmo":  ("UKMO Global", "UKMO", "#b8860b", "Met Office (CC BY-SA 4.0)"),
}
ORDER = list(MODELS)
MATCH_KM, CONS_MIN = 350.0, 3
WAIT_HOURS = 16                                   # stop waiting for a missing model this long after the run time
C_CONS, C_OBS = E.C_MEAN, "#222222"
SRC = f"Data: ECMWF (CC BY 4.0), NOAA, ECCC, DWD, Met Office (CC BY-SA 4.0) · same tracker for all models · {SITE}"


def name(m): return MODELS[m][0]
def short(m): return MODELS[m][1]


# ---------------------------------------------------------------- observed storms (JTWC best track via NCAR/RAL TCGP)
TCGP = "https://hurricanes.ral.ucar.edu/realtime/"


def jtwc_storms(run):
    """North Indian Ocean storms / invests in TCGP's current list: [dict(sid, name, fixes DataFrame t, lat, lon, w)]."""
    out = []
    try:
        html = urllib.request.urlopen(urllib.request.Request(TCGP + "current/", headers=mme_fields.UA), timeout=60).read().decode("latin-1")
    except Exception as e:
        print("  (TCGP list not available:", e, ")"); return out
    for path in sorted(set(re.findall(r'plots/northindian/(\d{4})/(io\d{6})/', html))):
        y, sid = path
        try:
            txt = urllib.request.urlopen(urllib.request.Request(f"{TCGP}plots/northindian/{y}/{sid}/b{sid}.dat",
                                                                headers=mme_fields.UA), timeout=60).read().decode("latin-1")
        except Exception:
            continue
        rows, nm = [], ""
        for l in txt.splitlines():
            f = [x.strip() for x in l.split(",")]
            if len(f) < 9 or not f[6][:-1].isdigit(): continue
            la = int(f[6][:-1]) / 10 * (1 if f[6][-1] == "N" else -1); lo = int(f[7][:-1]) / 10 * (1 if f[7][-1] == "E" else -1)
            rows.append(dict(t=pd.Timestamp(dt.datetime.strptime(f[2], "%Y%m%d%H")), lat=la, lon=lo, w=float(f[8] or 0)))
            if len(f) > 27 and f[27] and f[27] not in ("INVEST", "NONAME"): nm = f[27]
        if not rows: continue
        df = pd.DataFrame(rows).drop_duplicates("t").sort_values("t").reset_index(drop=True)
        num = int(sid[2:4]); basin = "B" if df.lon.iloc[-1] >= 77.5 else "A"
        out.append(dict(sid=f"{num:02d}{basin}" if num < 90 else f"Invest {num}{basin}", name=nm.upper() if nm else "", fixes=df))
    return out


# ---------------------------------------------------------------- grouping the models' tracks into systems
def tdf(points):
    d = pd.DataFrame(points)
    d["t"] = pd.to_datetime(d.t, format="%Y%m%d%H")
    return d.set_index("t")[["lat", "lon", "w", "p"]]


def match_km(a, b):
    """Median distance (km) between two tracks at their common times (None if they share fewer than 2 times)."""
    c = a.index.intersection(b.index)
    if len(c) < 2: return None
    return float(np.median(mme_track.km(a.lat[c].to_numpy(), a.lon[c].to_numpy(), b.lat[c].to_numpy(), b.lon[c].to_numpy())))


def consensus(members):
    a = pd.concat([d.assign(m=m) for m, d in members.items()]).reset_index()
    g = a.groupby("t")
    c = pd.DataFrame(dict(n=g.size(), lat=g.lat.mean(), lon=g.lon.mean(), w=g.w.mean(), p=g.p.mean()))
    need = min(CONS_MIN, len(members))
    return c[c.n >= need]


def find_systems(tracks):
    """tracks: {model: [list of points]} -> systems [dict(members={model: DataFrame}, cons=DataFrame)] with >= 2 models."""
    pool = [(m, tdf(tr["points"])) for m in ORDER for tr in tracks.get(m, [])]
    pool.sort(key=lambda x: -x[1].w.max())
    used, systems = set(), []
    for k, (m0, d0) in enumerate(pool):
        if k in used: continue
        mem, ref = {m0: d0}, d0; used.add(k)
        for m in ORDER:
            if m in mem: continue
            best = None
            for j, (mj, dj) in enumerate(pool):
                if j in used or mj != m: continue
                d = match_km(ref, dj)
                if d is not None and d <= MATCH_KM and (best is None or d < best[0]): best = (d, j)
            if best:
                used.add(best[1]); mem[m] = pool[best[1]][1]
                c = consensus(mem); ref = c if len(c) >= 2 else ref
        if len(mem) >= 2:
            c = consensus(mem)
            if len(c): systems.append(dict(members={m: mem[m] for m in ORDER if m in mem}, cons=c))
    for s in systems:
        r0 = s["cons"].iloc[0]; s["basin"] = "B" if r0.lon >= 77.5 else "A"; s["la0"], s["lo0"] = float(r0.lat), float(r0.lon)
    systems.sort(key=lambda s: ("BA".index(s["basin"]), -len(s["members"]), s["lo0"]))
    n = {}
    for s in systems:
        n[s["basin"]] = n.get(s["basin"], 0) + 1; s["id"] = f"{s['basin'].lower()}{n[s['basin']]}"
    return systems


def label_systems(systems, run, storms):
    """Name each system after the observed storm it starts at (JTWC number + name), else 'system n'."""
    t0 = pd.Timestamp(run)
    for s in systems:
        s["sid"], s["storm"], s["obs"] = "", "", None
        c = s["cons"]
        if c.index[0] > t0 + pd.Timedelta(hours=12): continue          # forms later in the forecast
        best = None
        for st in storms:
            fx = st["fixes"]; fx = fx[fx.t <= t0]
            if fx.empty or fx.t.iloc[-1] < t0 - pd.Timedelta(hours=12): continue
            d = float(mme_track.km(fx.lat.iloc[-1], fx.lon.iloc[-1], c.lat.iloc[0], c.lon.iloc[0]))
            if d < 300 and (best is None or d < best[0]): best = (d, st, fx)
        if best:
            st, fx = best[1], best[2]
            s["sid"], s["storm"] = st["sid"], st["name"]
            s["obs"] = fx[fx.t >= t0 - pd.Timedelta(hours=72)]
    for b in "BA":
        same = [s for s in systems if s["basin"] == b]
        for k, s in enumerate(same):
            tag = (f"{s['sid']} {s['storm']}".strip() if s["sid"] else (f"system {k + 1}" if len(same) > 1 else ""))
            s["name"] = E.BASINS[b] + (f" · {tag}" if tag else "")


# ---------------------------------------------------------------- products
def products(s, run, out, nmodels_run):
    os.makedirs(out, exist_ok=True)
    b = s["basin"]; IC = f"{run:%H} UTC {run:%d %b %Y}"; BN = s["name"]; pc = ccrs.PlateCarree()
    M, C = s["members"], s["cons"]
    allp = pd.concat([d.assign(model=m) for m, d in M.items()]).reset_index()
    ext = E.frame(b, allp)
    if s["obs"] is not None and len(s["obs"]):                                  # keep the observed track in view
        o = s["obs"]; ext = [min(ext[0], o.lon.min() - 1), max(ext[1], o.lon.max() + 1), min(ext[2], o.lat.min() - 1), max(ext[3], o.lat.max() + 1)]

    def src(ax, y=0.005):
        ax.text(0.005, y, SRC, transform=ax.transAxes, fontsize=7, color="#333", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    nm = len(M)
    # M1 tracks
    fig, ax = E.make_axes(ext, False); E.basemap(ax, ext, b)
    if s["obs"] is not None and len(s["obs"]):
        o = s["obs"]
        ax.plot(o.lon, o.lat, "--", color=C_OBS, lw=1.6, zorder=3, transform=pc, label=f"Observed {s['sid']} (JTWC, last 72 h)")
        ax.plot(o.lon, o.lat, "o", color=C_OBS, ms=3, zorder=3, transform=pc)
    for m, d in M.items():
        hrs = int((d.index[-1] - d.index[0]).total_seconds() // 3600)
        ax.plot(d.lon, d.lat, "-", color=MODELS[m][2], lw=1.9, zorder=4, transform=pc, label=f"{name(m)} ({hrs} h)")
        z = d[d.index.hour == 0]
        ax.plot(z.lon, z.lat, "o", color=MODELS[m][2], ms=4, zorder=4, transform=pc)
    ax.plot(C.lon, C.lat, "-", color=C_CONS, lw=3.4, zorder=5, transform=pc, label=f"Consensus (mean of ≥{min(CONS_MIN, nm)} models)")
    for t, r in C[C.index.hour == 0].iterrows():
        ax.plot(r.lon, r.lat, "o", mfc="white", mec=C_CONS, mew=2, ms=7, zorder=6, transform=pc)
        ax.annotate(f"{t:%d/%H}", (r.lon, r.lat), xycoords=pc._as_mpl_transform(ax), xytext=(6, -10), textcoords="offset points",
                    fontsize=8, fontweight="bold", zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.set_title(f"Multi-model tracks (open models, {nm} of {nmodels_run}) — IC: {IC}\n{BN}", fontsize=12.5); src(ax)
    plt.savefig(f"{out}/M1_MME_tracks.png", dpi=150, bbox_inches="tight"); plt.close()

    # M2 / M3 plumes
    def plume(var, fn, ylab, title):
        fig, (ax, an) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[5, 1], hspace=0.08))
        if var == "w":
            top = max(np.nanmax(allp.w) + 8, 40); ax.set_ylim(10, top)
            for lo, hi, n, col in E.CATS_NIO:
                if lo <= top:
                    ax.axhspan(lo, min(hi + 1, top), color=col, zorder=0)
                    ax.text(1.005, (lo + min(hi + 1, top)) / 2, n, transform=ax.get_yaxis_transform(), va="center", fontsize=9, fontweight="bold")
        for m, d in M.items():
            ax.plot(d.index, d[var], "-o", color=MODELS[m][2], lw=1.8, ms=3, label=name(m))
        ax.plot(C.index, C[var], "-", color=C_CONS, lw=3.2, label="Consensus")
        if var == "p": ax.invert_yaxis()
        ax.set_ylabel(ylab); ax.legend(loc="upper left", fontsize=9, ncol=4); E.tfmt(ax)
        ax.set_title(f"{title} — IC: {IC}\n{BN}", fontsize=13)
        cnt = allp.groupby("t").size()
        an.bar(cnt.index, cnt.values, width=0.2, color=C_CONS, alpha=0.6); an.set_ylabel("Models"); an.set_ylim(0, len(MODELS) + 0.5); an.set_yticks(range(0, len(MODELS) + 1, 2)); E.tfmt(an)
        src(an)
        plt.savefig(f"{out}/{fn}", dpi=150, bbox_inches="tight"); plt.close()
    plume("w", "M2_MME_MSW.png", "Maximum sustained wind (kt)", "Multi-model Maximum Sustained Wind (10 m, 0.25° grids)")
    plume("p", "M3_MME_MSLP.png", "Central MSLP (hPa)", "Multi-model Central Pressure")

    # M4 spread around the consensus
    fig, ax = plt.subplots(figsize=(12, 5.6))
    dist = {}
    for m, d in M.items():
        c = d.index.intersection(C.index)
        if len(c):
            dist[m] = pd.Series(mme_track.km(d.lat[c].to_numpy(), d.lon[c].to_numpy(), C.lat[c].to_numpy(), C.lon[c].to_numpy()), index=c)
            ax.plot(dist[m].index, dist[m].values, "-o", color=MODELS[m][2], lw=1.8, ms=3, label=name(m))
    if dist:
        mean = pd.concat(dist.values(), axis=1).mean(axis=1)
        ax.plot(mean.index, mean.values, "--", color=C_CONS, lw=3, label="Mean distance (spread)")
    ax.set_ylim(bottom=0); ax.set_ylabel("Distance from the consensus position (km)"); E.tfmt(ax)
    ax.legend(loc="upper left", fontsize=9, ncol=4)
    ax.set_title(f"Track spread around the consensus — IC: {IC}\n{BN}", fontsize=13); src(ax)
    plt.savefig(f"{out}/M4_MME_spread.png", dpi=150, bbox_inches="tight"); plt.close()

    # landfall of every model and of the consensus
    bb = box(min(allp.lon.min(), ext[0]) - 5, max(-80, allp.lat.min() - 5), max(allp.lon.max(), ext[1]) + 5, min(80, allp.lat.max() + 5))
    LAND = prep(unary_union(E.shifted(E.land_geoms(), bb, False)))
    names = E.coast_names(b, ext)
    lf_df = lambda d: d.reset_index()[["t", "lat", "lon", "w", "p"]]
    c_lf = E.landfall(lf_df(C), LAND)
    LF, mrows, mt = {}, [], []
    for m, d in M.items():
        l = E.landfall(lf_df(d), LAND); LF[m] = l
        pk = d.w.idxmax()
        place = E.near(l["lat"], l["lon"], names) if l else ""
        mrows.append([f"{short(m)} landfall", f"{l['t']:%HZ %d %b}, {place}, {l['w']:.0f} kt" if l else "no landfall"])
        mt.append(dict(m=m, n=name(m), c=MODELS[m][2], h=int((d.index[-1] - d.index[0]).total_seconds() // 3600),
                       pk=f"{d.w.max():.0f} kt ({E.cat(d.w.max(), b)}), {pk:%HZ %d %b}", pmin=f"{d.p.min():.0f} hPa",
                       lf=f"{l['t']:%HZ %d %b}" if l else "–", lfp=place or "no landfall",
                       lfw=f"{l['w']:.0f} kt ({E.cat(l['w'], b)})" if l else "–"))
    L = {m: l for m, l in LF.items() if l}
    cpk = C.w.idxmax()
    rows = [["Models tracking the system", f"{nm} of {nmodels_run} ({', '.join(short(m) for m in M)})"],
            ["Consensus peak", f"{C.w.max():.0f} kt ({E.cat(C.w.max(), b)}), {cpk:%HZ %d %b}, {C.p.loc[cpk]:.0f} hPa"],
            ["Consensus landfall", f"{c_lf['t']:%HZ %d %b}, {E.near(c_lf['lat'], c_lf['lon'], names)}" if c_lf else "no landfall"],
            ["Models making landfall", f"{len(L)} / {nm}"]]
    if L:
        ts = sorted(l["t"] for l in L.values()); las = sorted(L.values(), key=lambda l: l["lat"])
        nearest = lambda l: E.near(l["lat"], l["lon"], names).split(" of ")[-1].replace("near ", "")
        rows += [["Landfall window", f"{ts[0]:%HZ %d %b} – {ts[-1]:%HZ %d %b}"],
                 ["Landfall coast span", f"{nearest(las[0])} to {nearest(las[-1])} ({E.fmt_lat(las[0]['lat'])}–{E.fmt_lat(las[-1]['lat'])})"
                  if len(L) > 1 else f"{nearest(las[0])} ({E.fmt_lat(las[0]['lat'])})"],
                 ["MSW at landfall", f"{min(l['w'] for l in L.values()):.0f}–{max(l['w'] for l in L.values()):.0f} kt"
                  f" (mean {np.mean([l['w'] for l in L.values()]):.0f} kt, {E.cat(np.mean([l['w'] for l in L.values()]), b)})"]]
    strongest = max(M, key=lambda m: M[m].w.max())
    rows.append(["Strongest model", f"{short(strongest)} {M[strongest].w.max():.0f} kt ({E.cat(M[strongest].w.max(), b)}), "
                                    f"{M[strongest].w.idxmax():%HZ %d %b}"])
    if dist:
        sp = pd.concat(dist.values(), axis=1).mean(axis=1)
        sel = [h for h in (24, 48, 72, 96, 120) if pd.Timestamp(run) + pd.Timedelta(hours=h) in sp.index]
        if sel: rows.append(["Track spread", ", ".join(f"{h} h: {sp.loc[pd.Timestamp(run) + pd.Timedelta(hours=h)]:.0f} km" for h in sel)])

    # landfall summary PNG: summary table + one row per model
    nh = len(rows); nmr = len(mt)
    fig = plt.figure(figsize=(9.6, 1.55 + 0.48 * nh + 0.42 * (nmr + 1) + 0.75), dpi=150)
    H = fig.get_figheight()
    fig.text(0.5, 1 - 0.32 / H, "Multi-model ensemble (open models) — landfall summary", ha="center", va="top", fontsize=15, fontweight="bold", color=E.C_HEAD)
    fig.text(0.5, 1 - 0.72 / H, f"{BN} · IC {IC}", ha="center", va="top", fontsize=11)
    y1 = 1 - 1.05 / H; h1 = 0.48 * nh / H
    ax = fig.add_axes([0.03, y1 - h1, 0.94, h1]); ax.axis("off")
    cells = [[k, v] for k, v in rows]
    if c_lf: cells[2][1] += f"\n{E.fmt_lat(c_lf['lat'])} {E.fmt_lon(c_lf['lon'])}, {c_lf['w']:.0f} kt"
    tb = ax.table(cellText=cells, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[0.30, 0.70])
    tb.auto_set_font_size(False); tb.set_fontsize(9)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#bbb"); c.set_facecolor("#eef1f5" if j == 0 else E.C_HIL)
        if j == 0: c.get_text().set_fontweight("bold")
    y2 = y1 - h1 - 0.35 / H; h2 = 0.42 * (nmr + 1) / H
    ax = fig.add_axes([0.03, y2 - h2, 0.94, h2]); ax.axis("off")
    hdr = ["Model", "Track", "Peak MSW", "Min MSLP", "Landfall", "Place", "MSW at landfall"]
    body = [[x["n"], f"{x['h']} h", x["pk"], x["pmin"], x["lf"], x["lfp"], x["lfw"]] for x in mt]
    tb = ax.table(cellText=body, colLabels=hdr, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[.13, .07, .2, .09, .11, .26, .14])
    tb.auto_set_font_size(False); tb.set_fontsize(8)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#bbb")
        if i == 0: c.set_facecolor(E.C_HEAD); c.get_text().set_color("white"); c.get_text().set_fontweight("bold")
        elif j == 0: c.set_facecolor("#eef1f5"); c.get_text().set_fontweight("bold"); c.get_text().set_color(mt[i - 1]["c"])
    fig.text(0.5, 0.25 / H, "Winds: strongest 10 m wind within 250 km of the centre on 0.25° grids (weaker than official estimates for strong storms)\n" + SRC,
             ha="center", fontsize=7.5, style="italic", color="#555")
    plt.savefig(f"{out}/MME_landfall_summary.png", dpi=150, facecolor="white"); plt.close()

    # files
    files = []
    with open(f"{out}/MME_consensus_track.txt", "w") as f:
        f.write(f"# Multi-model consensus (open models: {', '.join(name(m) for m in M)}), {BN}, IC {run:%Y%m%d %H}Z\n"
                "# columns: YYYYMMDDHH lat lon(°E) MSW(kt) MSLP(hPa) models\n")
        for t, r in C.iterrows(): f.write(f"{t:%Y%m%d%H} {r.lat:.1f} {r.lon:.1f} {r.w:.0f} {r.p:.0f} {int(r.n)}\n")
    files.append(("Consensus track", "MME_consensus_track.txt"))
    allp.assign(model=allp.model.map(name), t=allp.t.dt.strftime("%Y%m%d%H"))[["model", "t", "lat", "lon", "w", "p"]] \
        .round(dict(lat=2, lon=2, w=1, p=1)).rename(columns=dict(w="msw_kt", p="mslp_hpa")).to_csv(f"{out}/MME_model_tracks.csv", index=False)
    files.append(("All model tracks (csv)", "MME_model_tracks.csv"))
    items = [("Landfall summary", "MME_landfall_summary.png"), ("Tracks", "M1_MME_tracks.png"), ("MSW plume", "M2_MME_MSW.png"),
             ("MSLP plume", "M3_MME_MSLP.png"), ("Track spread", "M4_MME_spread.png")]
    pkg = f"MME_{s['id']}_{run:%Y%m%d%H}.zip"
    with zipfile.ZipFile(f"{out}/{pkg}", "w", zipfile.ZIP_DEFLATED) as z:
        for _, f in items + files: z.write(f"{out}/{f}", f)
    files.insert(0, ("Package (zip)", pkg))
    track = [[f"{t:%Y%m%d%H}", round(r.lat, 2), round(r.lon, 2)] for t, r in C.iterrows()]
    print(f"  {BN}: {nm} models, consensus peak {C.w.max():.0f} kt, landfalls {len(L)}/{nm}"
          + (f", consensus landfall {c_lf['t']:%HZ %d %b} {E.near(c_lf['lat'], c_lf['lon'], names)}" if c_lf else ""))
    return dict(id=s["id"], b=b, name=BN, sid=s["sid"], lat=round(s["la0"], 2), lon=round(s["lo0"], 2), models=list(M),
                rows=rows, mrows=mrows, mt=mt, items=[{"t": t, "f": f} for t, f in items], files=[{"t": t, "f": f} for t, f in files],
                track=track)


def compare_png(cols, out, name_, updated):
    keys = []
    for c in cols:
        for k, v in c["rows"]:
            if k not in keys: keys.append(k)
    mk = [f"{short(m)} landfall" for m in ORDER]                     # model rows always in the same order, last
    keys = [k for k in keys if k not in mk] + [k for k in mk if k in keys]
    wrap = lambda v: "\n".join(textwrap.wrap(v, 20)) if v else "–"
    rows = [[k] + [wrap(dict(c["rows"]).get(k, "")) for c in cols] for k in keys]
    hdr = [""] + [c["label"].rsplit(" ", 1)[0].replace(" UTC ", " UTC\n") for c in cols]
    w = 2.6 + 2.1 * len(cols)
    fig = plt.figure(figsize=(w, 0.95 * len(rows) + 2.2), dpi=150)
    fig.text(0.5, 0.975, f"Multi-model ensemble — run comparison ({name_})", ha="center", va="top", fontsize=15, fontweight="bold", color=E.C_HEAD)
    fig.text(0.5, 0.94, f"Last {len(cols)} runs (oldest → newest)  ·  {updated}", ha="center", va="top", fontsize=10)
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.85]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=hdr, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=[2.6 / w] + [2.1 / w] * len(cols))
    tb.auto_set_font_size(False); tb.set_fontsize(8.2)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor("#bbb")
        if i == 0: c.set_facecolor(E.C_HEAD if j else "white"); c.get_text().set_color("white"); c.get_text().set_fontweight("bold")
        elif j == 0: c.set_facecolor("#eef1f5"); c.get_text().set_fontweight("bold")
        elif j == len(cols): c.set_facecolor(E.C_HIL); c.get_text().set_fontweight("bold")
    fig.text(0.5, 0.015, SRC, ha="center", fontsize=7.5, style="italic", color="#555")
    plt.savefig(out, dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.2); plt.close()


# ---------------------------------------------------------------- one run
def models_for(run, st, a):
    """Load the stored tracks of this run, fetch + track the models that are ready now. Returns (state, changed)."""
    ic = f"{run:%Y%m%d%H}"
    state = st.get_json(f"{PREFIX}/{ic}/tracks.json", {"ic": ic, "models": {}})
    if a.force: state["models"] = {}
    changed = False
    for m in ORDER:
        if m in state["models"] or (a.models and m not in a.models.split(",")): continue
        t0 = time.time()
        try:
            f = None
            cf = os.path.join(a.cache, f"{m}_{ic}.npz") if a.cache else None
            if cf and os.path.exists(cf):
                f = dict(np.load(cf)); f["steps"] = [int(x) for x in f["steps"]]
            else:
                f = mme_fields.fetch(m, run)
                if f is not None and cf:
                    os.makedirs(a.cache, exist_ok=True); np.savez_compressed(cf, **f)
            if f is None:
                print(f"  {name(m)} {ic}: not complete yet"); continue
            tr = mme_track.track(f, run)
            for x in tr:
                for p in x["points"]: p.pop("land", None); p["vort"] = round(p["vort"] * 1e5, 1)
                x["points"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()} for p in x["points"]]
            state["models"][m] = {"steps": len(f["steps"]), "tracks": tr}
            changed = True
            print(f"  {name(m)} {ic}: {len(tr)} track(s), {time.time() - t0:.0f} s", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {name(m)} {ic}: FAILED {e}")
    if changed:
        st.put(f"{PREFIX}/{ic}/tracks.json", json.dumps(state, separators=(",", ":")), "application/json", "public, max-age=300")
    return state, changed


def publish_run(run, st, a, storms):
    ic = f"{run:%Y%m%d%H}"
    state, changed = models_for(run, st, a)
    index = st.get_json(f"{PREFIX}/index.json", {"runs": []})
    have = [m for m in ORDER if m in state["models"]]
    old = next((r for r in index["runs"] if r["ic"] == ic), None)
    age = (dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - run).total_seconds() / 3600
    if not changed and old and not a.force and not a.redraw:
        print(f"MME {ic}: nothing new ({len(have)} models)"); return
    if len(have) < 3 and age < WAIT_HOURS and not a.force:
        print(f"MME {ic}: only {len(have)} model(s) so far — waiting for at least 3"); return
    if not have:
        print(f"MME {ic}: no model available"); return
    tracks = {m: state["models"][m]["tracks"] for m in have}
    systems = find_systems(tracks)
    label_systems(systems, run, storms)
    print(f"run {ic}: {len(have)} models, {len(systems)} system(s): " + ", ".join(f"{s['name']} ({len(s['members'])} models)" for s in systems))
    label = f"{run:%H} UTC {run.day} {run:%b %Y}"
    ist = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=5, minutes=30)
    updated = f"{ist:%d %b %Y %H:%M} IST"
    work = os.path.join(a.work, ic)
    import shutil; shutil.rmtree(work, ignore_errors=True)
    missing = [m for m in ORDER if m not in have]
    entry = {"ic": ic, "label": label, "models": have, "missing": missing, "waiting": bool(missing) and age < WAIT_HOURS,
             "v": int(time.time()) % 100000, "systems": []}
    for s in systems:
        try:
            entry["systems"].append(products(s, run, os.path.join(work, s["id"]), len(have)))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {s['name']}: FAILED {e}")
    runs = [r for r in index["runs"] if r["ic"] != ic] + [entry]
    runs.sort(key=lambda r: r["ic"], reverse=True)
    cut = (run - dt.timedelta(days=KEEP_DAYS)).strftime("%Y%m%d%H")
    gone = [r for r in runs if r["ic"] < cut]; runs = [r for r in runs if r["ic"] >= cut]
    pos = [r["ic"] for r in runs].index(ic); byic = {r["ic"]: r for r in runs}
    for sd in entry["systems"]:
        sd["prev"] = None
        for r in runs[pos + 1:pos + 4]:
            m = E.match_prev(sd, r)
            if m: sd["prev"] = {"ic": r["ic"], "id": m}; break
        cols = [dict(label=label, rows=sd["rows"] + sd["mrows"])]; p = sd["prev"]
        while p and len(cols) < 6 and p["ic"] in byic:
            o = next((x for x in byic[p["ic"]]["systems"] if x["id"] == p["id"]), None)
            if not o: break
            cols.insert(0, dict(label=byic[p["ic"]]["label"], rows=o["rows"] + o.get("mrows", []))); p = o.get("prev")
        if len(cols) >= 2:
            fn = f"MME_comparison_{sd['id']}.png"
            compare_png(cols, os.path.join(work, sd["id"], fn), sd["name"], updated); sd["cmp"] = fn
    # upload: the run's old figures go first (systems may have changed), then the files, then the index
    if old:
        for sd in old.get("systems", []): st.delete_prefix(f"{PREFIX}/{ic}/{sd['id']}/")
    n = 0
    for sd in entry["systems"]:
        d = os.path.join(work, sd["id"])
        for f in os.listdir(d):
            st.put(f"{PREFIX}/{ic}/{sd['id']}/{f}", open(os.path.join(d, f), "rb").read(),
                   E.CTYPE.get(os.path.splitext(f)[1], "application/octet-stream"), "public, max-age=600"); n += 1
    index = {"updated": updated, "keep_days": KEEP_DAYS, "models": {m: [name(m), MODELS[m][2], MODELS[m][3], short(m)] for m in ORDER}, "runs": runs}
    st.put(f"{PREFIX}/index.json", json.dumps(index, separators=(",", ":")), "application/json", "public, max-age=120")
    for r in gone: st.delete_prefix(f"{PREFIX}/{r['ic']}/")
    print(f"published MME {ic}: {len(have)} models ({', '.join(short(m) for m in have)}), {len(entry['systems'])} system(s), {n} files")


def recent_runs(now=None):
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    t = (now - dt.timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
    t = t.replace(hour=12 if t.hour >= 12 else 0)
    return [t - dt.timedelta(hours=12), t]                 # the older one first (it may still miss a model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--force", action="store_true"); ap.add_argument("--redraw", action="store_true")
    ap.add_argument("--local"); ap.add_argument("--cache"); ap.add_argument("--work", default="work_mme")
    ap.add_argument("--models", help="only these models this time, e.g. gfs,icon (tests)")
    a = ap.parse_args()
    st = E.Store(a.local)
    index = st.get_json(f"{PREFIX}/index.json", {"runs": []})
    if a.run:
        todo = [dt.datetime.strptime(a.run, "%Y%m%d%H")]
    else:
        todo = recent_runs()
        if not index["runs"]: todo = [todo[0] - dt.timedelta(hours=24), todo[0] - dt.timedelta(hours=12)] + todo   # first time: 2 days
        done = {r["ic"]: r for r in index["runs"]}
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        finished = lambda r: f"{r:%Y%m%d%H}" in done and (not done[f"{r:%Y%m%d%H}"].get("missing")
                                                          or (now - r).total_seconds() > WAIT_HOURS * 3600)
        todo = [r for r in todo if not finished(r)]
    storms = jtwc_storms(todo[-1]) if todo else []
    for r in todo:
        try:
            publish_run(r, st, a, storms)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"MME {r:%Y%m%d%H} failed: {e}")


if __name__ == "__main__":
    main()
