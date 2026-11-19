"""A simple, uniform vortex tracker for the North Indian Ocean, used the same way for every open model
(cyclone_mme.py). Input: MSLP and 10 m wind every 6 h on the 0.25° grid of mme_fields.py.

Centre at each step = a local minimum of (lightly smoothed) MSLP that
  * is a low: mean MSLP 250-350 km away is at least 1 hPa higher than at the centre,
  * has cyclonic 10 m relative vorticity (0.5° smoothed) of at least 3e-5 s-1 within 150 km,
  * lies below 500 m of terrain (no heat / orographic lows over the plateaus).
Centres are linked step to step: from the last position plus the last 6-h motion, the nearest centre within
350 km is taken (one missed step allowed). A track is kept if it lasts at least 24 h, starts over the sea
(or is at sea for at least 12 h), and over the Arabian Sea / Bay of Bengal reaches 20 kt and a
vorticity of 1e-4 s-1 (this drops the weak, nearly stationary lows of the Gulf of Aden and the Gulf).
Intensity at each point: MSW = strongest 10 m wind within 250 km of the centre (kt), MSLP = lowest MSLP within 100 km.
"""
import os
import numpy as np
from scipy import ndimage

from mme_fields import LAT, LON

HERE = os.path.dirname(os.path.abspath(__file__))
KT = 1.94384
R_EARTH = 6.371e6
VORT_MIN, DEFICIT_MIN, ORO_MAX = 3e-5, 1.0, 500.0
LINK_KM, MSW_KM, MSLP_KM = 350.0, 250.0, 100.0
MIN_HOURS, MIN_PEAK_KT, MIN_PEAK_VORT = 24, 20.0, 1e-4
REGION = dict(lat=(1.0, 25.0), lon=(42.0, 100.0))       # Arabian Sea + Bay of Bengal (genesis / time at sea)

LA2, LO2 = np.meshgrid(LAT, LON, indexing="ij")
_STATIC = {}


def static():
    """Terrain height (m) and land mask on the NIO grid."""
    if not _STATIC:
        z = np.load(os.path.join(HERE, "orography_0p25.npz"))["z"].astype(np.float32)   # 90..-90, -180..179.75
        rows = np.round((90 - LAT) / 0.25).astype(int); cols = np.round((LON + 180) / 0.25).astype(int)
        _STATIC["oro"] = z[np.ix_(rows, cols)]
        try:
            import shapely
            from cyclone_ens import land_geoms
            from shapely.ops import unary_union
            from shapely.geometry import box
            bb = box(LON[0] - 1, LAT[0] - 1, LON[-1] + 1, LAT[-1] + 1)
            land = unary_union([g.intersection(bb) for g in land_geoms() if g.intersects(bb)])
            _STATIC["land"] = shapely.contains_xy(land, LO2, LA2)
        except Exception as e:                                     # fall back to terrain > 0 m
            print("  (land mask from terrain:", e, ")")
            _STATIC["land"] = _STATIC["oro"] > 1
    return _STATIC["oro"], _STATIC["land"]


def km(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def vorticity(u, v):
    """Relative vorticity (s-1) on the lat/lon grid, then smoothed (Gaussian, sigma 0.5°)."""
    phi = np.radians(LA2)
    dlam = np.radians(0.25); dphi = np.radians(0.25)
    dvdx = np.gradient(v, axis=1) / dlam / (R_EARTH * np.cos(phi))
    dudy = np.gradient(u * np.cos(phi), axis=0) / dphi / (R_EARTH * np.cos(phi))
    return ndimage.gaussian_filter(dvdx - dudy, 2.0, mode="nearest")


def _disc(radius_km, lat):
    """Boolean footprint (odd square) of grid points within radius_km at latitude lat."""
    ny = int(np.ceil(radius_km / 27.8)); nx = int(np.ceil(radius_km / (27.8 * max(np.cos(np.radians(lat)), 0.2))))
    jj, ii = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    return jj, ii, np.hypot(jj * 27.8, ii * 27.8 * np.cos(np.radians(lat))) <= radius_km


def _window(a, j, i, radius_km):
    jj, ii, m = _disc(radius_km, LAT[j])
    J, I = j + jj, i + ii
    ok = m & (J >= 0) & (J < a.shape[0]) & (I >= 0) & (I < a.shape[1])
    return a[J[ok], I[ok]]


def candidates(msl, u, v):
    """Possible cyclone centres at one step: list of dict(lat, lon, p, vort)."""
    oro, _ = static()
    ms = ndimage.gaussian_filter(msl, 1.0, mode="nearest")
    zeta = vorticity(u, v) * np.sign(LA2 + 1e-6)                    # cyclonic positive (NH and SH)
    zmax = ndimage.maximum_filter(zeta, size=11)                    # within ~150 km
    mins = (ms == ndimage.minimum_filter(ms, size=9)) & (zmax >= VORT_MIN) & (oro < ORO_MAX)
    mins[:6, :] = mins[-6:, :] = False; mins[:, :6] = mins[:, -6:] = False
    out = []
    for j, i in zip(*np.nonzero(mins)):
        if abs(LAT[j]) < 2: continue
        jj, ii, outer = _disc(350, LAT[j])
        r = np.hypot(jj * 27.8, ii * 27.8 * np.cos(np.radians(LAT[j])))
        J, I = j + jj, i + ii; ring = outer & (r > 250) & (J >= 0) & (J < ms.shape[0]) & (I >= 0) & (I < ms.shape[1])
        if np.nanmean(ms[J[ring], I[ring]]) - ms[j, i] < DEFICIT_MIN: continue
        # sub-grid position: parabola through the 3 x 3 minimum of the smoothed field
        dy = dx = 0.0
        if 0 < j < ms.shape[0] - 1:
            a, b, c = ms[j - 1, i], ms[j, i], ms[j + 1, i]; den = a - 2 * b + c
            dy = 0.5 * (a - c) / den if den > 0 else 0.0
        if 0 < i < ms.shape[1] - 1:
            a, b, c = ms[j, i - 1], ms[j, i], ms[j, i + 1]; den = a - 2 * b + c
            dx = 0.5 * (a - c) / den if den > 0 else 0.0
        out.append(dict(lat=float(LAT[j] + 0.25 * np.clip(dy, -.5, .5)), lon=float(LON[i] + 0.25 * np.clip(dx, -.5, .5)),
                        j=int(j), i=int(i), vort=float(zmax[j, i])))
    return out


def intensity(msl, spd, j, i):
    return float(np.nanmax(_window(spd, j, i, MSW_KM)) * KT), float(np.nanmin(_window(msl, j, i, MSLP_KM)))


def track(fields, run):
    """All tracks of one model run: list of dict(points=[dict(h, lat, lon, w, p, vort)])."""
    import pandas as pd
    steps = fields["steps"]
    _, land = static()
    tracks, active = [], []
    for k, s in enumerate(steps):
        msl, u, v = fields["msl"][k], fields["u"][k], fields["v"][k]
        spd = np.hypot(u, v)
        cand = candidates(msl, u, v)
        for c in cand: c["w"], c["p"] = intensity(msl, spd, c["j"], c["i"])
        used = set(); still = []
        # longest-lived tracks pick first
        for tr in sorted(active, key=lambda t: -len(t["points"])):
            last = tr["points"][-1]; gap = (s - last["h"]) // 6
            if len(tr["points"]) >= 2:
                prev = tr["points"][-2]; f = gap * 6 / max(last["h"] - prev["h"], 6)
                pla, plo = last["lat"] + f * (last["lat"] - prev["lat"]), last["lon"] + f * (last["lon"] - prev["lon"])
            else:
                pla, plo = last["lat"], last["lon"]
            best = None
            for n, c in enumerate(cand):
                if n in used: continue
                d = km(pla, plo, c["lat"], c["lon"])
                if d <= LINK_KM * (1 if gap == 1 else 1.4) and (best is None or d < best[0]): best = (d, n)
            if best:
                used.add(best[1]); c = cand[best[1]]
                tr["points"].append(dict(h=s, lat=c["lat"], lon=c["lon"], w=c["w"], p=c["p"], vort=c["vort"],
                                         land=bool(land[c["j"], c["i"]])))
                still.append(tr)
            elif gap < 2:
                still.append(tr)                                    # allow one missed step
            else:
                tracks.append(tr)
        for n, c in enumerate(cand):
            if n not in used:
                still.append(dict(points=[dict(h=s, lat=c["lat"], lon=c["lon"], w=c["w"], p=c["p"], vort=c["vort"],
                                               land=bool(land[c["j"], c["i"]]))]))
        active = still
    tracks += active
    keep = []
    for tr in tracks:
        P = tr["points"]
        if P[-1]["h"] - P[0]["h"] < MIN_HOURS: continue
        sea = [p for p in P if not p["land"] and REGION["lat"][0] <= p["lat"] <= REGION["lat"][1]
               and REGION["lon"][0] <= p["lon"] <= REGION["lon"][1]]
        if not sea: continue
        if P[0]["land"] and sea[-1]["h"] - sea[0]["h"] < 12: continue        # born over land and hardly at sea
        if max(p["w"] for p in sea) < MIN_PEAK_KT or max(p["vort"] for p in sea) < MIN_PEAK_VORT: continue
        for p in P: p["t"] = (run + pd.Timedelta(hours=p["h"])).strftime("%Y%m%d%H")
        keep.append(tr)
    return keep


if __name__ == "__main__":
    import sys, datetime as dt
    run = dt.datetime.strptime(sys.argv[2], "%Y%m%d%H")
    f = dict(np.load(f"mme_cache/{sys.argv[1]}_{sys.argv[2]}.npz"))
    f["steps"] = list(f["steps"])
    for tr in track(f, run):
        P = tr["points"]; pk = max(P, key=lambda p: p["w"])
        print(f"{P[0]['h']:3d}-{P[-1]['h']:3d}h start {P[0]['lat']:.1f}N {P[0]['lon']:.1f}E  end {P[-1]['lat']:.1f}N {P[-1]['lon']:.1f}E  "
              f"peak {pk['w']:.0f} kt at {pk['h']}h, min p {min(p['p'] for p in P):.0f}")
