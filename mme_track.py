"""A simple, uniform vortex tracker used the same way for every open model (cyclone_mme.py).
Input: MSLP and 10 m wind every 6 h on the global 0.25° grid of mme_fields.py (50 S..50 N, 0..359.75 E).

Centre at each step = a local minimum of (lightly smoothed) MSLP that
  * is a low: mean MSLP 250-350 km away is at least 1 hPa higher than at the centre,
  * has cyclonic 10 m relative vorticity (0.5° smoothed) of at least 3e-5 s-1 within 150 km,
  * lies below 500 m of terrain (no heat / orographic lows over the plateaus), within 2-45° of the equator.
Centres are linked step to step: from the last position plus the last 6-h motion, the nearest centre within
350 km is taken (one missed step allowed). A track is kept if it lasts at least 24 h, starts within 28° of the
equator, starts over the sea (or is at sea for at least 12 h), and while over the tropical sea (within 30° of the
equator) reaches 20 kt and a vorticity of 1e-4 s-1 (this drops weak, nearly stationary heat lows and the
mid-latitude lows that reach the subtropics).
Intensity at each point: MSW = strongest 10 m wind within 250 km of the centre (kt), MSLP = lowest MSLP within 100 km.
Longitudes are 0..360 E.
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
DETECT_LAT, START_LAT, TROPIC_LAT = 45.0, 28.0, 30.0
MODE = ("nearest", "wrap")                                    # filters: edge in latitude, wrap in longitude

LA2, LO2 = np.meshgrid(LAT, LON, indexing="ij")
NY, NX = LA2.shape
_STATIC = {}


def static():
    """Terrain height (m) and land mask on the grid."""
    if not _STATIC:
        z = np.load(os.path.join(HERE, "orography_0p25.npz"))["z"].astype(np.float32)   # 90..-90, -180..179.75
        rows = np.round((90 - LAT) / 0.25).astype(int); cols = np.round(((LON + 180) % 360) / 0.25).astype(int) % 1440
        _STATIC["oro"] = z[np.ix_(rows, cols)]
        try:
            import shapely
            from cyclone_ens import land_geoms
            from shapely import affinity
            land = np.zeros(LA2.shape, bool)
            for g in land_geoms():
                for off in (0, 360):                                 # Natural Earth is -180..180; the grid 0..360
                    gg = affinity.translate(g, xoff=off) if off else g
                    x0, y0, x1, y1 = gg.bounds
                    if y1 < LAT[0] or y0 > LAT[-1] or x1 < 0 or x0 > 360: continue
                    j = np.where((LAT >= y0) & (LAT <= y1))[0]; i = np.where((LON >= x0) & (LON <= x1))[0]
                    if not len(j) or not len(i): continue
                    sub = np.ix_(j, i)
                    land[sub] |= shapely.contains_xy(gg, LO2[sub], LA2[sub])
            _STATIC["land"] = land
        except Exception as e:                                       # fall back to terrain > 0 m
            print("  (land mask from terrain:", e, ")")
            _STATIC["land"] = _STATIC["oro"] > 1
    return _STATIC["oro"], _STATIC["land"]


def km(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lo2) - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def dlon(a, b):
    """b - a in degrees, wrapped to -180..180."""
    return (b - a + 180) % 360 - 180


def vorticity(u, v):
    """Relative vorticity (s-1) on the lat/lon grid (periodic in longitude), then smoothed (Gaussian, sigma 0.5°)."""
    phi = np.radians(LA2); d = np.radians(0.25)
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * d) / (R_EARTH * np.cos(phi))
    dudy = np.gradient(u * np.cos(phi), axis=0) / d / (R_EARTH * np.cos(phi))
    return ndimage.gaussian_filter(dvdx - dudy, 2.0, mode=MODE)


def _disc(radius_km, lat):
    """Offsets (rows, columns) of the grid points within radius_km at latitude lat, and their distances."""
    c = max(np.cos(np.radians(lat)), 0.2)
    ny = int(np.ceil(radius_km / 27.8)); nx = int(np.ceil(radius_km / (27.8 * c)))
    jj, ii = np.mgrid[-ny:ny + 1, -nx:nx + 1]
    r = np.hypot(jj * 27.8, ii * 27.8 * c)
    return jj, ii, r


def _window(a, j, i, radius_km, rmin=-1.0):
    jj, ii, r = _disc(radius_km, LAT[j])
    J = j + jj; ok = (r <= radius_km) & (r > rmin) & (J >= 0) & (J < NY)
    return a[J[ok], (i + ii[ok]) % NX]


def candidates(msl, u, v):
    """Possible cyclone centres at one step: list of dict(lat, lon, j, i, vort)."""
    oro, _ = static()
    ms = ndimage.gaussian_filter(msl, 1.0, mode=MODE)
    zeta = vorticity(u, v) * np.sign(LA2 + 1e-6)                    # cyclonic positive (NH and SH)
    zmax = ndimage.maximum_filter(zeta, size=11, mode=MODE)         # within ~150 km
    mins = (ms == ndimage.minimum_filter(ms, size=9, mode=MODE)) & (zmax >= VORT_MIN) & (oro < ORO_MAX)
    mins &= (np.abs(LA2) >= 2) & (np.abs(LA2) <= DETECT_LAT)
    out = []
    for j, i in zip(*np.nonzero(mins)):
        if np.nanmean(_window(ms, j, i, 350, 250)) - ms[j, i] < DEFICIT_MIN: continue
        # sub-grid position: parabola through the 3 x 3 minimum of the smoothed field
        a, b, c = ms[j - 1, i], ms[j, i], ms[j + 1, i]; den = a - 2 * b + c
        dy = 0.5 * (a - c) / den if den > 0 else 0.0
        a, b, c = ms[j, (i - 1) % NX], ms[j, i], ms[j, (i + 1) % NX]; den = a - 2 * b + c
        dx = 0.5 * (a - c) / den if den > 0 else 0.0
        out.append(dict(lat=float(LAT[j] + 0.25 * np.clip(dy, -.5, .5)), lon=float((LON[i] + 0.25 * np.clip(dx, -.5, .5)) % 360),
                        j=int(j), i=int(i), vort=float(zmax[j, i])))
    return out


def intensity(msl, spd, j, i):
    return float(np.nanmax(_window(spd, j, i, MSW_KM)) * KT), float(np.nanmin(_window(msl, j, i, MSLP_KM)))


def track(fields, run):
    """All tracks of one model run: list of dict(points=[dict(h, t, lat, lon, w, p, vort, land)])."""
    import pandas as pd
    steps = fields["steps"]
    _, land = static()
    tracks, active = [], []

    def point(s, c):
        return dict(h=s, lat=c["lat"], lon=c["lon"], w=c["w"], p=c["p"], vort=c["vort"], land=bool(land[c["j"], c["i"]]))
    for k, s in enumerate(steps):
        msl, u, v = fields["msl"][k], fields["u"][k], fields["v"][k]
        spd = np.hypot(u, v)
        cand = candidates(msl, u, v)
        for c in cand: c["w"], c["p"] = intensity(msl, spd, c["j"], c["i"])
        clat = np.array([c["lat"] for c in cand]); clon = np.array([c["lon"] for c in cand])
        used = set(); still = []
        for tr in sorted(active, key=lambda t: -len(t["points"])):          # longest-lived tracks pick first
            last = tr["points"][-1]; gap = (s - last["h"]) // 6
            if len(tr["points"]) >= 2:
                prev = tr["points"][-2]; f = gap * 6 / max(last["h"] - prev["h"], 6)
                pla, plo = last["lat"] + f * (last["lat"] - prev["lat"]), last["lon"] + f * dlon(prev["lon"], last["lon"])
            else:
                pla, plo = last["lat"], last["lon"]
            best = None
            if len(cand):
                d = km(pla, plo, clat, clon)
                for n in np.argsort(d):
                    if d[n] > LINK_KM * (1 if gap == 1 else 1.4): break
                    if n not in used: best = n; break
            if best is not None:
                used.add(best); tr["points"].append(point(s, cand[best])); still.append(tr)
            elif gap < 2:
                still.append(tr)                                             # allow one missed step
            else:
                tracks.append(tr)
        for n, c in enumerate(cand):
            if n not in used: still.append(dict(points=[point(s, c)]))
        active = still
    tracks += active
    keep = []
    for tr in tracks:
        P = tr["points"]
        if P[-1]["h"] - P[0]["h"] < MIN_HOURS or abs(P[0]["lat"]) > START_LAT: continue
        sea = [p for p in P if not p["land"] and abs(p["lat"]) <= TROPIC_LAT]
        if not sea: continue
        if P[0]["land"] and sea[-1]["h"] - sea[0]["h"] < 12: continue        # born over land and hardly at sea
        if max(p["w"] for p in sea) < MIN_PEAK_KT or max(p["vort"] for p in sea) < MIN_PEAK_VORT: continue
        for p in P: p["t"] = (run + pd.Timedelta(hours=p["h"])).strftime("%Y%m%d%H")
        keep.append(tr)
    return keep


if __name__ == "__main__":
    import sys, time, datetime as dt
    run = dt.datetime.strptime(sys.argv[2], "%Y%m%d%H")
    f = dict(np.load(f"mme_cache/{sys.argv[1]}_{sys.argv[2]}.npz"))
    f["steps"] = list(f["steps"])
    t0 = time.time(); trs = track(f, run)
    for tr in trs:
        P = tr["points"]; pk = max(P, key=lambda p: p["w"])
        print(f"{P[0]['h']:3d}-{P[-1]['h']:3d}h start {P[0]['lat']:5.1f} {P[0]['lon']:6.1f}E  end {P[-1]['lat']:5.1f} {P[-1]['lon']:6.1f}E  "
              f"peak {pk['w']:3.0f} kt at {pk['h']}h, min p {min(p['p'] for p in P):.0f}")
    print(len(trs), "tracks", f"{time.time() - t0:.0f} s")
