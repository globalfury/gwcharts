"""tphi_feed.py - global radiosonde ascents from the WMO GTS, as relayed by the US National Weather Service.

Source (US public domain, real time, worldwide):
  https://tgftp.nws.noaa.gov/SL.us008001/DF.bf/DC.vsndn/DS.raobf/RE.high/   BUFR TEMP (high resolution)
  https://tgftp.nws.noaa.gov/SL.us008001/DF.bf/DC.vsndn/DS.raobf/RE.low/    BUFR TEMP
  https://tgftp.nws.noaa.gov/SL.us008001/DF.an/DC.vsndn/DS.raobf/           coded TEMP text (TTAA/TTBB, India etc.)
Each folder is a rolling set of files sn.0000 ... sn.0300, each holding the bulletins of a few minutes.

collect(hours) -> {(wmo, 'YYYYMMDD_HH'): Ascent}; the best report of each station and ascent wins
(full-resolution BUFR > BUFR from coded TEMP > coded TEMP text).
"""
import datetime as dt, io, os, re, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BASE = "https://tgftp.nws.noaa.gov/SL.us008001/"
FEEDS = (("bufr-high", "DF.bf/DC.vsndn/DS.raobf/RE.high/"),
         ("bufr-low", "DF.bf/DC.vsndn/DS.raobf/RE.low/"),
         ("temp", "DF.an/DC.vsndn/DS.raobf/"))
MISS = -1e99
K0 = 273.15
KT = 0.514444
MON = {m: i for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


class Ascent:
    def __init__(self, wmo, launch, kind):
        self.wmo, self.launch, self.kind = wmo, launch, kind
        self.lat = self.lon = self.elev = None
        self.lev = []            # (p hPa, z m, T C, Td C, wdir deg, wspd m/s); nan = missing

    @property
    def slot(self):              # nominal synoptic time: launch time to the nearest 6 h (launch is ~45 min early)
        t = self.launch + dt.timedelta(hours=3)
        t = t.replace(hour=t.hour // 6 * 6, minute=0, second=0, microsecond=0)
        return t.strftime("%Y%m%d_%H")

    def score(self):             # which report of the same ascent to keep
        rank = {"bufr-high": 3, "bufr-low": 2, "temp": 1}[self.kind]
        nt = sum(1 for l in self.lev if np.isfinite(l[2]))
        return (min(nt, 100), rank, nt)          # enough levels first, then the better format


# ------------------------------------------------------------------ download
def session():
    import requests
    s = requests.Session(); s.headers["User-Agent"] = "globalweathercharts.com T-phi grams"
    return s


def listing(s, folder):
    """[(file name, modified UTC)] from the folder's ls-l (times are UTC, the year is not shown for recent files)."""
    txt = s.get(BASE + folder + "ls-l", timeout=60).text
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); out = []
    for line in txt.splitlines():
        f = line.split()
        if len(f) < 9 or not f[8].startswith("sn."): continue
        try:
            mo, day = MON[f[5]], int(f[6])
            if ":" in f[7]:
                hh, mm = map(int, f[7].split(":")); t = dt.datetime(now.year, mo, day, hh, mm)
                if t > now + dt.timedelta(days=1): t = t.replace(year=now.year - 1)
            else:
                t = dt.datetime(int(f[7]), mo, day)
        except Exception:
            continue
        out.append((f[8], t))
    return out


def download(hours=30, cache="feed_cache"):
    """Files modified in the last `hours` hours of all three feeds -> [(kind, path)]. Already cached files are reused."""
    s = session(); cut = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=hours)
    jobs = []
    for kind, folder in FEEDS:
        os.makedirs(os.path.join(cache, kind), exist_ok=True)
        for name, t in listing(s, folder):
            if t < cut: continue
            path = os.path.join(cache, kind, f"{t:%Y%m%d%H%M}_{name}")
            jobs.append((kind, folder + name, path))

    def get(job):
        kind, url, path = job
        if os.path.exists(path): return kind, path
        for attempt in range(3):
            try:
                r = s.get(BASE + url, timeout=120); r.raise_for_status()
                open(path, "wb").write(r.content); return kind, path
            except Exception:
                if attempt == 2: return kind, None
                time.sleep(3)
    with ThreadPoolExecutor(8) as ex:
        got = [g for g in ex.map(get, jobs) if g[1]]
    print(f"feed: {len(got)} files from the last {hours} h", flush=True)
    return got


# ------------------------------------------------------------------ BUFR TEMP (ecCodes)
def _arr(h, key, n=None):
    import eccodes
    try: a = np.asarray(eccodes.codes_get_array(h, key), float)
    except Exception: return None
    a = np.where((a < MISS) | (a >= 2147483647), np.nan, a)
    return a[:n] if n is not None else a


def _val(h, key):
    import eccodes
    try:
        v = eccodes.codes_get(h, key)
        return None if v is None or (isinstance(v, float) and v < MISS) or v == 2147483647 else v
    except Exception:
        return None


def read_bufr(path, kind):
    import eccodes
    out = []
    with open(path, "rb") as f:
        while True:
            try: h = eccodes.codes_bufr_new_from_file(f)
            except Exception: break
            if h is None: break
            try:
                eccodes.codes_set(h, "unpack", 1)
                ns = eccodes.codes_get(h, "numberOfSubsets")
                comp = eccodes.codes_get(h, "compressedData")
                pre = [""] if ns == 1 else ([f"/subsetNumber={k}/" for k in range(1, ns + 1)] if not comp else [])
                for px in pre:
                    a = _bufr_subset(h, px, kind)
                    if a: out.append(a)
            except Exception:
                pass
            finally:
                eccodes.codes_release(h)
    return out


def _bufr_subset(h, px, kind):
    b, s = _val(h, px + "blockNumber"), _val(h, px + "stationNumber")
    if px:   # per-subset scalars come back as arrays of one
        b = (_arr(h, px + "blockNumber") or [None])[0]; s = (_arr(h, px + "stationNumber") or [None])[0]
    if b is None or s is None or not np.isfinite(b) or not np.isfinite(s) or b <= 0: return None
    wmo = f"{int(b):02d}{int(s):03d}"
    g = lambda k: (_arr(h, px + k) if px else None)
    def scal(k):
        if px:
            v = g(k); return None if v is None or not len(v) or not np.isfinite(v[0]) else v[0]
        return _val(h, k)
    y, mo, d, hh, mi = (scal(k) for k in ("year", "month", "day", "hour", "minute"))
    if None in (y, mo, d, hh): return None
    try: launch = dt.datetime(int(y), int(mo), int(d), int(hh), int(mi or 0))
    except ValueError: return None
    n = scal("extendedDelayedDescriptorReplicationFactor")
    p = _arr(h, px + "pressure", None)
    if n is None or p is None: return None
    n = int(n); p = p[:n] / 100
    get = lambda k: (lambda a: a if a is not None and len(a) == n else np.full(n, np.nan))(_arr(h, px + k, n))
    z, t, td = get("nonCoordinateGeopotentialHeight"), get("airTemperature") - K0, get("dewpointTemperature") - K0
    wd, ws = get("windDirection"), get("windSpeed")
    a = Ascent(wmo, launch, kind)
    a.lat, a.lon = scal("latitude"), scal("longitude")
    a.elev = scal("heightOfStationGroundAboveMeanSeaLevel") or scal("height")
    a.lev = list(zip(p, z, t, td, wd, ws))
    return a


# ------------------------------------------------------------------ coded TEMP (FM 35) text
STD = {"00": 1000, "92": 925, "85": 850, "70": 700, "50": 500, "40": 400, "30": 300, "25": 250, "20": 200, "15": 150, "10": 100}
ID_P = {"1": 100, "2": 200, "3": 300, "4": 400, "5": 500, "6": 600, "7": 700, "8": 850, "9": 925, "0": 1000}


def _t(g):
    if len(g) != 5 or "/" in g[:3]: return np.nan, np.nan
    v = int(g[:3]) / 10; v = -v if int(g[2]) % 2 else v
    dd = g[3:]
    if "/" in dd: return v, np.nan
    d = int(dd); dep = d / 10 if d <= 50 else (d - 50 if d >= 56 else np.nan)
    return v, v - dep


def _w(g, knots):
    if len(g) != 5 or "/" in g: return np.nan, np.nan
    ddd, ff = int(g[:3]), int(g[3:]); ff += 100 * (ddd % 5); ddd -= ddd % 5
    if ddd > 360: return np.nan, np.nan
    return float(ddd), ff * (KT if knots else 1.0)


def _std_height(code, hhh):
    if "/" in hhh: return np.nan
    h = int(hhh)
    if code == "00": return -(h - 500) if h >= 500 else h
    if code == "92": return h
    if code == "85": return 1000 + h
    if code == "70": return 3000 + h if h < 500 else 2000 + h
    if code in ("50", "40"): return h * 10
    if code in ("30", "25"): return (h + 1000) * 10 if h < 500 else h * 10
    return (h + 1000) * 10


def _yygg(g, ref):
    """YYGG -> launch datetime (month/year from the file time ref), knots flag, last indicator digit."""
    if len(g) != 5 or not g[:4].isdigit(): return None
    yy, gg = int(g[:2]), int(g[2:4]); knots = yy > 50; day = yy - 50 if knots else yy
    if not (1 <= day <= 31 and gg <= 23): return None
    for back in range(0, 3):                  # same month, or the month before (ref early in a month)
        m = (ref.month - back - 1) % 12 + 1; y = ref.year - (1 if ref.month - back < 1 else 0)
        try: t = dt.datetime(y, m, day, gg)
        except ValueError: continue
        if t <= ref + dt.timedelta(hours=6): return t, knots, g[4]
    return None


def parse_ttaa(gs, ref):
    r = _yygg(gs[0], ref)
    if not r: return None
    t0, knots, idc = r; wmo = gs[1]; lev = []; i = 2
    wind_to = ID_P.get(idc)        # winds reported down to this pressure (None: no winds)
    while i < len(gs):
        g = gs[i]
        if g[:2] == "99" and i == 2:
            p = int(g[2:]) if g[2:].isdigit() else None
            if p is None: return None
            p = p + 1000 if p < 100 else p
            T, Td = _t(gs[i + 1]) if i + 1 < len(gs) else (np.nan, np.nan)
            wd = ws = np.nan; i += 2
            if wind_to is not None and i < len(gs): wd, ws = _w(gs[i], knots); i += 1
            lev.append((p, np.nan, T, Td, wd, ws, "sfc")); continue
        code = g[:2]
        if code in STD:
            p = STD[code]; z = _std_height(code, g[2:])
            T, Td = _t(gs[i + 1]) if i + 1 < len(gs) else (np.nan, np.nan)
            i += 2; wd = ws = np.nan
            if wind_to is not None and p >= wind_to and i < len(gs): wd, ws = _w(gs[i], knots); i += 1
            lev.append((p, z, T, Td, wd, ws, "std")); continue
        if code in ("88", "77", "66") or g in ("31313", "51515", "41414", "21212"):
            break
        i += 1
    return wmo, t0, lev


def parse_ttbb(gs, ref):
    r = _yygg(gs[0], ref)
    if not r: return None
    t0, knots, _ = r; wmo = gs[1]; lev = []; i = 2; wind = False
    while i + 1 < len(gs):
        g = gs[i]
        if g == "21212": wind = True; i += 1; continue
        if g in ("31313", "41414", "51515", "61616"): break
        if len(g) == 5 and g[0] == g[1] and g[0].isdigit() and g[2:].isdigit():
            p = int(g[2:]); p = p + 1000 if p < 100 else p
            if wind:
                wd, ws = _w(gs[i + 1], knots); lev.append((p, np.nan, np.nan, np.nan, wd, ws, "sigw"))
            else:
                T, Td = _t(gs[i + 1]); lev.append((p, np.nan, T, Td, np.nan, np.nan, "sigt"))
            i += 2; continue
        i += 1
    return wmo, t0, lev


HDR = re.compile(r"^[A-Z]{4}\d\d [A-Z]{4} (\d{6})", re.M)


def read_temp(path, ref):
    """Coded TEMP text file -> list of Ascent (TTAA and TTBB parts merged per station and launch)."""
    txt = open(path, encoding="latin-1", errors="ignore").read().replace("\r", "")
    parts = {}
    starts = [m.start() for m in HDR.finditer(txt)] + [len(txt)]
    for a, b in zip(starts[:-1], starts[1:]):
        bull = txt[a:b]; body = bull.split("\n", 1)[1] if "\n" in bull else ""
        code = None
        for rep in body.split("="):
            gs = rep.split()
            if not gs: continue
            if gs[0] in ("TTAA", "TTBB", "TTCC", "TTDD", "PPAA", "PPBB", "PPCC", "PPDD"): code = gs[0]; gs = gs[1:]
            if code not in ("TTAA", "TTBB") or len(gs) < 3 or "NIL" in gs[:3]: continue
            try:
                r = (parse_ttaa if code == "TTAA" else parse_ttbb)(gs, ref)
            except Exception:
                r = None
            if not r or not re.fullmatch(r"\d{5}", r[0]): continue
            parts[(r[0], r[1], code)] = r[2]               # a later copy (correction) replaces an earlier one
    return parts


def merge_temp(parts):
    by = {}
    for (wmo, t0, code), lev in parts.items():
        by.setdefault((wmo, t0), {})[code] = lev
    out = []
    for (wmo, t0), d in by.items():
        if "TTAA" not in d: continue                       # need the standard levels (heights) to place the rest
        a = Ascent(wmo, t0, "temp")
        a.lev = [l[:6] for l in d["TTAA"] + d.get("TTBB", [])]
        out.append(a)
    return out


# ------------------------------------------------------------------ all feeds -> best ascent per station and time
def collect(hours=30, cache="feed_cache"):
    files = download(hours, cache)
    best = {}; temp_parts = {}

    def keep(a):
        k = (a.wmo, a.slot)
        if k not in best or a.score() > best[k].score(): best[k] = a
    import pickle
    for kind, path in sorted(files, key=lambda x: x[1]):
        pk = path + ".pkl"                                    # each feed file is decoded once (BUFR decoding is slow)
        try: got = pickle.load(open(pk, "rb"))
        except Exception:
            if kind == "temp":
                m = re.match(r"(\d{12})_", os.path.basename(path))
                ref = dt.datetime.strptime(m[1], "%Y%m%d%H%M") if m else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                got = read_temp(path, ref)
            else:
                got = read_bufr(path, kind)
            try: pickle.dump(got, open(pk, "wb"))
            except Exception: pass
        if kind == "temp": temp_parts.update(got)
        else:
            for a in got: keep(a)
    for a in merge_temp(temp_parts): keep(a)
    return best


# ------------------------------------------------------------------ levels -> sounding for tphi_core.make_chart
Rd, G = 287.04, 9.80665


def to_sounding(a):
    """Ascent -> dict(p,t,td,rh,z,d,s,tdok) with surface first, plus notes. Raises ValueError if not usable."""
    L = np.array([l for l in a.lev if np.isfinite(l[0]) and l[0] > 0], float).reshape(-1, 6)
    if not len(L): raise ValueError("no levels")
    L = L[np.argsort(-L[:, 0], kind="stable")]
    # merge reports of the same pressure (e.g. a standard level that is also a significant level)
    P = np.round(L[:, 0], 1); M = []
    for p in np.unique(P)[::-1]:
        rows = L[P == p]; M.append([p] + [next((v for v in rows[:, j] if np.isfinite(v)), np.nan) for j in range(1, 6)])
    L = np.array(M)
    ok_t = np.isfinite(L[:, 2])
    if ok_t.sum() < 6: raise ValueError(f"only {ok_t.sum()} temperature levels")
    p_sfc = L[ok_t, 0].max()
    L = L[L[:, 0] <= p_sfc]                                   # drop levels below ground (1000 hPa at high stations)
    ok_t = np.isfinite(L[:, 2]); top = L[ok_t, 0].min()
    if top > 400: raise ValueError(f"ascent ends at {top:.0f} hPa")
    L = L[L[:, 0] >= top]
    lp = np.log(L[:, 0]); ok_t = np.isfinite(L[:, 2])
    ip = lambda m, y: np.interp(-lp, -lp[m], y[m])            # log-p interpolation (x increasing)
    T = ip(ok_t, L[:, 2])
    ok_d = np.isfinite(L[:, 3]) & (L[:, 3] > -100)
    notes = []
    if ok_d.sum() >= 3:
        Td = ip(ok_d, L[:, 3]); tdok = L[:, 0] >= L[ok_d, 0].min() - 1e-6
        if L[ok_d, 0].min() > top + 50:
            notes.append(f"no dew point above {L[ok_d, 0].min():.0f} hPa (taken as dry)")
    else:
        Td = T - 30; tdok = np.zeros(len(L), bool); notes.append("no dew point reported (taken as dry)")
    Td = np.where(tdok, np.minimum(Td, T), np.minimum(T - 30, Td if ok_d.sum() >= 3 else T - 30))
    ok_w = np.isfinite(L[:, 4]) & np.isfinite(L[:, 5])
    if ok_w.sum() >= 2:
        u = -L[:, 5] * np.sin(np.radians(L[:, 4])); v = -L[:, 5] * np.cos(np.radians(L[:, 4]))
        uu, vv = ip(ok_w, u), ip(ok_w, v)
        S = np.hypot(uu, vv); D = np.degrees(np.arctan2(-uu, -vv)) % 360
    else:
        S = np.zeros(len(L)); D = np.zeros(len(L)); notes.append("no winds reported")
    # heights: reported where (nearly) every level has one, else hypsometric from the surface
    z = L[:, 1]; okz = np.isfinite(z)
    if okz.mean() >= 0.9 and okz.sum() >= 10:
        Z = ip(okz, z)
    else:
        es = 6.112 * np.exp(17.67 * Td / (Td + 243.5)); w = 0.622 * es / (L[:, 0] - es)
        tv = (T + K0) * (1 + w / 0.622) / (1 + w)
        dz = Rd / G * 0.5 * (tv[1:] + tv[:-1]) * np.log(L[:-1, 0] / L[1:, 0])
        Z = np.r_[0, np.cumsum(dz)]
        z0 = z[0] if np.isfinite(z[0]) else (a.elev if a.elev is not None and np.isfinite(a.elev) and a.elev > -500 else None)
        if z0 is None and okz.any():                          # anchor to the reported standard-level heights
            z0 = float(np.median(z[okz] - Z[okz]))
        Z = Z + (z0 or 0.0)
    RH = 100 * np.exp(17.67 * Td / (Td + 243.5)) / np.exp(17.67 * T / (T + 243.5))
    R = np.c_[L[:, 0], T, RH, Td, Z, D, S, tdok.astype(float)]
    return R, notes
