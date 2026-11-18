"""Download and open the hourly full-disk scans used by the Satellite page.

  Meteosat-9 (45.5°E, Indian Ocean)  EUMETSAT Data Store, hourly SEVIRI Level 1.5 scans (the scan that starts
                                     on the hour). Hourly Meteosat data are EUMETSAT "Core" data: free for any
                                     use under CC BY 4.0. Needs a free EUMETSAT account: EUMETSAT_KEY / EUMETSAT_SECRET.
  Himawari-9 (140.7°E)               NOAA Open Data on AWS, s3://noaa-himawari9 (AHI Level 1b, HSD format)
  GOES-19 (75.2°W), GOES-18 (137.2°W) NOAA Open Data on AWS, s3://noaa-goes19 / noaa-goes18 (ABI L2 CMIP full disk)

Every loader returns {"ir": Band, "wv": Band, "vis": Band} (only the ones asked for) or raises FeedMissing.
"""
import datetime as dt, glob, os, re, shutil, threading, time, warnings, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from sat_core import Band, SATS

warnings.filterwarnings("ignore")
try:
    import dask; dask.config.set(scheduler="threads", num_workers=2)
except Exception:
    pass
UA = {"User-Agent": "globalweathercharts.com satellite job"}
S = requests.Session(); S.headers.update(UA)


NC_LOCK = threading.Lock()          # the netCDF/HDF5 library is not thread-safe


class FeedMissing(Exception):
    pass


def _get(url, path, tries=3):
    for i in range(tries):
        try:
            with S.get(url, stream=True, timeout=120) as r:
                if r.status_code == 404: raise FeedMissing(url)
                r.raise_for_status()
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20): f.write(chunk)
                os.replace(tmp, path); return path
        except FeedMissing:
            raise
        except Exception:
            if i == tries - 1: raise
            time.sleep(3 * (i + 1))


def _s3_list(bucket, prefix):
    keys, token = [], None
    while True:
        p = {"list-type": "2", "prefix": prefix}
        if token: p["continuation-token"] = token
        r = S.get(f"https://{bucket}.s3.amazonaws.com/", params=p, timeout=60); r.raise_for_status()
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m: return keys
        token = m.group(1)


# ------------------------------------------------------------------------------------------ GOES
def goes_files(sat, when):
    bucket = {"g19": "noaa-goes19", "g18": "noaa-goes18"}[sat]
    keys = _s3_list(bucket, f"ABI-L2-CMIPF/{when:%Y}/{when.timetuple().tm_yday:03d}/{when:%H}/")
    tag = f"_s{when:%Y}{when.timetuple().tm_yday:03d}{when:%H%M}"
    out = {}
    for k in keys:
        m = re.search(r"-M\dC(\d\d)_G\d\d" + tag, k)
        if m: out[int(m.group(1))] = f"https://{bucket}.s3.amazonaws.com/{k}"
    return out


def load_goes(sat, when, need, cache):
    import netCDF4
    files = goes_files(sat, when)
    chan = {"ir": 13, "wv": 9, "vis": 2}
    if any(chan[n] not in files for n in need): raise FeedMissing(f"{sat} {when:%H%M} not complete yet")
    os.makedirs(cache, exist_ok=True)
    paths = {n: os.path.join(cache, files[chan[n]].rsplit("/", 1)[1]) for n in need}
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda n: os.path.exists(paths[n]) or _get(files[chan[n]], paths[n]), need))
    out = {}
    for n in need:
        f = netCDF4.Dataset(paths[n]); p = f["goes_imager_projection"]
        h = float(p.perspective_point_height)
        from pyproj import CRS
        crs = CRS.from_proj4(f"+proj=geos +lon_0={float(p.longitude_of_projection_origin)} +h={h} "
                             f"+a={float(p.semi_major_axis)} +b={float(p.semi_minor_axis)} +sweep=x +units=m +no_defs")
        x = f["x"][:].astype(float) * h; y = f["y"][:].astype(float) * h
        dx = x[1] - x[0]; dy = y[1] - y[0]
        ext = (x[0] - dx / 2, y[-1] + dy / 2, x[-1] + dx / 2, y[0] - dy / 2)
        shape = f["CMI"].shape; f.close()
        km = SATS[sat]["km"][n]

        def read(r0, r1, c0, c1, path=paths[n], n=n):
            with NC_LOCK:
                g = netCDF4.Dataset(path)
                a = g["CMI"][r0:r1, c0:c1]; g.close()
            a = np.ma.filled(a.astype(np.float32), np.nan)
            return a if n == "vis" else a - 273.15
        out[n] = Band(crs, ext, shape, read, km, when)
    return out


# ------------------------------------------------------------------------------------------ Himawari
HIM_BAND = {"ir": ("B13", "R20"), "wv": ("B09", "R20"), "vis": ("B03", "R05")}
HIM_EXT = 5499999.9


def him_segments(boxes):
    """Which of the 10 full-disk segments (1..10) cover the given (lat0, lat1, lon0, lon1) boxes."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "+proj=geos +lon_0=140.7 +h=35785863 +a=6378137 +rf=298.257024882273 +units=m",
                              always_xy=True)
    segs = set()
    for la0, la1, lo0, lo1 in boxes:
        lo, la = np.meshgrid(np.linspace(lo0, lo1, 30), np.linspace(la0, la1, 30))
        x, y = tr.transform(np.where(lo > 180, lo - 360, lo), la)
        y = np.asarray(y)[np.isfinite(y)]
        if not y.size: continue
        for v in (y.min(), y.max()):
            segs.add(int(np.clip((HIM_EXT - v) / (2 * HIM_EXT) * 10, 0, 9)) + 1)
        segs.update(range(min(segs), max(segs) + 1))
    return sorted(segs)


def load_him(when, need, cache, vis_boxes=None):
    from satpy import Scene
    base = f"AHI-L1b-FLDK/{when:%Y/%m/%d/%H%M}/"
    keys = set(k.rsplit("/", 1)[1] for k in _s3_list("noaa-himawari9", base))
    out = {}
    os.makedirs(cache, exist_ok=True)
    for n in need:
        b, r = HIM_BAND[n]
        segs = him_segments(vis_boxes) if (n == "vis" and vis_boxes) else range(1, 11)
        names = [f"HS_H09_{when:%Y%m%d_%H%M}_{b}_FLDK_{r}_S{s:02d}10.DAT.bz2" for s in segs]
        if any(nm not in keys for nm in names): raise FeedMissing(f"him9 {when:%H%M} {b} not complete yet")
        with ThreadPoolExecutor(8) as ex:
            paths = list(ex.map(lambda nm: os.path.join(cache, nm) if os.path.exists(os.path.join(cache, nm)) else
                                _get(f"https://noaa-himawari9.s3.amazonaws.com/{base}{nm}", os.path.join(cache, nm)), names))
        sc = Scene(reader="ahi_hsd", filenames=paths)
        sc.load([b])
        da = sc[b]; area = da.attrs["area"]
        H, W = area.shape; per = H // 10
        k = 2 if n == "vis" else 1                      # 0.5 km visible -> 1 km block means (plenty for the images, 1/4 memory)
        arr = np.full((H // k, W // k), np.nan, np.float32)
        for sg in segs:                                  # one segment at a time keeps memory low
            a = np.asarray(da.data[(sg - 1) * per: sg * per].compute(), np.float32)
            if k > 1: a = np.nanmean(a.reshape(per // k, k, W // k, k), axis=(1, 3))
            arr[(sg - 1) * per // k: sg * per // k] = a / 100.0 if n == "vis" else a - 273.15
        del sc, da
        out[n] = Band(area.crs, area.area_extent, arr.shape, lambda r0, r1, c0, c1, arr=arr: arr[r0:r1, c0:c1],
                      SATS["him9"]["km"][n] * k, when)
    return out


# ------------------------------------------------------------------------------------------ Meteosat-9 IODC
EUM = "https://api.eumetsat.int"
EUM_COLL = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"
MET_BAND = {"ir": "IR_108", "wv": "WV_062", "vis": "VIS006"}


def eum_ready():
    return bool(os.environ.get("EUMETSAT_KEY") and os.environ.get("EUMETSAT_SECRET"))


_tok = {}


def eum_token():
    if _tok.get("exp", 0) > time.time() + 60: return _tok["t"]
    r = S.post(f"{EUM}/token", data={"grant_type": "client_credentials"},
               auth=(os.environ["EUMETSAT_KEY"], os.environ["EUMETSAT_SECRET"]), timeout=60)
    r.raise_for_status(); j = r.json()
    _tok.update(t=j["access_token"], exp=time.time() + int(j.get("expires_in", 3600)))
    return _tok["t"]


def met_product(when):
    """The IODC scan that starts on the hour `when` (an "hourly" scan), or None if not in the Data Store yet."""
    p = {"format": "json", "pi": EUM_COLL, "dtstart": (when - dt.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "dtend": (when + dt.timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    r = S.get(f"{EUM}/data/search-products/1.0.0/os", params=p, timeout=60); r.raise_for_status()
    for f in r.json().get("features", []):
        start = f["properties"]["date"].split("/")[0]
        if start.startswith(when.strftime("%Y-%m-%dT%H:00")):
            return f["id"], f["properties"]["links"]["data"][0]["href"]
    return None


def load_met(when, need, cache):
    if not eum_ready(): raise FeedMissing("no EUMETSAT key")
    from satpy import Scene
    prod = met_product(when)
    if not prod: raise FeedMissing(f"met9 {when:%H%M} not in the Data Store yet")
    pid, href = prod
    os.makedirs(cache, exist_ok=True)
    nat = glob.glob(os.path.join(cache, pid + "*.nat"))
    if not nat:
        z = os.path.join(cache, pid + ".zip")
        for i in range(3):
            try:
                with S.get(href, headers={"Authorization": "Bearer " + eum_token()}, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(z + ".part", "wb") as f:
                        for c in r.iter_content(1 << 20): f.write(c)
                os.replace(z + ".part", z); break
            except Exception:
                if i == 2: raise
                time.sleep(10)
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".nat"))
            with zf.open(name) as src, open(os.path.join(cache, pid + ".nat"), "wb") as dst: shutil.copyfileobj(src, dst)
        os.remove(z)
        nat = [os.path.join(cache, pid + ".nat")]
    sc = Scene(reader="seviri_l1b_native", filenames=nat)
    names = [MET_BAND[n] for n in need]
    sc.load(names)
    out = {}
    for n in need:
        da = sc[MET_BAND[n]]; area = da.attrs["area"]
        arr = np.asarray(da.data.compute(), np.float32)
        arr = arr / 100.0 if n == "vis" else arr - 273.15
        out[n] = Band(area.crs, area.area_extent, arr.shape, lambda r0, r1, c0, c1, arr=arr: arr[r0:r1, c0:c1],
                      SATS["met9"]["km"][n], when)
    del sc
    return out


# ------------------------------------------------------------------------------------------ NOAA GMGSI
GM_DIR = {"ir": "LW", "wv": "WV", "vis": "VIS"}


def load_gmgsi(when, need, cache):
    """NOAA/NESDIS Global Mosaic of Geostationary Satellite Imagery (AWS noaa-gmgsi-pds), hourly, ~8 km.
    Values are 0-255 image counts; IR/WV follow the McIDAS brightness scale (cold = high), 255 / dqf 20 = no data."""
    import netCDF4
    from sat_core import LatLonBand
    os.makedirs(cache, exist_ok=True)
    out = {}
    for n in need:
        d = GM_DIR[n]
        keys = _s3_list("noaa-gmgsi-pds", f"GMGSI_{d}/{when:%Y/%m/%d/%H}/")
        keys = [k for k in keys if f"_s{when:%Y%m%d%H}00" in k]
        if not keys: raise FeedMissing(f"gmgsi {d} {when:%H%M} not out yet")
        path = os.path.join(cache, keys[-1].rsplit("/", 1)[1])
        if not os.path.exists(path): _get(f"https://noaa-gmgsi-pds.s3.amazonaws.com/{keys[-1]}", path)
        with NC_LOCK:
            f = netCDF4.Dataset(path)
            c = np.ma.filled(f["data"][0].astype(np.float32), np.nan)
            q = np.ma.filled(f["dqf"][0], 20)
            lats = f["lat"][:, 0].astype(float); lons = f["lon"][0, :].astype(float); f.close()
        c[(q >= 20) | (c >= 255) | (c < 0)] = np.nan
        if n == "vis":
            arr = c / 255.0
        else:
            k = np.where(c <= 176, 330.0 - c / 2.0, 418.0 - c)
            arr = (k - 273.15).astype(np.float32)
        dlon = float(np.median(np.diff(lons[1:50])))
        out[n] = LatLonBand(lats, float(lons[0]), dlon, c.shape[1], lambda r0, r1, c0, c1, arr=arr: arr[r0:r1, c0:c1],
                            8, when, pre=(n == "vis"))
    return out


def load(sat, when, need, cache, vis_boxes=None):
    need = sorted(set(need))
    if sat == "gmgsi": return load_gmgsi(when, need, cache)
    if sat == "him9": return load_him(when, need, cache, vis_boxes)
    if sat == "met9": return load_met(when, need, cache)
    return load_goes(sat, when, need, cache)


def available(sat):
    return eum_ready() if sat == "met9" else True
