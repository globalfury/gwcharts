"""Satellite imagery for globalweathercharts.com/sat - sectors, colour tables and drawing.

Every image is a latitude-longitude (equirectangular) view of one sector, sampled straight from the
satellite's own fixed grid (nearest pixel, with block averaging where the satellite pixels are much
smaller than the image pixels). Drawn with Pillow: title line, panel with coastlines and India-view
country boundaries, latitude/longitude labels, colour bar and credit line.
"""
import datetime as dt, json, math, os
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get("SITE_NAME", "globalweathercharts.com")

# ------------------------------------------------------------------------------------------ satellites
# lon: sub-satellite longitude. Bands: which channel draws each product.
SATS = {
    "met9": dict(name="Meteosat-9", lon=45.5, credit="Contains modified EUMETSAT Meteosat-9 data ({y}), CC BY 4.0",
                 ir=("IR_108", "Channel 9 (IR 10.8 µm)"), wv=("WV_062", "Channel 5 (Water Vapour 6.2 µm)"),
                 vis=("VIS006", "Channel 1 (Visible 0.6 µm)"), km=dict(ir=3, wv=3, vis=3)),
    "him9": dict(name="Himawari-9", lon=140.7, credit="Data: JMA Himawari-9, via NOAA",
                 ir=("B13", "Band 13 (IR 10.4 µm)"), wv=("B09", "Band 9 (Water Vapour 6.9 µm)"),
                 vis=("B03", "Band 3 (Visible 0.64 µm)"), km=dict(ir=2, wv=2, vis=0.5)),
    "g19": dict(name="GOES-19", lon=-75.2, credit="Data: NOAA GOES-19",
                ir=("C13", "Channel 13 (IR 10.3 µm)"), wv=("C09", "Channel 9 (Water Vapour 6.9 µm)"),
                vis=("C02", "Channel 2 (Visible 0.64 µm)"), km=dict(ir=2, wv=2, vis=0.5)),
    "g18": dict(name="GOES-18", lon=-137.2, credit="Data: NOAA GOES-18",
                ir=("C13", "Channel 13 (IR 10.3 µm)"), wv=("C09", "Channel 9 (Water Vapour 6.9 µm)"),
                vis=("C02", "Channel 2 (Visible 0.64 µm)"), km=dict(ir=2, wv=2, vis=0.5)),
    "gmgsi": dict(name="NOAA Global Mosaic", lon=None, credit="Data: NOAA/NESDIS Global Mosaic of Geostationary Satellite Imagery (GOES, Himawari, Meteosat)",
                  ir=("LW", "Longwave IR (~11 µm)"), wv=("WV", "Water Vapour (~6.7 µm)"),
                  vis=("VIS", "Visible"), km=dict(ir=8, wv=8, vis=8)),
}

# ------------------------------------------------------------------------------------------ sectors
# id: (name, group, satellites in order of preference, lat0, lat1, lon0, lon1, degrees per pixel)
# Longitudes east of 180 are written as >180 (e.g. 190 = 170°W) so a sector can cross the date line.
GROUPS = [("india", "India & Indian Ocean"), ("pacific", "Pacific & Asia"), ("atlantic", "Atlantic & Americas"),
          ("global", "Global, Europe & Africa")]
SECTORS = {
    "india":   ("India", "india", ["met9", "gmgsi"], 5, 38, 65, 100, 0.04),
    "nwindia": ("North-West India & Western Himalaya", "india", ["met9", "gmgsi"], 22, 40, 60, 86, 0.03),
    "bob":     ("Bay of Bengal", "india", ["met9", "him9"], 4, 26, 76, 100, 0.03),
    "arb":     ("Arabian Sea", "india", ["met9", "gmgsi"], 4, 28, 50, 78, 0.03),
    "nio":     ("North Indian Ocean", "india", ["met9", "gmgsi"], -5, 32, 40, 105, 0.06),
    "swio":    ("South-West Indian Ocean", "india", ["met9", "gmgsi"], -35, -3, 30, 80, 0.06),
    "sio":     ("South-East Indian Ocean", "india", ["met9", "him9"], -35, -3, 75, 125, 0.06),
    "wpac":    ("West Pacific", "pacific", ["him9"], 0, 35, 110, 165, 0.06),
    "scs":     ("South China Sea & Philippines", "pacific", ["him9"], 0, 25, 100, 132, 0.04),
    "easia":   ("China, Korea & Japan", "pacific", ["him9"], 18, 50, 100, 150, 0.06),
    "seasia":  ("South-East Asia & Indonesia", "pacific", ["him9"], -12, 22, 90, 135, 0.05),
    "aus":     ("Australia", "pacific", ["him9"], -45, -5, 105, 160, 0.06),
    "spac":    ("South-West Pacific", "pacific", ["him9"], -35, -5, 150, 195, 0.06),
    "atl":     ("Tropical Atlantic", "atlantic", ["g19"], 5, 32, -85, -15, 0.07),
    "gulf":    ("Gulf of Mexico & Caribbean", "atlantic", ["g19"], 8, 32, -100, -58, 0.05),
    "conus":   ("United States", "atlantic", ["g19", "g18"], 22, 52, -127, -65, 0.06),
    "epac":    ("East Pacific", "atlantic", ["g18"], 5, 30, -140, -90, 0.06),
    "cpac":    ("Central Pacific & Hawaii", "atlantic", ["g18"], 8, 30, -180, -140, 0.05),
    "samer":   ("South America", "atlantic", ["g19"], -45, 12, -85, -30, 0.08),
    "world":   ("World", "global", ["gmgsi"], -60, 70, -180, 180, 0.25),
    "europe":  ("Europe", "global", ["gmgsi"], 34, 72, -25, 45, 0.08),
    "mideast": ("Middle East & Central Asia", "global", ["gmgsi"], 12, 45, 30, 80, 0.06),
    "africa":  ("Africa", "global", ["gmgsi"], -36, 38, -20, 55, 0.1),
}
FLOATER = dict(w=18.0, h=13.0, res=0.02)        # storm-centred box, 900 x 650 px (2 km)
PRODUCTS = [("ir", "Longwave IR"), ("dvorak", "Longwave IR [Dvorak]"), ("vis", "Visible"), ("wv", "Water Vapour")]
BAND_OF = {"ir": "ir", "dvorak": "ir", "vis": "vis", "wv": "wv"}


def sat_view_ok(sat, lat0, lat1, lon0, lon1, limit=72.0):
    """True if the whole box is seen at an earth-central angle under `limit` degrees."""
    s = SATS[sat]["lon"]
    if s is None: return lat0 >= -73 and lat1 <= 73          # global mosaic
    worst = 0.0
    for la in (lat0, lat1, (lat0 + lat1) / 2):
        for lo in (lon0, lon1, (lon0 + lon1) / 2):
            d = math.degrees(math.acos(max(-1, min(1, math.cos(math.radians(la)) * math.cos(math.radians(lo - s))))))
            worst = max(worst, d)
    return worst < limit


def best_sat(lat, lon, avail):
    best, bd = None, 1e9
    for k in avail:
        if SATS[k]["lon"] is None: continue
        d = math.degrees(math.acos(max(-1, min(1, math.cos(math.radians(lat)) * math.cos(math.radians(lon - SATS[k]["lon"]))))))
        if d < bd: best, bd = k, d
    if bd < 70: return best
    return "gmgsi" if "gmgsi" in avail and abs(lat) < 70 else None


# ------------------------------------------------------------------------------------------ colour tables
def _hex(c): return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
T_MIN, T_MAX, T_STEP = -110.0, 60.0, 0.1


def _lut(stops):
    """stops: [(°C, '#rrggbb'), ...] ascending; a repeated temperature makes a sharp step."""
    t = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
    xs = [s[0] for s in stops]; cs = np.array([_hex(s[1]) for s in stops], float)
    return np.stack([np.interp(t, xs, cs[:, i]) for i in range(3)], 1).round().astype(np.uint8)


def _steps(levels, warm):
    """Dvorak-type table: levels [(upper °C, grey)], coldest first; warm ramp above the last level."""
    st = []; lo = T_MIN
    for up, g in levels:
        c = "#%02x%02x%02x" % (g, g, g); st += [(lo, c), (up, c)]; lo = up + 1e-3
    st += [(lo, warm[0]), (T_MAX, warm[1])]
    return st


IR_STOPS = [(-110, "#16052e"), (-100, "#3a0c6e"), (-90, "#a020c0"), (-85, "#ff7ce0"), (-80, "#ffffff"),
            (-75, "#8c8c8c"), (-70, "#1e1e1e"), (-64, "#7a0000"), (-58, "#e00000"), (-52, "#ff9a00"),
            (-45, "#f5f500"), (-40, "#3cc83c"), (-30, "#0b6e1f"), (-29.99, "#d4d4d4"), (50, "#000000"), (60, "#000000")]
# Dvorak BD enhancement (grey shades): CDG < -85, CMG -81..-85, W -76..-80, B -70..-75, LG -64..-69,
# MG -54..-63, DG -42..-53, OW -31..-41, WMG -30..+9, warmer than +9 °C a dark ramp.
DVORAK_STOPS = _steps([(-85.5, 70), (-80.5, 130), (-75.5, 255), (-69.5, 0), (-63.5, 200), (-53.5, 150),
                       (-41.5, 100), (-30.5, 230), (9.0, 135)], ("#5a5a5a", "#0a0a0a"))
WV_STOPS = [(-110, "#ffffff"), (-80, "#ffffff"), (-72, "#00d2d2"), (-64, "#0a50b4"), (-56, "#3c8ce6"),
            (-47, "#a5cdf0"), (-40, "#f2f2f2"), (-34, "#ffd583"), (-27, "#e4872d"), (-18, "#8c3c0a"),
            (-5, "#2d1200"), (60, "#000000")]
LUTS = {"ir": _lut(IR_STOPS), "dvorak": _lut(DVORAK_STOPS), "wv": _lut(WV_STOPS)}
CBAR = {"ir": (-100, 40, 10), "dvorak": (-90, 30, 10), "wv": (-90, 0, 10)}


def colourise(prod, val, pre=False):
    """val: °C (IR/WV) or reflectance 0..1.x (vis); NaN = no data. Returns HxWx3 uint8."""
    bad = ~np.isfinite(val)
    if prod == "vis":
        g = (np.clip(np.nan_to_num(val), 0, 1) if pre else np.clip((np.nan_to_num(val) - 0.02) / 0.98, 0, 1) ** 0.62) * 255
        rgb = np.repeat(g.round().astype(np.uint8)[..., None], 3, 2)
    else:
        i = np.clip(np.round((np.nan_to_num(val, nan=0) - T_MIN) / T_STEP), 0, len(LUTS[prod]) - 1).astype(np.int32)
        rgb = LUTS[prod][i]
    rgb[bad] = (40, 40, 48)
    return rgb


# ------------------------------------------------------------------------------------------ sampling
class Band:
    """One channel on a geostationary fixed grid.
    crs: pyproj CRS; extent: (x_ll, y_ll, x_ur, y_ur) in metres (satpy convention); shape (rows, cols);
    read(r0, r1, c0, c1) -> float32 array (°C for IR/WV, reflectance 0..1 for visible), NaN outside the disk."""
    def __init__(self, crs, extent, shape, read, km, time):
        self.crs, self.extent, self.shape, self.read, self.km, self.time = crs, extent, shape, read, km, time

    def indices(self, lon, lat):
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x, y = tr.transform(np.where(lon > 180, lon - 360, lon), lat)
        x0, y0, x1, y1 = self.extent; H, W = self.shape
        col = (np.asarray(x) - x0) / (x1 - x0) * W
        row = (y1 - np.asarray(y)) / (y1 - y0) * H
        ok = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
        return row, col, ok


class LatLonBand(Band):
    """A channel on a latitude-longitude grid: regular longitudes, latitudes given as a (descending) table."""
    def __init__(self, lats, lon0, dlon, ncol, read, km, time, pre=False):
        super().__init__(None, None, (len(lats), ncol), read, km, time)
        self.lats, self.lon0, self.dlon, self.pre = np.asarray(lats, float), lon0, dlon, pre

    def indices(self, lon, lat):
        la = self.lats
        row = np.interp(-lat, -la, np.arange(len(la), dtype=float)) + 0.5     # +0.5: the caller truncates
        col = ((((lon - self.lon0) % 360) / self.dlon) + 0.5) % self.shape[1]
        ok = (lat <= la[0]) & (lat >= la[-1])
        return np.where(ok, row, 0), col, ok


def grid(lat0, lat1, lon0, lon1, res):
    W = int(round((lon1 - lon0) / res)); H = int(round((lat1 - lat0) / res))
    lon = lon0 + (np.arange(W) + 0.5) * res; lat = lat1 - (np.arange(H) + 0.5) * res
    return np.meshgrid(lon, lat)


def sample(band, lon, lat, out_km):
    """Values of `band` at the lon/lat grid. Satellite pixels much finer than the output are block-averaged."""
    row, col, ok = band.indices(lon, lat)
    out = np.full(lon.shape, np.nan, np.float32)
    if not ok.any(): return out
    k = max(1, int(out_km / (band.km * 1.6)))
    r = row[ok]; c = col[ok]
    r0 = int(r.min()) // k * k; c0 = int(c.min()) // k * k
    r1 = min(band.shape[0], (int(r.max()) // k + 1) * k); c1 = min(band.shape[1], (int(c.max()) // k + 1) * k)
    win = band.read(r0, r1, c0, c1).astype(np.float32)
    if k > 1:
        h, w = win.shape[0] // k, win.shape[1] // k
        with np.errstate(invalid="ignore"):
            win = np.nanmean(win[:h * k, :w * k].reshape(h, k, w, k), axis=(1, 3))
    ri = np.clip(((r - r0) / k).astype(np.int64), 0, win.shape[0] - 1)
    ci = np.clip(((c - c0) / k).astype(np.int64), 0, win.shape[1] - 1)
    out[ok] = win[ri, ci]
    return out


def sun_cos(lon, lat, when):
    """Cosine of the solar zenith angle (NOAA simple formula, good to ~0.5°)."""
    d = when.timetuple().tm_yday; h = when.hour + when.minute / 60
    g = 2 * math.pi / 365 * (d - 1 + (h - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    ha = np.radians((h * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.sin(la) * math.sin(decl) + np.cos(la) * math.cos(decl) * np.cos(ha)


def daylight_fraction(lat0, lat1, lon0, lon1, when):
    lo, la = np.meshgrid(np.linspace(lon0, lon1, 25), np.linspace(lat0, lat1, 25))
    return float((sun_cos(lo, la, when) > 0.1).mean())


# ------------------------------------------------------------------------------------------ boundaries
@lru_cache(1)
def _bounds():
    raw = json.load(open(os.path.join(HERE, "boundaries_india_view.json")))
    out = {}
    for kind in ("coast", "borders"):
        for q in ("fine", "coarse"):
            out[(kind, q)] = [np.array(l, float) for l in raw[f"{kind}_{q}"]]
    return out


def _lines(kind, lat0, lat1, lon0, lon1, fine):
    for l in _bounds()[(kind, "fine" if fine else "coarse")]:
        for shift in ((0, 360) if lon1 > 180 else (0,)):
            x = l[:, 0] + shift
            if x.max() < lon0 - 1 or x.min() > lon1 + 1 or l[:, 1].max() < lat0 - 1 or l[:, 1].min() > lat1 + 1: continue
            yield np.column_stack([x, l[:, 1]])


# ------------------------------------------------------------------------------------------ drawing
@lru_cache(8)
def _font(size, bold=False):
    import matplotlib
    p = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(p, size)


def _fmt_lon(x):
    x = (x + 180) % 360 - 180
    return "180°" if abs(abs(x) - 180) < 1e-6 else ("0°" if abs(x) < 1e-6 else f"{abs(x):g}°{'E' if x > 0 else 'W'}")


def _fmt_lat(y): return "0°" if abs(y) < 1e-6 else f"{abs(y):g}°{'N' if y > 0 else 'S'}"


def render(prod, val, box, sat, when, sector_name, quality=86, pre=False):
    """Draw one finished image and return JPEG bytes. box = (lat0, lat1, lon0, lon1, res)."""
    lat0, lat1, lon0, lon1, res = box
    H, W = val.shape
    S = SATS[sat]; ch = S[BAND_OF[prod]][1]
    L, T, R, B = 50, 26, (78 if prod != "vis" else 14), 40
    img = Image.new("RGB", (L + W + R, T + H + B), "white")
    panel = Image.fromarray(colourise(prod, val, pre)); pd = ImageDraw.Draw(panel)
    fine = res <= 0.035
    px = lambda lo, la: ((lo - lon0) / res, (lat1 - la) / res)
    line_col = {"ir": (0, 0, 0), "wv": (0, 0, 0), "dvorak": (255, 220, 0), "vis": (255, 220, 0)}[prod]
    for kind in ("coast", "borders"):
        for l in _lines(kind, lat0, lat1, lon0, lon1, fine):
            pd.line([px(a, b) for a, b in l], fill=line_col if kind == "coast" else tuple(int(c * .85) for c in line_col), width=1)
    img.paste(panel, (L, T))
    d = ImageDraw.Draw(img)
    # graticule
    span = max(lon1 - lon0, lat1 - lat0)
    step = 2 if span <= 20 else (5 if span <= 45 else (10 if span <= 100 else 30))
    f = _font(11)
    for lo in np.arange(math.ceil(lon0 / step) * step, lon1 + 1e-6, step):
        x = L + (lo - lon0) / res
        for y in range(T, T + H, 6): d.line([(x, y), (x, min(y + 2, T + H))], fill=(190, 190, 190))
        tw = d.textlength(_fmt_lon(lo), font=f); d.text((min(x - tw / 2, L + W - tw), T + H + 4), _fmt_lon(lo), fill="black", font=f)
    for la in np.arange(math.ceil(lat0 / step) * step, lat1 + 1e-6, step):
        y = T + (lat1 - la) / res
        for x in range(L, L + W, 6): d.line([(x, y), (min(x + 2, L + W), y)], fill=(190, 190, 190))
        tw = d.textlength(_fmt_lat(la), font=f); d.text((L - 5 - tw, y - 7), _fmt_lat(la), fill="black", font=f)
    d.rectangle([L - 1, T - 1, L + W, T + H], outline="black")
    # title
    kind = {"ir": "Brightness Temperature (°C)", "dvorak": "BD Enhancement (°C)",
            "wv": "Brightness Temperature (°C)", "vis": "Reflectance"}[prod]
    title = f"{S['name']} {ch} {kind} · {when:%H:%MZ %d %b %Y}"
    tw = d.textlength(SITE.upper(), font=_font(11, True))
    fs = 12
    while fs > 9 and d.textlength(title, font=_font(fs, True)) > W - tw - 14: fs -= 1
    d.text((L, 6 + (12 - fs)), title, fill="black", font=_font(fs, True))
    d.text((L + W - tw, 7), SITE.upper(), fill=(11, 92, 173), font=_font(11, True))
    # credit
    cred = S["credit"].format(y=when.year) + " · " + sector_name + " · International boundaries as per India's official map"
    d.text((L, T + H + 22), cred, fill=(90, 90, 90), font=_font(10))
    # colour bar
    if prod != "vis":
        lo, hi, stp = CBAR[prod]; x0 = L + W + 12; bw = 16
        lut = LUTS[prod]
        for y in range(H):
            t = hi - (hi - lo) * y / (H - 1)
            d.line([(x0, T + y), (x0 + bw, T + y)], fill=tuple(int(c) for c in lut[int(round((t - T_MIN) / T_STEP))]))
        d.rectangle([x0, T, x0 + bw, T + H - 1], outline="black")
        for t in range(lo, hi + 1, stp):
            y = T + (hi - t) / (hi - lo) * (H - 1)
            d.line([(x0 + bw, y), (x0 + bw + 4, y)], fill="black"); d.text((x0 + bw + 6, y - 7), f"{t}", fill="black", font=f)
    bio = __import__("io").BytesIO()
    img.save(bio, "JPEG", quality=quality, optimize=True, progressive=True)
    return bio.getvalue()


def make(prod, band, box, sat, sector_name):
    """Sample + draw. Returns (jpeg bytes, info) or (None, reason)."""
    lat0, lat1, lon0, lon1, res = box
    lon, lat = grid(lat0, lat1, lon0, lon1, res)
    out_km = res * 111.0 * max(0.35, math.cos(math.radians((lat0 + lat1) / 2)))
    val = sample(band, lon, lat, out_km)
    if np.isfinite(val).mean() < 0.3: return None, "outside the satellite's view"
    if prod == "vis":
        mu = sun_cos(lon, lat, band.time)
        if (mu > 0.1).mean() < 0.15: return None, "night"
        pre = getattr(band, "pre", False)
        if not pre: val = np.where(mu > 0, val / np.maximum(mu, 0.3), 0.0)
        return render(prod, val, box, sat, band.time, sector_name, pre=pre), "ok"
    return render(prod, val, box, sat, band.time, sector_name), "ok"
