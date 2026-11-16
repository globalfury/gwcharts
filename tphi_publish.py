"""Automatic job: the charts for the T-phi grams page (globalweathercharts.com/tphi).

Every run: read the radiosonde ascents of the last HOURS hours from the WMO GTS feed (tphi_feed.py),
draw a T-phi gram + 39 indices (tphi_core.py) for every ascent that is new or has more data than before,
upload the charts to Cloudflare R2 and rewrite the list the web page reads.

Layout in the bucket:
  tphi/<YYYYMMDD_HH>/<wmo>.jpg     the charts
  tphi/index.json                  written last: stations + every ascent kept, with a few indices

Usage: python tphi_publish.py [--hours 30] [--only 42182,43279] [--dry-run]
Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (same GitHub secrets as the model charts).
"""
import argparse, datetime as dt, glob, io, json, os, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

KEEP_DAYS = int(os.environ.get("TPHI_KEEP_DAYS", "3"))      # matches the R2 lifecycle rule (objects deleted after 3 days)
SITE = os.environ.get("SITE_NAME", "globalweathercharts.com")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {"bufr-high": "BUFR, high resolution", "bufr-low": "BUFR", "temp": "TEMP code"}
STN = json.load(open(os.path.join(HERE, "tphi_stations.json"), encoding="utf-8"))
BLOCK_COUNTRY = {"42": "India", "43": "India"}


def station(wmo, a=None):
    if wmo in STN: return STN[wmo]
    lat = round(float(a.lat), 3) if a is not None and a.lat is not None else None
    lon = round(float(a.lon), 3) if a is not None and a.lon is not None else None
    return [f"Station {wmo}", BLOCK_COUNTRY.get(wmo[:2], ""), lat, lon]


def draw(job):
    """(wmo, slot, ascent) -> (wmo, slot, jpeg bytes, row) or (wmo, slot, None, error). Runs in a worker process."""
    import tphi_feed as F, tphi_core as C
    wmo, slot, a = job
    try:
        R, notes = F.to_sounding(a)
        R = C.quality_control(R)
        S = dict(p=R[:, 0], t=R[:, 1], rh=R[:, 2], td=R[:, 3], z=R[:, 4], d=R[:, 5], s=R[:, 6], tdok=R[:, 7] > 0.5)
        d = dt.datetime.strptime(slot, "%Y%m%d_%H")
        when = f"{d:%H} UTC, {d.day} {d:%B %Y}"
        nm, cty = station(wmo, a)[:2]
        label = f"{nm}, {cty}" if cty else nm
        src = (f"Source: computed from the WMO GTS radiosonde report ({SRC[a.kind]}, launched {a.launch:%H:%M} UTC), "
               f"received via the US National Weather Service")
        buf = io.BytesIO()
        r = C.make_chart(S, label, wmo, when, buf, source=src, site=SITE, extra_issues=notes)
        from PIL import Image
        buf.seek(0); im = Image.open(buf).convert("RGB"); w = 1000
        im = im.resize((w, int(im.size[1] * w / im.size[0])), Image.LANCZOS)
        out = io.BytesIO(); im.save(out, "JPEG", quality=80, optimize=True)
        f = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 1)
        row = {"k": a.kind, "n": int(len(R)), "w": 1 if r.get("suspect") else 0,
               "cape": f(r["SBCAPE"]), "mu": f(r["MUCAPE"]), "ki": f(r["KI"]), "pw": f(r["PW"]), "li": f(r["LI"])}
        return wmo, slot, out.getvalue(), row
    except Exception as e:
        return wmo, slot, None, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=30); ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--out", default="tphi_out")
    ap.add_argument("--cache", default="feed_cache"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    import tphi_feed as F
    from publish import r2, get_json, put, BUCKET

    s3 = None if a.dry_run else r2()
    index = get_json(s3, "tphi/index.json", {}) if s3 else (json.load(open(os.path.join(a.out, "index.json")))
                                                          if os.path.exists(os.path.join(a.out, "index.json")) else {})
    asc = index.get("ascents", {})

    best = F.collect(a.hours, a.cache)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); cutoff = (now - dt.timedelta(days=KEEP_DAYS)).strftime("%Y%m%d_%H")
    jobs = []
    for (wmo, slot), asc_ in sorted(best.items(), key=lambda kv: kv[0][1], reverse=True):
        if slot < cutoff or slot > (now + dt.timedelta(hours=3)).strftime("%Y%m%d_%H"): continue
        if only and wmo not in only: continue
        old = asc.get(slot, {}).get(wmo)
        nlev = len(asc_.lev)
        if old and old.get("k") == asc_.kind and old.get("src_n", 0) >= nlev: continue     # nothing new for this ascent
        jobs.append((wmo, slot, asc_))
    if a.limit: jobs = jobs[:a.limit]
    print(f"{len(best)} ascents in the feed, {len(jobs)} to draw", flush=True)

    done = fail = 0; uploads = []
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(os.cpu_count() or 2) as ex:
        for (wmo, slot, img, row), job in zip(ex.map(draw, jobs, chunksize=4), jobs):
            if img is None:
                fail += 1; print(f"  x {wmo} {slot}: {row}"); continue
            row["src_n"] = len(job[2].lev)
            old = asc.get(slot, {}).get(wmo)
            row["v"] = (old.get("v", 0) + 1) if old else 1          # version number: browsers fetch the new image
            asc.setdefault(slot, {})[wmo] = row
            key = f"tphi/{slot}/{wmo}.jpg"; uploads.append((key, img)); done += 1
            if a.dry_run:
                os.makedirs(os.path.join(a.out, slot), exist_ok=True)
                open(os.path.join(a.out, slot, f"{wmo}.jpg"), "wb").write(img)
    print(f"drawn {done}, failed {fail}, {time.time() - t0:.0f}s", flush=True)

    if s3 and uploads:
        def up(kv): put(s3, kv[0], kv[1], "image/jpeg", "public, max-age=604800")
        with ThreadPoolExecutor(16) as ex: list(ex.map(up, uploads))

    old_slots = [s for s in asc if s < cutoff]
    for s in old_slots: asc.pop(s)
    used = {w for s in asc.values() for w in s}
    index = {"updated": now.strftime("%Y-%m-%dT%H:%MZ"), "keep_days": KEEP_DAYS,
             "source": "WMO GTS radiosonde reports via the US National Weather Service (tgftp.nws.noaa.gov)",
             "stations": {w: station(w, best.get(next(((w, s) for s in asc if (w, s) in best), None))) for w in sorted(used)},
             "ascents": dict(sorted(asc.items(), reverse=True))}
    body = json.dumps(index, separators=(",", ":"), ensure_ascii=False)
    if s3:
        put(s3, "tphi/index.json", body.encode("utf-8"), "application/json", "public, max-age=120")
        for s in old_slots:                                     # delete charts of ascents no longer listed
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"tphi/{s}/"):
                keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                if keys: s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
    else:
        open(os.path.join(a.out, "index.json"), "w", encoding="utf-8").write(body)
    # keep the feed cache small: drop files that have left the time window
    lim = time.time() - (a.hours + 6) * 3600
    for p in glob.glob(os.path.join(a.cache, "*", "*")):
        if os.path.getmtime(p) < lim: os.remove(p)
    print(f"published {sum(len(v) for v in asc.values())} ascents in {len(asc)} times, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
