"""Automatic job: images for the Satellite page (globalweathercharts.com/sat).

Every run: for each of the last HOURS clock hours, take the full-disk scan that starts on the hour from each
satellite (Meteosat-9, Himawari-9, GOES-19, GOES-18), draw every sector and storm floater that does not have
that hour yet (Longwave IR, IR Dvorak BD, Visible in daylight, Water Vapour), upload to Cloudflare R2 and
rewrite the list the page reads.

Layout in the bucket:
  sat/<target>/<product>/<YYYYMMDD_HHMM>.jpg    target = sector id (india, bob, wpac, ...) or storm id (io052026)
  sat/index.json                                written last

Storm floaters: active storms and invests from the NCAR/RAL Tropical Cyclone Guidance Project list of
best-track (b-deck) files; the box follows the storm, interpolated between fixes.

Usage: python sat_publish.py [--hours 3] [--only india,bob] [--dry-run] [--out sat_out]
Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY; for Meteosat-9 also EUMETSAT_KEY, EUMETSAT_SECRET.
"""
import argparse, datetime as dt, json, os, re, shutil, time
from concurrent.futures import ThreadPoolExecutor
import sat_core as C
import sat_feed as F

KEEP_HOURS = int(os.environ.get("SAT_KEEP_HOURS", "24"))
TCGP = "https://hurricanes.ral.ucar.edu/realtime/"
BASIN_NAME = {"AL": "L", "EP": "E", "CP": "C", "WP": "W"}
NUMWORDS = {"ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN",
            "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY", "", "NONAME", "UNNAMED"}


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# ------------------------------------------------------------------------------------------ storms
def storms():
    """{id: {label, basin, track: [(time, lat, lon, vmax)]}} for systems active in the last 24 h."""
    out = {}
    try:
        page = F.S.get(TCGP + "current/", timeout=60).text
    except Exception as e:
        print("  storm list unavailable:", e); return None
    for path, sid in sorted(set(re.findall(r'href="\.\./(plots/[a-z]+/\d{4}/([a-z]{2}\d{6}))/"', page))):
        try:
            txt = F.S.get(f"{TCGP}{path}/b{sid}.dat", timeout=60).text
        except Exception:
            continue
        tr, name = {}, ""
        for ln in txt.splitlines():
            c = [x.strip() for x in ln.split(",")]
            if len(c) < 9 or not c[2].isdigit(): continue
            try:
                t = dt.datetime.strptime(c[2], "%Y%m%d%H")
                la = int(c[6][:-1]) / 10 * (1 if c[6][-1] == "N" else -1)
                lo = int(c[7][:-1]) / 10 * (1 if c[7][-1] == "E" else -1)
                v = int(c[8] or 0)
            except Exception:
                continue
            tr[t] = (t, la, lo, max(v, tr.get(t, (0, 0, 0, 0))[3]))
            if len(c) > 27 and c[27]: name = c[27].upper()
        if not tr: continue
        track = sorted(tr.values())
        if (now_utc() - track[-1][0]).total_seconds() > 24 * 3600: continue
        basin, num = sid[:2].upper(), sid[2:4]
        lo = track[-1][2]
        suf = ("A" if lo < 78 else "B") if basin == "IO" else (("S" if 0 <= lo < 135 else "P") if basin == "SH" else BASIN_NAME.get(basin, ""))
        v = track[-1][3]
        if name.startswith("INVEST") or int(num) >= 90:
            label = f"Invest {num}{suf}"
        elif name in NUMWORDS:
            label = f"Tropical Depression {num}{suf}"
        elif basin in ("AL", "EP", "CP"):
            label = f"{'Hurricane' if v >= 64 else 'Tropical Storm' if v >= 34 else 'Tropical Depression'} {name}"
        elif basin == "WP":
            label = f"{'Typhoon' if v >= 64 else 'Tropical Storm' if v >= 34 else 'Tropical Depression'} {name}"
        else:
            label = f"Tropical Cyclone {name}"
        out[sid] = dict(label=label, basin=basin, short=f"{num}{suf}", track=track)
    return out


def storm_at(track, when):
    """Position at `when`: interpolated between fixes, carried forward on the last motion for up to 12 h."""
    ts = [p[0] for p in track]
    if when <= ts[0]: return track[0][1], track[0][2]
    for a, b in zip(track, track[1:]):
        if a[0] <= when <= b[0]:
            f = (when - a[0]).total_seconds() / max(1, (b[0] - a[0]).total_seconds())
            dlo = ((b[2] - a[2] + 180) % 360) - 180
            return a[1] + f * (b[1] - a[1]), a[2] + f * dlo
    last = track[-1]; h = min(12, (when - last[0]).total_seconds() / 3600)
    if len(track) >= 2:
        prev = track[-2]; span = max(1, (last[0] - prev[0]).total_seconds() / 3600)
        dlo = ((last[2] - prev[2] + 180) % 360) - 180
        return last[1] + (last[1] - prev[1]) / span * h, last[2] + dlo / span * h
    return last[1], last[2]


def floater_box(lat, lon):
    lon = ((lon + 180) % 360) - 180
    if abs(lon) > 170: lon %= 360                       # near the date line: use 0..360 so the box is continuous
    w, h, r = C.FLOATER["w"], C.FLOATER["h"], C.FLOATER["res"]
    lat = max(-60 + h / 2, min(60 - h / 2, lat))
    lat, lon = round(lat, 1), round(lon, 1)
    return (lat - h / 2, lat + h / 2, lon - w / 2, lon + w / 2, r)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=3); ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--out", default="sat_out")
    ap.add_argument("--cache", default="sat_cache"); ap.add_argument("--sats", default="met9,him9,g19,g18,gmgsi")
    a = ap.parse_args(); t0 = time.time()
    from publish import r2, get_json, put, BUCKET
    s3 = None if a.dry_run else r2()
    ipath = os.path.join(a.out, "index.json")
    index = get_json(s3, "sat/index.json", {}) if s3 else (json.load(open(ipath)) if os.path.exists(ipath) else {})
    T = index.get("targets", {})
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    avail = [s for s in a.sats.split(",") if F.available(s)]
    print("satellites:", ", ".join(avail), flush=True)

    now = now_utc()
    hours = [(now - dt.timedelta(hours=i)).replace(minute=0, second=0, microsecond=0) for i in range(a.hours)][::-1]
    oldest = (now - dt.timedelta(hours=KEEP_HOURS + 1)).strftime("%Y%m%d_%H%M")

    # ---- what to draw: (sat, hour) -> [(target id, box, products)]
    st = storms()
    if st is not None:
        for sid, s in st.items():
            T.setdefault(sid, {}).update(group="floater", name=s["label"], short=s["short"], basin=s["basin"],
                                         lat=s["track"][-1][1], lon=s["track"][-1][2], vmax=s["track"][-1][3],
                                         fix=s["track"][-1][0].strftime("%Y-%m-%dT%H:%MZ"))
    jobs = {}
    for h in hours:
        fr = h.strftime("%Y%m%d_%H%M")
        todo = []
        for sid, (name, grp, prefs, la0, la1, lo0, lo1, res) in C.SECTORS.items():
            sat = next((p for p in prefs if p in avail and C.sat_view_ok(p, la0, la1, lo0, lo1)), None)
            if sat: todo.append((sid, sat, (la0, la1, lo0, lo1, res), name))
        for sid, s in (st or {}).items():
            if h < s["track"][0][0] - dt.timedelta(hours=6): continue
            la, lo = storm_at(s["track"], h)
            sat = C.best_sat(la, lo, avail)
            if sat: todo.append((sid, sat, floater_box(la, lo), s["label"]))
        for sid, sat, box, name in todo:
            if only and sid not in only: continue
            if fr in T.get(sid, {}).get("done", []): continue
            prods = ["ir", "dvorak", "wv"] + (["vis"] if C.daylight_fraction(*box[:4], h) >= 0.15 else [])
            jobs.setdefault((sat, h), []).append((sid, box, name, prods))
    print(f"{sum(len(v) for v in jobs.values())} images sets to draw in {len(jobs)} satellite scans", flush=True)

    uploads, drawn = [], 0
    os.makedirs(a.out, exist_ok=True)
    for (sat, h), lst in sorted(jobs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        fr = h.strftime("%Y%m%d_%H%M"); t1 = time.time()
        need = sorted({C.BAND_OF[p] for _, _, _, ps in lst for p in ps})
        vis_boxes = [b[:4] for _, b, _, ps in lst if "vis" in ps]
        cache = os.path.join(a.cache, f"{sat}_{fr}")
        try:
            bands = F.load(sat, h, need, cache, vis_boxes=vis_boxes)
        except F.FeedMissing as e:
            print(f"  {sat} {fr}: waiting ({e})", flush=True); shutil.rmtree(cache, ignore_errors=True); continue
        except Exception as e:
            print(f"  {sat} {fr}: FAILED {type(e).__name__}: {e}", flush=True); shutil.rmtree(cache, ignore_errors=True); continue

        def one(job):
            sid, box, name, prods = job; res = []
            for p in prods:
                try:
                    img, why = C.make(p, bands[C.BAND_OF[p]], box, sat, name)
                except Exception as e:
                    img, why = None, f"{type(e).__name__}: {e}"
                res.append((p, img, why))
            return job, res
        with ThreadPoolExecutor(4) as ex:
            for (sid, box, name, prods), res in ex.map(one, lst):
                t = T.setdefault(sid, {})
                if sid in C.SECTORS:
                    nm, grp, _, la0, la1, lo0, lo1, r = C.SECTORS[sid]
                    t.update(group=grp, name=nm, box=[la0, la1, lo0, lo1])
                t["sat"] = sat
                fx = t.setdefault("frames", {})
                for p, img, why in res:
                    if img is None:
                        if why not in ("night", "outside the satellite's view"): print(f"  x {sid} {p} {fr}: {why}")
                        continue
                    key = f"sat/{sid}/{p}/{fr}.jpg"; uploads.append((key, img)); drawn += 1
                    fx.setdefault(p, []); fx[p] = sorted(set(fx[p]) | {fr})
                    if a.dry_run:
                        os.makedirs(os.path.join(a.out, sid, p), exist_ok=True)
                        open(os.path.join(a.out, sid, p, fr + ".jpg"), "wb").write(img)
                if sid not in C.SECTORS: t.setdefault("pos", {})[fr] = [round(box[0] + C.FLOATER["h"] / 2, 1), round(box[2] + C.FLOATER["w"] / 2, 1)]
                t["done"] = sorted(set(t.get("done", [])) | {fr})
        shutil.rmtree(cache, ignore_errors=True)
        print(f"  {C.SATS[sat]['name']} {fr}: {len(lst)} targets, {time.time() - t1:.0f}s", flush=True)
        if s3 and uploads:                                     # upload as we go so a long run shows progress
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda kv: put(s3, kv[0], kv[1], "image/jpeg", "public, max-age=604800"), uploads))
            uploads = []

    # ---- prune frames older than KEEP_HOURS, drop storms with nothing left
    dead = []
    for sid in list(T):
        t = T[sid]
        for p, fl in list(t.get("frames", {}).items()):
            dead += [f"sat/{sid}/{p}/{f}.jpg" for f in fl if f < oldest]
            t["frames"][p] = [f for f in fl if f >= oldest]
            if not t["frames"][p]: t["frames"].pop(p)
        t["done"] = [f for f in t.get("done", []) if f >= oldest]
        if "pos" in t: t["pos"] = {k: v for k, v in t["pos"].items() if k >= oldest}
        if not t.get("frames") and not t.get("done"): T.pop(sid)
    for sid, t in T.items():                                   # storms that left the active list stay until their frames expire
        if t.get("group") == "floater": t["active"] = bool(st and sid in st) if st is not None else t.get("active", True)

    index = {"updated": now.strftime("%Y-%m-%dT%H:%MZ"), "keep_hours": KEEP_HOURS,
             "groups": [["floater", "Floaters"]] + [list(g) for g in C.GROUPS],
             "products": [list(p) for p in C.PRODUCTS],
             "sats": {k: {"name": v["name"], "km": v["km"]} for k, v in C.SATS.items()},
             "sectors": {k: [v[0], v[1]] for k, v in C.SECTORS.items()},
             "order": list(C.SECTORS), "targets": T}
    body = json.dumps(index, separators=(",", ":"), ensure_ascii=False).encode()
    if s3:
        put(s3, "sat/index.json", body, "application/json", "public, max-age=60")
        for i in range(0, len(dead), 1000):
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in dead[i:i + 1000]], "Quiet": True})
    else:
        open(ipath, "wb").write(body)
        for k in dead:
            p = os.path.join(a.out, *k.split("/")[1:])
            if os.path.exists(p): os.remove(p)
    print(f"drew {drawn} images, removed {len(dead)} old ones, {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
