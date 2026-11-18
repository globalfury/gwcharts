"""Automatic job for the ensemble charts (ECMWF EPS, NOAA GEFS): find the newest 00Z/12Z run; if it is not on
R2 yet, download it (fetch_ens.py), draw every chart (ens_charts.py), upload, update the run list and delete runs
older than KEEP_DAYS. Same bucket layout as publish.py, so the website reads them like any other model:

  <eps|gefs>/<run>/<field>/<region>/<step>.png
  <eps|gefs>/<run>/manifest.json        written last = run complete
  <eps|gefs>/runs.json                  newest first

Usage: python ens_publish.py eps|gefs [--run YYYYMMDDHH] [--dry-run]
Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (GitHub secrets).
"""
import os, sys, json, time, calendar, argparse, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor
from publish import r2, exists, get_json, put, BUCKET, KEEP_DAYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["eps", "gefs"]); ap.add_argument("--run"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    s3 = None if a.dry_run else r2()
    import fetch_ens
    run = a.run or fetch_ens.latest_run(a.model).strftime("%Y%m%d%H")
    print(f"{a.model}: newest run {run}", flush=True)
    if s3 and exists(s3, f"{a.model}/{run}/manifest.json"):
        print("already published - nothing to do"); return
    try:
        run_dir = fetch_ens.fetch(a.model, run)[0]
    except Exception as e:
        print(f"run {run} not complete yet ({e}); will retry next hour"); return
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)

    out = "out_ens"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, "ens_charts.py", run_dir, "--out", out, "--jobs", str(os.cpu_count() or 2)]
    if os.environ.get("CHART_REGIONS"): cmd += ["--regions", os.environ["CHART_REGIONS"]]
    subprocess.run(cmd, check=True)
    base = os.path.join(out, a.model, run)
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(base) for f in fs if f.endswith(".png")]
    print(f"{len(files)} charts drawn, {time.time() - t0:.0f}s so far", flush=True)
    if a.dry_run:
        print("dry run: not uploading"); return

    def up(path):
        key = os.path.relpath(path, out).replace(os.sep, "/")
        with open(path, "rb") as fh:
            put(s3, key, fh.read(), "image/png", "public, max-age=86400, immutable")
    with ThreadPoolExecutor(24) as ex:
        for n, _ in enumerate(ex.map(up, files), 1):
            if n % 500 == 0: print(f"  uploaded {n}/{len(files)}", flush=True)

    manifest = json.load(open(os.path.join(base, "manifest.json")))
    manifest["published"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    put(s3, f"{a.model}/{run}/manifest.json", json.dumps(manifest), "application/json", "public, max-age=300")

    runs = sorted(set(get_json(s3, f"{a.model}/runs.json", [])) | {run}, reverse=True)
    newest = time.strptime(runs[0], "%Y%m%d%H")
    cut = time.strftime("%Y%m%d%H", time.gmtime(calendar.timegm(newest) - KEEP_DAYS * 86400))
    keep = [r for r in runs if r >= cut]; drop = [r for r in runs if r < cut]
    put(s3, f"{a.model}/runs.json", json.dumps(keep), "application/json", "public, max-age=120")
    for old in drop:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{a.model}/{old}/"):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys: s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
        print(f"deleted old run {old}", flush=True)
    print(f"published {a.model} {run}: {len(files)} charts in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
