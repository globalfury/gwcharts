"""Automatic job: for one model, find the newest run; if it is not on R2 yet, download it,
draw every chart, upload to R2, update the run list and delete runs older than the newest KEEP.

Usage: python publish.py ifs|gfs|aifs|aigfs|cmc|icon [--run YYYYMMDDHH] [--dry-run]
Needs environment variables R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (GitHub secrets).

Layout in the bucket:
  <model>/<run>/<field>/<region>/<step>.png
  <model>/<run>/manifest.json        written last = run complete
  <model>/runs.json                  newest first, what the website reads
"""
import os, sys, json, time, argparse, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor

BUCKET = os.environ.get("R2_BUCKET", "gwcharts")
KEEP = 2


def env(name):
    return os.environ[name].strip()        # ignore stray spaces / line breaks pasted into the secrets


def r2():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=os.environ.get("R2_ENDPOINT") or f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"), aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        region_name="auto", config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, max_pool_connections=32))


def exists(s3, key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key); return True
    except Exception:
        return False


def get_json(s3, key, default):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:
        return default


def put(s3, key, body, ctype, cache):
    s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ctype, CacheControl=cache)


AI_HOURS = (0, 12)        # the AI models are charted for their 00Z and 12Z runs only (keeps R2 use in the free tier)


def newest_run(model):
    import datetime as dt
    if model in ("cmc", "icon"):
        from fetch_more import latest_run
        return latest_run(model).strftime("%Y%m%d%H")
    if model in ("gfs", "aigfs"):
        from fetch_gfs import latest_run
        return latest_run(model, AI_HOURS if model == "aigfs" else (0, 6, 12, 18)).strftime("%Y%m%d%H")
    from ecmwf.opendata import Client
    if model == "aifs":
        run = Client(source="ecmwf", model="aifs-single", resol="0p25").latest(type="fc", step=240, param="msl")
        while run.hour not in AI_HOURS: run -= dt.timedelta(hours=6)
    else:
        run = Client(source="ecmwf", model="ifs", resol="0p25").latest(type="fc", step=90, param="msl")
    return run.strftime("%Y%m%d%H")


def download(model, run):
    if model in ("cmc", "icon"):
        from fetch_more import fetch
        return fetch(model, run)[0]
    if model in ("gfs", "aigfs"):
        from fetch_gfs import fetch
        return fetch(run, model=model)[0]
    from fetch_ifs import fetch
    last = None
    for source in ("ecmwf", "aws"):                       # ECMWF portal first, AWS mirror as fallback
        try:
            return fetch(run, source=source, model=model)[0]
        except Exception as e:
            last = e; print(f"download from {source} failed: {e}", flush=True)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["ifs", "gfs", "aifs", "aigfs", "cmc", "icon"]); ap.add_argument("--run"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    s3 = None if a.dry_run else r2()

    run = a.run or newest_run(a.model)
    print(f"{a.model}: newest run {run}", flush=True)
    if s3 and exists(s3, f"{a.model}/{run}/manifest.json"):
        print("already published - nothing to do"); return

    try:
        run_dir = download(a.model, run)
    except Exception as e:
        print(f"run {run} not complete yet ({e}); will retry next hour"); return
    print(f"downloaded in {time.time() - t0:.0f}s", flush=True)

    out = "out"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, "charts.py", run_dir, "--out", out, "--jobs", str(os.cpu_count() or 2)]
    if os.environ.get("CHART_REGIONS"): cmd += ["--regions", os.environ["CHART_REGIONS"]]   # for quick tests
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
            if n % 1000 == 0: print(f"  uploaded {n}/{len(files)}", flush=True)

    manifest = json.load(open(os.path.join(base, "manifest.json")))
    manifest["published"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    put(s3, f"{a.model}/{run}/manifest.json", json.dumps(manifest), "application/json", "public, max-age=300")

    runs = get_json(s3, f"{a.model}/runs.json", [])
    runs = sorted(set(runs) | {run}, reverse=True)
    keep, drop = runs[:KEEP], runs[KEEP:]
    put(s3, f"{a.model}/runs.json", json.dumps(keep), "application/json", "public, max-age=120")

    for old in drop:                                       # remove older runs (deletes are free on R2)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{a.model}/{old}/"):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys: s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
        print(f"deleted old run {old}", flush=True)
    print(f"published {a.model} {run}: {len(files)} charts in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
