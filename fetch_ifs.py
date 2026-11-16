"""Download one ECMWF IFS open-data run (0.25°) for the chart steps.
Usage: python fetch.py [YYYYMMDDHH]   (default: latest available run)
Writes data/ifs/<run>/sfc.grib2 and data/ifs/<run>/const.grib2"""
import os, sys, datetime as dt
from ecmwf.opendata import Client

SFC = ["msl", "tp", "10u", "10v", "mucape", "tcwv"]
PL = {"gh": [850, 500, 200], "t": [850], "u": [850, 500, 200], "v": [850, 500, 200], "vo": [500]}


def chart_steps(hour):
    """6-hourly to 240 h for 00/12 UTC runs, to 90 h for 06/18 UTC runs (what ECMWF publishes)."""
    return list(range(0, 241, 6)) if hour in (0, 12) else list(range(0, 91, 6))


def fetch(run=None, root="data/ifs", source="ecmwf"):
    c = Client(source=source, model="ifs", resol="0p25")
    if run is None:
        run = c.latest(type="fc", step=90, param="msl")          # newest run whose files are complete
    else:
        run = dt.datetime.strptime(run, "%Y%m%d%H")
    out = os.path.join(root, run.strftime("%Y%m%d%H"))
    os.makedirs(out, exist_ok=True)
    steps = chart_steps(run.hour)
    sfc = os.path.join(out, "sfc.grib2")
    if not os.path.exists(sfc):
        c.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", step=steps, param=SFC, target=sfc + ".part")
        os.replace(sfc + ".part", sfc)
    pl = os.path.join(out, "pl.grib2")
    if not os.path.exists(pl):
        levels = sorted({l for v in PL.values() for l in v})
        c.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", step=steps, levtype="pl",
                   levelist=levels, param=list(PL), target=pl + ".part")
        os.replace(pl + ".part", pl)
    const = os.path.join(out, "const.grib2")
    if not os.path.exists(const):
        c.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", step=0, param=["z", "lsm"], levtype="sfc", target=const)
    return out, run, steps


if __name__ == "__main__":
    print(*fetch(sys.argv[1] if len(sys.argv) > 1 else None))
