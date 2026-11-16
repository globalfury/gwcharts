"""Download one ECMWF open-data run (0.25°) for the chart steps: IFS (physics model) or AIFS (AI model).
Usage: python fetch_ifs.py [YYYYMMDDHH] [ifs|aifs]   (default: latest available IFS run)
Writes data/<model>/<run>/sfc.grib2, pl.grib2 (+ const.grib2 for IFS)"""
import os, sys, datetime as dt
from ecmwf.opendata import Client

SFC = ["msl", "tp", "10u", "10v", "mucape", "tcwv"]
PL = {"gh": [850, 500, 200], "t": [850], "u": [850, 500, 200], "v": [850, 500, 200], "vo": [500]}


def chart_steps(hour):
    """6-hourly to 240 h for 00/12 UTC runs, to 90 h for 06/18 UTC runs (what ECMWF publishes)."""
    return list(range(0, 241, 6)) if hour in (0, 12) else list(range(0, 91, 6))


# AIFS has no CAPE or vorticity output; vorticity is computed from the winds, tcw stands in for tcwv
AIFS_SFC = ["msl", "tp", "10u", "10v", "tcw"]
AIFS_PL = {"gh": [850, 500, 200], "t": [850], "u": [850, 500, 200], "v": [850, 500, 200]}


def fetch(run=None, root=None, source="ecmwf", model="ifs"):
    ai = model == "aifs"
    root = root or f"data/{model}"
    c = Client(source=source, model="aifs-single" if ai else "ifs", resol="0p25")
    if run is None:
        run = c.latest(type="fc", step=240 if ai else 90, param="msl")   # newest run whose files are complete
    else:
        run = dt.datetime.strptime(run, "%Y%m%d%H")
    out = os.path.join(root, run.strftime("%Y%m%d%H"))
    os.makedirs(out, exist_ok=True)
    steps = list(range(0, 241, 6)) if ai else chart_steps(run.hour)
    SFC_, PL_ = (AIFS_SFC, AIFS_PL) if ai else (SFC, PL)
    sfc = os.path.join(out, "sfc.grib2")
    if not os.path.exists(sfc):
        c.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", step=steps, param=SFC_, target=sfc + ".part")
        os.replace(sfc + ".part", sfc)
    pl = os.path.join(out, "pl.grib2")
    if not os.path.exists(pl):
        levels = sorted({l for v in PL_.values() for l in v})
        c.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", step=steps, levtype="pl",
                   levelist=levels, param=list(PL_), target=pl + ".part")
        os.replace(pl + ".part", pl)
    const = os.path.join(out, "const.grib2")
    if not ai and not os.path.exists(const):
        c.retrieve(date=run.strftime("%Y%m%d"), time=run.hour, type="fc", step=0, param=["z", "lsm"], levtype="sfc", target=const)
    return out, run, steps


if __name__ == "__main__":
    a = sys.argv[1:]
    print(*fetch(a[0] if a and a[0].isdigit() else None, model=a[-1] if a and not a[-1].isdigit() else "ifs"))
