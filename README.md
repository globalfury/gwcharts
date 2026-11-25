# Global Weather Charts — chart maker

Draws weather model charts for globalweathercharts.com.

- **Models:** ECMWF IFS and AIFS 0.25° (ECMWF open data, CC BY 4.0); NOAA GFS and AIGFS 0.25° (public domain); CMC GDPS 15 km (ECCC, Open Government Licence – Canada) and DWD ICON 13 km (Deutscher Wetterdienst), both regridded to 0.25°
- **Ensembles:** ECMWF EPS (50 members, 0.25°, ECMWF open data, CC BY 4.0) and NOAA GEFS (31 members, 0.25°, public domain), 00Z and 12Z runs: ensemble-mean MSLP with mean 24-h precipitation, and the probability of more than 10, 25 and 50 mm in 24 h, +24 h to +240 h every 6 h
- **Charts:** MSLP & 6-h precipitation, total precipitation, 500 hPa height & vorticity, 850 hPa temperature, 200 hPa jet, CAPE, precipitable water
- **Cyclones:** ECMWF 51-member ensemble cyclone tracks for every basin (Bay of Bengal, Arabian Sea, West/East/Central Pacific, North Atlantic, South-West Indian Ocean, Australian region, South Pacific): spaghetti, strike probability, wind and pressure plumes, landfall summary, run comparison. Kept 15 days. Charts are kept 7 days.
- **Regions:** Global, India, North Indian Ocean, Mediterranean to India, East Asia, West Pacific, Australia, Europe, Africa & Middle East, North America, Tropical Atlantic, South America

Every hour GitHub Actions runs `publish.py` for each model. When a new run is available it is downloaded, every chart is drawn, and the charts are stored in Cloudflare R2. Runs are kept for 7 days.

Files: `fetch_ifs.py`, `fetch_gfs.py`, `fetch_more.py` (download), `charts.py` (drawing), `publish.py` (the automatic job), `.github/workflows/charts.yml` (the schedule); `cyclone_ens.py` + `places.json` with `.github/workflows/cyclone.yml` (cyclone tab). Ensembles: `fetch_ens.py` (download + ensemble mean and probabilities), `ens_charts.py` (drawing, reuses `charts.py`), `ens_publish.py` with `.github/workflows/ens.yml` (hourly at :52).

## Global T-phi grams (globalweathercharts.com/tphi)

Every hour (`.github/workflows/tphi.yml`) `tphi_publish.py` reads the radiosonde ascents reported worldwide on the WMO GTS, as relayed in real time by the US National Weather Service (tgftp.nws.noaa.gov, public domain): high-resolution BUFR where a country sends it, otherwise the TEMP message (TTAA/TTBB). For each new ascent it draws a T-phi gram with 39 thermodynamic indices (`tphi_core.py`) and stores it in R2 under `tphi/`. Station names come from NOAA's IGRA station list (`tphi_stations.json`). The page globalweathercharts.com/tphi, and its link in the top menu, come from the Worker `gwcharts-tphi`.

Files: `tphi_feed.py` (download + decoding), `tphi_core.py` (chart), `tphi_publish.py` (the automatic job), `tphi_stations.json`.

## Cyclones — multi-model ensemble of open models (globalweathercharts.com/cyclone#mme)

Twice a day (`.github/workflows/mme.yml`, hourly 03:50–11:50 and 15:50–23:50 UTC) `cyclone_mme.py` takes mean-sea-level pressure and 10 m wind from seven open global models as soon as each model's 00/12 UTC run is complete (`mme_fields.py`): ECMWF HRES and AIFS (ECMWF open data, CC BY 4.0), NCEP GFS and NOAA AIGFS (public domain, AWS Open Data), CMC GDPS (ECCC, OGL-Canada), DWD ICON (DWD Open Data) and the Met Office global model (CC BY-SA 4.0, AWS Open Data). All are put on one global 0.25° grid (50°S–50°N) and tracked with the same tracker (`mme_track.py`). Tracks of the same storm from different models are grouped and sorted into the basins of the ECMWF ENS tab (Bay of Bengal, Arabian Sea, West/East/Central Pacific, North Atlantic, South-West Indian Ocean, Australian region, South Pacific). Shown: in the Bay of Bengal and Arabian Sea every depression tracked by at least 2 models; elsewhere storms reaching 34 kt over the tropical sea in at least 2 models, or numbered by JTWC. For every system it draws: landfall summary (consensus and every model), tracks, wind and pressure plumes, track spread, run comparison (last 6 runs) and a zip with the consensus track and all model tracks. The consensus is the equal-weight mean of the models tracking the system (at least 3). Storm numbers and observed tracks come from the JTWC best track (via NCAR/RAL TCGP). Each model's tracks are stored as soon as they are made, so later calls only fetch the models that were not ready yet. Stored in R2 under `cyclone/mme/`, kept 15 days; the page tab comes from the Worker `gwcharts-mme`.

Files: `mme_fields.py` (download), `mme_track.py` (tracker), `cyclone_mme.py` (systems, consensus, figures, upload).

## Live map — animated wind (globalweathercharts.com/map)

A Windy-style map of the whole world: coloured wind speed with moving particles, ECMWF IFS or GFS,
every 6 hours for 10 days, city temperatures (2 m) and the values at any tapped point. India is the default view.
Levels: GFS 10 m, 925, 850, 700, 500, 300, 250, 200, 150, 100, 70, 50, 30, 10 hPa;
ECMWF the same except 70 and 30 hPa (not in ECMWF open data).

| Piece | What it does |
|---|---|
| `map_publish.py ifs` / `map_publish.py gfs` | Takes the newest complete 00/12 UTC run (ECMWF open data, CC BY 4.0, via `ecmwf-opendata`; GFS from NOAA on AWS Open Data via `fetch_gfs.py`), u/v wind and temperature for 0–240 h every 6 h, smooths them to a 0.5° global grid and stores each level and hour as one PNG (red = u, green = v, blue = temperature) in R2 under `map/<model>/<run>/<level>/<hhh>.png`, plus `map/<model>/<run>/meta.json`, `map/places.json` and finally `map/<model>/latest.json`. Keeps the newest 2 runs per model. About 100 MB and 3–6 minutes per run. |
| `.github/workflows/map.yml` | "Live map (wind)": both models at :25 past 04–10 and 16–22 UTC (a run is published once, later tries skip it). **Run workflow** for a manual run. |
| Worker `gwcharts-map` | Route `globalweathercharts.com/map*`, binding `CHARTS` → R2 `gwcharts`. Serves the page and `/map/data/…`. The browser draws the wind field and particles; boundaries are the site's India-view borders. |
