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
