# Global Weather Charts — chart maker

Draws weather model charts for globalweathercharts.com.

- **Models:** ECMWF IFS and AIFS 0.25° (ECMWF open data, CC BY 4.0); NOAA GFS and AIGFS 0.25° (public domain)
- **Charts:** MSLP & 6-h precipitation, total precipitation, 500 hPa height & vorticity, 850 hPa temperature, 200 hPa jet, CAPE, precipitable water
- **Regions:** Global, India, North Indian Ocean, Mediterranean to India, East Asia, West Pacific, Australia, Europe, Africa & Middle East, North America, Tropical Atlantic, South America

Every hour GitHub Actions runs `publish.py` for each model. When a new run is available it is downloaded, every chart is drawn, and the charts are stored in Cloudflare R2. Only the newest two runs of each model are kept.

Files: `fetch_ifs.py`, `fetch_gfs.py` (download), `charts.py` (drawing), `publish.py` (the automatic job), `.github/workflows/charts.yml` (the schedule).

## Global T-phi grams (globalweathercharts.com/tphi)

Every hour (`.github/workflows/tphi.yml`) `tphi_publish.py` reads the radiosonde ascents reported worldwide on the WMO GTS, as relayed in real time by the US National Weather Service (tgftp.nws.noaa.gov, public domain): high-resolution BUFR where a country sends it, otherwise the TEMP message (TTAA/TTBB). For each new ascent it draws a T-phi gram with 39 thermodynamic indices (`tphi_core.py`) and stores it in R2 under `tphi/`. Station names come from NOAA's IGRA station list (`tphi_stations.json`). The page globalweathercharts.com/tphi, and its link in the top menu, come from the Worker `gwcharts-tphi`.

Files: `tphi_feed.py` (download + decoding), `tphi_core.py` (chart), `tphi_publish.py` (the automatic job), `tphi_stations.json`.
