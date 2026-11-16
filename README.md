# Global Weather Charts — chart maker

Draws weather model charts for globalweathercharts.com.

- **Models:** ECMWF IFS and AIFS 0.25° (ECMWF open data, CC BY 4.0); NOAA GFS and AIGFS 0.25° (public domain)
- **Charts:** MSLP & 6-h precipitation, total precipitation, 500 hPa height & vorticity, 850 hPa temperature, 200 hPa jet, CAPE, precipitable water
- **Regions:** Global, India, North Indian Ocean, Mediterranean to India, East Asia, West Pacific, Australia, Europe, Africa & Middle East, North America, Tropical Atlantic, South America

Every hour GitHub Actions runs `publish.py` for each model. When a new run is available it is downloaded, every chart is drawn, and the charts are stored in Cloudflare R2. Only the newest two runs of each model are kept.

Files: `fetch_ifs.py`, `fetch_gfs.py` (download), `charts.py` (drawing), `publish.py` (the automatic job), `.github/workflows/charts.yml` (the schedule).
