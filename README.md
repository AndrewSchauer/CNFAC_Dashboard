# CNFAC Avalanche Forecast Dashboard

An interactive avalanche risk assessment dashboard built with Python and Plotly Dash, developed for the Chugach National Forest Avalanche Center (CNFAC).

## Features

- **Likelihood Matrix** — plots a point on a 3×4 Sensitivity × Distribution matrix based on slider input. Supports half-step positions between named categories. Click and drag directly on the matrix to reposition the point and automatically update the sliders.
- **Danger Rating Matrix** — a 9×9 Likelihood × Size grid where each cell is colour-coded by avalanche danger level using official GNFAC/CAA colour standards. The highlighted box updates automatically based on slider and likelihood matrix inputs.
- **Configurable Danger Grid** — switch to the Settings tab to customise the danger level assigned to any cell. Click a cell to open a dropdown and select from No Rating, Low, Moderate, Considerable, High, or Extreme. Changes reflect immediately in the Forecast tab.
- **Forecast Summary** — live readout of selected sensitivity, distribution, likelihood range, size range, and the maximum danger level within the selected box.

## Running Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
python CMAH_dash.py
```

Open your browser and go to:
```
http://127.0.0.1:8050
```

## Deploying Online

### Render (free tier)
1. Push this repository to GitHub
2. Create an account at [render.com](https://render.com)
3. New → Web Service → connect your GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn CMAH_dash:server`

### Railway
1. Push to GitHub
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set start command: `gunicorn CMAH_dash:server`

### Heroku
Add a `Procfile` containing:
```
web: gunicorn CMAH_dash:server
```
Then deploy via the Heroku CLI.

## Adding Password Protection

Add the following to `CMAH_dash.py` before `app.layout`, then set environment variables `DASHBOARD_PASSWORD` and `SECRET_KEY` on your hosting platform:

```python
import os
from flask import request, redirect, session

app.server.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "yourpassword")

@app.server.before_request
def require_login():
    if request.path == "/login":
        return
    if not session.get("logged_in"):
        return redirect("/login")

@app.server.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        return '<p style="color:red">Wrong password</p>' + LOGIN_HTML
    return LOGIN_HTML

LOGIN_HTML = """
<html><body style="background:#080f1a;display:flex;justify-content:center;align-items:center;height:100vh">
<form method="POST" style="background:#0d1b2a;padding:40px;border:1px solid #1e3a4a">
  <h2 style="color:#00e5ff;font-family:sans-serif;letter-spacing:0.2em">AVALANCHE FORECAST</h2>
  <input name="password" type="password" placeholder="Enter password"
    style="display:block;width:100%;padding:10px;margin:16px 0;background:#111;color:#fff;border:1px solid #1e3a4a">
  <button type="submit"
    style="background:#00e5ff;color:#111;padding:10px 24px;border:none;cursor:pointer;font-weight:bold">
    LOGIN
  </button>
</form></body></html>
"""
```

## Danger Level Colours

| Level | Hex |
|---|---|
| Low | `#50B848` |
| Moderate | `#FFF200` |
| Considerable | `#F7941E` |
| High | `#ED1C24` |
| Extreme | `#231F20` |

## Dependencies

| Package | Purpose |
|---|---|
| `dash` | Web app framework |
| `dash-bootstrap-components` | UI layout and styling |
| `plotly` | Interactive charts and matrices |
| `numpy` | Matrix data handling |
| `gunicorn` | Production WSGI server |
