# CNFAC Avalanche Forecast Dashboard

An interactive avalanche risk assessment dashboard built with Python and Plotly Dash, developed for the Chugach National Forest Avalanche Center (CNFAC) using data from the National Avalanche Center Avalanche Forecasting Platform. The tool is designed to guide a CMAH workflow.

## Features

- **Likelihood Matrix** — plots a point on a 3×4 Sensitivity × Distribution matrix based on slider input. Supports half-step positions between named categories. Click and drag directly on the matrix to reposition the point and automatically update the sliders.
- **Danger Rating Matrix** — a 5×7 size/likelihood grid where each cell displays a histogram of danger ratings assigned to days with the specified size/likelihood combination based on the historical data. The background color of each cell is defined by the mode of the distribution of danger ratings for the specified size/likelihood. The highlighted box updates automatically based on slider and likelihood matrix inputs.
- **Configurable Danger Grid** — switch to the Settings tab to customise the danger level assigned to any cell. Click a cell to open a dropdown and select from No Rating, Low, Moderate, Considerable, High, or Extreme. Changes reflect immediately in the Forecast tab.
- **Summary** — live readout of selected sensitivity, distribution, likelihood range, size range, and the maximum danger level within the selected box.

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
