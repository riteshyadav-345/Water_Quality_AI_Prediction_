# AquaCheck — Water Quality and Sanitation Toolkit

A Streamlit educational app for exploring drinking-water measurements, a screening model, household water use, rainwater harvesting, purifier options, and sanitation practices.

## Run locally

Use Python 3.10 or newer. From this folder, create and activate a virtual environment, then install dependencies and start Streamlit:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local URL (usually <http://localhost:8501>).
Use the **Appearance** control at the top of the sidebar to switch between Light and Dark mode.

## Optional real dataset

In the sidebar, choose **Upload Kaggle CSV** and upload the Kaggle Water Potability dataset (`water_potability.csv`). The expected fields are `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Turbidity`, and `Potability` (labels `0` or `1`). Missing measurement values are filled with the column median. Invalid rows and unsupported label values are skipped; upload errors are shown in the app.

## How the estimates work

- The default Logistic Regression model is trained on generated examples whose labels are based on the app's configured parameter ranges. This demonstrates a workflow; it is not a validated drinking-water classifier.
- The uploaded-data option trains a Random Forest and reports a holdout accuracy. Dataset quality, class balance, and sampling affect that score.
- The Water Quality Index is a simple weighted deviation score defined in the app, not an official regulatory index.

These results cannot detect pathogens or contaminants that are not measured by the listed inputs. Do not use the app to certify water as safe. For drinking decisions, use a certified laboratory and local public-health guidance.

## Project files

- `app.py` — Streamlit application, data preparation, and models.
- `requirements.txt` — Python dependencies.
- `.gitignore` — local environments, caches, and generated files excluded from version control.
