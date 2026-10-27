# NBM ventilation maps (AZ / NM PSAs)

Every morning this builds a webpage with NBM max afternoon ventilation rate (knot-ft) for
Days 1–7: one overview of all Arizona and New Mexico PSAs, plus one map per PSA.

Webpage: `https://<your-username>.github.io/vent-maps/`

## Files
- `make_vent_maps.py`: the maps and webpage. Settings are at the top (categories, colors,
  afternoon hours, PSAs to include or exclude).
- `requirements.txt`: Python packages
- `.github/workflows/vent-maps.yml`: daily schedule (UTC) and publishing

## Make the page now
Actions tab > Daily ventilation maps > Run workflow > Run workflow (about 5–10 minutes).

## If a run shows a red X
Click the run > build-and-publish > the step with the red X. The webpage keeps the last good
version until a run succeeds.

## Run it on your own computer
    pip install -r requirements.txt
    python make_vent_maps.py
Then open `site/index.html`.
