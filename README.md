# HyperCube-Nash Sports Predictor

A working prototype sports betting / prediction app based on the conceptual platform from the handwritten notes.

## Features
- Select two teams
- Instant win probability based on season stats
- Elo + Net Rating model (practical) or HyperCube-Nash attractor mode (conceptual)
- Visualizations: probability bars, radar comparison, attractor basin
- Full stats comparison table
- Adjustable home advantage

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501

## Notes
This is a functional demo using realistic mock NBA data.  
The core user flow works: pick two teams → get win chance from season stats.  
The advanced mathematical ideas (hypercube, Krasnosel'skii-Mann, Arnold tongues, basins of attractors) are referenced in the UI and lightly applied in the conceptual mode.
