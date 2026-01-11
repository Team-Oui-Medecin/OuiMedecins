# Conversion into parquet file for visualization
from inspect_ai.analysis import evals_df
from inspect_viz.mark import bar_x
from inspect_viz.plot import plot, write_html
from inspect_viz import Data
import pandas as pd

# Extract a dataframe from your local /logs directory
df = evals_df("./logs")

# Filter for the specific task if needed
df_health_scenarios = df[df['task_name'] == 'health_scenarios']

# For duplicate model names, keep only the most recent log
# Sort by created date (most recent first), then take first occurrence of each model
df_health_scenarios = df_health_scenarios.sort_values('created', ascending=False)
df_health_scenarios = df_health_scenarios.drop_duplicates(subset=['model'], keep='first')

# Save as the parquet file your visualization expects
df_health_scenarios.to_parquet("docs/assets/health_scenarios.parquet")

# Load the data for visualization
evals = Data.from_file("docs/assets/health_scenarios.parquet")

# Create the plot
chart = plot(
    bar_x(
        evals, 
        x="score_headline_value", 
        y="model",
        sort={"y": "x", "reverse": True},
        fill="#3266ae"
    ),
    title="Health Scenarios - LLM Score Comparison",
    x_label="Score",
    y_label=None,
    margin_left=300,  # Space for y-axis labels
    width=1000,        # Total plot width (adjust to spread bars more)
    height=600,        # Total plot height
)

# Save the visualization as an HTML file
write_html("docs/visualization.html", chart)