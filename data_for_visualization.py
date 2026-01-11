# Grouped bar chart showing model performance across datasets using matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def model_score_plot():
    # Read the CSV file with average scores per dataset
    # Get the script's directory and construct path relative to it
    script_dir = Path(__file__).parent
    csv_path = script_dir / "visualization_data" / "health_scenarios_for_export - Sheet6.csv"
    df = pd.read_csv(csv_path)

    # Clean up the data - remove empty rows and fix column names
    df = df.dropna(subset=['model_name', 'dataset', 'mean score'])
    df = df[df['model_name'].str.strip() != '']  # Remove empty model rows

    # Clean column names
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'mean score': 'mean_score'})

    # Ensure mean_score is numeric
    df['mean_score'] = pd.to_numeric(df['mean_score'], errors='coerce')
    df = df.dropna(subset=['mean_score'])

    print("Loaded data:")
    print(df)
    print(f"\nModels: {df['model_name'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")

    # Order datasets for consistent grouping
    dataset_order = ['baseline', 'ai_amplified', 'human_amplified']
    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
    df = df.sort_values(['model_name', 'dataset'])

    # Pivot the data to create a matrix suitable for grouped bar chart
    # Rows = models, Columns = datasets
    df_pivot = df.pivot(index='model_name', columns='dataset', values='mean_score')
    df_pivot = df_pivot.reindex(columns=dataset_order)  # Ensure correct column order

    print("\nPivoted data (models × datasets):")
    print(df_pivot)

    # Create the grouped bar chart using matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set up the x positions for models
    models = df_pivot.index.tolist()
    x = np.arange(len(models))
    width = 0.25  # Width of each bar group

    # Create bars for each dataset
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    datasets = df_pivot.columns.tolist()

    for i, dataset in enumerate(datasets):
        offset = (i - 1) * width  # Center the bars around each model position
        bars = ax.bar(x + offset, df_pivot[dataset], width, label=dataset, color=colors[i])

    # Customize the chart
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
    ax.set_title('Health Scenarios - Model Performance Across 3 Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Datasets', title_fontsize=11, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the visualization
    output_path = script_dir / "docs" / "visualization.html"
    # For HTML output, we'll use matplotlib's interactive backend or save as PNG/PDF
    # Save as both PNG and create an HTML wrapper
    png_path = script_dir / "docs" / "visualization.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')

    # Create an HTML file that displays the image
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Health Scenarios - Model Performance Score Across 3 Datasets</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Health Scenarios - Model Performance Across 3 Datasets</h1>
            <img src="visualization.png" alt="Grouped Bar Chart">
        </div>
    </body>
    </html>
    """
    with open(output_path, 'w') as f:
        f.write(html_content)

    print("\nGrouped bar chart saved:")
    print(f"  - PNG: {png_path}")
    print(f"  - HTML: {output_path}")
    print("\nEach model shows 3 bars side-by-side (one for each dataset: baseline, ai_amplified, human_amplified)")

if __name__ == "__main__":
    model_score_plot()