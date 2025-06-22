import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your dataset
data = pd.read_csv("C:\\Users\\ARPAN MANDAL\\Downloads\\petroleum_all.csv")

# Set cutoff year
cutoff_year = 2022
data['period'] = pd.to_numeric(data['period'], errors='coerce')  # ensure numeric

# Folder to save plots
plot_folder = "C:\\Users\\ARPAN MANDAL\\Downloads\\RDD_Plots"
os.makedirs(plot_folder, exist_ok=True)

# Loop over countries
for country in data['Country'].unique():
    df_country = data[data['Country'] == country].copy()

    if df_country.empty or len(df_country) < 6:
        continue  # skip small groups

    # Create RDD variables
    df_country['post'] = (df_country['period'] >= cutoff_year).astype(int)
    df_country['running'] = df_country['period'] - cutoff_year
    df_country['post_running'] = df_country['post'] * df_country['running']
    df_country['log_primary'] = np.log(df_country['primary value'])

    # Fit the model (optional)
    model = smf.ols('log_primary ~ post + running + post_running', data=df_country).fit()

    # Plot
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df_country, x='period', y='log_primary', marker='o', linewidth=2)
    plt.axvline(x=cutoff_year, color='red', linestyle='--', label='RDD Cutoff (2022)')
    plt.title(f"{country} - Log(Primary Export Value) with RDD Cutoff")
    plt.xlabel("Year")
    plt.ylabel("log(Primary Value)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save each plot as a PNG
    plt.savefig(os.path.join(plot_folder, f"{country}_RDD_plot.png"))
    plt.close()

print(f"✅ RDD plots saved to: {plot_folder}")

