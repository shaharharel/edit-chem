#!/usr/bin/env python3
"""
Create visualizations of ChEMBL property statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
props_df = pd.read_csv('data/analysis/properties_with_1k_molecules.csv')

# Create output directory
output_dir = Path('data/analysis/plots')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loaded {len(props_df)} properties")

# 1. Distribution by activity type
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Count by category
category_counts = props_df['category'].value_counts()
axes[0, 0].bar(range(len(category_counts)), category_counts.values, color='steelblue')
axes[0, 0].set_xticks(range(len(category_counts)))
axes[0, 0].set_xticklabels(category_counts.index, rotation=45, ha='right')
axes[0, 0].set_ylabel('Number of Properties')
axes[0, 0].set_title('Properties by Activity Type (≥1k molecules)')
axes[0, 0].grid(axis='y', alpha=0.3)

# Total molecules by category
category_molecules = props_df.groupby('category')['unique_molecules'].sum()
axes[0, 1].bar(range(len(category_molecules)), category_molecules.values, color='coral')
axes[0, 1].set_xticks(range(len(category_molecules)))
axes[0, 1].set_xticklabels(category_molecules.index, rotation=45, ha='right')
axes[0, 1].set_ylabel('Total Molecules')
axes[0, 1].set_title('Total Molecules by Activity Type')
axes[0, 1].grid(axis='y', alpha=0.3)

# Distribution of molecule counts
for category in props_df['category'].unique():
    subset = props_df[props_df['category'] == category]['unique_molecules']
    axes[1, 0].hist(subset, bins=30, alpha=0.6, label=category)
axes[1, 0].set_xlabel('Molecules per Property')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Molecule Counts by Category')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 35000)

# Top 20 properties
top20 = props_df.nlargest(20, 'unique_molecules')
colors = [plt.cm.Set3(i/20) for i in range(20)]
axes[1, 1].barh(range(20), top20['unique_molecules'].values, color=colors)
axes[1, 1].set_yticks(range(20))
axes[1, 1].set_yticklabels([name[:30] for name in top20['property_name']], fontsize=8)
axes[1, 1].set_xlabel('Unique Molecules')
axes[1, 1].set_title('Top 20 Properties by Molecule Count')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'property_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'property_overview.png'}")

# 2. Activity type deep dive
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

activity_types = ['experimental_IC50', 'experimental_Ki', 'experimental_EC50', 'experimental_Kd']
for idx, activity in enumerate(activity_types):
    ax = axes[idx // 2, idx % 2]
    subset = props_df[props_df['category'] == activity].nlargest(15, 'unique_molecules')

    if len(subset) > 0:
        ax.barh(range(len(subset)), subset['unique_molecules'].values,
                color=plt.cm.viridis(np.linspace(0, 0.8, len(subset))))
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels([name[:35] for name in subset['property_name']], fontsize=9)
        ax.set_xlabel('Unique Molecules')
        ax.set_title(f'Top {activity.replace("experimental_", "")} Properties')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'No {activity} data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(output_dir / 'activity_type_breakdown.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'activity_type_breakdown.png'}")

# 3. Property tiers
fig, ax = plt.subplots(figsize=(12, 6))

# Define tiers
props_df['tier'] = pd.cut(props_df['unique_molecules'],
                           bins=[0, 1000, 2500, 5000, 10000, 50000],
                           labels=['1k-2.5k', '2.5k-5k', '5k-10k', '10k-50k', 'Excluded'],
                           include_lowest=True)

tier_counts = props_df.groupby(['tier', 'category']).size().unstack(fill_value=0)

tier_counts.plot(kind='bar', stacked=True, ax=ax,
                 colormap='Set2', width=0.7)
ax.set_xlabel('Property Tier (by molecule count)')
ax.set_ylabel('Number of Properties')
ax.set_title('Property Distribution Across Tiers by Activity Type')
ax.legend(title='Activity Type', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(output_dir / 'property_tiers.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'property_tiers.png'}")

# 4. Mean pChEMBL distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall distribution
axes[0].hist(props_df['mean_value'], bins=40, color='teal', alpha=0.7, edgecolor='black')
axes[0].axvline(props_df['mean_value'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {props_df["mean_value"].mean():.2f}')
axes[0].set_xlabel('Mean pChEMBL Value')
axes[0].set_ylabel('Number of Properties')
axes[0].set_title('Distribution of Mean pChEMBL Values Across Properties')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# By activity type
for category in props_df['category'].unique():
    subset = props_df[props_df['category'] == category]['mean_value']
    axes[1].hist(subset, bins=20, alpha=0.5, label=category)
axes[1].set_xlabel('Mean pChEMBL Value')
axes[1].set_ylabel('Number of Properties')
axes[1].set_title('pChEMBL Distribution by Activity Type')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pchembl_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'pchembl_distribution.png'}")

# 5. Summary statistics table
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary = props_df.groupby('category').agg({
    'unique_molecules': ['count', 'sum', 'mean', 'median', 'min', 'max'],
    'mean_value': ['mean', 'std'],
    'std_value': ['mean']
}).round(2)

print(summary)

print("\n" + "="*80)
print("TIER DISTRIBUTION")
print("="*80)

tier_summary = props_df.groupby('tier').agg({
    'property_name': 'count',
    'unique_molecules': ['sum', 'mean']
}).round(0)

print(tier_summary)

print(f"\n✓ All plots saved to: {output_dir}/")
