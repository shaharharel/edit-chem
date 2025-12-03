#!/usr/bin/env python3
"""
Generate interactive HTML report for antibody datasets.

This script analyzes the collected antibody datasets and generates
a comprehensive HTML report with interactive visualizations.

Usage:
    python reports/generate_antibody_dataset_report.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
import re
import math

# Data paths (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "antibody"

def load_datasets():
    """Load all antibody datasets."""
    datasets = {}

    # 1. AbBiBench
    abbibench_path = DATA_DIR / "abbibench" / "train.csv"
    if abbibench_path.exists():
        datasets['abbibench'] = pd.read_csv(abbibench_path)
        print(f"Loaded AbBiBench: {len(datasets['abbibench']):,} samples")

    # 2. SKEMPI2
    skempi_path = DATA_DIR / "skempi2" / "skempi_v2.csv"
    if skempi_path.exists():
        skempi = pd.read_csv(skempi_path, sep=';')
        # Filter for antibody entries
        antibody_keywords = ['antibody', 'fab', 'igg', 'fv', 'scfv', 'nanobody', 'vhh', 'immunoglobulin']
        is_antibody = skempi.apply(lambda r: any(
            kw in str(r.get('Protein 1', '')).lower() or kw in str(r.get('Protein 2', '')).lower()
            for kw in antibody_keywords
        ), axis=1)
        datasets['skempi2'] = skempi[is_antibody].copy()
        print(f"Loaded SKEMPI2: {len(datasets['skempi2']):,} antibody entries")

    # 3. AbAgym
    abagym_path = DATA_DIR / "abagym" / "AbAgym_data_non-redundant.csv"
    if abagym_path.exists():
        datasets['abagym'] = pd.read_csv(abagym_path)
        print(f"Loaded AbAgym: {len(datasets['abagym']):,} mutations")

    abagym_meta_path = DATA_DIR / "abagym" / "AbAgym_metadata.csv"
    if abagym_meta_path.exists():
        datasets['abagym_meta'] = pd.read_csv(abagym_meta_path)

    # 4. Trastuzumab DMS
    trast_pos_path = DATA_DIR / "trastuzumab_dms" / "data" / "mHER_H3_AgPos.csv"
    trast_neg_path = DATA_DIR / "trastuzumab_dms" / "data" / "mHER_H3_AgNeg.csv"
    if trast_pos_path.exists() and trast_neg_path.exists():
        pos = pd.read_csv(trast_pos_path)
        neg = pd.read_csv(trast_neg_path)
        pos['binding'] = 1
        neg['binding'] = 0
        datasets['trastuzumab'] = pd.concat([pos, neg], ignore_index=True)
        print(f"Loaded Trastuzumab: {len(datasets['trastuzumab']):,} CDRH3 variants")

    return datasets


def analyze_abbibench(df):
    """Analyze AbBiBench dataset."""
    analysis = {
        'total_samples': len(df),
        'unique_heavy': df['heavy_chain_seq'].nunique(),
        'unique_light': df['light_chain_seq'].nunique(),
        'unique_pairs': df.groupby(['heavy_chain_seq', 'light_chain_seq']).ngroups,
    }

    # Binding score stats
    analysis['binding_score'] = {
        'min': float(df['binding_score'].min()),
        'max': float(df['binding_score'].max()),
        'mean': float(df['binding_score'].mean()),
        'std': float(df['binding_score'].std()),
        'median': float(df['binding_score'].median()),
        'q25': float(df['binding_score'].quantile(0.25)),
        'q75': float(df['binding_score'].quantile(0.75)),
    }

    # Sequence lengths
    df['heavy_len'] = df['heavy_chain_seq'].str.len()
    df['light_len'] = df['light_chain_seq'].str.len()

    analysis['heavy_length'] = {
        'min': int(df['heavy_len'].min()),
        'max': int(df['heavy_len'].max()),
        'mean': float(df['heavy_len'].mean()),
        'std': float(df['heavy_len'].std()),
    }

    analysis['light_length'] = {
        'min': int(df['light_len'].min()),
        'max': int(df['light_len'].max()),
        'mean': float(df['light_len'].mean()),
        'std': float(df['light_len'].std()),
    }

    # Distribution data for plotting
    analysis['binding_score_hist'] = df['binding_score'].value_counts(bins=50).sort_index().to_dict()
    analysis['heavy_len_hist'] = df['heavy_len'].value_counts().sort_index().to_dict()
    analysis['light_len_hist'] = df['light_len'].value_counts().sort_index().to_dict()

    # Amino acid composition
    all_heavy = ''.join(df['heavy_chain_seq'].tolist())
    all_light = ''.join(df['light_chain_seq'].tolist())
    analysis['heavy_aa_freq'] = dict(Counter(all_heavy))
    analysis['light_aa_freq'] = dict(Counter(all_light))

    # Samples per heavy chain (to understand grouping)
    heavy_counts = df.groupby('heavy_chain_seq').size()
    analysis['samples_per_heavy'] = {
        'single': int((heavy_counts == 1).sum()),
        'multiple': int((heavy_counts > 1).sum()),
        'max': int(heavy_counts.max()),
        'mean': float(heavy_counts.mean()),
    }

    return analysis


def analyze_skempi2(df):
    """Analyze SKEMPI2 dataset."""
    analysis = {
        'total_entries': len(df),
        'unique_pdbs': df['#Pdb'].nunique(),
        'unique_antibodies': df['Protein 1'].nunique(),
        'unique_antigens': df['Protein 2'].nunique(),
    }

    # Parse temperature
    def parse_temp(temp_str):
        if pd.isna(temp_str):
            return 298
        match = re.search(r'(\d+(?:\.\d+)?)', str(temp_str))
        return float(match.group(1)) if match else 298

    # Calculate ddG
    R = 0.001987  # kcal/(mol*K)
    ddg_values = []
    for _, row in df.iterrows():
        kd_mut = row.get('Affinity_mut_parsed', None)
        kd_wt = row.get('Affinity_wt_parsed', None)
        temp = parse_temp(row.get('Temperature'))

        if pd.notna(kd_mut) and pd.notna(kd_wt) and kd_mut > 0 and kd_wt > 0:
            ddg = R * temp * math.log(kd_mut / kd_wt)
            ddg_values.append(ddg)
        else:
            ddg_values.append(None)

    df['ddG_computed'] = ddg_values
    valid_ddg = df[df['ddG_computed'].notna()]['ddG_computed']

    analysis['ddg_stats'] = {
        'valid_count': len(valid_ddg),
        'min': float(valid_ddg.min()) if len(valid_ddg) > 0 else None,
        'max': float(valid_ddg.max()) if len(valid_ddg) > 0 else None,
        'mean': float(valid_ddg.mean()) if len(valid_ddg) > 0 else None,
        'std': float(valid_ddg.std()) if len(valid_ddg) > 0 else None,
        'median': float(valid_ddg.median()) if len(valid_ddg) > 0 else None,
    }

    # Mutation count distribution
    def count_mutations(mut_str):
        if pd.isna(mut_str):
            return 0
        return len(str(mut_str).split(','))

    df['num_mutations'] = df['Mutation(s)_cleaned'].apply(count_mutations)
    analysis['mutation_counts'] = df['num_mutations'].value_counts().sort_index().to_dict()

    # ddG histogram
    if len(valid_ddg) > 0:
        hist, bins = np.histogram(valid_ddg, bins=30)
        analysis['ddg_hist'] = {
            'counts': hist.tolist(),
            'bins': bins.tolist(),
        }

    # Top antigens
    analysis['top_antigens'] = df['Protein 2'].value_counts().head(10).to_dict()

    # Top antibodies
    analysis['top_antibodies'] = df['Protein 1'].value_counts().head(10).to_dict()

    return analysis


def analyze_abagym(df, meta_df=None):
    """Analyze AbAgym dataset."""
    analysis = {
        'total_mutations': len(df),
        'unique_experiments': df['DMS_name'].nunique(),
    }

    # DMS score stats
    analysis['dms_score'] = {
        'min': float(df['DMS_score'].min()),
        'max': float(df['DMS_score'].max()),
        'mean': float(df['DMS_score'].mean()),
        'std': float(df['DMS_score'].std()),
        'median': float(df['DMS_score'].median()),
    }

    # Chain distribution
    analysis['chain_dist'] = df['chains'].value_counts().head(10).to_dict()

    # Interface vs non-interface
    interface_threshold = 5.0  # Angstroms
    interface = df[df['closest_interface_atom_distance'] < interface_threshold]
    analysis['interface_mutations'] = len(interface)
    analysis['non_interface_mutations'] = len(df) - len(interface)

    # Mutations per experiment
    muts_per_exp = df.groupby('DMS_name').size()
    analysis['muts_per_experiment'] = {
        'min': int(muts_per_exp.min()),
        'max': int(muts_per_exp.max()),
        'mean': float(muts_per_exp.mean()),
    }

    # Wildtype amino acid distribution
    analysis['wildtype_aa_freq'] = df['wildtype'].value_counts().to_dict()

    # Mutation amino acid distribution
    analysis['mutation_aa_freq'] = df['mutation'].value_counts().to_dict()

    # DMS score histogram
    # Clip extreme values for better visualization
    clipped_scores = df['DMS_score'].clip(-5, 5)
    hist, bins = np.histogram(clipped_scores, bins=50)
    analysis['dms_score_hist'] = {
        'counts': hist.tolist(),
        'bins': bins.tolist(),
    }

    # Interface distance histogram
    hist, bins = np.histogram(df['closest_interface_atom_distance'].clip(0, 30), bins=30)
    analysis['interface_dist_hist'] = {
        'counts': hist.tolist(),
        'bins': bins.tolist(),
    }

    # Antigen distribution from metadata
    if meta_df is not None:
        analysis['antigen_dist'] = meta_df['antigen_name'].value_counts().to_dict()

    # Substitution matrix (which AA changes to which)
    sub_matrix = df.groupby(['wildtype', 'mutation']).size().unstack(fill_value=0)
    analysis['substitution_matrix'] = {
        'index': sub_matrix.index.tolist(),
        'columns': sub_matrix.columns.tolist(),
        'values': sub_matrix.values.tolist(),
    }

    return analysis


def analyze_trastuzumab(df):
    """Analyze Trastuzumab CDRH3 DMS dataset."""
    analysis = {
        'total_variants': len(df),
        'binders': int(df['binding'].sum()),
        'non_binders': int((df['binding'] == 0).sum()),
        'binding_ratio': float(df['binding'].mean()),
    }

    # Sequence length (should be constant)
    df['seq_len'] = df['AASeq'].str.len()
    analysis['sequence_length'] = int(df['seq_len'].mode().iloc[0])

    # Unique sequences
    analysis['unique_sequences'] = df['AASeq'].nunique()

    # Position-specific amino acid frequencies for binders vs non-binders
    binders = df[df['binding'] == 1]['AASeq'].tolist()
    non_binders = df[df['binding'] == 0]['AASeq'].tolist()

    seq_len = analysis['sequence_length']
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'

    # Position frequency matrices
    def calc_position_freq(sequences):
        freq = {pos: {aa: 0 for aa in aa_list} for pos in range(seq_len)}
        for seq in sequences:
            for pos, aa in enumerate(seq):
                if aa in aa_list:
                    freq[pos][aa] += 1
        # Normalize
        for pos in freq:
            total = sum(freq[pos].values())
            if total > 0:
                for aa in freq[pos]:
                    freq[pos][aa] /= total
        return freq

    analysis['binder_pos_freq'] = calc_position_freq(binders)
    analysis['non_binder_pos_freq'] = calc_position_freq(non_binders)

    # Overall amino acid frequency difference
    all_binder_aa = ''.join(binders)
    all_non_binder_aa = ''.join(non_binders)

    binder_freq = Counter(all_binder_aa)
    non_binder_freq = Counter(all_non_binder_aa)

    # Normalize
    binder_total = sum(binder_freq.values())
    non_binder_total = sum(non_binder_freq.values())

    analysis['binder_aa_freq'] = {aa: binder_freq.get(aa, 0) / binder_total for aa in aa_list}
    analysis['non_binder_aa_freq'] = {aa: non_binder_freq.get(aa, 0) / non_binder_total for aa in aa_list}

    # Enrichment (log2 ratio)
    analysis['aa_enrichment'] = {}
    for aa in aa_list:
        b_freq = analysis['binder_aa_freq'].get(aa, 0.001)
        nb_freq = analysis['non_binder_aa_freq'].get(aa, 0.001)
        if b_freq > 0 and nb_freq > 0:
            analysis['aa_enrichment'][aa] = math.log2(b_freq / nb_freq)

    return analysis


def generate_html_report(datasets, analyses):
    """Generate comprehensive HTML report."""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Antibody Dataset Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-light: #f8fafc;
            --bg-card: #ffffff;
            --border: #e2e8f0;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-light);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background: linear-gradient(135deg, var(--primary), #1d4ed8);
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
            border-radius: 12px;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }

        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .summary-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary);
        }

        .summary-card h3 {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .summary-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }

        .summary-card .subtitle {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 4px;
        }

        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0;
        }

        .tab {
            padding: 12px 24px;
            cursor: pointer;
            background: none;
            border: none;
            font-size: 1rem;
            color: var(--text-secondary);
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }

        .tab:hover {
            color: var(--primary);
        }

        .tab.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
            font-weight: 600;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .card h3 {
            font-size: 1.1rem;
            margin: 20px 0 12px;
            color: var(--text-secondary);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }

        .stat-item {
            text-align: center;
            padding: 16px;
            background: var(--bg-light);
            border-radius: 8px;
        }

        .stat-item .label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .stat-item .value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .plot-container {
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }

        .plot-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg-light);
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
        }

        tr:hover {
            background: var(--bg-light);
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        .badge-success { background: #dcfce7; color: #166534; }
        .badge-warning { background: #fef3c7; color: #92400e; }
        .badge-info { background: #dbeafe; color: #1e40af; }

        .description {
            background: var(--bg-light);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid var(--primary);
        }

        .heatmap-container {
            overflow-x: auto;
        }

        footer {
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Antibody Dataset Analysis Report</h1>
            <p>Comprehensive analysis of collected antibody mutation and binding datasets</p>
        </header>

        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="summary-card">
                <h3>Total Samples</h3>
                <div class="value" id="total-samples">-</div>
                <div class="subtitle">Across all datasets</div>
            </div>
            <div class="summary-card">
                <h3>Datasets</h3>
                <div class="value">4</div>
                <div class="subtitle">AbBiBench, SKEMPI2, AbAgym, Trastuzumab</div>
            </div>
            <div class="summary-card">
                <h3>DMS Experiments</h3>
                <div class="value" id="total-dms">-</div>
                <div class="subtitle">From AbAgym</div>
            </div>
            <div class="summary-card">
                <h3>Unique Antigens</h3>
                <div class="value" id="total-antigens">-</div>
                <div class="subtitle">COVID-19, HIV, HER2, etc.</div>
            </div>
        </div>

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab active" onclick="showTab('abbibench')">AbBiBench</button>
            <button class="tab" onclick="showTab('skempi2')">SKEMPI2</button>
            <button class="tab" onclick="showTab('abagym')">AbAgym</button>
            <button class="tab" onclick="showTab('trastuzumab')">Trastuzumab DMS</button>
            <button class="tab" onclick="showTab('comparison')">Comparison</button>
        </div>

        <!-- AbBiBench Tab -->
        <div id="abbibench" class="tab-content active">
            <div class="card">
                <h2>AbBiBench - Antibody Binding Benchmark</h2>
                <div class="description">
                    Large-scale antibody binding dataset with full heavy and light chain sequences paired with binding scores.
                    Ideal for pre-training and general binding affinity prediction.
                </div>

                <div class="stats-grid" id="abbibench-stats"></div>

                <h3>Binding Score Distribution</h3>
                <div id="abbibench-binding-hist" class="plot-container"></div>

                <div class="plot-row">
                    <div>
                        <h3>Heavy Chain Length Distribution</h3>
                        <div id="abbibench-heavy-len" class="plot-container"></div>
                    </div>
                    <div>
                        <h3>Light Chain Length Distribution</h3>
                        <div id="abbibench-light-len" class="plot-container"></div>
                    </div>
                </div>

                <h3>Amino Acid Composition</h3>
                <div id="abbibench-aa-freq" class="plot-container"></div>
            </div>
        </div>

        <!-- SKEMPI2 Tab -->
        <div id="skempi2" class="tab-content">
            <div class="card">
                <h2>SKEMPI2 - Structural Kinetic and Energetic database of Mutant Protein Interactions</h2>
                <div class="description">
                    Curated database of protein-protein binding affinity changes upon mutation.
                    Antibody subset contains explicit WT/Mutant pairs with ddG values and PDB structures.
                    Ideal for edit-effect prediction with structural context.
                </div>

                <div class="stats-grid" id="skempi2-stats"></div>

                <h3>ddG Distribution (kcal/mol)</h3>
                <div id="skempi2-ddg-hist" class="plot-container"></div>

                <h3>Mutation Count Distribution</h3>
                <div id="skempi2-mut-count" class="plot-container"></div>

                <div class="plot-row">
                    <div>
                        <h3>Top Antibodies</h3>
                        <div id="skempi2-top-ab" class="plot-container"></div>
                    </div>
                    <div>
                        <h3>Top Antigens</h3>
                        <div id="skempi2-top-ag" class="plot-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- AbAgym Tab -->
        <div id="abagym" class="tab-content">
            <div class="card">
                <h2>AbAgym - Antibody-Antigen Gym</h2>
                <div class="description">
                    Comprehensive collection of 68 deep mutational scanning (DMS) experiments on antibody-antigen complexes.
                    Contains 324k single-site mutations with PDB structures and interface information.
                    The largest curated antibody DMS dataset available.
                </div>

                <div class="stats-grid" id="abagym-stats"></div>

                <h3>DMS Score Distribution (clipped to [-5, 5])</h3>
                <div id="abagym-dms-hist" class="plot-container"></div>

                <div class="plot-row">
                    <div>
                        <h3>Interface Distance Distribution</h3>
                        <div id="abagym-interface-dist" class="plot-container"></div>
                    </div>
                    <div>
                        <h3>Antigen Distribution</h3>
                        <div id="abagym-antigen-dist" class="plot-container"></div>
                    </div>
                </div>

                <h3>Amino Acid Substitution Preferences</h3>
                <div id="abagym-aa-freq" class="plot-container"></div>

                <h3>Substitution Matrix (Wildtype → Mutant)</h3>
                <div id="abagym-sub-matrix" class="plot-container" style="height: 500px;"></div>
            </div>
        </div>

        <!-- Trastuzumab Tab -->
        <div id="trastuzumab" class="tab-content">
            <div class="card">
                <h2>Trastuzumab CDRH3 DMS (Mason et al. 2021)</h2>
                <div class="description">
                    Deep mutational scanning of the CDRH3 region (10 amino acids) of therapeutic antibody Trastuzumab (Herceptin).
                    Binary classification: binders vs non-binders to HER2 antigen.
                    Classic dataset for antibody CDR optimization studies.
                </div>

                <div class="stats-grid" id="trastuzumab-stats"></div>

                <h3>Binding Class Distribution</h3>
                <div id="trastuzumab-binding-pie" class="plot-container"></div>

                <h3>Amino Acid Enrichment in Binders (log2 ratio)</h3>
                <div id="trastuzumab-enrichment" class="plot-container"></div>

                <h3>Position-Specific Amino Acid Preferences</h3>
                <div class="plot-row">
                    <div>
                        <h3>Binders</h3>
                        <div id="trastuzumab-binder-logo" class="plot-container" style="height: 300px;"></div>
                    </div>
                    <div>
                        <h3>Non-Binders</h3>
                        <div id="trastuzumab-nonbinder-logo" class="plot-container" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Comparison Tab -->
        <div id="comparison" class="tab-content">
            <div class="card">
                <h2>Dataset Comparison</h2>

                <table>
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Samples</th>
                            <th>Data Type</th>
                            <th>Target</th>
                            <th>Structure</th>
                            <th>Use Case</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>AbBiBench</strong></td>
                            <td id="comp-abbibench-n">-</td>
                            <td>Full sequences</td>
                            <td>Binding score (continuous)</td>
                            <td><span class="badge badge-warning">No</span></td>
                            <td>Pre-training, representation learning</td>
                        </tr>
                        <tr>
                            <td><strong>SKEMPI2</strong></td>
                            <td id="comp-skempi2-n">-</td>
                            <td>WT/Mutant pairs</td>
                            <td>ddG (kcal/mol)</td>
                            <td><span class="badge badge-success">Yes (PDB)</span></td>
                            <td>Edit effect prediction</td>
                        </tr>
                        <tr>
                            <td><strong>AbAgym</strong></td>
                            <td id="comp-abagym-n">-</td>
                            <td>Single-site mutations</td>
                            <td>DMS score (continuous)</td>
                            <td><span class="badge badge-success">Yes (PDB)</span></td>
                            <td>Mutation effect prediction, benchmark</td>
                        </tr>
                        <tr>
                            <td><strong>Trastuzumab</strong></td>
                            <td id="comp-trast-n">-</td>
                            <td>CDRH3 variants</td>
                            <td>Binary (binder/non-binder)</td>
                            <td><span class="badge badge-warning">No</span></td>
                            <td>CDR optimization, classification</td>
                        </tr>
                    </tbody>
                </table>

                <h3>Sample Size Comparison</h3>
                <div id="comparison-samples" class="plot-container"></div>

                <h3>Recommendations by Task</h3>
                <div class="description">
                    <strong>For Edit-Based Learning (Mutation Effect Prediction):</strong>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>AbAgym</strong> - Best choice with 324k single-site mutations from 68 DMS experiments</li>
                        <li><strong>SKEMPI2</strong> - Gold standard with biophysically meaningful ddG values</li>
                    </ul>
                </div>
                <div class="description">
                    <strong>For Representation Learning / Pre-training:</strong>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>AbBiBench</strong> - Largest dataset (186k) with diverse sequences</li>
                    </ul>
                </div>
                <div class="description">
                    <strong>For CDR Engineering:</strong>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>Trastuzumab DMS</strong> - Focused CDRH3 optimization data</li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            Generated by edit-chem antibody dataset analysis pipeline
        </footer>
    </div>

    <script>
        // Data from Python analysis
        const analysisData = ANALYSIS_DATA_PLACEHOLDER;

        // Tab switching
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        // Plotly config
        const plotConfig = {responsive: true, displayModeBar: false};
        const plotLayout = {
            margin: {l: 50, r: 30, t: 30, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'}
        };

        // Initialize dashboard
        function initDashboard() {
            // Summary cards
            let totalSamples = 0;
            if (analysisData.abbibench) totalSamples += analysisData.abbibench.total_samples;
            if (analysisData.skempi2) totalSamples += analysisData.skempi2.total_entries;
            if (analysisData.abagym) totalSamples += analysisData.abagym.total_mutations;
            if (analysisData.trastuzumab) totalSamples += analysisData.trastuzumab.total_variants;

            document.getElementById('total-samples').textContent = totalSamples.toLocaleString();
            document.getElementById('total-dms').textContent = analysisData.abagym ? analysisData.abagym.unique_experiments : '-';
            document.getElementById('total-antigens').textContent = analysisData.abagym && analysisData.abagym.antigen_dist ?
                Object.keys(analysisData.abagym.antigen_dist).length : '-';

            // Comparison table
            if (analysisData.abbibench) document.getElementById('comp-abbibench-n').textContent = analysisData.abbibench.total_samples.toLocaleString();
            if (analysisData.skempi2) document.getElementById('comp-skempi2-n').textContent = analysisData.skempi2.total_entries.toLocaleString();
            if (analysisData.abagym) document.getElementById('comp-abagym-n').textContent = analysisData.abagym.total_mutations.toLocaleString();
            if (analysisData.trastuzumab) document.getElementById('comp-trast-n').textContent = analysisData.trastuzumab.total_variants.toLocaleString();

            // Initialize all plots
            initAbBiBench();
            initSKEMPI2();
            initAbAgym();
            initTrastuzumab();
            initComparison();
        }

        function initAbBiBench() {
            const data = analysisData.abbibench;
            if (!data) return;

            // Stats grid
            const statsHtml = `
                <div class="stat-item"><div class="label">Total Samples</div><div class="value">${data.total_samples.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Unique Heavy</div><div class="value">${data.unique_heavy.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Unique Light</div><div class="value">${data.unique_light.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Unique Pairs</div><div class="value">${data.unique_pairs.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Binding Mean</div><div class="value">${data.binding_score.mean.toFixed(2)}</div></div>
                <div class="stat-item"><div class="label">Binding Std</div><div class="value">${data.binding_score.std.toFixed(2)}</div></div>
            `;
            document.getElementById('abbibench-stats').innerHTML = statsHtml;

            // Binding score histogram
            Plotly.newPlot('abbibench-binding-hist', [{
                x: Object.keys(data.binding_score_hist).map(k => {
                    const match = k.match(/\\(([^,]+),/);
                    return match ? parseFloat(match[1]) : 0;
                }),
                y: Object.values(data.binding_score_hist),
                type: 'bar',
                marker: {color: '#2563eb'}
            }], {...plotLayout, xaxis: {title: 'Binding Score'}, yaxis: {title: 'Count'}}, plotConfig);

            // Heavy length histogram
            Plotly.newPlot('abbibench-heavy-len', [{
                x: Object.keys(data.heavy_len_hist).map(Number),
                y: Object.values(data.heavy_len_hist),
                type: 'bar',
                marker: {color: '#10b981'}
            }], {...plotLayout, xaxis: {title: 'Length (aa)'}, yaxis: {title: 'Count'}}, plotConfig);

            // Light length histogram
            Plotly.newPlot('abbibench-light-len', [{
                x: Object.keys(data.light_len_hist).map(Number),
                y: Object.values(data.light_len_hist),
                type: 'bar',
                marker: {color: '#f59e0b'}
            }], {...plotLayout, xaxis: {title: 'Length (aa)'}, yaxis: {title: 'Count'}}, plotConfig);

            // AA frequency comparison
            const aas = 'ACDEFGHIKLMNPQRSTVWY'.split('');
            const heavyTotal = Object.values(data.heavy_aa_freq).reduce((a,b) => a+b, 0);
            const lightTotal = Object.values(data.light_aa_freq).reduce((a,b) => a+b, 0);

            Plotly.newPlot('abbibench-aa-freq', [
                {
                    x: aas,
                    y: aas.map(aa => (data.heavy_aa_freq[aa] || 0) / heavyTotal * 100),
                    name: 'Heavy Chain',
                    type: 'bar',
                    marker: {color: '#2563eb'}
                },
                {
                    x: aas,
                    y: aas.map(aa => (data.light_aa_freq[aa] || 0) / lightTotal * 100),
                    name: 'Light Chain',
                    type: 'bar',
                    marker: {color: '#10b981'}
                }
            ], {...plotLayout, barmode: 'group', xaxis: {title: 'Amino Acid'}, yaxis: {title: 'Frequency (%)'}}, plotConfig);
        }

        function initSKEMPI2() {
            const data = analysisData.skempi2;
            if (!data) return;

            // Stats grid
            const statsHtml = `
                <div class="stat-item"><div class="label">Total Entries</div><div class="value">${data.total_entries.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Valid ddG</div><div class="value">${data.ddg_stats.valid_count}</div></div>
                <div class="stat-item"><div class="label">Unique PDBs</div><div class="value">${data.unique_pdbs}</div></div>
                <div class="stat-item"><div class="label">ddG Mean</div><div class="value">${data.ddg_stats.mean ? data.ddg_stats.mean.toFixed(2) : '-'}</div></div>
                <div class="stat-item"><div class="label">ddG Std</div><div class="value">${data.ddg_stats.std ? data.ddg_stats.std.toFixed(2) : '-'}</div></div>
                <div class="stat-item"><div class="label">ddG Range</div><div class="value">${data.ddg_stats.min ? data.ddg_stats.min.toFixed(1) : '-'} to ${data.ddg_stats.max ? data.ddg_stats.max.toFixed(1) : '-'}</div></div>
            `;
            document.getElementById('skempi2-stats').innerHTML = statsHtml;

            // ddG histogram
            if (data.ddg_hist) {
                const binCenters = data.ddg_hist.bins.slice(0, -1).map((b, i) => (b + data.ddg_hist.bins[i+1]) / 2);
                Plotly.newPlot('skempi2-ddg-hist', [{
                    x: binCenters,
                    y: data.ddg_hist.counts,
                    type: 'bar',
                    marker: {color: '#2563eb'}
                }], {...plotLayout, xaxis: {title: 'ddG (kcal/mol)'}, yaxis: {title: 'Count'}}, plotConfig);
            }

            // Mutation count distribution
            Plotly.newPlot('skempi2-mut-count', [{
                x: Object.keys(data.mutation_counts),
                y: Object.values(data.mutation_counts),
                type: 'bar',
                marker: {color: '#10b981'}
            }], {...plotLayout, xaxis: {title: 'Number of Mutations'}, yaxis: {title: 'Count'}}, plotConfig);

            // Top antibodies
            const abNames = Object.keys(data.top_antibodies).slice(0, 8);
            Plotly.newPlot('skempi2-top-ab', [{
                y: abNames,
                x: abNames.map(n => data.top_antibodies[n]),
                type: 'bar',
                orientation: 'h',
                marker: {color: '#2563eb'}
            }], {...plotLayout, margin: {l: 200}}, plotConfig);

            // Top antigens
            const agNames = Object.keys(data.top_antigens).slice(0, 8);
            Plotly.newPlot('skempi2-top-ag', [{
                y: agNames,
                x: agNames.map(n => data.top_antigens[n]),
                type: 'bar',
                orientation: 'h',
                marker: {color: '#f59e0b'}
            }], {...plotLayout, margin: {l: 200}}, plotConfig);
        }

        function initAbAgym() {
            const data = analysisData.abagym;
            if (!data) return;

            // Stats grid
            const statsHtml = `
                <div class="stat-item"><div class="label">Total Mutations</div><div class="value">${data.total_mutations.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">DMS Experiments</div><div class="value">${data.unique_experiments}</div></div>
                <div class="stat-item"><div class="label">Interface Mutations</div><div class="value">${data.interface_mutations.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">DMS Mean</div><div class="value">${data.dms_score.mean.toFixed(2)}</div></div>
                <div class="stat-item"><div class="label">DMS Std</div><div class="value">${data.dms_score.std.toFixed(2)}</div></div>
                <div class="stat-item"><div class="label">DMS Median</div><div class="value">${data.dms_score.median.toFixed(2)}</div></div>
            `;
            document.getElementById('abagym-stats').innerHTML = statsHtml;

            // DMS score histogram
            if (data.dms_score_hist) {
                const binCenters = data.dms_score_hist.bins.slice(0, -1).map((b, i) => (b + data.dms_score_hist.bins[i+1]) / 2);
                Plotly.newPlot('abagym-dms-hist', [{
                    x: binCenters,
                    y: data.dms_score_hist.counts,
                    type: 'bar',
                    marker: {color: '#2563eb'}
                }], {...plotLayout, xaxis: {title: 'DMS Score (clipped)'}, yaxis: {title: 'Count'}}, plotConfig);
            }

            // Interface distance histogram
            if (data.interface_dist_hist) {
                const binCenters = data.interface_dist_hist.bins.slice(0, -1).map((b, i) => (b + data.interface_dist_hist.bins[i+1]) / 2);
                Plotly.newPlot('abagym-interface-dist', [{
                    x: binCenters,
                    y: data.interface_dist_hist.counts,
                    type: 'bar',
                    marker: {color: '#10b981'}
                }], {...plotLayout, xaxis: {title: 'Distance to Interface (Å)'}, yaxis: {title: 'Count'}}, plotConfig);
            }

            // Antigen distribution
            if (data.antigen_dist) {
                const antigens = Object.keys(data.antigen_dist);
                Plotly.newPlot('abagym-antigen-dist', [{
                    labels: antigens,
                    values: antigens.map(a => data.antigen_dist[a]),
                    type: 'pie',
                    textinfo: 'label+percent',
                    marker: {colors: ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16']}
                }], {...plotLayout}, plotConfig);
            }

            // AA frequency
            const aas = 'ACDEFGHIKLMNPQRSTVWY'.split('');
            const wtTotal = Object.values(data.wildtype_aa_freq).reduce((a,b) => a+b, 0);
            const mutTotal = Object.values(data.mutation_aa_freq).reduce((a,b) => a+b, 0);

            Plotly.newPlot('abagym-aa-freq', [
                {
                    x: aas,
                    y: aas.map(aa => (data.wildtype_aa_freq[aa] || 0) / wtTotal * 100),
                    name: 'Wildtype',
                    type: 'bar',
                    marker: {color: '#2563eb'}
                },
                {
                    x: aas,
                    y: aas.map(aa => (data.mutation_aa_freq[aa] || 0) / mutTotal * 100),
                    name: 'Mutant',
                    type: 'bar',
                    marker: {color: '#ef4444'}
                }
            ], {...plotLayout, barmode: 'group', xaxis: {title: 'Amino Acid'}, yaxis: {title: 'Frequency (%)'}}, plotConfig);

            // Substitution matrix
            if (data.substitution_matrix) {
                const sm = data.substitution_matrix;
                // Normalize by row
                const normalized = sm.values.map(row => {
                    const total = row.reduce((a,b) => a+b, 0);
                    return total > 0 ? row.map(v => v / total) : row;
                });

                Plotly.newPlot('abagym-sub-matrix', [{
                    x: sm.columns,
                    y: sm.index,
                    z: normalized,
                    type: 'heatmap',
                    colorscale: 'Blues',
                    showscale: true
                }], {
                    ...plotLayout,
                    xaxis: {title: 'Mutant AA', side: 'bottom'},
                    yaxis: {title: 'Wildtype AA', autorange: 'reversed'},
                    margin: {l: 60, r: 30, t: 30, b: 60}
                }, plotConfig);
            }
        }

        function initTrastuzumab() {
            const data = analysisData.trastuzumab;
            if (!data) return;

            // Stats grid
            const statsHtml = `
                <div class="stat-item"><div class="label">Total Variants</div><div class="value">${data.total_variants.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Binders</div><div class="value">${data.binders.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Non-Binders</div><div class="value">${data.non_binders.toLocaleString()}</div></div>
                <div class="stat-item"><div class="label">Binding Ratio</div><div class="value">${(data.binding_ratio * 100).toFixed(1)}%</div></div>
                <div class="stat-item"><div class="label">CDRH3 Length</div><div class="value">${data.sequence_length} aa</div></div>
                <div class="stat-item"><div class="label">Unique Sequences</div><div class="value">${data.unique_sequences.toLocaleString()}</div></div>
            `;
            document.getElementById('trastuzumab-stats').innerHTML = statsHtml;

            // Binding pie chart
            Plotly.newPlot('trastuzumab-binding-pie', [{
                labels: ['Binders', 'Non-Binders'],
                values: [data.binders, data.non_binders],
                type: 'pie',
                marker: {colors: ['#10b981', '#ef4444']},
                textinfo: 'label+percent'
            }], {...plotLayout}, plotConfig);

            // AA enrichment
            const aas = Object.keys(data.aa_enrichment).sort((a, b) => data.aa_enrichment[b] - data.aa_enrichment[a]);
            const enrichments = aas.map(aa => data.aa_enrichment[aa]);

            Plotly.newPlot('trastuzumab-enrichment', [{
                x: aas,
                y: enrichments,
                type: 'bar',
                marker: {color: enrichments.map(e => e > 0 ? '#10b981' : '#ef4444')}
            }], {
                ...plotLayout,
                xaxis: {title: 'Amino Acid'},
                yaxis: {title: 'Log2 Enrichment (Binder/Non-Binder)'},
                shapes: [{
                    type: 'line',
                    x0: -0.5,
                    x1: aas.length - 0.5,
                    y0: 0,
                    y1: 0,
                    line: {color: '#64748b', dash: 'dash'}
                }]
            }, plotConfig);

            // Position-specific heatmaps
            const positions = Array.from({length: data.sequence_length}, (_, i) => `Pos ${i+1}`);
            const aaList = 'ACDEFGHIKLMNPQRSTVWY'.split('');

            // Binder logo
            const binderMatrix = aaList.map(aa =>
                positions.map((_, pos) => data.binder_pos_freq[pos][aa] || 0)
            );

            Plotly.newPlot('trastuzumab-binder-logo', [{
                x: positions,
                y: aaList,
                z: binderMatrix,
                type: 'heatmap',
                colorscale: 'Greens',
                showscale: true
            }], {
                ...plotLayout,
                xaxis: {title: 'Position'},
                yaxis: {title: 'Amino Acid'},
                margin: {l: 50, r: 30, t: 10, b: 50}
            }, plotConfig);

            // Non-binder logo
            const nonBinderMatrix = aaList.map(aa =>
                positions.map((_, pos) => data.non_binder_pos_freq[pos][aa] || 0)
            );

            Plotly.newPlot('trastuzumab-nonbinder-logo', [{
                x: positions,
                y: aaList,
                z: nonBinderMatrix,
                type: 'heatmap',
                colorscale: 'Reds',
                showscale: true
            }], {
                ...plotLayout,
                xaxis: {title: 'Position'},
                yaxis: {title: 'Amino Acid'},
                margin: {l: 50, r: 30, t: 10, b: 50}
            }, plotConfig);
        }

        function initComparison() {
            const datasets = ['AbBiBench', 'SKEMPI2', 'AbAgym', 'Trastuzumab'];
            const samples = [
                analysisData.abbibench ? analysisData.abbibench.total_samples : 0,
                analysisData.skempi2 ? analysisData.skempi2.total_entries : 0,
                analysisData.abagym ? analysisData.abagym.total_mutations : 0,
                analysisData.trastuzumab ? analysisData.trastuzumab.total_variants : 0
            ];

            Plotly.newPlot('comparison-samples', [{
                x: datasets,
                y: samples,
                type: 'bar',
                marker: {color: ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6']},
                text: samples.map(s => s.toLocaleString()),
                textposition: 'outside'
            }], {
                ...plotLayout,
                xaxis: {title: 'Dataset'},
                yaxis: {title: 'Number of Samples', type: 'log'}
            }, plotConfig);
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
"""

    return html


def main():
    print("Loading datasets...")
    datasets = load_datasets()

    print("\nAnalyzing datasets...")
    analyses = {}

    if 'abbibench' in datasets:
        print("  Analyzing AbBiBench...")
        analyses['abbibench'] = analyze_abbibench(datasets['abbibench'])

    if 'skempi2' in datasets:
        print("  Analyzing SKEMPI2...")
        analyses['skempi2'] = analyze_skempi2(datasets['skempi2'])

    if 'abagym' in datasets:
        print("  Analyzing AbAgym...")
        meta_df = datasets.get('abagym_meta')
        analyses['abagym'] = analyze_abagym(datasets['abagym'], meta_df)

    if 'trastuzumab' in datasets:
        print("  Analyzing Trastuzumab...")
        analyses['trastuzumab'] = analyze_trastuzumab(datasets['trastuzumab'])

    print("\nGenerating HTML report...")
    html = generate_html_report(datasets, analyses)

    # Convert analyses to JSON and inject into HTML
    # Handle non-serializable types
    def convert_keys(obj):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                # Convert interval keys to strings
                new_key = str(k) if not isinstance(k, (str, int, float)) else k
                new_obj[new_key] = convert_keys(v)
            return new_obj
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    analyses_clean = convert_keys(analyses)
    analyses_json = json.dumps(analyses_clean, indent=2)
    html = html.replace('ANALYSIS_DATA_PLACEHOLDER', analyses_json)

    # Save report
    output_path = REPO_ROOT / "reports" / "antibody_datasets_report.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nReport saved to: {output_path}")
    print("Open in browser to view interactive visualizations.")


if __name__ == '__main__':
    main()
