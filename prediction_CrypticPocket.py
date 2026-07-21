#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import urllib.request
import urllib.error
import gc
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import get_surface
from scipy.spatial import cKDTree, distance_matrix
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve
from unimol_tools import UniMolRepr
from rdkit import Chem
import networkx as nx
from Bio.PDB.Polypeptide import standard_aa_names
from utils import calEval, calMass
from model import MDBind
from readData import shortest_path_matrix_from_smiles_no_hs, pad_matrix_to_size, mapSS, calc_pseudo_cb, get_chi1_angle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

# Base model and state definitions
STATES = ['Holo', 'Apo', 'AF']

# Unified color scheme
COLOR_MAP = {
    'Holo': '#E64B35',
    'Apo': '#4DBBD5',
    'AF': '#35e64b'
}

# Configuration section (paths updated)
nn_config = {
    'proj_dir': './datasets/CrypticPocket',
    'data_details': './datasets/CrypticPocket/Details.xlsx',
    'output_dir': './Results/CrypticPocket',

    'model_dir': './weights/',
    'dssp_exec': './tools/mkdssp',
    'msms_exec': './tools/msms',
    'ankh_path': './tools/ankh-large/',
    'dssp_max_repr': np.load('./tools/dssp_max_repr.npy'),
    'dssp_min_repr': np.load('./tools/dssp_min_repr.npy'),
    'ankh_max_repr': np.load('./tools/ankh_max_repr.npy'),
    'ankh_min_repr': np.load('./tools/ankh_min_repr.npy'),

    'rfeat_dim': 1556, 'ligand_dim': 512, 'hidden_dim': 256, 'heads': 4,
    'augment_eps': 0.0, 'rbf_num': 8, 'top_k': 30, 'attn_drop': 0.0,
    'dropout': 0.0, 'num_layers': 5, 'max_distance': 15, 'batch_size': 4,
}


# =========================================================================
# 2. Basic Auxiliary Functions
# =========================================================================
def set_nature_style():
    """Configure Nature-level global plotting parameters."""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['axes.grid'] = False


def extract_true_labels(label_str):
    """Extract ['12', '16'] from 'LEU:12;TYR:16'."""
    if pd.isna(label_str) or not str(label_str).strip() or str(label_str).strip().lower() == 'nan':
        return []
    return [str(item.split(':')[1]) for item in str(label_str).split(';') if ':' in item]

def collect_all_predictions():
    print("🚀 Loading Details.xlsx and filtering training set leakage samples...")
    df = pd.read_excel(nn_config['data_details'])
    if 'Is_In_Training_Set' in df.columns:
        valid_df = df[df['Is_In_Training_Set'] == False]
        print(f"✅ Filtered {len(valid_df)} strictly non-leaked targets out of {len(df)} total samples.")
    else:
        valid_df = df
        print("⚠️ 'Is_In_Training_Set' column not detected; all samples will be used for evaluation!")

    global_data = {s: {'y_true': [], 'y_prob': []} for s in STATES}

    # Sample-level AUPR data pool for boxplots/violin plots
    sample_aupr_records = []

    print("📊 Parsing MDBind prediction results...")

    for idx, row in valid_df.iterrows():
        holo_pid = str(row['Holo PDB ID']).strip().lower()
        apo_pid = str(row['Apo PDB ID']).strip().lower()
        lig_code = str(row['ligand code']).strip().upper()
        uniprot_id = str(row['Uniprot ID']).strip() if pd.notna(row['Uniprot ID']) else None

        holo_cid = str(row['Holo chain ID']).strip()
        apo_cid = str(row['Apo chain ID']).strip()
        af_cid = 'A'

        # Extract true labels for the three states of the current sample
        gt_dict = {
            'Holo': extract_true_labels(row.get('Holo Binding Sites', '')),
            'Apo': extract_true_labels(row.get('Apo Binding Sites', '')),
            'AF': extract_true_labels(row.get('AF Binding Sites', ''))
        }

        for state in STATES:
            true_ids = gt_dict[state]
            if not true_ids: continue

            # ---------------- Determine Entity ID for Current State ----------------
            if state == 'Holo':
                target_pid, target_cid = holo_pid, holo_cid
            elif state == 'Apo':
                target_pid, target_cid = apo_pid, apo_cid
            elif state == 'AF':
                if not uniprot_id: continue
                target_pid, target_cid = uniprot_id, af_cid

            y_true_sample = []
            y_prob_sample = []
            csv_path = os.path.join(nn_config['output_dir'], f"{state}_raw_preds", f"{target_pid}_{lig_code}.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join(nn_config['output_dir'], f"{state}_raw_preds", f"{holo_pid}_{lig_code}.csv")

            if os.path.exists(csv_path):
                try:
                    df_pred = pd.read_csv(csv_path)
                    if 'residue_label' in df_pred.columns:
                        res_labels = df_pred['residue_label'].astype(str).str.replace(r'\.0$', '',
                                                                                      regex=True).values
                        y_true_sample = [1 if rl in true_ids else 0 for rl in res_labels]
                        y_prob_sample = df_pred['EarlyFusion_Predicted_Prob'].values
                    else:
                        y_true_sample = df_pred['True_Label'].values
                        y_prob_sample = df_pred['EarlyFusion_Predicted_Prob'].values
                except Exception:
                    pass

            # ---------------- Aggregate Sample Data ----------------
            if len(y_true_sample) > 0 and len(y_prob_sample) > 0 and sum(y_true_sample) > 0:
                y_true_arr = np.array(y_true_sample)
                y_prob_arr = np.array(y_prob_sample)

                global_data[state]['y_true'].extend(y_true_arr)
                global_data[state]['y_prob'].extend(y_prob_arr)

                aupr = average_precision_score(y_true_arr, y_prob_arr)
                sample_aupr_records.append({
                    'Sample': f"{holo_pid}_{lig_code}",
                    'State': state,
                    'Method': 'MDBind',
                    'AUPR': aupr
                })

    print("✅ Data collection completed! Preparing plots...")
    df_sample = pd.DataFrame(sample_aupr_records)
    return global_data, df_sample


# =========================================================================
# 4. Visualization Chart 1: Single-Panel Global PR Curve Combined Plot
# =========================================================================
def plot_global_pr_curves_single(global_data):
    set_nature_style()
    fig, ax_pr = plt.subplots(1, 1, figsize=(4.8, 4.2))

    # Configure highly distinguishable line styles, colors, and markers
    styles = {
        'Holo': {'color': '#E64B35', 'linestyle': '-', 'lw': 2.0, 'marker': '', 'label': 'Holo'},
        'Apo': {'color': '#E64B35', 'linestyle': '--', 'lw': 1.8, 'marker': 'v', 'label': 'Apo'},
        'AF': {'color': '#E64B35', 'linestyle': '-.', 'lw': 1.8, 'marker': 'o', 'label': 'AF'},
    }

    print("\n" + "=" * 50)
    print("📊 Global AUPR Performance Report for Each Model")
    print("=" * 50)

    for state in STATES:
        y_true = np.array(global_data[state]['y_true'])
        y_prob = np.array(global_data[state]['y_prob'])

        if len(y_true) == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob)

        key = f"{state}"
        style = styles[key]

        print(f"{key:<15}: {aupr:.4f}")

        ax_pr.plot(recall, precision,
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=style['lw'],
                   marker=style['marker'],
                   markersize=4,
                   markevery=0.1,  # Sparse markers on curves to avoid clutter
                   alpha=0.9,
                   label=f"{style['label']} (AUPR={aupr:.3f})")

    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.set_xlabel('Recall', weight='bold')
    ax_pr.set_ylabel('Precision', weight='bold')
    ax_pr.set_title('Precision-Recall Curve', weight='bold')

    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    png_path = os.path.join(nn_config['output_dir'], 'MDBind_PR_curve.png')
    pdf_path = os.path.join(nn_config['output_dir'], 'MDBind_PR_curve.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print("=" * 50)
    print(f"🎉 Single-panel global PR curve plot saved to:\n  👉 {png_path}")

# =========================================================================
# 5. Visualization Chart 2: Three-Panel Sample-Level AUPR Violin/Box Plot
# =========================================================================
def plot_sample_aupr_distributions(df_sample):
    if df_sample.empty:
        print("⚠️ Sample-level data is empty; skipping AUPR distribution plot.")
        return

    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharey=True)

    for i, state in enumerate(STATES):
        ax = axes[i]
        df_sub = df_sample[df_sample['State'] == state]

        if df_sub.empty:
            continue

        sns.violinplot(
            data=df_sub, x='State', y='AUPR', ax=ax,
            palette=COLOR_MAP, inner=None, linewidth=0, alpha=0.3
        )

        sns.boxplot(
            data=df_sub, x='State', y='AUPR', ax=ax,
            palette=COLOR_MAP, width=0.3, boxprops={'zorder': 2, 'alpha': 0.8},
            whiskerprops={'linewidth': 1.2}, capprops={'linewidth': 1.2},
            medianprops={'color': 'white', 'linewidth': 1.5}, showfliers=False
        )

        sns.stripplot(
            data=df_sub, x='State', y='AUPR', ax=ax,
            color='black', size=2.5, alpha=0.4, jitter=True, zorder=3
        )

        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('')

        ax.set_title(f'{state} Conformation', weight='bold')

        if i == 0:
            ax.set_ylabel('Sample-level AUPR', weight='bold')
        else:
            ax.set_ylabel('')

        ax.set_xticklabels([])

    plt.tight_layout()
    png_path = os.path.join(nn_config['output_dir'], 'MDBind_boxplot.png')
    pdf_path = os.path.join(nn_config['output_dir'], 'MDBind_boxplot.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"🎉 Sample-level AUPR distribution plot (Violin + Boxplot) saved to:\n  👉 {png_path}")

#_____________________________________________________________________________________

class SuppressOutput:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        os.close(self.null_fd)
        os.close(self.save_stdout)
        os.close(self.save_stderr)


# =========================================================================
# Configuration
# =========================================================================
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

D3TO1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def get_one_letter_code(resname):
    return D3TO1.get(resname.upper(), 'X')


def is_standard_aa(res):
    return res.get_resname() in standard_aa_names

# =========================================================================
# Feature Extraction Helpers
# =========================================================================
def get_smiles_dict(smiles_file):
    smiles_dict = {}
    if os.path.exists(smiles_file):
        with open(smiles_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    smiles_dict[parts[0]] = parts[1]
    return smiles_dict


def process_pdb_chain_features(pdb_file, target_chain_id, dssp_exec, msms_exec):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", pdb_file)
    model = structure[0]

    try:
        dssp = DSSP(model, pdb_file, dssp=dssp_exec)
        dssp_keys = set(dssp.keys())
    except Exception:
        dssp = {}
        dssp_keys = set()

    dssp_res, xyz_res, ca_list, cb_list, angles_list, res_ids, res_names = [], [], [], [], [], [], []
    chain_atom = ['N', 'CA', 'C', 'O']

    target_chain = model[target_chain_id]

    try:
        with SuppressOutput():
            surf = get_surface(target_chain, MSMS=msms_exec)
        surf_tree = cKDTree(surf) if surf is not None and len(surf) > 0 else None
    except Exception:
        surf = np.empty(0)
        surf_tree = None

    residues = sorted([r for r in target_chain if r.id[0] == " "], key=lambda r: r.id[1])

    for residue in residues:
        res_ids.append(residue.id[1])
        res_names.append(residue.get_resname())

        res_key = (target_chain.id, (' ', residue.id[1], residue.id[2]))
        if res_key in dssp_keys:
            tuple_dssp = dssp[res_key]
            ss_vec = mapSS.get(tuple_dssp[2], mapSS[' '])
            other_vals = [float(x) if x != "NA" else 0.0 for x in tuple_dssp[3:]]
            dssp_res.append(ss_vec + other_vals)
        else:
            dssp_res.append(np.zeros(20, dtype=float))

        line = []
        atoms = list(residue.get_atoms())
        atoms_coord = np.array([atom.get_coord() for atom in atoms]) if len(atoms) > 0 else np.empty((0, 3))

        if surf.size != 0 and surf_tree is not None and atoms_coord.size > 0:
            dist, _ = surf_tree.query(atoms_coord)
            closest_atom = int(np.argmin(dist))
            closest_pos = atoms_coord[closest_atom]
        else:
            closest_pos = atoms_coord[-1] if atoms_coord.size > 0 else np.zeros(3)

        ca_pos = residue['CA'].get_coord() if 'CA' in residue else np.zeros(3)
        pos_s, un_s = np.zeros(3), 0.0

        for atom in atoms:
            if atom.name in chain_atom:
                line.append(atom.get_coord())
            else:
                pos_s += calMass(atom, True)
                un_s += calMass(atom, False)

        if len(line) != 4:
            line = line + [list(ca_pos)] * (4 - len(line))
        R_pos = pos_s / un_s if un_s != 0 else ca_pos
        line.append(R_pos)
        line.append(closest_pos)
        xyz_res.append(line)

        ca_list.append(ca_pos)
        if 'CB' in residue:
            cb_coord = residue['CB'].get_coord()
        elif all(atom in residue for atom in ['N', 'CA', 'C']):
            cb_coord = calc_pseudo_cb(residue['N'].get_coord(), residue['CA'].get_coord(), residue['C'].get_coord())
        else:
            cb_coord = ca_pos
        cb_list.append(cb_coord)

        chi1 = get_chi1_angle(residue)
        angles_list.append([math.sin(chi1), math.cos(chi1)] if chi1 is not None else [0.0, 0.0])

    dssp_arr = np.array(dssp_res, dtype=np.float64)
    xyz_arr = np.array(xyz_res, dtype=np.float64)
    angles_arr = np.array(angles_list, dtype=np.float32)

    ca_coords = np.array(ca_list, dtype=np.float32)
    cb_coords = np.array(cb_list, dtype=np.float32)
    ca_dist = distance_matrix(ca_coords, ca_coords)
    cb_dist = distance_matrix(cb_coords, cb_coords)
    cmap_arr = np.stack([ca_dist, cb_dist], axis=-1).astype(np.float32)

    seq = "".join([D3TO1.get(res.resname, 'X') for res in residues])

    return dssp_arr, xyz_arr, cmap_arr, angles_arr, res_ids, res_names, seq


# =========================================================================
# Custom Dataset for PocketMiner Data
# =========================================================================
class PocketMinerDataset(Dataset):
    def __init__(self, t_type, details_df, config, ankh_model, tokenizer, unimol_clf, smiles_dict):
        self.t_type = t_type
        self.details_df = details_df
        self.config = config
        self.ankh_model = ankh_model
        self.tokenizer = tokenizer
        self.unimol_clf = unimol_clf
        self.smiles_dict = smiles_dict

    def __len__(self):
        return len(self.details_df)

    def normalize(self, arr, max_value, min_value):
        max_v = np.array(max_value, dtype=float)
        min_v = np.array(min_value, dtype=float)
        scalar = max_v - min_v
        scalar[scalar == 0] = 1.0
        return (arr - min_v) / scalar

    def _extract_labels(self, label_str):
        if pd.isna(label_str) or not str(label_str).strip() or str(label_str).strip().lower() == 'nan':
            return []
        return [int(item.split(':')[1]) for item in str(label_str).split(';') if ':' in item]

    def __getitem__(self, idx):
        row = self.details_df.iloc[idx]
        holo_pdb_id = str(row['Holo PDB ID']).strip().lower()
        ligand_code = str(row['ligand code']).strip().upper()

        try:
            # ---------------- Parse Physical Paths and Precise Labels from New Excel ----------------
            if self.t_type == 'Holo':
                target_pdb_id = holo_pdb_id
                target_chain_id = str(row['Holo chain ID']).strip()
                pdb_file = os.path.join(self.config['proj_dir'], holo_pdb_id,
                                        f"{target_pdb_id}_{target_chain_id}_chain.pdb")
                true_res_ids = self._extract_labels(row.get('Holo Binding Sites', ''))

            elif self.t_type == 'Apo':
                target_pdb_id = str(row['Apo PDB ID']).strip().lower()
                target_chain_id = str(row['Apo chain ID']).strip()
                pdb_file = os.path.join(self.config['proj_dir'], holo_pdb_id,
                                        f"{target_pdb_id}_{target_chain_id}_chain.pdb")
                true_res_ids = self._extract_labels(row.get('Apo Binding Sites', ''))

            elif self.t_type == 'AF':
                if pd.isna(row['Uniprot ID']):
                    raise ValueError(f"Skipping: {holo_pdb_id} lacks Uniprot ID")
                uniprot_id = str(row['Uniprot ID']).strip()
                target_pdb_id = uniprot_id

                af_target_dir = os.path.join(self.config['proj_dir'], holo_pdb_id)
                pdb_file = os.path.join(af_target_dir, f"AF-{uniprot_id}-F1-model_v6.pdb")
                if not os.path.exists(pdb_file):
                    raise ValueError(f"Skipping: AlphaFold file not found {pdb_file}")

                af_struct = PDBParser(QUIET=True).get_structure("af", pdb_file)
                chains = list(af_struct[0].get_chains())
                if not chains:
                    raise ValueError(f"Skipping: No chains found in AF")
                target_chain_id = chains[0].id

                # Directly read pre-mapped AF labels without re-alignment!
                true_res_ids = self._extract_labels(row.get('AF Binding Sites', ''))

            # ---------------- Extract Protein Structure and Features ----------------
            dssp_arr, xyz, cmap, angles, res_ids, res_names, seq = process_pdb_chain_features(
                pdb_file, target_chain_id, self.config['dssp_exec'], self.config['msms_exec']
            )

            y_true_arr = np.zeros(len(res_ids), dtype=int)
            for i, r_id in enumerate(res_ids):
                if r_id in true_res_ids:
                    y_true_arr[i] = 1

            ids = self.tokenizer.batch_encode_plus([list(seq)], add_special_tokens=True, padding=True,
                                                   is_split_into_words=True, return_tensors="pt")
            input_ids = ids['input_ids'].to(DEVICE)
            attention_mask = ids['attention_mask'].to(DEVICE)
            with torch.no_grad():
                embedding_repr = self.ankh_model(input_ids=input_ids, attention_mask=attention_mask)
                ankh_arr = embedding_repr.last_hidden_state[0, :len(seq)].cpu().numpy()

            dssp = self.normalize(dssp_arr, self.config['dssp_max_repr'], self.config['dssp_min_repr'])
            ankh = self.normalize(ankh_arr, self.config['ankh_max_repr'], self.config['ankh_min_repr'])

            min_len = min(dssp.shape[0], ankh.shape[0])
            feature = np.concatenate([dssp[:min_len], ankh[:min_len]], axis=1)
            xyz = xyz[:min_len]
            angles = angles[:min_len]
            cmap = cmap[:min_len, :min_len, :]
            y_true_arr = y_true_arr[:min_len]
            res_ids = res_ids[:min_len]
            res_names = res_names[:min_len]

            # ---------------- Extract Ligand UniMol Features ----------------
            smiles = self.smiles_dict.get(ligand_code, None)
            if not smiles or smiles == "NOT_FOUND_IN_PAGE" or smiles == "SMILES_NOT_FOUND":
                raise ValueError(f"SMILES not found for {ligand_code}")

            unimol_repr = self.unimol_clf.get_repr([smiles], return_atomic_reprs=True)
            atomic_reprs = unimol_repr.get('atomic_reprs', None) if isinstance(unimol_repr, dict) else unimol_repr[
                0].get('atomic_reprs', None)
            ligand_arr = np.array(atomic_reprs)
            if ligand_arr.ndim == 3 and ligand_arr.shape[0] == 1: ligand_arr = ligand_arr[0]
            ligand = ligand_arr[:-1]

            return target_pdb_id, ligand_code, target_chain_id, res_ids, res_names, feature, angles, ligand, xyz, cmap, y_true_arr

        except Exception as e:
            print(f"Error processing {holo_pdb_id} (Type: {self.t_type}): {e}")
            return self.__getitem__((idx + 1) % len(self))

    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return torch.tensor(padded, dtype=torch.float)

    def collate_fn(self, batch):
        pdbids, ligids, chain_ids, res_ids_list, res_names_list, features, angles, ligands, xyzs, cmaps, y_trues = zip(
            *batch)

        maxprotlen = max([f.shape[0] for f in features])
        maxliglen = max([f.shape[0] for f in ligands])

        batch_feat, batch_angle, batch_xyz, batch_mask, batch_cmap = [], [], [], [], []
        batch_ligand, batch_lig_mask, batch_lig_node_paths = [], [], []
        batch_y_true = []

        for idx in range(len(batch)):
            batch_feat.append(self._padding(features[idx], maxprotlen))
            batch_angle.append(self._padding(angles[idx], maxprotlen))
            batch_xyz.append(self._padding(xyzs[idx], maxprotlen))

            cmap_arr = cmaps[idx]
            N = cmap_arr.shape[0]
            padded_cmap = np.zeros((maxprotlen, maxprotlen, cmap_arr.shape[-1]), dtype=np.float32)
            padded_cmap[:N, :N, :] = cmap_arr
            batch_cmap.append(torch.tensor(padded_cmap, dtype=torch.float))

            mask = np.zeros(maxprotlen, dtype=np.int64)
            mask[: features[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype=torch.long))

            smiles = self.smiles_dict[ligids[idx]]
            batch_ligand.append(self._padding(ligands[idx], maxliglen))
            node_paths = shortest_path_matrix_from_smiles_no_hs(smiles, max_distance=self.config['max_distance'])
            node_paths_padded = pad_matrix_to_size(node_paths, maxliglen, pad_value=0)
            batch_lig_node_paths.append(torch.tensor(node_paths_padded, dtype=torch.long))

            lig_mask = np.zeros(maxliglen, dtype=bool)
            lig_mask[:ligands[idx].shape[0]] = True
            batch_lig_mask.append(torch.tensor(lig_mask, dtype=torch.bool))

            pad_y = np.zeros(maxprotlen, dtype=np.float32)
            pad_y[: y_trues[idx].shape[0]] = y_trues[idx]
            batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))

        return (
            pdbids, ligids, chain_ids, res_ids_list, res_names_list,
            torch.stack(batch_feat), torch.stack(batch_xyz), torch.stack(batch_mask),
            torch.stack(batch_angle), torch.stack(batch_cmap),
            torch.stack(batch_ligand), torch.stack(batch_lig_node_paths), torch.stack(batch_lig_mask),
            torch.stack(batch_y_true),
        )


# =========================================================================
# Load Models & Test Logic
# =========================================================================
def load_models_for_ensemble(model_path, num_folds=3):
    models = []
    for fold in range(num_folds):
        ckpt = os.path.join(model_path, f'fold{fold}.ckpt')
        if not os.path.exists(ckpt): continue
        state_dict = torch.load(ckpt, map_location=DEVICE)
        model = MDBind(
            rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'],
            hidden_dim=nn_config['hidden_dim'], heads=nn_config['heads'],
            augment_eps=nn_config['augment_eps'], rbf_num=nn_config['rbf_num'],
            top_k=nn_config['top_k'], attn_drop=nn_config['attn_drop'],
            dropout=nn_config['dropout'], num_layers=nn_config['num_layers']
        ).to(DEVICE)

        if any(k.startswith('module.') for k in state_dict.keys()):
            model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.module
        else:
            model.load_state_dict(state_dict)

        model.eval()
        models.append(model)
    return models


def test_evaluate_with_pdb_alignment(models, data_loader, best_th=0.35, save_raw_preds=False, raw_save_dir=None):
    results = []
    all_y_true_global, all_y_score_early = [], []

    if save_raw_preds and raw_save_dir is not None:
        os.makedirs(raw_save_dir, exist_ok=True)

    with torch.no_grad():
        for pdbids, ligids, chain_ids, res_ids_list, res_names_list, rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true in tqdm(
                data_loader, desc='Testing & Aligning', unit='batch'):

            tensors = [t.to(DEVICE) for t in
                       [rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true]]
            rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true = tensors

            evidences = [m(rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask) for m in models]

            # EARLY FUSION
            avg_evidence = torch.stack(evidences, 0).mean(0)
            alpha = avg_evidence + 1.0
            S = torch.sum(alpha, dim=-1, keepdim=True)
            early_prob_binding = alpha[..., 1] / S.squeeze(-1)
            early_uncertainty = 2.0 / S.squeeze(-1)

            mask_cpu = mask.cpu().numpy()
            y_true_cpu = y_true.cpu().numpy()
            early_prob_cpu = early_prob_binding.cpu().numpy()
            early_uncert_cpu = early_uncertainty.cpu().numpy()

            batch_size = y_true.shape[0]
            for i in range(batch_size):
                sample_mask = mask_cpu[i] == 1
                y_true_np = y_true_cpu[i][sample_mask]
                early_score_np = early_prob_cpu[i][sample_mask]
                early_uncert_np = early_uncert_cpu[i][sample_mask]

                cur_res_ids = res_ids_list[i]
                cur_res_names = res_names_list[i]
                cur_chain = chain_ids[i]

                all_y_true_global.extend(y_true_np)
                all_y_score_early.extend(early_score_np)

                if save_raw_preds and raw_save_dir is not None:
                    df_dict = {
                        'pdbid': [pdbids[i]] * len(y_true_np),
                        'ligid': [ligids[i]] * len(y_true_np),
                        'chain': [cur_chain] * len(y_true_np),
                        'residue_label': cur_res_ids,
                        'residue_name': cur_res_names,
                        'True_Label': y_true_np,
                        'EarlyFusion_Predicted_Prob': early_score_np,
                        'EarlyFusion_Uncertainty': early_uncert_np
                    }
                    df_pdb = pd.DataFrame(df_dict)
                    df_pdb.to_csv(os.path.join(raw_save_dir, f"{pdbids[i]}_{ligids[i]}.csv"), index=False)

                if len(y_true_np) > 0:
                    data_early = calEval(y_true_np, early_score_np, best_th=best_th)
                    data_early['pdbid'] = pdbids[i]
                    data_early['ligid'] = ligids[i]
                    results.append(data_early)

    df = pd.DataFrame(results)
    if len(all_y_true_global) > 0:
        overall_early = calEval(all_y_true_global, all_y_score_early, best_th=best_th)
        overall_early.update({'pdbid': 'OVERALL', 'ligid': 'OVERALL'})
        df = pd.concat([df, pd.DataFrame([overall_early])], ignore_index=True)

    return df, all_y_true_global, all_y_score_early


# =========================================================================
# Main Execution Pipeline
# =========================================================================
def main():
    print("1. Loading large model for feature extraction preparation...")
    tokenizer = AutoTokenizer.from_pretrained(nn_config['ankh_path'])
    ankh_model = T5EncoderModel.from_pretrained(nn_config['ankh_path']).to(DEVICE)
    ankh_model.eval()
    unimol_clf = UniMolRepr(data_type='molecule', remove_hs=True, model_name='unimolv1', model_size='164')

    smiles_dict = get_smiles_dict(os.path.join(nn_config['proj_dir'], "ligand_smiles.txt"))

    print("\n2. Loading trained ensemble weights...")
    models = load_models_for_ensemble(nn_config['model_dir'], num_folds=3)
    if not models:
        raise RuntimeError(f"Model weights not found in {nn_config['model_dir']}; please check the path.")

    print("\n3. Reading threshold calculated from validation set...")
    threshold_file = os.path.join(nn_config['model_dir'], 'MD_Best_Threshold.txt')
    best_threshold = 0.35
    if os.path.exists(threshold_file):
        with open(threshold_file, 'r') as f:
            best_threshold = float(f.read().strip().split()[0])

    # Build output root directory
    output_dir = nn_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    details_excel = nn_config['data_details']
    if not os.path.exists(details_excel):
        print(f"Warning: Updated Excel file not found: {details_excel}")
        return

    details_df = pd.read_excel(details_excel)
    test_types = ['Holo', 'Apo', 'AF']

    for t_type in test_types:
        print(f"\n========== Starting testing for {t_type} dataset ==========")
        dataset = PocketMinerDataset(t_type, details_df, nn_config, ankh_model, tokenizer, unimol_clf, smiles_dict)
        dataloader = DataLoader(dataset, batch_size=nn_config['batch_size'], collate_fn=dataset.collate_fn,
                                num_workers=0)

        raw_save_dir = os.path.join(output_dir, f'{t_type}_raw_preds')

        results_df, _, _ = test_evaluate_with_pdb_alignment(
            models=models,
            data_loader=dataloader,
            best_th=best_threshold,
            save_raw_preds=True,
            raw_save_dir=raw_save_dir
        )

        metrics_csv = os.path.join(output_dir, f'{t_type}_Metrics.csv')
        results_df.to_csv(metrics_csv, index=False)
        print(f"{t_type} evaluation completed, metrics saved to {metrics_csv}")

    print(f"\n🎉 All tasks executed successfully! Results saved in {output_dir}")


if __name__ == '__main__':
    main()
    global_data, df_sample = collect_all_predictions()
    plot_global_pr_curves_single(global_data)
    plot_sample_aupr_distributions(df_sample)