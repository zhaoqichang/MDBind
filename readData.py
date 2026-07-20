import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
import os
import random
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch.utils.data import Dataset
from Bio.PDB.vectors import calc_dihedral

mapSS = {
    ' ': [0,0,0,0,0,0,0,0,0],
    '-': [1,0,0,0,0,0,0,0,0],
    'H': [0,1,0,0,0,0,0,0,0],
    'B': [0,0,1,0,0,0,0,0,0],
    'E': [0,0,0,1,0,0,0,0,0],
    'G': [0,0,0,0,1,0,0,0,0],
    'I': [0,0,0,0,0,1,0,0,0],
    'P': [0,0,0,0,0,0,1,0,0],
    'T': [0,0,0,0,0,0,0,1,0],
    'S': [0,0,0,0,0,0,0,0,1]
}

# helper functions
def safe_load(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None

def calc_pseudo_cb(n_coord, ca_coord, c_coord):
    v_n = n_coord - ca_coord
    v_c = c_coord - ca_coord
    v_n = v_n / (np.linalg.norm(v_n) + 1e-8)
    v_c = v_c / (np.linalg.norm(v_c) + 1e-8)
    bisector = v_n + v_c
    bisector = bisector / (np.linalg.norm(bisector) + 1e-8)
    perp = np.cross(v_c, v_n)
    perp = perp / (np.linalg.norm(perp) + 1e-8)
    vec = -bisector * np.sqrt(1 / 3) - perp * np.sqrt(2 / 3)
    return ca_coord + vec * 1.522


def get_chi1_angle(residue):
    cg_names = ['CG', 'CG1', 'SG', 'OG', 'OG1']
    try:
        n = residue['N'].get_vector()
        ca = residue['CA'].get_vector()
        cb = residue['CB'].get_vector()
        cg = next((residue[name].get_vector() for name in cg_names if name in residue), None)
        if cg is None: return None
        return calc_dihedral(n, ca, cb, cg)
    except KeyError:
        return None

def shortest_path_matrix_from_smiles_no_hs(smiles: str, max_distance: int = None) -> np.ndarray:
    """
    Calculate the shortest path matrix (number of bonds) from hydrogen-removed SMILES.
    Unreachable or padding values are represented by 0.
    Returns a numpy array of shape (N, N).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    # Ensure hydrogens are removed
    mol = Chem.RemoveHs(mol)
    N = mol.GetNumAtoms()
    if N == 0:
        return np.zeros((0, 0), dtype=np.int64)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        G.add_edge(a1, a2)

    mat = np.zeros((N, N), dtype=np.int64)  # Default 0 means no path / padding
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=max_distance)
        for j, l in lengths.items():
            # lengths includes i->i with length 0
            mat[i, j] = l
    # Diagonal is already 0
    return mat

def pad_matrix_to_size(mat: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """
    Pad the square matrix mat to (size, size), filling the bottom-right corner with pad_value.
    """
    N = mat.shape[0]
    if N == size:
        return mat
    if N > size:
        # Truncate if the actual number of atoms exceeds maxliglen in the batch (should usually not happen)
        return mat[:size, :size]
    pad0 = size - N
    return np.pad(mat, ((0, pad0), (0, pad0)), mode='constant', constant_values=pad_value)

def parse_index_list(s):
    """Parse a string like "77 83 86" into integer indices list.
       Return empty list on failure or empty string.
    """
    if s is None:
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []
    parts = s.split()
    idxs = []
    for p in parts:
        try:
            v = int(p)
            if v > 0:
                idxs.append(v)
        except Exception:
            continue
    return idxs

def build_label_from_indices(length, indices):
    """Build binary label array of given length with ones at indices (0-based)."""
    lab = np.zeros(length, dtype=np.int64)
    for i in indices:
        if 0 <= i < length:
            lab[i] = 1
    return lab

class MDBindDataset(Dataset):
    """
    MDBind Dataset: Dynamically reads pre-extracted features (.npy).
    Samples MD conformations based on the 'all_lengths_match' flag and the specified 'md_prob' probability.
    If MD feature files are corrupted or dimensions mismatch, it automatically falls back to Holo conformation features.
    """

    def __init__(self, dataset_split: str, summary_xlsx: str, feat_dir: str,
                 lig_dict: dict, smiles_dict: dict, nn_config: dict,
                 md_frame_count: int = 100, seed: int = 42):

        self.df = pd.read_excel(summary_xlsx)
        self.feat_dir = feat_dir
        self.lig_dict = lig_dict
        self.smiles_dict = smiles_dict
        self.nn_config = nn_config
        self.md_prob = nn_config.get('md_prob', 0.5)
        self.md_frame_count = md_frame_count
        self.max_distance = nn_config.get('max_distance', 15)

        self.name_list = []
        for _, row in self.df.iterrows():
            pdb = str(row.get("PDB_ID", "")).strip()
            lig = str(row.get("Ligand", "")).strip()
            if not pdb or not lig or lig not in self.lig_dict:
                continue
            if str(row.get("Dataset", "")).strip() == dataset_split:
                self.name_list.append((pdb, lig))

        random.seed(seed)

    def __len__(self):
        return len(self.name_list)

    def _is_dimension_valid(self, a, d, x, c, ang, s_len):
        if any(v is None for v in [a, d, x, c, ang]): return False
        return (a.shape[0] == d.shape[0] == x.shape[0] == c.shape[0] == c.shape[1] == ang.shape[0] == s_len)

    def _load_conformation(self, pdbid, is_md=False, frame_idx=0):
        pdb_feat_root = os.path.join(self.feat_dir, pdbid)
        if is_md:
            ankh = safe_load(os.path.join(pdb_feat_root, "ankh", "frames_ankh.npy"))
            dssp = safe_load(os.path.join(pdb_feat_root, "dssp", f"frame{frame_idx}_dssp.npy"))
            pos = safe_load(os.path.join(pdb_feat_root, "pos", f"frame{frame_idx}_pos.npy"))
            cmap = safe_load(os.path.join(pdb_feat_root, "cmap", f"frame{frame_idx}_cmap.npy"))
            ang = safe_load(os.path.join(pdb_feat_root, "angle", f"frame{frame_idx}_angles.npy"))
        else:
            ankh = safe_load(os.path.join(pdb_feat_root, "ankh", f"{pdbid}_ankh.npy"))
            dssp = safe_load(os.path.join(pdb_feat_root, "dssp", f"{pdbid}_dssp.npy"))
            pos = safe_load(os.path.join(pdb_feat_root, "pos", f"{pdbid}_pos.npy"))
            cmap = safe_load(os.path.join(pdb_feat_root, "cmap", f"{pdbid}_cmap.npy"))
            ang = safe_load(os.path.join(pdb_feat_root, "angle", f"{pdbid}_angles.npy"))
        return ankh, dssp, pos, cmap, ang

    def __getitem__(self, idx):
        try:
            pdbid, lig = self.name_list[idx]
            row = self.df[self.df["PDB_ID"].astype(str).str.strip() == pdbid].iloc[0]

            all_match = str(row.get("all_lengths_match", "False")).strip().lower() in ["1", "true", "t", "yes", "y"]
            use_md = all_match and (random.random() < self.md_prob)

            seq_len = len(str(row.get("protein_seq", "")))
            binding_str = row.get("binding_indices_protein", "")

            if use_md:
                frame_idx = random.randint(0, max(0, self.md_frame_count - 1))
                ankh_arr, dssp_arr, xyz, cmap, angles = self._load_conformation(pdbid, is_md=True, frame_idx=frame_idx)
                if not self._is_dimension_valid(ankh_arr, dssp_arr, xyz, cmap, angles, seq_len):
                    # MD read failure or dimension mismatch -> silently fallback to Holo features
                    ankh_arr, dssp_arr, xyz, cmap, angles = self._load_conformation(pdbid, is_md=False)
            else:
                ankh_arr, dssp_arr, xyz, cmap, angles = self._load_conformation(pdbid, is_md=False)

            if not self._is_dimension_valid(ankh_arr, dssp_arr, xyz, cmap, angles, seq_len):
                raise ValueError(f"Holo feature corruption for {pdbid}")

            ligand = self.lig_dict[lig]["atomic_reprs"]
            y_true_arr = np.asarray(build_label_from_indices(seq_len, parse_index_list(binding_str)), dtype=int)

            dssp = self.Normalize(dssp_arr, self.nn_config['dssp_max_repr'], self.nn_config['dssp_min_repr'])
            ankh = self.Normalize(ankh_arr, self.nn_config['ankh_max_repr'], self.nn_config['ankh_min_repr'])
            feature = np.concatenate([dssp, ankh], axis=1)

            return pdbid, lig, feature, angles, ligand, xyz, cmap, y_true_arr

        except Exception as e:
            # Silently fetch the next record when an exception occurs to prevent DataLoader interruption
            return self.__getitem__((idx + 1) % len(self))

    def Normalize(self, arr, max_value, min_value):
        max_v = np.array(max_value, dtype=float)
        min_v = np.array(min_value, dtype=float)
        scalar = max_v - min_v
        scalar[scalar == 0] = 1.0
        return (arr - min_v) / scalar

    def collate_fn(self, batch):
        pdbids, ligids, features, angles, ligands, xyzs, cmaps, y_trues = zip(*batch)

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

            node_paths = shortest_path_matrix_from_smiles_no_hs(smiles, max_distance=self.max_distance)
            node_paths_padded = pad_matrix_to_size(node_paths, maxliglen, pad_value=0)
            batch_lig_node_paths.append(torch.tensor(node_paths_padded, dtype=torch.long))

            lig_mask = np.zeros(maxliglen, dtype=bool)
            lig_mask[:ligands[idx].shape[0]] = True
            batch_lig_mask.append(torch.tensor(lig_mask, dtype=torch.bool))

            pad_y = np.zeros(maxprotlen, dtype=np.float32)
            pad_y[: y_trues[idx].shape[0]] = y_trues[idx]
            batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))

        return (
            pdbids, ligids,
            torch.stack(batch_feat),
            torch.stack(batch_xyz),
            torch.stack(batch_mask),
            torch.stack(batch_angle),
            torch.stack(batch_cmap),
            torch.stack(batch_ligand),
            torch.stack(batch_lig_node_paths),
            torch.stack(batch_lig_mask),
            torch.stack(batch_y_true),
        )

    def _padding(self, arr, maxlen):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return torch.tensor(padded, dtype=torch.float)