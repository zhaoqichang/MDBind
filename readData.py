import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import RDLogger
# 关闭所有 rdApp 相关的日志输出
RDLogger.DisableLog('rdApp.*')
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from rdkit import Chem
import networkx as nx
# from config import nn_config
# data_class = nn_config['pdb_class']

import numpy as np
import torch
from rdkit import Chem
import networkx as nx
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import get_surface
from scipy.spatial import cKDTree
from utils import calMass

import os
import sys

class SuppressOutput:
    """上下文管理器，在操作系统级别屏蔽 C/C++ 子进程的输出"""
    def __enter__(self):
        # 打开黑洞设备
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        # 保存原本的 stdout (编号 1) 和 stderr (编号 2)
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        # 将 1 和 2 重定向到黑洞
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        # 恢复 stdout 和 stderr
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        # 清理文件描述符
        os.close(self.null_fd)
        os.close(self.save_stdout)
        os.close(self.save_stderr)

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
def get_chain_order(model):
    """
    Return list of chain objects sorted by their minimum residue sequence number.
    This ensures chain order follows residue order across chains.
    """
    chain_min = []
    for chain in model.get_chains():
        resnums = [res.id[1] for res in chain if res.id[0] == " "]
        mn = min(resnums) if resnums else float('inf')
        chain_min.append((chain, mn))
    chain_min.sort(key=lambda x: x[1])
    return [c for c, _ in chain_min]

def process_dssp_file(pdb_file, dssp_exec = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/mkdssp"):
    """
    Process a single PDB file with DSSP and save per-residue features to save_file (.npy).
    Chain order and residue order are enforced.
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("tmp", pdb_file)
        model = structure[0]
        try:
            dssp = DSSP(model, pdb_file, dssp=dssp_exec)
            keys = set(dssp.keys())
        except Exception:
            keys = set()
        res_np = []
        for chain in get_chain_order(model):
            residues = sorted([r for r in chain if r.id[0] == " "], key=lambda r: r.id[1])
            for residue in residues:
                res_key = (chain.id, (' ', residue.id[1], residue.id[2]))
                if res_key in keys:
                    tuple_dssp = dssp[res_key]
                    # tuple_dssp[2] is secondary structure letter
                    ss_vec = mapSS.get(tuple_dssp[2], mapSS[' '])
                    other_vals = [float(x) if x != "NA" else 0.0 for x in tuple_dssp[3:]]
                    res_np.append(ss_vec + other_vals)
                else:
                    res_np.append(np.zeros(20, dtype=float))
        return np.array(res_np, dtype=np.float64)
    except Exception:
        return None

def process_msms_file(pdb_file, msms_exec = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/msms"):
    """
    Process a single PDB file with MSMS/residue depth and save per-residue position features to save_file (.npy).
    Chain order and residue order are enforced.
    """
    try:
        parser = PDBParser(QUIET=True)
        model = parser.get_structure('model', pdb_file)[0]
        X = []
        for chain in get_chain_order(model):
            try:
                with SuppressOutput():
                    surf = get_surface(chain, MSMS=msms_exec)
                surf_tree = cKDTree(surf) if surf is not None and len(surf) > 0 else None
            except Exception:
                surf = np.empty(0)
                surf_tree = None
            chain_atom = ['N', 'CA', 'C', 'O']
            residues = sorted([r for r in chain if r.id[0] == " "], key=lambda r: r.id[1])
            for residue in residues:
                line = []
                atoms_coord = np.array([atom.get_coord() for atom in residue]) if len(list(residue.get_atoms()))>0 else np.empty((0,3))
                if surf.size != 0 and surf_tree is not None and atoms_coord.size > 0:
                    dist, _ = surf_tree.query(atoms_coord)
                    closest_atom = int(np.argmin(dist))
                    closest_pos = atoms_coord[closest_atom]
                else:
                    closest_pos = atoms_coord[-1] if atoms_coord.size else np.zeros(3)
                atoms = list(residue.get_atoms())
                ca_pos = residue['CA'].get_coord() if 'CA' in residue else np.zeros(3)
                pos_s = np.zeros(3)
                un_s = 0.0
                for atom in atoms:
                    if atom.name in chain_atom:
                        line.append(atom.get_coord())
                    else:
                        # calMass fallback returns scalar; multiply by coord to accumulate weighted sum
                        # m = calMass(atom, True)
                        # pos_s += atom.get_coord() * m
                        # un_s += m
                        pos_s += calMass(atom, True)
                        un_s += calMass(atom, False)
                if len(line) != 4:
                    # ensure 4 backbone coords (N, CA, C, O) order may vary; pad with CA
                    line = line + [list(ca_pos)] * (4 - len(line))
                if un_s == 0:
                    R_pos = ca_pos
                else:
                    R_pos = pos_s / un_s
                line.append(R_pos)
                line.append(closest_pos)
                X.append(line)
        return np.array(X, dtype=np.float64)
    except Exception:
        return None


def shortest_path_matrix_from_smiles_no_hs(smiles: str, max_distance: int = None) -> np.ndarray:
    """
    从去氢 SMILES 计算最短路径矩阵（键数），不可达或 padding 用 0 表示。
    返回 numpy array shape (N, N)。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    # 确保去氢
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

    mat = np.zeros((N, N), dtype=np.int64)  # 默认 0 表示无路径 / padding
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=max_distance)
        for j, l in lengths.items():
            # lengths 包含 i->i 的 0
            mat[i, j] = l
    # 对角已为 0
    return mat


def pad_matrix_to_size(mat: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """
    将方阵 mat pad 到 (size, size)，右下方填充 pad_value
    """
    N = mat.shape[0]
    if N == size:
        return mat
    if N > size:
        # 若实际原子数超过 batch 中的 maxliglen，截断（通常不应发生）
        return mat[:size, :size]
    pad0 = size - N
    return np.pad(mat, ((0, pad0), (0, pad0)), mode='constant', constant_values=pad_value)


class readData(Dataset):  # 用于训练
    def __init__(self, name_list, proj_dir, lig_dict, smiles_dict, true_file, nn_config, data_check=False):
        self.label_dict = self._read(true_file, skew=1)
        self.name_list = name_list
        self.proj_dir = proj_dir
        self.lig_dict = lig_dict
        self.smiles_dict = smiles_dict
        self.data_class = nn_config['pdb_class']
        self.nn_config = nn_config
        self.max_distance = nn_config['max_distance']

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        try:
            pdbid, lig = self.name_list[idx]  # name是pdb_name
            ankh = self.Normalize(np.load(f'{self.proj_dir}/ankh/{pdbid}.npy'),
                                  self.nn_config[f'ankh_max_repr'], self.nn_config[f'ankh_min_repr'])
            dssp = self.Normalize(np.load(f'{self.proj_dir}/dssp/{pdbid}_dssp.npy'),
                                  self.nn_config[f'dssp_max_repr'], self.nn_config[f'dssp_min_repr'])
            sc = np.load(f'{self.proj_dir}/sidechain/{pdbid}_sc.npy')
            angles = sc[:, :2]
            feature = np.concatenate([dssp, ankh], axis=1)
            cmap = np.load(f'{self.proj_dir}/cmap/{pdbid}_cmap.npy')

            ligand = self.lig_dict[lig]["atomic_reprs"]
            xyz = np.load(f'{self.proj_dir}/pos/{pdbid}_pos.npy')
            y_true_arr = np.asarray(list(self.label_dict[(pdbid, lig)]), dtype=int)

            N_feat = feature.shape[0]
            N_angles = angles.shape[0]
            N_xyz = xyz.shape[0]
            N_cmap_0, N_cmap_1 = cmap.shape[0], cmap.shape[1]  # cmap 是 N x N 的矩阵
            N_y = y_true_arr.shape[0]

            # 检查所有蛋白质级别的特征长度是否一致
            if not (N_feat == N_angles == N_xyz == N_cmap_0 == N_cmap_1 == N_y):
                error_msg = (
                    f"Dimension mismatch for PDBID {pdbid}, Ligand {lig}:\n"
                    f"  feature: {feature.shape} (Expected N, D)\n"
                    f"  angles: {angles.shape} (Expected N, 2)\n"
                    f"  xyz: {xyz.shape} (Expected N, P, 3)\n"
                    f"  cmap: {cmap.shape} (Expected N, N, 2)\n"
                    f"  y_true: {y_true_arr.shape} (Expected N,)\n"
                    f"Note: ligand shape is {ligand.shape} and does not need to match."
                )
                print(error_msg)
                raise ValueError(error_msg)

            return pdbid, lig, feature, angles, ligand, xyz, cmap, y_true_arr

        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar == 0] = 1
        return (arr - min_value) / scalar

    def collate_fn(self, batch):
        pdbids, ligids, rfeats, angles, ligands, xyzs, cmaps, y_trues = zip(*batch)

        maxprotlen = max([f.shape[0] for f in rfeats])
        maxliglen = max([f.shape[0] for f in ligands])

        batch_feat = []
        batch_angle = []
        batch_xyz = []
        batch_mask = []
        batch_cmap = []

        batch_ligand = []
        batch_lig_mask = []
        batch_lig_node_paths = []

        batch_y_true = []
        for idx in range(len(batch)):
            # protein features / angles / xyz padding
            batch_feat.append(self._padding(rfeats[idx], maxprotlen))
            batch_angle.append(self._padding(angles[idx], maxprotlen)) # [新增] 角度 Padding
            batch_xyz.append(self._padding(xyzs[idx], maxprotlen))

            # 接触图 (cmap) 的二维 Padding: [N, N, 2] -> [maxprotlen, maxprotlen, 2]
            cmap_arr = cmaps[idx]
            N = cmap_arr.shape[0]
            padded_cmap = np.zeros((maxprotlen, maxprotlen, cmap_arr.shape[-1]), dtype=np.float32)
            padded_cmap[:N, :N, :] = cmap_arr
            batch_cmap.append(torch.tensor(padded_cmap, dtype=torch.float))

            # per-residue mask (0/1 -> long)
            mask = np.zeros(maxprotlen, dtype=np.int64)
            mask[: rfeats[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype=torch.long))

            smiles = self.smiles_dict[ligids[idx]]
            batch_ligand.append(self._padding(ligands[idx], maxliglen))
            node_paths = shortest_path_matrix_from_smiles_no_hs(smiles, max_distance=self.max_distance)
            node_paths_padded = pad_matrix_to_size(node_paths, maxliglen, pad_value=0)
            batch_lig_node_paths.append(torch.tensor(node_paths_padded, dtype=torch.long))

            lig_mask = np.zeros(maxliglen, dtype=bool)
            lig_mask[:ligands[idx].shape[0]] = True
            batch_lig_mask.append(torch.tensor(lig_mask, dtype=torch.bool))

            # labels padded
            pad_y = np.zeros(maxprotlen, dtype=np.float32)
            pad_y[: y_trues[idx].shape[0]] = y_trues[idx]
            batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))

        return (
            pdbids,
            ligids,
            torch.stack(batch_feat),     # [B, maxlen, C] (语义特征: DSSP + Ankh + UniMol)
            torch.stack(batch_xyz),      # [B, maxlen, P, 3]
            torch.stack(batch_mask),     # [B, maxlen] long (0/1)
            torch.stack(batch_angle),  # [B, maxlen, 2] (纯几何角度: sin, cos)
            torch.stack(batch_cmap),     # [B, maxlen, maxlen, 2] float
            torch.stack(batch_ligand),   # [B, LIG_MAX, D2]
            torch.stack(batch_lig_node_paths),  # [B, LIG_MAX, LIG_MAX] long
            torch.stack(batch_lig_mask), # [B, LIG_MAX] bool
            torch.stack(batch_y_true),   # [B, maxlen] float
        )

    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded, dtype=torch.float)
        return res

    def _read(self, file_name, skew=0):
        lab_dict = {}
        with open(file_name, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2 + skew]:
                name = content[idx].replace('>', '').replace('\n', '')
                id, lig = name.split(' ')[0], name.split(' ')[1]
                lab = content[idx + 1 + skew].replace('\n', '')
                lab_dict[(id, lig)] = lab
        return lab_dict

class LoadData(Dataset):  # 用于测试
    '''
        name_list: list of tuple, [(pdb_name,lig_name),...]
        proj_dir: str, path of label file
        lig_dict: 配体字典
        repr_dict: 归一化字典
    '''

    def __init__(self, name_list, proj_dir, lig_dict, repr_dict):
        self.name_list = name_list
        self.proj_dir = proj_dir
        self.repr_dict = repr_dict
        self.lig_dict = lig_dict

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name, lig = self.name_list[idx]  # name是pdb_name
        feature_list = []
        feature_list.append(self.Normalize(np.load(f'{self.proj_dir}/dssp/{name}.npy'), self.repr_dict['dssp_max_repr'],
                                           self.repr_dict['dssp_min_repr']))
        feature_list.append(self.Normalize(np.load(f'{self.proj_dir}/ankh/{name}.npy'), self.repr_dict['ankh_max_repr'],
                                           self.repr_dict['ankh_min_repr']))
        feature = np.concatenate(feature_list, axis=1)  # rfeat
        xyz = np.load(os.path.join(self.proj_dir, 'pos', name + '.npy'))  # xyz
        # ligand 信息
        ligand = self.lig_dict[lig]["atomic_reprs"]
        return name, lig, feature, ligand, xyz

    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar == 0] = 1
        return (arr - min_value) / scalar

    def collate_fn(self, batch):
        names, ligs, features, ligands, xyzs = zip(*batch)
        maxlen = 1500
        batch_names = []
        batch_ligs = []
        batch_rfeat = []
        batch_ligand = []
        batch_xyz = []
        batch_mask = []
        for idx in range(len(batch)):
            if len(features[idx]) <= maxlen:
                batch_names.append(names[idx])
                batch_ligs.append(ligs[idx])
                batch_rfeat.append(self._padding(features[idx], maxlen))  # [ L, D]
                batch_ligand.append(torch.tensor(ligands[idx], dtype=torch.float))
                batch_xyz.append(self._padding(xyzs[idx], maxlen))
                mask = np.zeros(maxlen)
                mask[:features[idx].shape[0]] = 1
                batch_mask.append(torch.tensor(mask, dtype=torch.long))
        return batch_names, batch_ligs, torch.stack(batch_rfeat), torch.stack(batch_ligand), torch.stack(
            batch_xyz), torch.stack(batch_mask)

    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded, dtype=torch.float)
        return res


import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# helper functions
def safe_load(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


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


def try_md_file_patterns(base_dir, pdb_id, frame_idx, suffixes):
    """
    Try several common patterns for MD feature files.
    suffixes: list of filename templates, e.g. ["frame{idx}.npy", "{pdb}_frame{idx}.npy"]
    Returns full path if exists, else None.
    """
    # try subfolder pattern: base_dir/<pdb_id>/frame{idx}.npy
    subdir = os.path.join(base_dir, pdb_id)
    for tmpl in suffixes:
        p = os.path.join(subdir, tmpl.format(idx=frame_idx, pdb=pdb_id))
        if os.path.exists(p):
            return p
    # try flat pattern: base_dir/<pdb_id>_frame{idx}.npy or base_dir/<pdb_id>.npy
    for tmpl in suffixes:
        p = os.path.join(base_dir, tmpl.format(idx=frame_idx, pdb=pdb_id))
        if os.path.exists(p):
            return p
    # try single-file (no frame) fallback
    p_single = os.path.join(base_dir, f"{pdb_id}.npy")
    if os.path.exists(p_single):
        return p_single
    return None


def sample_row_by_pocket_rms(csv_path, random_state=None):
    """
    从 apoholo.csv 中按 pocket_rms 大小加权随机抽取一行。

    参数:
        csv_path: str, apoholo.csv 文件路径
        random_state: int 或 None, 随机种子，便于复现

    返回:
        pandas.Series, 抽取的一行
    """
    df = pd.read_csv(csv_path)
    # 将 pocket_rms 转为数值，缺失或非法值设为 0
    weights = pd.to_numeric(df["pocket_rms"], errors="coerce").fillna(0).values

    # 如果所有权重都是 0，则退化为均匀随机
    if np.all(weights == 0):
        probs = None
    else:
        probs = weights / weights.sum()

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df), p=probs)
    return df.iloc[idx]


import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
from scipy.spatial import distance_matrix


# 假设外部有定义最短路径等函数
# from your_utils import safe_load, parse_index_list, build_label_from_indices, sample_row_by_pocket_rms, shortest_path_matrix_from_smiles_no_hs, pad_matrix_to_size

# ==========================================
# [新增] 几何与 PDB 解析 Helpers
# ==========================================
def get_chain_order(model):
    chain_min = []
    for chain in model.get_chains():
        resnums = [res.id[1] for res in chain if res.id[0] == " "]
        mn = min(resnums) if resnums else float('inf')
        chain_min.append((chain, mn))
    chain_min.sort(key=lambda x: x[1])
    return [c for c, _ in chain_min]


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


import numpy as np
import math
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import get_surface
from scipy.spatial import cKDTree, distance_matrix


def process_pdb_all_features(pdb_file,
                             dssp_exec="/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/mkdssp",
                             msms_exec="/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/msms"):
    """
    一次性解析 PDB 文件，提取 DSSP, XYZ (MSMS), CMAP 和 Angles。
    """
    try:
        # 1. 仅进行一次 I/O 和 PDB 结构解析
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("tmp", pdb_file)
        model = structure[0]

        # 2. 运行 DSSP (DSSP 需要 model 级别)
        try:
            dssp = DSSP(model, pdb_file, dssp=dssp_exec)
            dssp_keys = set(dssp.keys())
        except Exception:
            dssp = {}
            dssp_keys = set()

        dssp_res = []
        xyz_res = []
        ca_list = []
        cb_list = []
        angles_list = []

        chain_atom = ['N', 'CA', 'C', 'O']

        # 3. 遍历 Chain 和 Residues
        for chain in get_chain_order(model):
            # 运行 MSMS (get_surface 仅支持 chain 级别)
            try:
                with SuppressOutput():
                    surf = get_surface(chain, MSMS=msms_exec)
                surf_tree = cKDTree(surf) if surf is not None and len(surf) > 0 else None
            except Exception:
                surf = np.empty(0)
                surf_tree = None

            residues = sorted([r for r in chain if r.id[0] == " "], key=lambda r: r.id[1])

            for residue in residues:
                # --- A. DSSP 特征提取 ---
                res_key = (chain.id, (' ', residue.id[1], residue.id[2]))
                if res_key in dssp_keys:
                    tuple_dssp = dssp[res_key]
                    ss_vec = mapSS.get(tuple_dssp[2], mapSS[' '])
                    other_vals = [float(x) if x != "NA" else 0.0 for x in tuple_dssp[3:]]
                    dssp_res.append(ss_vec + other_vals)
                else:
                    dssp_res.append(np.zeros(20, dtype=float))

                # --- B. XYZ & MSMS 特征提取 ---
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
                pos_s = np.zeros(3)
                un_s = 0.0

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

                # --- C. CMAP 坐标收集 ---
                ca_list.append(ca_pos)
                if 'CB' in residue:
                    cb_coord = residue['CB'].get_coord()
                elif all(atom in residue for atom in ['N', 'CA', 'C']):
                    cb_coord = calc_pseudo_cb(residue['N'].get_coord(), residue['CA'].get_coord(),
                                              residue['C'].get_coord())
                else:
                    cb_coord = ca_pos
                cb_list.append(cb_coord)

                # --- D. 角度收集 ---
                chi1 = get_chi1_angle(residue)
                angle_feat = [math.sin(chi1), math.cos(chi1)] if chi1 is not None else [0.0, 0.0]
                angles_list.append(angle_feat)

        if len(ca_list) == 0:
            return None, None, None, None

        # 4. 组装为 Numpy 数组
        dssp_arr = np.array(dssp_res, dtype=np.float64)
        xyz_arr = np.array(xyz_res, dtype=np.float64)
        angles_arr = np.array(angles_list, dtype=np.float32)

        ca_coords = np.array(ca_list, dtype=np.float32)
        cb_coords = np.array(cb_list, dtype=np.float32)
        ca_dist = distance_matrix(ca_coords, ca_coords)
        cb_dist = distance_matrix(cb_coords, cb_coords)
        cmap_arr = np.stack([ca_dist, cb_dist], axis=-1).astype(np.float32)

        return dssp_arr, xyz_arr, cmap_arr, angles_arr

    except Exception as e:
        # 为了保证 DataLoader 不崩溃，这里可以增加 logging
        print(f"Error processing {pdb_file}: {e}")
        return None, None, None, None

class readData_MD(Dataset):
    """
    Dataset that mixes original PDBbind data and MD frames.
    All features (DSSP, POS, CMAP, Angles, Ankh) are loaded directly from disk.
    If MD features are corrupted or dimensions mismatch, it automatically falls back to Original features.
    """

    def __init__(self,
                 dataset: str,
                 summary_xlsx: str,
                 proj_dir: str,
                 lig_dict: dict,
                 smiles_dict: dict,
                 true_file: str,
                 nn_config: dict,
                 md_frame_count: int = 100,
                 apohold_dir: str = None,
                 apohold_feature_dir: str = None,
                 seed: int = 42,
                 conformation_type: str = None
                 ):
        self.label_dict = self._read(true_file, skew=1) if true_file is not None else {}
        self.df = pd.read_excel(summary_xlsx, dtype=str)
        self.proj_dir = proj_dir  # Base directory, e.g., /.../Datasets/PDBbind
        self.lig_dict = lig_dict
        self.smiles_dict = smiles_dict
        self.nn_config = nn_config
        self.max_distance = nn_config.get('max_distance', 6)
        self.md_frame_count = md_frame_count
        self.conformation_type = conformation_type
        self.md_prob = nn_config.get('md_prob', 0.5)
        # 定义 Original 和 Misato (MD) 特征的根目录
        # self.orig_dir = os.path.join(self.proj_dir, "original")
        self.orig_dir = os.path.join(self.proj_dir, "processed")
        self.misato_dir = os.path.join(self.proj_dir, "misato")

        self.apohold_dir = apohold_dir
        self.apoholo_feature_dir = apohold_feature_dir

        self.name_list = []
        for _, row in self.df.iterrows():
            pdb = str(row.get("PDB_ID", "")).strip()
            lig = str(row.get("Ligand", "")).strip()
            if pdb == "" or lig == "" or lig not in self.lig_dict:
                continue
            if str(row.get("Dataset", "")).strip() == dataset:
                self.name_list.append((pdb, lig))
        random.seed(seed)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        try:
            pdbid, lig = self.name_list[idx]
            row = self.df[(self.df["PDB_ID"].astype(str).str.strip() == pdbid)].iloc[0]
            all_match = str(row.get("all_lengths_match", "0")).strip()
            use_md = True if all_match in ("1", "True", "true", "T", "t", "YES", "yes", "Y", "y") else False

            frame_idx = random.randint(0, max(0, self.md_frame_count - 1))

            # ==========================================
            # 辅助函数：读取 Original 数据
            # ==========================================
            def load_original():
                ankh_arr = safe_load(os.path.join(self.orig_dir, "ankh", f"{pdbid}.npy"))
                dssp_arr = safe_load(os.path.join(self.orig_dir, "dssp", pdbid, f"{pdbid}_dssp.npy"))
                xyz = safe_load(os.path.join(self.orig_dir, "pos", pdbid, f"{pdbid}_pos.npy"))
                cmap = safe_load(os.path.join(self.orig_dir, "cmap", pdbid, f"{pdbid}_cmap.npy"))
                angles = safe_load(os.path.join(self.orig_dir, "angle", pdbid, f"{pdbid}_angles.npy"))

                # seq_len = len(str(row.get("original_seq", "")))
                # binding_str = row.get("binding_indices_original", "")
                seq_len = len(str(row.get("processed_seq", "")))
                binding_str = row.get("binding_indices_processed", "")
                return ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str

            # ==========================================
            # 辅助函数：读取 MD 数据
            # ==========================================
            def load_md(f_idx):
                ankh_arr = safe_load(os.path.join(self.misato_dir, "ankh", f"{pdbid}.npy"))
                dssp_arr = safe_load(os.path.join(self.misato_dir, "dssp", pdbid, f"frame{f_idx}_dssp.npy"))
                xyz = safe_load(os.path.join(self.misato_dir, "pos", pdbid, f"frame{f_idx}_pos.npy"))
                cmap = safe_load(os.path.join(self.misato_dir, "cmap", pdbid, f"frame{f_idx}_cmap.npy"))
                angles = safe_load(os.path.join(self.misato_dir, "angle", pdbid, f"frame{f_idx}_angles.npy"))

                seq_len = len(str(row.get("frame0_seq", "")))
                binding_str = row.get("binding_indices_frame", "")
                return ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str

            # ==========================================
            # 辅助函数：静默校验维度 (用于决定是否 Fallback)
            # ==========================================
            def is_dimension_valid(a, d, x, c, ang, s_len):
                if any(v is None for v in [a, d, x, c, ang]): return False
                return (a.shape[0] == d.shape[0] == x.shape[0] == c.shape[0] == c.shape[1] == ang.shape[0] == s_len)

            # --- 主干读取逻辑 ---
            if self.conformation_type == "MISATO" and use_md:
                # 尝试加载 MD 数据
                ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_md(frame_idx)
                # 如果 MD 维度校验失败，回退到 Original
                if not is_dimension_valid(ankh_arr, dssp_arr, xyz, cmap, angles, seq_len):
                    ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_original()

            elif self.conformation_type == "Mix" and use_md:
                if random.random() < self.md_prob:
                    # 尝试加载 MD 数据
                    ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_md(frame_idx)
                    # 如果 MD 维度校验失败，回退到 Original
                    if not is_dimension_valid(ankh_arr, dssp_arr, xyz, cmap, angles, seq_len):
                        ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_original()
                else:
                    # 直接加载 Original 数据
                    ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_original()

            elif self.conformation_type == "ApoHolo" and use_md:
                apoholo_csv = os.path.join(self.apohold_dir, f"{pdbid}/apoholo.csv")
                input_row = sample_row_by_pocket_rms(apoholo_csv) if os.path.exists(apoholo_csv) else None

                if input_row is not None:
                    apoholoid = input_row.get("src_pdb_id")
                    apoholochain = input_row.get("src_chain_id")
                    ankh_arr = safe_load(
                        os.path.join(self.apoholo_feature_dir, f"{pdbid}", "ankh", f"{apoholoid}{apoholochain}.npy"))
                    dssp_arr = safe_load(os.path.join(self.apoholo_feature_dir, f"{pdbid}", "dssp",
                                                      f"{apoholoid}{apoholochain}_dssp.npy"))
                    xyz = safe_load(
                        os.path.join(self.apoholo_feature_dir, f"{pdbid}", "pos", f"{apoholoid}{apoholochain}_pos.npy"))
                    cmap = safe_load(os.path.join(self.apoholo_feature_dir, f"{pdbid}", "cmap",
                                                  f"{apoholoid}{apoholochain}_cmap.npy"))
                    angles = safe_load(os.path.join(self.apoholo_feature_dir, f"{pdbid}", "angles",
                                                    f"{apoholoid}{apoholochain}_angles.npy"))
                    seq_len = len(input_row.get("sequence"))
                    binding_str = input_row.get("mapped_binding_indices")
                else:
                    # Fallback to MD or Original if ApoHolo missing
                    if random.random() < 0.5:
                        ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_md(frame_idx)
                        if not is_dimension_valid(ankh_arr, dssp_arr, xyz, cmap, angles, seq_len):
                            ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_original()
                    else:
                        ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_original()
            else:
                # 默认读取 Original 数据
                ankh_arr, dssp_arr, xyz, cmap, angles, seq_len, binding_str = load_original()

            # ==========================================
            # 数据拼接与最后的安全拦截
            # ==========================================
            # 获取 Label 与 Ligand
            ligand = self.lig_dict[lig]["atomic_reprs"]
            y_true = build_label_from_indices(seq_len, parse_index_list(binding_str))
            y_true_arr = np.asarray(y_true, dtype=int)

            # 语义特征拼接
            dssp = self.Normalize(dssp_arr, self.nn_config['dssp_max_repr'], self.nn_config['dssp_min_repr'])
            ankh = self.Normalize(ankh_arr, self.nn_config['ankh_max_repr'], self.nn_config['ankh_min_repr'])
            feature = np.concatenate([dssp, ankh], axis=1)

            # --- 维度一致性最终检查 ---
            # 提取所有残基维度 (N)
            N_feat = feature.shape[0]
            N_angles = angles.shape[0]
            N_xyz = xyz.shape[0]
            N_cmap_0, N_cmap_1 = cmap.shape[0], cmap.shape[1]  # cmap 是 N x N 的矩阵
            N_y = y_true_arr.shape[0]

            # 检查所有蛋白质级别的特征长度是否一致
            if not (N_feat == N_angles == N_xyz == N_cmap_0 == N_cmap_1 == N_y):
                error_msg = (
                    f"Dimension mismatch for PDBID {pdbid}, Ligand {lig}:\n"
                    f"  feature: {feature.shape} (Expected N, D)\n"
                    f"  angles: {angles.shape} (Expected N, 2)\n"
                    f"  xyz: {xyz.shape} (Expected N, P, 3)\n"
                    f"  cmap: {cmap.shape} (Expected N, N, 2)\n"
                    f"  y_true: {y_true_arr.shape} (Expected N,)\n"
                    f"Note: ligand shape is {ligand.shape} and does not need to match."
                )
                print(error_msg)
                raise ValueError(error_msg)

            return pdbid, lig, feature, angles, ligand, xyz, cmap, y_true_arr

        except Exception as e:
            # 遇到脏数据、文件损坏或缺失时，自动递归调用获取下一个有效样本，避免DataLoader崩溃
            # print(f"Skipping index {idx} due to error: {e}")
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

        batch_feat = []
        batch_angle = []
        batch_xyz = []
        batch_mask = []
        batch_cmap = []

        batch_ligand = []
        batch_lig_mask = []
        batch_lig_node_paths = []

        batch_y_true = []

        for idx in range(len(batch)):
            batch_feat.append(self._padding(features[idx], maxprotlen))
            batch_angle.append(self._padding(angles[idx], maxprotlen))
            batch_xyz.append(self._padding(xyzs[idx], maxprotlen))

            # CMAP 2D Padding [N, N, 2] -> [maxprotlen, maxprotlen, 2]
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
            # 注意：需确保外部已导入最短路径等相关函数
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
            pdbids,
            ligids,
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

    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        return torch.tensor(padded, dtype=torch.float)

    def _read(self, file_name, skew=0):
        lab_dict = {}
        with open(file_name, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2 + skew]:
                name = content[idx].replace('>', '').replace('\n', '')
                id, lig = name.split(' ')[0], name.split(' ')[1]
                lab = content[idx + 1 + skew].replace('\n', '')
                lab_dict[(id, lig)] = lab
        return lab_dict

# class readData_MD(Dataset):
#     """
#     Dataset that mixes original PDBbind data and MD frames for entries where all_lengths_match==1.
#     """
#
#     def __init__(self,
#                  dataset: str,
#                  summary_csv: str,
#                  proj_dir: str,
#                  lig_dict: dict,
#                  smiles_dict: dict,
#                  true_file: str,
#                  nn_config: dict,
#                  md_dirs: dict = None,
#                  md_frame_count: int = 100,
#                  apohold_dir: str = None,
#                  apohold_feature_dir: str = None,
#                  seed: int = 42,
#                  conformation_type: str = None
#                  ):
#         self.label_dict = self._read(true_file, skew=1) if true_file is not None else {}
#         self.df = pd.read_csv(summary_csv, dtype=str)
#         self.proj_dir = proj_dir
#         self.lig_dict = lig_dict
#         self.smiles_dict = smiles_dict
#         self.data_class = nn_config['pdb_class']
#         self.nn_config = nn_config
#         self.max_distance = nn_config.get('max_distance', 6)
#         self.md_dirs = md_dirs
#         self.md_frame_count = md_frame_count
#
#         self.name_list = []
#         for _, row in self.df.iterrows():
#             pdb = str(row.get("PDB_ID", "")).strip()
#             lig = str(row.get("Ligand", "")).strip()
#             if pdb == "" or lig == "" or lig not in self.lig_dict:
#                 continue
#             if str(row.get("Dataset", "")).strip() == dataset:
#                 self.name_list.append((pdb, lig))
#         random.seed(seed)
#         self.apohold_dir = apohold_dir
#         self.apohold_feature_dir = apohold_feature_dir
#         self.conformation_type = conformation_type
#         self.pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/processed"
#         # self.pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/PDBs/"
#         self.misato_pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/processed"
#         self.apoholo_pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/ApoHolo"
#
#     def __len__(self):
#         return len(self.name_list)
#
#     # ==========================================
#     # [新增] 动态从 PDB 文件提取 CMAP 和 Angles
#     # ==========================================
#     def _extract_cmap_and_angles(self, pdb_file):
#         """实时解析 PDB 文件提取 CMAP 和角度，并保证长度匹配 target_len"""
#         parser = PDBParser(QUIET=True)
#         model = parser.get_structure('model', pdb_file)[0]
#         ca_list, cb_list, angles_list = [], [], []
#
#         for chain in get_chain_order(model):
#             residues = sorted([r for r in chain if r.id[0] == " "], key=lambda r: r.id[1])
#             for residue in residues:
#                 # 1. 坐标获取 (CMAP)
#                 ca_coord = residue['CA'].get_coord() if 'CA' in residue else np.zeros(3)
#                 ca_list.append(ca_coord)
#                 if 'CB' in residue:
#                     cb_coord = residue['CB'].get_coord()
#                 elif all(atom in residue for atom in ['N', 'CA', 'C']):
#                     cb_coord = calc_pseudo_cb(residue['N'].get_coord(), residue['CA'].get_coord(),
#                                               residue['C'].get_coord())
#                 else:
#                     cb_coord = ca_coord
#                 cb_list.append(cb_coord)
#
#                 # 2. 角度获取 (Angles)
#                 chi1 = get_chi1_angle(residue)
#                 angle_feat = [math.sin(chi1), math.cos(chi1)] if chi1 is not None else [0.0, 0.0]
#                 angles_list.append(angle_feat)
#
#         if len(ca_list) == 0:
#             raise ValueError("No residues found.")
#
#         ca_coords = np.array(ca_list, dtype=np.float32)
#         cb_coords = np.array(cb_list, dtype=np.float32)
#         ca_dist = distance_matrix(ca_coords, ca_coords)
#         cb_dist = distance_matrix(cb_coords, cb_coords)
#         cmap = np.stack([ca_dist, cb_dist], axis=-1).astype(np.float32)
#         angles = np.array(angles_list, dtype=np.float32)
#
#         # # [关键安全机制] 防止 PDB 实际残基数与预提取的 dssp 特征数有微小偏差
#         # actual_len = cmap.shape[0]
#         # if actual_len != target_len:
#         #     padded_cmap = np.zeros((target_len, target_len, 2), dtype=np.float32)
#         #     padded_angles = np.zeros((target_len, 2), dtype=np.float32)
#         #     min_len = min(actual_len, target_len)
#         #     padded_cmap[:min_len, :min_len, :] = cmap[:min_len, :min_len, :]
#         #     padded_angles[:min_len, :] = angles[:min_len, :]
#         #     return padded_cmap, padded_angles
#         return cmap, angles
#
#     def __getitem__(self, idx):
#         pdbid, lig = self.name_list[idx]
#         row = self.df[(self.df["PDB_ID"].astype(str).str.strip() == pdbid)].iloc[0]
#         seq_len = len(str(row.get("protein_seq", "")))
#         all_match = str(row.get("all_lengths_match", "0")).strip()
#         use_md = True if all_match in ("1", "True", "true", "T", "t", "YES", "yes", "Y", "y") else False
#
#         orig_ankh_path = os.path.join(self.proj_dir, "ankh", f"{pdbid}.npy")
#         orig_dssp_path = os.path.join(self.proj_dir, f"{self.data_class}_dssp", f"{pdbid}.npy")
#         orig_pos_path = os.path.join(self.proj_dir, f"{self.data_class}_pos", f"{pdbid}.npy")
#
#         md_ankh_root = os.path.join(self.md_dirs, "MISATO_ankh") if self.md_dirs else ""
#         # md_dssp_root = os.path.join(self.md_dirs, "MISATO_dssp") if self.md_dirs else ""
#         # md_pos_root = os.path.join(self.md_dirs, "MISATO_pos") if self.md_dirs else ""
#
#         frame_idx = random.randint(0, max(0, self.md_frame_count - 1))
#         md_ankh_path = os.path.join(md_ankh_root, f"{pdbid}.npy")
#         # md_dssp_path = os.path.join(md_dssp_root, f"{pdbid}/frame{frame_idx}_dssp.npy")
#         # md_pos_path = os.path.join(md_pos_root, f"{pdbid}/frame{frame_idx}_pos.npy")
#
#         pdb_path = ""  # 初始化追踪 pdb_path
#
#         if self.conformation_type == "MISATO" and use_md:
#             pdb_path = os.path.join(self.pdb_path, pdbid, f"frame{frame_idx}.pdb")
#             ankh_arr = safe_load(md_ankh_path)
#             # dssp_arr = safe_load(md_dssp_path)
#             # xyz = safe_load(md_pos_path)
#             # dssp_arr = process_dssp_file(pdb_path)
#             # xyz = process_msms_file(pdb_path)
#             dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#             seq_len = len(row.get("frame0_seq", ""))
#             binding_frame_str = row.get("binding_indices_frame", "")
#             y_true = build_label_from_indices(seq_len, parse_index_list(binding_frame_str))
#
#         elif self.conformation_type == "Mix" and use_md:
#             if random.random() < 0.5:
#                 pdb_path = os.path.join(self.pdb_path, pdbid, f"frame{frame_idx}.pdb")
#                 ankh_arr = safe_load(md_ankh_path)
#                 # dssp_arr = safe_load(md_dssp_path)
#                 # xyz = safe_load(md_pos_path)
#                 # dssp_arr = process_dssp_file(pdb_path)
#                 # xyz = process_msms_file(pdb_path)
#                 dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#                 seq_len = len(row.get("frame0_seq", ""))
#                 binding_frame_str = row.get("binding_indices_frame", "")
#                 y_true = build_label_from_indices(seq_len, parse_index_list(binding_frame_str))
#             else:
#                 pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#                 ankh_arr = safe_load(orig_ankh_path)
#                 # dssp_arr = safe_load(orig_dssp_path)
#                 # xyz = safe_load(orig_pos_path)
#                 # dssp_arr = process_dssp_file(pdb_path)
#                 # xyz = process_msms_file(pdb_path)
#                 dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#                 if ankh_arr.shape[0] != dssp_arr.shape[0]:
#                     print(pdbid, pdb_path, ankh_arr.shape, dssp_arr.shape)
#                 seq_len = len(str(row.get("protein_seq", "")))
#                 binding_prot_str = row.get("binding_indices_protein", "")
#                 y_true = build_label_from_indices(seq_len, parse_index_list(binding_prot_str))
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#
#         elif self.conformation_type == "ApoHolo" and use_md:
#             apoholo_csv = os.path.join(self.apohold_dir, f"{pdbid}/apoholo.csv")
#             input_row = sample_row_by_pocket_rms(apoholo_csv) if os.path.exists(apoholo_csv) else None
#             if input_row is not None:
#                 apoholoid = input_row.get("src_pdb_id")
#                 apoholochain = input_row.get("src_chain_id")
#                 apoholotype = input_row.get("type")
#                 pdb_path = os.path.join(self.apoholo_pdb_path, pdbid, f"{apoholotype}/{apoholoid}{apoholochain}.pdb")
#                 ankh_arr = safe_load(
#                     os.path.join(self.apohold_feature_dir, f"{pdbid}", "ankh", f"{apoholoid}{apoholochain}.npy"))
#                 dssp_arr = safe_load(
#                     os.path.join(self.apohold_feature_dir, f"{pdbid}", "dssp", f"{apoholoid}{apoholochain}_dssp.npy"))
#                 xyz = safe_load(
#                     os.path.join(self.apohold_feature_dir, f"{pdbid}", "pos", f"{apoholoid}{apoholochain}_pos.npy"))
#                 seq_len = len(input_row.get("sequence"))
#                 binding_frame_str = input_row.get("mapped_binding_indices")
#                 y_true = build_label_from_indices(seq_len, parse_index_list(binding_frame_str))
#             else:
#                 if random.random() < 0.5:
#                     pdb_path = os.path.join(self.pdb_path, pdbid, f"frame{frame_idx}.pdb")
#                     ankh_arr = safe_load(md_ankh_path)
#                     # dssp_arr = safe_load(md_dssp_path)
#                     # xyz = safe_load(md_pos_path)
#                     # dssp_arr = process_dssp_file(pdb_path)
#                     # xyz = process_msms_file(pdb_path)
#                     dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#                     md_seq_len = len(row.get("frame0_seq", ""))
#                     binding_frame_str = row.get("binding_indices_frame", "")
#                     y_true = build_label_from_indices(md_seq_len, parse_index_list(binding_frame_str))
#                 else:
#                     pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#                     ankh_arr = safe_load(orig_ankh_path)
#                     # dssp_arr = safe_load(orig_dssp_path)
#                     # xyz = safe_load(orig_pos_path)
#                     # dssp_arr = process_dssp_file(pdb_path)
#                     # xyz = process_msms_file(pdb_path)
#                     dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#                     seq_len = len(str(row.get("protein_seq", "")))
#                     binding_prot_str = row.get("binding_indices_protein", "")
#                     y_true = build_label_from_indices(seq_len, parse_index_list(binding_prot_str))
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#
#         else:
#             pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#             ankh_arr = safe_load(orig_ankh_path)
#             # dssp_arr = safe_load(orig_dssp_path)
#             # xyz = safe_load(orig_pos_path)
#             # dssp_arr = process_dssp_file(pdb_path)
#             # xyz = process_msms_file(pdb_path)
#             dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#             binding_prot_str = row.get("binding_indices_protein", "")
#             y_true = build_label_from_indices(seq_len, parse_index_list(binding_prot_str))
#
#         if xyz is None or dssp_arr is None or ankh_arr is None:
#             pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#             ankh_arr = safe_load(orig_ankh_path)
#             # dssp_arr = safe_load(orig_dssp_path)
#             # xyz = safe_load(orig_pos_path)
#             # dssp_arr = process_dssp_file(pdb_path)
#             # xyz = process_msms_file(pdb_path)
#             dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#             binding_prot_str = row.get("binding_indices_protein", "")
#             y_true = build_label_from_indices(seq_len, parse_index_list(binding_prot_str))
#
#         # # [新增] 修正可能存在的序列错位 (截断/补齐 Ankh 长度以匹配结构 DSSP 长度)
#         # N_pdb = dssp_arr.shape[0]
#         # N_seq = ankh_arr.shape[0]
#         # if N_seq < N_pdb:
#         #     ankh_arr = np.pad(ankh_arr, ((0, N_pdb - N_seq), (0, 0)), mode='constant')
#         # elif N_seq > N_pdb:
#         #     ankh_arr = ankh_arr[:N_pdb, :]
#
#         # # [新增] 调用提取函数实时解析 CMAP 和 Angles
#         # cmap, angles = self._extract_cmap_and_angles(pdb_path)
#         #
#         # dssp = self.Normalize(dssp_arr, self.nn_config['dssp_max_repr'], self.nn_config['dssp_min_repr'])
#         # ankh = self.Normalize(ankh_arr, self.nn_config['ankh_max_repr'], self.nn_config['ankh_min_repr'])
#         # feature = np.concatenate([dssp, ankh], axis=1)  # 纯语义特征拼装
#         #
#         # # [修改] 返回值增加了 angles 和 cmap
#         # return pdbid, lig, feature, angles, ligand, xyz, cmap, np.asarray(y_true, dtype=int)
#
#         # [新增] 调用提取函数实时解析 CMAP 和 Angles
#         # cmap, angles = self._extract_cmap_and_angles(pdb_path)
#
#         dssp = self.Normalize(dssp_arr, self.nn_config['dssp_max_repr'], self.nn_config['dssp_min_repr'])
#         ankh = self.Normalize(ankh_arr, self.nn_config['ankh_max_repr'], self.nn_config['ankh_min_repr'])
#         feature = np.concatenate([dssp, ankh], axis=1)  # 纯语义特征拼装
#
#         # 提前将 y_true 转换为 numpy array
#         y_true_arr = np.asarray(y_true, dtype=int)
#
#         # --- 维度一致性检查 ---
#         # 获取各个特征在维度 0 的长度 (即蛋白质的残基数 N)
#         N_feat = feature.shape[0]
#         N_angles = angles.shape[0]
#         N_xyz = xyz.shape[0]
#         N_cmap_0, N_cmap_1 = cmap.shape[0], cmap.shape[1]  # cmap 是 N x N 的矩阵
#         N_y = y_true_arr.shape[0]
#
#         # 检查所有蛋白质级别的特征长度是否一致
#         if not (N_feat == N_angles == N_xyz == N_cmap_0 == N_cmap_1 == N_y):
#             error_msg = (
#                 f"Dimension mismatch for PDBID {pdbid}, Ligand {lig}:\n"
#                 f"  feature: {feature.shape} (Expected N, D)\n"
#                 f"  angles: {angles.shape} (Expected N, 2)\n"
#                 f"  xyz: {xyz.shape} (Expected N, P, 3)\n"
#                 f"  cmap: {cmap.shape} (Expected N, N, 2)\n"
#                 f"  y_true: {y_true_arr.shape} (Expected N,)\n"
#                 f"Note: ligand shape is {ligand.shape} and does not need to match."
#             )
#             print(error_msg)
#             raise ValueError(error_msg)
#         # ----------------------
#
#         return pdbid, lig, feature, angles, ligand, xyz, cmap, y_true_arr
#
#     def Normalize(self, arr, max_value, min_value):
#         max_v = np.array(max_value, dtype=float)
#         min_v = np.array(min_value, dtype=float)
#         scalar = max_v - min_v
#         scalar[scalar == 0] = 1.0
#         return (arr - min_v) / scalar
#
#     def collate_fn(self, batch):
#         # [修改] 增加 angles, cmaps 的解包
#         pdbids, ligids, features, angles, ligands, xyzs, cmaps, y_trues = zip(*batch)
#
#         maxprotlen = max([f.shape[0] for f in features])
#         maxliglen = max([f.shape[0] for f in ligands])
#
#         batch_feat = []
#         batch_angle = []  # [新增]
#         batch_xyz = []
#         batch_mask = []
#         batch_cmap = []  # [新增]
#
#         batch_ligand = []
#         batch_lig_mask = []
#         batch_lig_node_paths = []
#
#         batch_y_true = []
#
#         for idx in range(len(batch)):
#             batch_feat.append(self._padding(features[idx], maxprotlen))
#             batch_angle.append(self._padding(angles[idx], maxprotlen))  # [新增] 1D Padding for angles
#             batch_xyz.append(self._padding(xyzs[idx], maxprotlen))
#
#             # [新增] CMAP 2D Padding [N, N, 2] -> [maxprotlen, maxprotlen, 2]
#             cmap_arr = cmaps[idx]
#             N = cmap_arr.shape[0]
#             padded_cmap = np.zeros((maxprotlen, maxprotlen, cmap_arr.shape[-1]), dtype=np.float32)
#             padded_cmap[:N, :N, :] = cmap_arr
#             batch_cmap.append(torch.tensor(padded_cmap, dtype=torch.float))
#
#             mask = np.zeros(maxprotlen, dtype=np.int64)
#             mask[: features[idx].shape[0]] = 1
#             batch_mask.append(torch.tensor(mask, dtype=torch.long))
#
#             smiles = self.smiles_dict[ligids[idx]]
#             batch_ligand.append(self._padding(ligands[idx], maxliglen))
#             node_paths = shortest_path_matrix_from_smiles_no_hs(smiles, max_distance=self.max_distance)
#             node_paths_padded = pad_matrix_to_size(node_paths, maxliglen, pad_value=0)
#             batch_lig_node_paths.append(torch.tensor(node_paths_padded, dtype=torch.long))
#
#             lig_mask = np.zeros(maxliglen, dtype=bool)
#             lig_mask[:ligands[idx].shape[0]] = True
#             batch_lig_mask.append(torch.tensor(lig_mask, dtype=torch.bool))
#
#             pad_y = np.zeros(maxprotlen, dtype=np.float32)
#             pad_y[: y_trues[idx].shape[0]] = y_trues[idx]
#             batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))
#
#         return (
#             pdbids,
#             ligids,
#             torch.stack(batch_feat),
#             torch.stack(batch_xyz),
#             torch.stack(batch_mask),
#             torch.stack(batch_angle),  # [新增] [B, maxlen, 2]
#             torch.stack(batch_cmap),  # [新增] [B, maxlen, maxlen, 2]
#             torch.stack(batch_ligand),
#             torch.stack(batch_lig_node_paths),
#             torch.stack(batch_lig_mask),
#             torch.stack(batch_y_true),
#         )
#
#     def _padding(self, arr, maxlen=1500):
#         padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
#         padded[:arr.shape[0]] = arr
#         return torch.tensor(padded, dtype=torch.float)
#
#     def _read(self, file_name, skew=0):
#         lab_dict = {}
#         with open(file_name, 'r') as file:
#             content = file.readlines()
#             lens = len(content)
#             for idx in range(lens)[::2 + skew]:
#                 name = content[idx].replace('>', '').replace('\n', '')
#                 id, lig = name.split(' ')[0], name.split(' ')[1]
#                 lab = content[idx + 1 + skew].replace('\n', '')
#                 lab_dict[(id, lig)] = lab
#         return lab_dict


class readData_ApoHolo(Dataset):  # 用于训练
    def __init__(self, name_list, proj_dir, lig_dict, smiles_dict, true_file,nn_config):
        self.label_dict = self._read(true_file, skew=1)
        if name_list is not None:
            self.name_list = name_list
        self.proj_dir = proj_dir
        self.lig_dict = lig_dict
        self.smiles_dict = smiles_dict
        self.nn_config = nn_config
        self.ankh_path = '/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/Datasets/PDBbind/ankh_apo_holo_test'
        self.dssp_path = '/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/Datasets/PDBbind/dssp_apo_holo_test'
        self.xyz_path = '/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/Datasets/PDBbind/pos_apo_holo_test'
        self.max_distance = nn_config['max_distance']
        self.apoholo_pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/ApoHolo"

    def __len__(self):
        return len(self.name_list)

    def _extract_cmap_and_angles(self, pdb_file):
        """实时解析 PDB 文件提取 CMAP 和角度，并保证长度匹配 target_len"""
        parser = PDBParser(QUIET=True)
        model = parser.get_structure('model', pdb_file)[0]
        ca_list, cb_list, angles_list = [], [], []

        for chain in get_chain_order(model):
            residues = sorted([r for r in chain if r.id[0] == " "], key=lambda r: r.id[1])
            for residue in residues:
                # 1. 坐标获取 (CMAP)
                ca_coord = residue['CA'].get_coord() if 'CA' in residue else np.zeros(3)
                ca_list.append(ca_coord)
                if 'CB' in residue:
                    cb_coord = residue['CB'].get_coord()
                elif all(atom in residue for atom in ['N', 'CA', 'C']):
                    cb_coord = calc_pseudo_cb(residue['N'].get_coord(), residue['CA'].get_coord(),
                                              residue['C'].get_coord())
                else:
                    cb_coord = ca_coord
                cb_list.append(cb_coord)

                # 2. 角度获取 (Angles)
                chi1 = get_chi1_angle(residue)
                angle_feat = [math.sin(chi1), math.cos(chi1)] if chi1 is not None else [0.0, 0.0]
                angles_list.append(angle_feat)

        if len(ca_list) == 0:
            raise ValueError("No residues found.")

        ca_coords = np.array(ca_list, dtype=np.float32)
        cb_coords = np.array(cb_list, dtype=np.float32)
        ca_dist = distance_matrix(ca_coords, ca_coords)
        cb_dist = distance_matrix(cb_coords, cb_coords)
        cmap = np.stack([ca_dist, cb_dist], axis=-1).astype(np.float32)
        angles = np.array(angles_list, dtype=np.float32)

        # # [关键安全机制] 防止 PDB 实际残基数与预提取的 dssp 特征数有微小偏差
        # actual_len = cmap.shape[0]
        # if actual_len != target_len:
        #     padded_cmap = np.zeros((target_len, target_len, 2), dtype=np.float32)
        #     padded_angles = np.zeros((target_len, 2), dtype=np.float32)
        #     min_len = min(actual_len, target_len)
        #     padded_cmap[:min_len, :min_len, :] = cmap[:min_len, :min_len, :]
        #     padded_angles[:min_len, :] = angles[:min_len, :]
        #     return padded_cmap, padded_angles
        return cmap, angles

    def __getitem__(self, idx):
        name = self.name_list[idx]
        parts = str(name).split('_')
        if len(parts) >= 4:
            target_pdb_id = parts[0]
            lig = parts[1]
            source_pdb_type = parts[-1]
            source_pdb_id = "_".join(parts[2:-1])
        pdb_path = os.path.join(self.apoholo_pdb_path, target_pdb_id, f"{source_pdb_type}/{source_pdb_id}.pdb")
        dssp_arr, xyz, cmap, angles = process_pdb_all_features(pdb_path)
        # dssp_arr = process_dssp_file(pdb_path)
        # xyz = process_msms_file(pdb_path)

        dssp = self.Normalize(dssp_arr,
            self.nn_config[f'dssp_max_repr'],
            self.nn_config[f'dssp_min_repr'])
        ankh = self.Normalize(np.load(os.path.join(self.ankh_path, name + '.npy')), self.nn_config[f'ankh_max_repr'],
                              self.nn_config[f'ankh_min_repr'])
        # cmap, angles = self._extract_cmap_and_angles(pdb_path)
        # sc = np.load(f'{self.proj_dir}/sidechain/{name}/{name}_sc.npy')
        # angles = sc[:, :2]

        feature = np.concatenate([dssp, ankh], axis=1)

        # cmap = np.load(f'{self.proj_dir}/cmap/{name}/{name}_cmap.npy')

        ligand = self.lig_dict[lig]["atomic_reprs"]
        y_true_arr = np.asarray(list(self.label_dict[name]), dtype=int)

        # --- 维度一致性最终检查 ---
        # 提取所有残基维度 (N)
        N_feat = feature.shape[0]
        N_angles = angles.shape[0]
        N_xyz = xyz.shape[0]
        N_cmap_0, N_cmap_1 = cmap.shape[0], cmap.shape[1]  # cmap 是 N x N 的矩阵
        N_y = y_true_arr.shape[0]

        # 检查所有蛋白质级别的特征长度是否一致
        if not (N_feat == N_angles == N_xyz == N_cmap_0 == N_cmap_1 == N_y):
            error_msg = (
                f"Dimension mismatch for PDBID {name}, Ligand {lig}:\n"
                f"  feature: {feature.shape} (Expected N, D)\n"
                f"  angles: {angles.shape} (Expected N, 2)\n"
                f"  xyz: {xyz.shape} (Expected N, P, 3)\n"
                f"  cmap: {cmap.shape} (Expected N, N, 2)\n"
                f"  y_true: {y_true_arr.shape} (Expected N,)\n"
                f"Note: ligand shape is {ligand.shape} and does not need to match."
            )
            print(error_msg)
            raise ValueError(error_msg)
        return name, lig, feature, angles, ligand, xyz, cmap, y_true_arr

    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar == 0] = 1
        return (arr - min_value) / scalar

    def collate_fn(self, batch):
        pdbids, ligids, rfeats, angles, ligands, xyzs, cmaps, y_trues = zip(*batch)

        maxprotlen = max([f.shape[0] for f in rfeats])
        maxliglen = max([f.shape[0] for f in ligands])

        batch_feat = []
        batch_angle = []
        batch_xyz = []
        batch_mask = []
        batch_cmap = []

        batch_ligand = []
        batch_lig_mask = []
        batch_lig_node_paths = []

        batch_y_true = []
        for idx in range(len(batch)):
            # protein features / angles / xyz padding
            batch_feat.append(self._padding(rfeats[idx], maxprotlen))
            batch_angle.append(self._padding(angles[idx], maxprotlen))  # [新增] 角度 Padding
            batch_xyz.append(self._padding(xyzs[idx], maxprotlen))

            # 接触图 (cmap) 的二维 Padding: [N, N, 2] -> [maxprotlen, maxprotlen, 2]
            cmap_arr = cmaps[idx]
            N = cmap_arr.shape[0]
            padded_cmap = np.zeros((maxprotlen, maxprotlen, cmap_arr.shape[-1]), dtype=np.float32)
            padded_cmap[:N, :N, :] = cmap_arr
            batch_cmap.append(torch.tensor(padded_cmap, dtype=torch.float))

            # per-residue mask (0/1 -> long)
            mask = np.zeros(maxprotlen, dtype=np.int64)
            mask[: rfeats[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype=torch.long))

            smiles = self.smiles_dict[ligids[idx]]
            batch_ligand.append(self._padding(ligands[idx], maxliglen))
            node_paths = shortest_path_matrix_from_smiles_no_hs(smiles, max_distance=self.max_distance)
            node_paths_padded = pad_matrix_to_size(node_paths, maxliglen, pad_value=0)
            batch_lig_node_paths.append(torch.tensor(node_paths_padded, dtype=torch.long))

            lig_mask = np.zeros(maxliglen, dtype=bool)
            lig_mask[:ligands[idx].shape[0]] = True
            batch_lig_mask.append(torch.tensor(lig_mask, dtype=torch.bool))

            # labels padded
            pad_y = np.zeros(maxprotlen, dtype=np.float32)
            pad_y[: y_trues[idx].shape[0]] = y_trues[idx]
            batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))

        return (
            pdbids,
            ligids,
            torch.stack(batch_feat),  # [B, maxlen, C] (语义特征: DSSP + Ankh + UniMol)
            torch.stack(batch_xyz),  # [B, maxlen, P, 3]
            torch.stack(batch_mask),  # [B, maxlen] long (0/1)
            torch.stack(batch_angle),  # [B, maxlen, 2] (纯几何角度: sin, cos)
            torch.stack(batch_cmap),  # [B, maxlen, maxlen, 2] float
            torch.stack(batch_ligand),  # [B, LIG_MAX, D2]
            torch.stack(batch_lig_node_paths),  # [B, LIG_MAX, LIG_MAX] long
            torch.stack(batch_lig_mask),  # [B, LIG_MAX] bool
            torch.stack(batch_y_true),  # [B, maxlen] float
        )

    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded, dtype=torch.float)
        return res

    def _read(self, file_name, skew=0):
        lab_dict = {}
        with open(file_name, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2 + skew]:
                name = content[idx].replace('>', '').replace('\n', '')
                lab = content[idx + 1 + skew].replace('\n', '')
                lab_dict[name] = lab
        return lab_dict


# -------------------------
# readData_MD Dataset
# -------------------------
# class readData_MD(Dataset):
#     """
#     Dataset that mixes original PDBbind data and MD frames for entries where all_lengths_match==1.
#
#     Behavior:
#     - Reads summary CSV to obtain PDB_ID, Ligand, protein_seq, binding_indices_protein,
#       frame0_seq, binding_indices_frame, all_lengths_match.
#     - For entries with all_lengths_match == 0: always use original (PDBbind) features:
#         proj_dir/<data_class>_dssp/<pdbid>.npy
#         proj_dir/ankh/<pdbid>.npy
#         proj_dir/<data_class>_pos/<pdbid>.npy
#       and labels come from binding_indices_protein column (if provided) or from true_file label_dict fallback.
#     - For entries with all_lengths_match == 1: with probability 0.5 keep original as above;
#       with probability 0.5 randomly pick a frame index in [0,99] and attempt to load MD features:
#         - ankh:  md_ankh_dir/<pdbid>/frame{idx}.npy  (or other patterns)
#         - dssp:  md_dssp_dir/<pdbid>/frame{idx}_dssp.npy
#         - pos:   md_pos_dir/<pdbid>/frame{idx}_pos.npy
#       Labels for MD come from binding_indices_frame column (parsed).
#       If MD files are missing or label parsing fails, fallback to original PDBbind data.
#     - The dataset still supports the original label file (true_file) for PDBbind labels when needed.
#     """
#
#     def __init__(self,
#                  dataset: str,
#                  summary_csv: str,
#                  proj_dir: str,
#                  lig_dict: dict,
#                  smiles_dict: dict,
#                  true_file: str,
#                  nn_config: dict,
#                  md_dirs: dict = None,
#                  md_frame_count: int = 100,
#                  apohold_dir: str = None,
#                  apohold_feature_dir: str = None,
#                  seed: int = 42,
#                  conformation_type: str = None
#                  ):
#         """
#         summary_csv: path to summary_binding_mapping_with_dataset_with_counts.csv
#         proj_dir: project directory where original PDBbind features live, e.g. proj_dir/<data_class>_dssp, proj_dir/ankh, proj_dir/<data_class>_pos
#         lig_dict: dictionary mapping ligand id -> ligand features (as in original code)
#         smiles_dict: mapping ligand id -> SMILES string
#         true_file: original label file used by _read (same format as before)
#         nn_config: network config dict (must contain 'pdb_class', 'dssp_max_repr', 'dssp_min_repr', 'ankh_max_repr', 'ankh_min_repr', 'max_distance')
#         md_dirs: dict with keys 'ankh','dssp','pos' pointing to MD feature root directories (MISATO_ankh, MISATO_dssp, MISATO_pos)
#                  If None, defaults to proj_dir/<data_class>_... with 'MISATO' prefix not assumed.
#         md_frame_count: number of frames available (default 100)
#         """
#         # load original label dict (for PDBbind labels)
#         self.label_dict = self._read(true_file, skew=1) if true_file is not None else {}
#         # read summary CSV
#         self.df = pd.read_csv(summary_csv, dtype=str)
#         self.proj_dir = proj_dir
#         self.lig_dict = lig_dict
#         self.smiles_dict = smiles_dict
#         self.data_class = nn_config['pdb_class']
#         self.nn_config = nn_config
#         self.max_distance = nn_config.get('max_distance', 6)
#         self.md_dirs = md_dirs
#         self.md_frame_count = md_frame_count
#         # build name_list as list of (pdbid, ligand) tuples
#         self.name_list = []
#         for _, row in self.df.iterrows():
#             pdb = str(row.get("PDB_ID", "")).strip()
#             lig = str(row.get("Ligand", "")).strip()
#             if pdb == "" or lig == "" or lig not in self.lig_dict:
#                 continue
#             if str(row.get("Dataset", "")).strip() == dataset:
#                 self.name_list.append((pdb, lig))
#         random.seed(seed)
#         self.apohold_dir = apohold_dir
#         self.apohold_feature_dir = apohold_feature_dir
#         self.conformation_type = conformation_type
#         self.pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/processed"
#         self.misato_pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/processed"
#         self.apoholo_pdb_path = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/ApoHolo"
#
#
#     def __len__(self):
#         return len(self.name_list)
#
#     def __getitem__(self, idx):
#         pdbid, lig = self.name_list[idx]
#         # find row in dataframe
#         row = self.df[(self.df["PDB_ID"].astype(str).str.strip() == pdbid)].iloc[0]
#         seq_len = len(str(row.get("protein_seq", "")))
#         all_match = str(row.get("all_lengths_match", "0")).strip()
#         if all_match in ("1", "True", "true", "T", "t", "YES", "yes", "Y", "y"):
#             use_md = True
#         else:
#             use_md = False
#
#         # prepare file paths
#
#         # original PDBbind feature paths
#         orig_dssp_path = os.path.join(self.proj_dir, f"{self.data_class}_dssp", f"{pdbid}.npy")
#         orig_ankh_path = os.path.join(self.proj_dir, "ankh", f"{pdbid}.npy")
#         orig_pos_path = os.path.join(self.proj_dir, f"{self.data_class}_pos", f"{pdbid}.npy")
#         # MD directories
#         md_ankh_root = os.path.join(self.md_dirs, "MISATO_ankh")
#         md_dssp_root = os.path.join(self.md_dirs, "MISATO_dssp")
#         md_pos_root = os.path.join(self.md_dirs, "MISATO_pos")
#         frame_idx = random.randint(0, max(0, self.md_frame_count - 1))
#         md_ankh_path = os.path.join(md_ankh_root, f"{pdbid}.npy")
#         md_dssp_path = os.path.join(md_dssp_root, f"{pdbid}/frame{frame_idx}_dssp.npy")
#         md_pos_path = os.path.join(md_pos_root, f"{pdbid}/frame{frame_idx}_pos.npy")
#
#         if self.conformation_type == "MISATO" and use_md:
#             pdb_path = os.path.join(self.pdb_path, pdbid, f"frame{frame_idx}.pdb")
#             ankh_arr = safe_load(md_ankh_path)
#             dssp_arr = safe_load(md_dssp_path)
#             xyz = safe_load(md_pos_path)
#
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#
#             seq_len = len(row.get("frame0_seq", ""))
#             binding_frame_str = row.get("binding_indices_frame", "")
#             indices = parse_index_list(binding_frame_str)
#             y_true = build_label_from_indices(seq_len, indices)
#
#         elif self.conformation_type == "Mix" and use_md:
#             if random.random() < 0.5:
#                 pdb_path = os.path.join(self.pdb_path, pdbid, f"frame{frame_idx}.pdb")
#                 ankh_arr = safe_load(md_ankh_path)
#                 dssp_arr = safe_load(md_dssp_path)
#                 xyz = safe_load(md_pos_path)
#
#                 seq_len = len(row.get("frame0_seq", ""))
#                 binding_frame_str = row.get("binding_indices_frame", "")
#                 indices = parse_index_list(binding_frame_str)
#                 y_true = build_label_from_indices(seq_len, indices)
#             else:
#                 pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#                 ankh_arr = safe_load(orig_ankh_path)
#                 dssp_arr = safe_load(orig_dssp_path)
#                 xyz = safe_load(orig_pos_path)
#
#                 seq_len = len(str(row.get("protein_seq", "")))
#                 binding_prot_str = row.get("binding_indices_protein", "")
#                 prot_indices = parse_index_list(binding_prot_str)
#                 y_true = build_label_from_indices(seq_len, prot_indices)
#
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#
#         elif self.conformation_type == "ApoHolo" and use_md:
#             apoholo_csv = os.path.join(self.apohold_dir, f"{pdbid}/apoholo.csv")
#             input_row = None
#             if os.path.exists(apoholo_csv):
#                 input_row = sample_row_by_pocket_rms(apoholo_csv)
#
#             if input_row is not None:
#                 apoholoid = input_row.get("src_pdb_id")
#                 apoholochain = input_row.get("src_chain_id")
#                 apoholotype = input_row.get("type")
#                 pdb_path = os.path.join(self.apoholo_pdb_path, pdbid, f"/{apoholotype}/{apoholoid}{apoholochain}.pdb")
#                 ankh_arr = safe_load(os.path.join(self.apohold_feature_dir, f"{pdbid}", "ankh",f"{apoholoid}{apoholochain}.npy"))
#                 dssp_arr = safe_load(os.path.join(self.apohold_feature_dir, f"{pdbid}", "dssp",f"{apoholoid}{apoholochain}_dssp.npy"))
#                 xyz = safe_load(os.path.join(self.apohold_feature_dir, f"{pdbid}", "pos",f"{apoholoid}{apoholochain}_pos.npy"))
#                 seq_len = len(input_row.get("sequence"))
#                 binding_frame_str = row.get("mapped_binding_indices")
#                 indices = parse_index_list(binding_frame_str)
#                 y_true = build_label_from_indices(seq_len, indices)
#             else:
#                 if random.random() < 0.5:
#                     pdb_path = os.path.join(self.pdb_path, pdbid, f"frame{frame_idx}.pdb")
#                     ankh_arr = safe_load(md_ankh_path)
#                     dssp_arr = safe_load(md_dssp_path)
#                     xyz = safe_load(md_pos_path)
#
#                     md_seq_len = len(row.get("frame0_seq", ""))
#                     binding_frame_str = row.get("binding_indices_frame", "")
#                     indices = parse_index_list(binding_frame_str)
#                     y_true = build_label_from_indices(md_seq_len, indices)
#                 else:
#                     pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#                     ankh_arr = safe_load(orig_ankh_path)
#                     dssp_arr = safe_load(orig_dssp_path)
#                     xyz = safe_load(orig_pos_path)
#
#                     seq_len = len(str(row.get("protein_seq", "")))
#                     binding_prot_str = row.get("binding_indices_protein", "")
#                     prot_indices = parse_index_list(binding_prot_str)
#                     y_true = build_label_from_indices(seq_len, prot_indices)
#
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#         else:
#             # load original features
#             pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#             dssp_arr = safe_load(orig_dssp_path)
#             ankh_arr = safe_load(orig_ankh_path)
#             pos_arr = safe_load(orig_pos_path)
#
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#             xyz = pos_arr
#             binding_prot_str = row.get("binding_indices_protein", "")
#             prot_indices = parse_index_list(binding_prot_str)
#             y_true = build_label_from_indices(seq_len, prot_indices)
#
#         if xyz is None or dssp_arr is None or ankh_arr is None:
#             pdb_path = os.path.join(self.pdb_path, pdbid, f"{pdbid}_protein.pdb")
#             dssp_arr = safe_load(orig_dssp_path)
#             ankh_arr = safe_load(orig_ankh_path)
#             pos_arr = safe_load(orig_pos_path)
#             ligand = self.lig_dict[lig]["atomic_reprs"]
#             xyz = pos_arr
#             binding_prot_str = row.get("binding_indices_protein", "")
#             prot_indices = parse_index_list(binding_prot_str)
#             y_true = build_label_from_indices(seq_len, prot_indices)
#         dssp = self.Normalize(dssp_arr, self.nn_config['dssp_max_repr'], self.nn_config['dssp_min_repr'])
#         ankh = self.Normalize(ankh_arr, self.nn_config['ankh_max_repr'], self.nn_config['ankh_min_repr'])
#         feature = np.concatenate([dssp, ankh], axis=1)
#         return pdbid, lig, feature, ligand, xyz, np.asarray(y_true, dtype=int)
#
#
#     def Normalize(self, arr, max_value, min_value):
#         # support scalar or array max/min
#         max_v = np.array(max_value, dtype=float)
#         min_v = np.array(min_value, dtype=float)
#         scalar = max_v - min_v
#         scalar[scalar == 0] = 1.0
#         return (arr - min_v) / scalar
#
#     def collate_fn(self, batch):
#         pdbids, ligids, features, ligands, xyzs, y_trues = zip(*batch)
#         maxprotlen = max([f.shape[0] for f in features])
#         maxliglen = max([f.shape[0] for f in ligands])
#
#         batch_feat = []
#         batch_xyz = []
#         batch_mask = []
#
#         batch_ligand = []
#         batch_lig_mask = []
#         batch_lig_node_paths = []
#
#         batch_y_true = []
#         for idx in range(len(batch)):
#             batch_feat.append(self._padding(features[idx], maxprotlen))
#             batch_xyz.append(self._padding(xyzs[idx], maxprotlen))
#             mask = np.zeros(maxprotlen, dtype=np.int64)
#             mask[: features[idx].shape[0]] = 1
#             batch_mask.append(torch.tensor(mask, dtype=torch.long))
#
#             smiles = self.smiles_dict[ligids[idx]]
#             batch_ligand.append(self._padding(ligands[idx], maxliglen))
#             node_paths = shortest_path_matrix_from_smiles_no_hs(smiles, max_distance=self.max_distance)
#             node_paths_padded = pad_matrix_to_size(node_paths, maxliglen, pad_value=0)
#             batch_lig_node_paths.append(torch.tensor(node_paths_padded, dtype=torch.long))
#             lig_mask = np.zeros(maxliglen, dtype=bool)
#             lig_mask[:ligands[idx].shape[0]] = True
#             batch_lig_mask.append(torch.tensor(lig_mask, dtype=torch.bool))
#
#             pad_y = np.zeros(maxprotlen, dtype=np.float32)
#             pad_y[: y_trues[idx].shape[0]] = y_trues[idx]
#             batch_y_true.append(torch.tensor(pad_y, dtype=torch.float))
#
#         return (
#             pdbids,
#             ligids,
#             torch.stack(batch_feat),
#             torch.stack(batch_xyz),
#             torch.stack(batch_mask),
#             torch.stack(batch_ligand),
#             torch.stack(batch_lig_node_paths),
#             torch.stack(batch_lig_mask),
#             torch.stack(batch_y_true),
#         )
#
#     def _padding(self, arr, maxlen=1500):
#         padded = np.zeros((maxlen, *arr.shape[1:]), dtype=np.float32)
#         padded[:arr.shape[0]] = arr
#         return torch.tensor(padded, dtype=torch.float)
#
#     def _read(self, file_name, skew=0):
#         lab_dict = {}
#         with open(file_name, 'r') as file:
#             content = file.readlines()
#             lens = len(content)
#             for idx in range(lens)[::2 + skew]:
#                 name = content[idx].replace('>', '').replace('\n', '')
#                 id, lig = name.split(' ')[0], name.split(' ')[1]
#                 lab = content[idx + 1 + skew].replace('\n', '')
#                 lab_dict[(id, lig)] = lab
#         return lab_dict

# Note: shortest_path_matrix_from_smiles_no_hs and pad_matrix_to_size must be available in scope,
# as in your original codebase. If not, import or implement them accordingly.
