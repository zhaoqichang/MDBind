#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDBind Feature Extraction Pipeline (Holo + MD Edition)
Extracts protein Holo conformations and MD conformations from PDB_ID-based subfolders,
and performs strict shape alignment validation across different features for each conformation.
"""

import os
import gc
import glob
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, T5EncoderModel

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.Polypeptide import three_to_index, index_to_one
from scipy.spatial import cKDTree, distance_matrix

import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)


def makeDir(p):
    os.makedirs(p, exist_ok=True)


import periodictable


def lowerElem(elem):
    if len(elem) == 1:
        return elem
    return elem[0] + elem[1].lower()


def calMass(atom, pos=True):
    if pos:
        return periodictable.elements.symbol(lowerElem(atom.element)).mass * np.array(atom.get_coord())
    return periodictable.elements.symbol(lowerElem(atom.element)).mass


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mapSS = {
    ' ': [0, 0, 0, 0, 0, 0, 0, 0, 0], '-': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    'H': [0, 1, 0, 0, 0, 0, 0, 0, 0], 'B': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0], 'G': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'I': [0, 0, 0, 0, 0, 1, 0, 0, 0], 'P': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 1, 0], 'S': [0, 0, 0, 0, 0, 0, 0, 0, 1]
}


def three_to_one(resname):
    try:
        return index_to_one(three_to_index(resname))
    except Exception:
        return "X"


def get_chain_order(protein_structure, order_by_resseq=True):
    chains = list(protein_structure.get_chains())
    if not order_by_resseq:
        return [c.id for c in chains]
    chain_min = []
    for c in chains:
        resnums = [r.id[1] for r in c if r.id[0] == " "]
        mn = min(resnums) if resnums else float('inf')
        chain_min.append((c.id, mn))
    chain_min.sort(key=lambda x: x[1])
    return [c for c, _ in chain_min]


def structure_to_sequence_and_residues(structure, chain_order=None):
    seq, residues = [], []
    chains_by_id = {c.id: c for c in structure.get_chains()}
    if chain_order is None:
        chain_order = list(chains_by_id.keys())

    for cid in chain_order:
        chain = chains_by_id.get(cid)
        if chain is None: continue
        sorted_res = sorted([r for r in chain if r.id[0] == " "], key=lambda r: r.id[1])
        for residue in sorted_res:
            seq.append(three_to_one(residue.get_resname()))
            residues.append(residue)
    return "".join(seq), residues


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
        n, ca, cb = residue['N'].get_vector(), residue['CA'].get_vector(), residue['CB'].get_vector()
        cg = next((residue[name].get_vector() for name in cg_names if name in residue), None)
        if cg is None: return None
        return calc_dihedral(n, ca, cb, cg)
    except KeyError:
        return None


# ================= Core Analysis Modules =================

def run_ankh_on_tasks(tasks: list, pretrain_dir: str, device=DEVICE):
    """
    Supports task lists in the format of (pdb_file, out_file) for flexible output path dispatching.
    """
    print(f"Loading Ankh model to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(pretrain_dir)
    model = T5EncoderModel.from_pretrained(pretrain_dir)
    model.to(device)
    model.eval()

    for pdb_file, out_file in tqdm(tasks, desc="Ankh embeddings"):
        if os.path.exists(out_file): continue
        makeDir(os.path.dirname(out_file))

        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("tmp", pdb_file)
            model_struct = structure[0]
            chain_order = get_chain_order(model_struct, order_by_resseq=True)
            seq, _ = structure_to_sequence_and_residues(model_struct, chain_order)

            if not seq: continue

            ids = tokenizer.batch_encode_plus([list(seq)], add_special_tokens=True, padding=True,
                                              is_split_into_words=True, return_tensors="pt")
            with torch.no_grad():
                emb = model(input_ids=ids['input_ids'].to(device), attention_mask=ids['attention_mask'].to(device))
                np.save(out_file, emb.last_hidden_state[0, :len(seq)].cpu().numpy())
        except Exception:
            continue

    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def process_dssp_file(pdb_file: str, save_file: str, dssp_exec: str):
    try:
        makeDir(os.path.dirname(save_file))
        model = PDBParser(QUIET=True).get_structure("tmp", pdb_file)[0]
        try:
            dssp, keys = DSSP(model, pdb_file, dssp=dssp_exec), set(DSSP(model, pdb_file, dssp=dssp_exec).keys())
        except Exception:
            keys = set()

        chain_order = get_chain_order(model, order_by_resseq=True)
        _, residues = structure_to_sequence_and_residues(model, chain_order)
        res_np = []
        for residue in residues:
            res_key = (residue.get_parent().id, (' ', residue.id[1], residue.id[2]))
            if res_key in keys:
                tuple_dssp = dssp[res_key]
                res_np.append(
                    mapSS.get(tuple_dssp[2], mapSS[' ']) + [float(x) if x != "NA" else 0.0 for x in tuple_dssp[3:]])
            else:
                res_np.append(np.zeros(20, dtype=float))
        np.save(save_file, np.array(res_np, dtype=np.float64))
    except Exception:
        pass


def process_msms_file(pdb_file: str, save_file: str, msms_exec: str):
    try:
        makeDir(os.path.dirname(save_file))
        model = PDBParser(QUIET=True).get_structure("tmp", pdb_file)[0]
        chain_order = get_chain_order(model, order_by_resseq=True)
        _, residues = structure_to_sequence_and_residues(model, chain_order)
        try:
            surf = get_surface(model, MSMS=msms_exec)
            surf_tree = cKDTree(surf) if surf is not None and len(surf) > 0 else None
        except Exception:
            surf, surf_tree = np.empty(0), None

        X = []
        for residue in residues:
            line = []
            atoms_coord = np.array([atom.get_coord() for atom in residue]) if len(
                list(residue.get_atoms())) > 0 else np.empty((0, 3))

            if surf.size != 0 and surf_tree is not None and atoms_coord.size > 0:
                closest_pos = atoms_coord[int(np.argmin(surf_tree.query(atoms_coord)[0]))]
            else:
                closest_pos = atoms_coord[-1] if atoms_coord.size else np.zeros(3)

            ca_pos = residue['CA'].get_coord() if 'CA' in residue else np.zeros(3)
            pos_s, un_s = np.zeros(3), 0.0

            for atom in list(residue.get_atoms()):
                if atom.name in ['N', 'CA', 'C', 'O']:
                    line.append(atom.get_coord())
                else:
                    pos_s += calMass(atom, True)
                    un_s += calMass(atom, False)

            line = line + [list(ca_pos)] * (4 - len(line))
            X.append(line + [ca_pos if un_s == 0 else pos_s / un_s, closest_pos])

        np.save(save_file, X)
    except Exception:
        pass


def process_cmap_file(pdb_file: str, save_file: str):
    try:
        makeDir(os.path.dirname(save_file))
        model = PDBParser(QUIET=True).get_structure("tmp", pdb_file)[0]
        chain_order = get_chain_order(model, order_by_resseq=True)
        _, residues = structure_to_sequence_and_residues(model, chain_order)

        ca_list, cb_list = [], []
        for residue in residues:
            ca = residue['CA'].get_coord() if 'CA' in residue else np.zeros(3)
            ca_list.append(ca)
            if 'CB' in residue:
                cb_list.append(residue['CB'].get_coord())
            elif all(a in residue for a in ['N', 'CA', 'C']):
                cb_list.append(
                    calc_pseudo_cb(residue['N'].get_coord(), residue['CA'].get_coord(), residue['C'].get_coord()))
            else:
                cb_list.append(ca)

        if ca_list:
            ca_dist = distance_matrix(np.array(ca_list, dtype=np.float32), np.array(ca_list, dtype=np.float32))
            cb_dist = distance_matrix(np.array(cb_list, dtype=np.float32), np.array(cb_list, dtype=np.float32))
            np.save(save_file, np.stack([ca_dist, cb_dist], axis=-1).astype(np.float32))
    except Exception:
        pass


def process_angle_file(pdb_file: str, save_file: str):
    try:
        makeDir(os.path.dirname(save_file))
        model = PDBParser(QUIET=True).get_structure("tmp", pdb_file)[0]
        chain_order = get_chain_order(model, order_by_resseq=True)
        _, residues = structure_to_sequence_and_residues(model, chain_order)

        node_angles = []
        for residue in residues:
            chi1 = get_chi1_angle(residue)
            node_angles.append([math.sin(chi1), math.cos(chi1)] if chi1 is not None else [0.0, 0.0])

        if node_angles: np.save(save_file, np.array(node_angles, dtype=np.float32))
    except Exception:
        pass


def run_parallel_feature(tasks: list, process_func, desc: str, n_workers: int, *args):
    """
    Receives tasks: a list of (pdb_file, save_file) tuples.
    """
    valid_tasks = [(f, s) for f, s in tasks if not os.path.exists(s)]
    if not valid_tasks: return

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_func, f_path, s_path, *args) for f_path, s_path in valid_tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=desc): pass


# ================= Feature Dimension Validation Module =================
def check_single_conformation(*file_paths):
    """
    Accepts 5 feature file paths for a single conformation.
    Returns True if all files exist and have consistent lengths; otherwise returns False.
    """
    shapes = []
    for p in file_paths:
        if not os.path.exists(p):
            return False
        try:
            # Fast reading via mmap to avoid OOM
            shapes.append(np.load(p, mmap_mode='r').shape[0])
        except Exception:
            return False
    # Check if the first dimension (sequence length) of all 5 features matches
    return len(set(shapes)) == 1


# ================= Command-Line Arguments =================
def parse_args():
    ap = argparse.ArgumentParser(description="MDBind Feature Extraction (Holo & MD Frames)")
    ap.add_argument("--pdb_root",
                    default="./../datasets/PDBbind/pdbs",
                    help="Root directory containing complex PDB files")
    ap.add_argument("--out_root",
                    default="./../datasets/PDBbind/Features",
                    help="Root output directory for extracted features")
    ap.add_argument("--summary_file",
                    default="./../datasets/PDBbind/summary.xlsx",
                    help="Excel file recording PDB_IDs and validation results")

    ap.add_argument("--pretrain_ankh",
                    default="/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/ankh-large/",
                    help="Ankh pretrained directory")
    ap.add_argument("--dssp_exec",
                    default="/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/mkdssp",
                    help="Path to mkdssp executable")
    ap.add_argument("--msms_exec",
                    default="/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Baselines/LABind/tools/msms",
                    help="Path to msms executable")
    ap.add_argument("--workers", type=int, default=64, help="General number of parallel workers")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load summary table
    print(f"🚀 Loading summary data: {args.summary_file}")
    df = pd.read_excel(args.summary_file)
    if 'PDB_ID' not in df.columns:
        raise ValueError("Missing 'PDB_ID' column in the Excel file!")

    pdb_ids = df['PDB_ID'].astype(str).tolist()

    # === Task Collection Lists ===
    ankh_tasks, dssp_tasks, msms_tasks, cmap_tasks, angle_tasks = [], [], [], [], []

    print("🔍 Scanning files and building feature extraction task pools...")
    for pdb_id in pdb_ids:
        pdb_dir = os.path.join(args.pdb_root, pdb_id)
        feat_dir = os.path.join(args.out_root, pdb_id)

        # 1. Assemble Holo tasks
        holo_pdb = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
        if os.path.exists(holo_pdb):
            ankh_tasks.append((holo_pdb, os.path.join(feat_dir, "ankh", f"{pdb_id}_ankh.npy")))
            dssp_tasks.append((holo_pdb, os.path.join(feat_dir, "dssp", f"{pdb_id}_dssp.npy")))
            msms_tasks.append((holo_pdb, os.path.join(feat_dir, "pos", f"{pdb_id}_pos.npy")))
            cmap_tasks.append((holo_pdb, os.path.join(feat_dir, "cmap", f"{pdb_id}_cmap.npy")))
            angle_tasks.append((holo_pdb, os.path.join(feat_dir, "angle", f"{pdb_id}_angles.npy")))

        # 2. Assemble MD conformation (frame{i}.pdb) tasks
        md_files = glob.glob(os.path.join(pdb_dir, "frame*.pdb"))

        # MD Ankh uses frame0 universally to extract once as frames_ankh.npy
        frame0_pdb = os.path.join(pdb_dir, "frame0.pdb")
        if os.path.exists(frame0_pdb):
            ankh_tasks.append((frame0_pdb, os.path.join(feat_dir, "ankh", "frames_ankh.npy")))

        for md_pdb in md_files:
            fname = os.path.basename(md_pdb).replace('.pdb', '')  # Obtain "frame0", "frame1", etc.
            # DSSP, MSMS(pos), CMAP, Angles extracted independently
            dssp_tasks.append((md_pdb, os.path.join(feat_dir, "dssp", f"{fname}_dssp.npy")))
            msms_tasks.append((md_pdb, os.path.join(feat_dir, "pos", f"{fname}_pos.npy")))
            cmap_tasks.append((md_pdb, os.path.join(feat_dir, "cmap", f"{fname}_cmap.npy")))
            angle_tasks.append((md_pdb, os.path.join(feat_dir, "angle", f"{fname}_angles.npy")))

    print(f"📦 Total tasks built: Ankh({len(ankh_tasks)}), Other features(approx. {len(dssp_tasks)})")

    # === Execute Feature Extraction ===
    print("\n[STEP 1] Extract Ankh embeddings (Holo & Frame0)")
    run_ankh_on_tasks(ankh_tasks, args.pretrain_ankh, device=DEVICE)

    print("\n[STEP 2] Run DSSP in parallel (Holo & All Frames)")
    run_parallel_feature(dssp_tasks, process_dssp_file, "DSSP Extraction", args.workers, args.dssp_exec)

    print("\n[STEP 3] Run MSMS in parallel (Holo & All Frames)")
    run_parallel_feature(msms_tasks, process_msms_file, "MSMS Extraction", args.workers, args.msms_exec)

    print("\n[STEP 4] Run CA/CB Contact Maps (CMAP)")
    run_parallel_feature(cmap_tasks, process_cmap_file, "CMAP Extraction", args.workers)

    print("\n[STEP 5] Run Dihedral Angles Extraction")
    run_parallel_feature(angle_tasks, process_angle_file, "Angles Extraction", args.workers)

    print("\n================ Feature Extraction Completed. Starting Dimension Consistency Validation ================")

    # === Validate and Write Back to summary.xlsx ===
    alignment_results = {}

    for pdb_id in tqdm(pdb_ids, desc="Checking Dimensions"):
        feat_dir = os.path.join(args.out_root, pdb_id)
        pdb_dir = os.path.join(args.pdb_root, pdb_id)

        # 1. Validate Holo
        holo_ok = check_single_conformation(
            os.path.join(feat_dir, "ankh", f"{pdb_id}_ankh.npy"),
            os.path.join(feat_dir, "dssp", f"{pdb_id}_dssp.npy"),
            os.path.join(feat_dir, "pos", f"{pdb_id}_pos.npy"),
            os.path.join(feat_dir, "angle", f"{pdb_id}_angles.npy"),
            os.path.join(feat_dir, "cmap", f"{pdb_id}_cmap.npy")
        )

        # 2. Validate all Frames
        frames_ok = True
        md_files = glob.glob(os.path.join(pdb_dir, "frame*.pdb"))
        ankh_frame_file = os.path.join(feat_dir, "ankh", "frames_ankh.npy")

        for md_pdb in md_files:
            fname = os.path.basename(md_pdb).replace('.pdb', '')
            f_ok = check_single_conformation(
                ankh_frame_file,  # Uses unified frames_ankh.npy
                os.path.join(feat_dir, "dssp", f"{fname}_dssp.npy"),
                os.path.join(feat_dir, "pos", f"{fname}_pos.npy"),
                os.path.join(feat_dir, "angle", f"{fname}_angles.npy"),
                os.path.join(feat_dir, "cmap", f"{fname}_cmap.npy")
            )
            if not f_ok:
                frames_ok = False
                break

        # Passed if Holo itself is aligned, and (if MD frames exist) all existing MD frames are aligned respectively
        alignment_results[pdb_id] = holo_ok and frames_ok

    # Update DataFrame
    df['all_lengths_match'] = df['PDB_ID'].map(alignment_results)

    # Statistics
    pass_count = df['all_lengths_match'].sum()
    print(f"\n✅ Validation complete: {pass_count} complexes can safely enable MD sampling (Holo and MD features are properly self-aligned).")
    print(f"⚠️ The remaining {len(df) - pass_count} complexes cannot use MD sampling (due to missing or anomalous dimensions) and will be restricted to Holo training only.")

    # Write back to original Excel file
    df.to_excel(args.summary_file, index=False)
    print(f"💾 Updated status successfully saved to: {args.summary_file}")


if __name__ == "__main__":
    main()