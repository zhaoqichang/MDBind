#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm

from rdkit import Chem

try:
    from unimol_tools import UniMolRepr
except Exception as e:
    raise ImportError(
        "Please install and configure unimol_tools first, ensuring unimol_tools.UniMolRepr is available.") from e


def getAtomEmbed(smiles_file="./../datasets/PDBbind/ligand_smiles.txt",
                 out_path="./../datasets/PDBbind/Features/ligand_atoms.pkl",
                 unimol_model_name='unimolv1',
                 remove_hs=True,
                 device='cpu'):
    """
    Read SMILES from ligand_smiles.txt, extract molecule and atomic representations using UniMol,
    and save them to ligand_atoms.pkl. Ensure the order of atomic_reprs matches RDKit mol.GetAtoms().
    """

    if not os.path.exists(smiles_file):
        raise FileNotFoundError(f"File not found: {smiles_file}")

    # Read ligand_smiles.txt
    smiles_dict = {}
    with open(smiles_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                lig_name, smiles = parts
                smiles_dict[lig_name] = smiles

    # Initialize UniMolRepr
    clf = UniMolRepr(data_type='molecule',
                     remove_hs=remove_hs,
                     model_name=unimol_model_name,
                     model_size='164', )

    res_dict = {}
    for lig_name, smiles in tqdm(smiles_dict.items(), desc='UniMol running', ncols=80, unit='molecules'):
        # Parse molecule with RDKit first and get atom order
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"⚠️ Failed to parse SMILES: {lig_name} {smiles}, skipping")
            continue

        # If user chooses not to remove hydrogens, explicitly add hydrogens to align RDKit atom indices with UniMol's remove_hs parameter
        if not remove_hs:
            mol_with_h = Chem.AddHs(mol)
            rdkit_mol = mol_with_h
        else:
            rdkit_mol = mol

        rdkit_atoms = list(rdkit_mol.GetAtoms())
        rdkit_num_atoms = len(rdkit_atoms)

        # UniMol accepts a list of smiles and returns atomic_reprs (aligned with RDKit atom order, per unimol_tools documentation)
        try:
            unimol_repr = clf.get_repr([smiles], return_atomic_reprs=True)
        except Exception as e:
            print(f"⚠️ UniMol extraction failed: {lig_name} {e}")
            continue

        # Handle common return formats: assume dict or list-of-dict
        # atomic_reprs may be a list (len == 1) or ndarray
        atomic_reprs = None
        cls_repr = None

        if isinstance(unimol_repr, dict):
            cls_repr = unimol_repr.get('cls_repr', None)
            atomic_reprs = unimol_repr.get('atomic_reprs', None)
        elif isinstance(unimol_repr, list) and len(unimol_repr) > 0 and isinstance(unimol_repr[0], dict):
            cls_repr = unimol_repr[0].get('cls_repr', None)
            atomic_reprs = unimol_repr[0].get('atomic_reprs', None)
        else:
            try:
                cls_repr = unimol_repr['cls_repr']
                atomic_reprs = unimol_repr['atomic_reprs']
            except Exception:
                print(f"⚠️ Unknown UniMol return format for {lig_name}, skipping")
                continue

        # Convert to numpy array
        if cls_repr is not None:
            cls_arr = np.array(cls_repr)
        else:
            cls_arr = None

        if atomic_reprs is None:
            print(f"⚠️ Failed to get atomic_reprs for {lig_name}, skipping")
            continue

        atomic_arr = np.array(atomic_reprs)

        # Some implementations return shape (1, N, D) or (N, D), handle uniformly
        if atomic_arr.ndim == 3 and atomic_arr.shape[0] == 1:
            atomic_arr = atomic_arr[0]
        atomic_arr = atomic_arr[:-1]

        # Check if the number of atoms matches RDKit
        if atomic_arr.shape[0] != rdkit_num_atoms:
            print(f"⚠️ Atom count mismatch for {lig_name}: RDKit={rdkit_num_atoms}, UniMol={atomic_arr.shape[0]}")

        # Save: ensure atomic_reprs order matches RDKit atom order
        res_dict[lig_name] = {
            "cls_repr": cls_arr,
            "atomic_reprs": atomic_arr,
            "rdkit_num_atoms": rdkit_num_atoms,
            "smiles": smiles
        }

    # Save to specified path
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'wb') as f:
        pkl.dump(res_dict, f)

    print(f"✅ Successfully saved ligand_atoms.pkl to {out_path}")

    # Cleanup
    del clf
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ligand atom and molecular features using UniMol.")
    parser.add_argument(
        "--smiles_file",
        type=str,
        default="./../datasets/PDBbind/ligand_smiles.txt",
        help="Path to the input SMILES file."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./../datasets/PDBbind/Features/ligand_atoms.pkl",
        help="Path to save the output pickle file."
    )
    args = parser.parse_args()

    getAtomEmbed(smiles_file=args.smiles_file,
                 out_path=args.out_path,
                 unimol_model_name='unimolv1',
                 remove_hs=True,
                 device='cpu')