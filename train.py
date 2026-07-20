#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import gc
import random
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Bio.PDB import PDBParser
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, average_precision_score, roc_auc_score, roc_curve

from readData import MDBindDataset
from model import MDBind
from func_help import get_std_opt


def setALlSeed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def get_smiles_dict(smiles_file):
    smiles_dict = {}
    with open(smiles_file, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                lig_name, smiles = parts
                smiles_dict[lig_name] = smiles
    return smiles_dict

def appendText(path:str, text:str):
    with open(path, 'a') as file:
        file.write(text)

def calEval(y_true, y_score, save_path=None, best_th=0.35):
    y_pred = [1 if i > best_th else 0 for i in
              y_score]  # default, The optimal threshold needs to be selected through validation

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    if save_path != None:
        result = '\nRec: ' + str(recall_score(y_true, y_pred)) + '\n' + \
                 'SPE: ' + str(TN / (TN + FP)) + '\n' + \
                 'Acc: ' + str(accuracy_score(y_true, y_pred)) + '\n' + \
                 'Pre: ' + str(precision_score(y_true, y_pred)) + '\n' + \
                 'F1: ' + str(f1_score(y_true, y_pred)) + '\n' + \
                 'MCC: ' + str(matthews_corrcoef(y_true, y_pred)) + '\n' + \
                 'AUC: ' + str(roc_auc_score(y_true, y_score)) + '\n' + \
                 'AUPR: ' + str(average_precision_score(y_true, y_score)) + '\n'
        appendText(save_path, result)
        return
    else:
        return {'Rec': recall_score(y_true, y_pred), 'SPE': TN / (TN + FP), 'Acc': accuracy_score(y_true, y_pred),
                'Pre': precision_score(y_true, y_pred), 'F1': f1_score(y_true, y_pred),
                'MCC': matthews_corrcoef(y_true, y_pred), 'AUC': roc_auc_score(y_true, y_score),
                'AUPR': average_precision_score(y_true, y_score)}

def getBestThreshold(y_true,y_score):
    best_threshold = 0
    best_mcc = -1
    best_pred = []
    for i in range(100):
        threshold = i/100
        y_pred = [1 if i > threshold else 0 for i in y_score]
        mcc = matthews_corrcoef(y_true,y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
            best_pred = y_pred
    return best_threshold,best_mcc,best_pred

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
summary_xlsx = "./../datasets/PDBbind/summary_updated.xlsx"

nn_config = {
    'feat_dir': './../datasets/PDBbind/Features',
    'lig_dict': pkl.load(open(f'./../datasets/PDBbind/Features/ligand_atoms.pkl', 'rb')),
    'smiles_dict': get_smiles_dict("./../datasets/PDBbind/ligand_smiles.txt"),
    'dssp_max_repr': np.load(f'./tools/dssp_max_repr.npy'),
    'dssp_min_repr': np.load(f'./tools/dssp_min_repr.npy'),
    'ankh_max_repr': np.load(f'./tools/ankh_max_repr.npy'),
    'ankh_min_repr': np.load(f'./tools/ankh_min_repr.npy'),
    # model parameters
    'rfeat_dim': 1556,
    'ligand_dim': 512,
    'hidden_dim': 256,
    'heads': 4,
    'augment_eps': 0.05,
    'rbf_num': 8,
    'top_k': 30,
    'attn_drop': 0.1,
    'dropout': 0.1,
    'num_layers': 5,
    'lr': 0.0004,
    'max_distance':15,
    # training parameters
    'batch_size': 12,
    'max_patience': 15,
    'device_ids': [0,1,2,3],  # adjust to available GPUs
    "md_prob": 0.5,
    "pos_weight" : 6,
    "lam": 1
}



class EvidentialLoss(nn.Module):
    def __init__(self, num_classes=2, annealing_epochs=10, pos_weight=None, lam=1):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_epochs = annealing_epochs
        self.lam = lam  # 核心加入：正则化系数

        if pos_weight is not None:
            class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, evidence, y_true, mask, current_epoch):
        y_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes).float()

        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)

        loss_nll = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)

        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)

        kl_loss = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True) \
                  + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=-1, keepdim=True) \
                  - torch.lgamma(torch.ones_like(S_tilde) * self.num_classes) \
                  + torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=-1,
                              keepdim=True)
        kl_loss = kl_loss.squeeze(-1)

        annealing_coef = min(1.0, current_epoch / self.annealing_epochs)

        # 核心改动：使用 self.lam 调节正则化惩罚强度
        loss = loss_nll + self.lam * annealing_coef * kl_loss

        if self.class_weights is not None:
            sample_weights = self.class_weights[y_true.long()]
            loss = loss * sample_weights

        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

def valid_process(model):
    """
    Compute AUPR on validation set.
    """
    model.to(DEVICE)
    model.eval()
    valid_data = MDBindDataset(
        dataset_split = "val",
        summary_xlsx = summary_xlsx,
        feat_dir=nn_config['feat_dir'],
        lig_dict=nn_config['lig_dict'],
        smiles_dict=nn_config['smiles_dict'],
        nn_config=nn_config,
    )
    valid_loader = DataLoader(valid_data, batch_size=nn_config['batch_size'],
                              shuffle=True, collate_fn=valid_data.collate_fn, num_workers=5)
    all_y_score = []
    all_y_true = []
    with torch.no_grad():
        for pdbids, ligids, rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true in valid_loader:
            tensors = [tensor.to(DEVICE) for tensor in [rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true]]
            rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true = tensors

            evidences = model(rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask)
            alpha = evidences + 1.0
            S = torch.sum(alpha, dim=-1, keepdim=True)
            prob_binding = alpha[..., 1] / S.squeeze(-1)
            y_score = torch.masked_select(prob_binding, mask == 1)
            y_true_masked = torch.masked_select(y_true, mask == 1)

            all_y_score.extend(y_score.cpu().detach().numpy())
            all_y_true.extend(y_true_masked.cpu().detach().numpy())

    aupr_value = average_precision_score(all_y_true, all_y_score) if len(all_y_true) > 0 else 0.0
    return aupr_value

def train_process(model=None, save_dir = None, epochs=50, fold_idx=0):
    """
    Train model with early stopping based on validation AUPR.
    """
    model.to(DEVICE)
    train_data = MDBindDataset(
        dataset_split = "train",
        summary_xlsx = summary_xlsx,
        feat_dir=nn_config['feat_dir'],
        lig_dict=nn_config['lig_dict'],
        smiles_dict=nn_config['smiles_dict'],
        nn_config=nn_config,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=nn_config['batch_size'],
        shuffle=True,
        collate_fn=train_data.collate_fn,
        num_workers=24,
        prefetch_factor=2,
        pin_memory=True,
        drop_last=True
    )

    loss_fn = EvidentialLoss(num_classes=2, annealing_epochs=10, pos_weight = nn_config["pos_weight"], lam=nn_config["lam"]).to(DEVICE)
    optimizer = get_std_opt(len(train_data), nn_config['batch_size'], model.parameters(), nn_config['hidden_dim'], nn_config['lr'])
    v_max_aupr = 0.0
    patience = 0
    if torch.cuda.device_count() > 1 and len(nn_config.get('device_ids', [])) > 0:
        model = nn.DataParallel(model, device_ids=nn_config['device_ids'])
        ckpt = os.path.join(save_dir, f'fold{fold_idx}.ckpt')
        if os.path.exists(ckpt):
            state_dict = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print("✅ load model:", ckpt)
    train_losses = []
    for epoch in range(epochs):
        all_loss = 0.0
        all_cnt = 0
        model.train()
        for pdbids, ligids, rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true in tqdm(train_loader,
                                                                                               desc=f'Epoch {epoch}',
                                                                                               unit='batch'):
            tensors = [tensor.to(DEVICE) for tensor in [rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true]]
            rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true = tensors
            optimizer.zero_grad()
            evidence = model(rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask)
            loss = loss_fn(evidence, y_true, mask, epoch)
            all_loss += loss.item()
            all_cnt += 1
            loss.backward()
            optimizer.step()
        avg_loss = all_loss / all_cnt if all_cnt > 0 else 0.0
        train_losses.append(avg_loss)
        print(f'Epoch {epoch} Loss: {avg_loss}')

        v_aupr = valid_process(model)
        print(f'Epoch {epoch} Valid AUPR: {v_aupr}')
        if v_aupr > v_max_aupr:
            v_max_aupr = v_aupr
            patience = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'fold{fold_idx}.ckpt'))
        else:
            patience += 1
        if patience >= nn_config['max_patience']:
            print('Early stopping triggered.')
            break

def load_models_for_ensemble(model_path: str, num_folds: int = 3):
    """
    Load saved fold models for ensemble evaluation.
    Returns list of models (DataParallel if multiple GPUs).
    """
    models = []
    for fold in range(num_folds):
        ckpt = os.path.join(model_path, f'weights/fold{fold}.ckpt')
        if not os.path.exists(ckpt):
            continue
        state_dict = torch.load(ckpt, map_location=DEVICE)
        model = MDBind(
            rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'],
            hidden_dim=nn_config['hidden_dim'], heads=nn_config['heads'],
            augment_eps=nn_config['augment_eps'], rbf_num=nn_config['rbf_num'],
            top_k=nn_config['top_k'], attn_drop=nn_config['attn_drop'],
            dropout=nn_config['dropout'], num_layers=nn_config['num_layers']
        ).to(DEVICE)
        if torch.cuda.device_count() > 1 and len(nn_config.get('device_ids', [])) > 0:
            model = nn.DataParallel(model, device_ids=nn_config['device_ids'])
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def evaluate_on_loader(models, data_loader):
    """
    Run ensemble models on data_loader and return flattened scores and truths.
    """
    all_y_score = []
    all_y_true = []
    with torch.no_grad():
        for pdbids, ligids, rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true in tqdm(
                data_loader,
                desc=f'evaluting',
                unit='batch'):
            tensors = [tensor.to(DEVICE) for tensor in
                       [rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true]]
            rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true = tensors

            # 1. 获取所有模型的 evidence 输出 (注意：移除了 .sigmoid())
            evidences = [m(rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask) for m in models]

            # 2. 对所有模型的 evidence 求均值 (集成策略)
            avg_evidence = torch.stack(evidences, 0).mean(0)  # 形状: [B, N, 2]

            # 3. 计算 Dirichlet 参数 alpha 和 S
            alpha = avg_evidence + 1.0
            S = torch.sum(alpha, dim=-1, keepdim=True)  # 形状: [B, N, 1]

            # 4. 计算正类(结合位点，索引为1)的预期概率
            prob_binding = alpha[..., 1] / S.squeeze(-1)  # 形状: [B, N]

            # 5. 使用 mask 提取有效残基
            y_score = torch.masked_select(prob_binding, mask == 1)
            y_true_masked = torch.masked_select(y_true, mask == 1)

            all_y_score.extend(y_score.cpu().detach().numpy())
            all_y_true.extend(y_true_masked.cpu().detach().numpy())

    return all_y_true, all_y_score



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



def test_evaluate_with_pdb_alignment(models, data_loader, processed_pdb_dir, best_th=0.35, save_raw_preds=False,
                                     raw_save_dir='raw_predictions'):

    results = []

    all_y_true_global = []
    all_y_score_late = []
    all_uncert_late = []

    if save_raw_preds and raw_save_dir is not None:
        os.makedirs(raw_save_dir, exist_ok=True)

    parser = PDBParser(QUIET=True)

    with torch.no_grad():
        for pdbids, ligids, rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true in tqdm(
                data_loader,
                desc='Testing & Aligning with PDB',
                unit='batch'):

            DEVICE = rfeat.device
            tensors = [tensor.to(DEVICE) for tensor in
                       [rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true]]
            rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask, y_true = tensors

            evidences = [m(rfeat, xyz, mask, angles, cmaps, ligand, lig_node_paths, lig_mask) for m in models]

            indiv_probs = []
            indiv_uncerts = []
            for ev in evidences:
                a = ev + 1.0
                s = torch.sum(a, dim=-1, keepdim=True)
                p = a[..., 1] / s.squeeze(-1)
                u = 2.0 / s.squeeze(-1)
                indiv_probs.append(p)
                indiv_uncerts.append(u)

            late_prob_binding = torch.stack(indiv_probs, 0).mean(0)
            late_uncertainty = torch.stack(indiv_uncerts, 0).mean(0)

            mask_cpu = mask.cpu().numpy()
            y_true_cpu = y_true.cpu().numpy()

            late_prob_cpu = late_prob_binding.cpu().numpy()
            late_uncert_cpu = late_uncertainty.cpu().numpy()

            indiv_probs_cpu = [p.cpu().numpy() for p in indiv_probs]
            indiv_uncerts_cpu = [u.cpu().numpy() for u in indiv_uncerts]

            batch_size = y_true.shape[0]
            for i in range(batch_size):
                sample_mask = mask_cpu[i] == 1

                y_true_np = y_true_cpu[i][sample_mask]
                late_score_np = late_prob_cpu[i][sample_mask]
                late_uncert_np = late_uncert_cpu[i][sample_mask]

                pdb_id = pdbids[i]
                lig_id = ligids[i]

                pdb_file = os.path.join(processed_pdb_dir, pdb_id, f"{pdb_id}_protein.pdb")
                chain_ids, res_seqs, res_names = [], [], []

                if os.path.exists(pdb_file):
                    structure = parser.get_structure(pdb_id, pdb_file)
                    model_pdb = structure[0]
                    ordered_chains = get_chain_order(model_pdb)

                    for chain in ordered_chains:
                        for residue in chain:
                            if residue.get_id()[0] == ' ':
                                chain_ids.append(chain.id)
                                res_seqs.append(residue.get_id()[1])
                                res_names.append(residue.resname)
                else:
                    chain_ids = ['UNK'] * len(y_true_np)
                    res_seqs = list(range(len(y_true_np)))
                    res_names = ['UNK'] * len(y_true_np)

                min_len = min(len(chain_ids), len(y_true_np))
                y_true_np = y_true_np[:min_len]
                late_score_np = late_score_np[:min_len]
                late_uncert_np = late_uncert_np[:min_len]

                if save_raw_preds and raw_save_dir is not None:
                    df_dict = {
                        'pdbid': [pdb_id] * min_len,
                        'ligid': [lig_id] * min_len,
                        'Chain_ID': chain_ids[:min_len],
                        'Residue_ID': res_seqs[:min_len],
                        'Residue_Name': res_names[:min_len],
                        'True_Label': y_true_np,
                    }

                    for m_idx in range(len(models)):
                        df_dict[f'Model_{m_idx}_Predicted_Prob'] = indiv_probs_cpu[m_idx][i][sample_mask][:min_len]
                        df_dict[f'Model_{m_idx}_Uncertainty'] = indiv_uncerts_cpu[m_idx][i][sample_mask][:min_len]

                    df_dict['LateFusion_Predicted_Prob'] = late_score_np
                    df_dict['LateFusion_Uncertainty'] = late_uncert_np

                    df_pdb = pd.DataFrame(df_dict)
                    save_name = os.path.join(raw_save_dir, f"{pdb_id}.csv")
                    df_pdb.to_csv(save_name, index=False)

                all_y_true_global.extend(y_true_np)
                all_y_score_late.extend(late_score_np)
                all_uncert_late.extend(late_uncert_np)

                if len(y_true_np) > 0:
                    data_late = calEval(y_true_np, late_score_np, best_th=best_th)
                    data_late['pdbid'] = pdb_id
                    data_late['ligid'] = lig_id
                    data_late['Mean_Uncertainty'] = float(np.mean(late_uncert_np))
                    results.append(data_late)

    columns = ['pdbid', 'ligid', 'Rec', 'SPE', 'Acc', 'Pre', 'F1', 'MCC', 'AUC', 'AUPR',
               'Mean_Uncertainty']
    df = pd.DataFrame(results, columns=columns)

    metrics_to_mean = ['Rec', 'SPE', 'Acc', 'Pre', 'F1', 'MCC', 'AUC', 'AUPR', 'Mean_Uncertainty']

    # 1. 计算 MEAN
    mean_late = df[metrics_to_mean].mean().to_dict()
    mean_late.update({'pdbid': 'MEAN', 'ligid': 'MEAN'})

    # 2. 计算 OVERALL
    overall_late = calEval(all_y_true_global, all_y_score_late, best_th=best_th)
    overall_late.update({'pdbid': 'OVERALL', 'ligid': 'OVERALL',
                         'Mean_Uncertainty': float(np.mean(all_uncert_late)) if len(all_uncert_late) > 0 else 0.0})

    # 合并到最终的 DataFrame
    df = pd.concat([
        df,
        pd.DataFrame([mean_late, overall_late])
    ], ignore_index=True)

    return df

def makeDir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# -------------------------
# Main pipeline
# -------------------------
def main():
    # 1) Train
    num_folds = 3
    save_dir = f"./Results/PDBbind/"
    get_threshold = True
    makeDir(save_dir)

    for fold_idx in range(num_folds):
        ckpt_path = os.path.join(save_dir, f'weights/fold{fold_idx}.ckpt')
        if os.path.exists(ckpt_path):
            print(f"Fold {fold_idx} 已经存在，跳过训练...")
            continue

        current_seed = 11 + fold_idx
        setALlSeed(current_seed)
        print(f"--- Starting fold {fold_idx} with seed {current_seed} ---")
        model = MDBind(
            rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'],
            hidden_dim=nn_config['hidden_dim'], heads=nn_config['heads'],
            augment_eps=nn_config['augment_eps'], rbf_num=nn_config['rbf_num'],
            top_k=nn_config['top_k'], attn_drop=nn_config['attn_drop'],
            dropout=nn_config['dropout'], num_layers=nn_config['num_layers']
        )
        train_process(model, save_dir, epochs=100, fold_idx=fold_idx)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(save_dir)
    models = load_models_for_ensemble(save_dir, num_folds=num_folds)
    if len(models) == 0:
        raise RuntimeError(save_dir, 'No trained model found. Check training step and checkpoint paths.')

    # 2) Determine best threshold on validation set
    if get_threshold:
        valid_data = MDBindDataset(
            dataset_split="val",
            summary_xlsx=summary_xlsx,
            feat_dir=nn_config['feat_dir'],
            lig_dict=nn_config['lig_dict'],
            smiles_dict=nn_config['smiles_dict'],
            nn_config=nn_config
        )
        valid_loader = DataLoader(valid_data, batch_size=nn_config['batch_size'], collate_fn=valid_data.collate_fn, num_workers=8)
        all_y_true, all_y_score = evaluate_on_loader(models, valid_loader)
        best_threshold, best_mcc, best_pred = getBestThreshold(all_y_true, all_y_score)
        appendText(os.path.join(save_dir, f'MD_Best_Threshold.txt'), f'{best_threshold} {best_mcc}\n')
    else:
        best_threshold = 0.34

    # 3) Test evaluation per ionic type
    processed_pdb_dir = "/home/lab_LiangSK/zhaoqc/Projects/BindingSitePrediction/Datasets/PDBbind/processed/"
    for test_name in ['test', 'val','train']:
    # for test_name in ['test']:
        test_data = MDBindDataset(
            dataset_split=test_name,
            summary_xlsx=summary_xlsx,
            feat_dir=nn_config['feat_dir'],
            lig_dict=nn_config['lig_dict'],
            smiles_dict=nn_config['smiles_dict'],
            nn_config=nn_config
        )
        test_loader = DataLoader(test_data, batch_size=nn_config['batch_size'], collate_fn=test_data.collate_fn, num_workers=8)
        print(f'{test_name} data length: {len(test_data)}')
        raw_save_dir = os.path.join(save_dir, f'{test_name}_predictions')
        results_df = test_evaluate_with_pdb_alignment(
            models=models,
            data_loader=test_loader,
            processed_pdb_dir=processed_pdb_dir,
            best_th=best_threshold,
            raw_save_dir=raw_save_dir,
            save_raw_preds=True
        )
        results_df.to_csv(os.path.join(save_dir, f'{test_name}.csv'), index=False)
        print('Evaluation finished. Results saved to:', os.path.join(save_dir, f'{test_name}.csv'))


if __name__ == '__main__':
    main()
