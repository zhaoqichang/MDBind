#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

# 1. Globally set the Hugging Face mirror endpoint to ensure all underlying requests use it
# If you're in China, please set https://hf-mirror.com/
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download


def download_all_weights(outpath):
    """
    Download pretrained weights for MDBind.
    """
    target_dir = os.path.join(outpath, "ankh-large")
    print(f"--> Using Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
    print(f"--> Downloading Ankh-large weights to {target_dir}...")

    # 2. snapshot_download will automatically use the HF_ENDPOINT environment variable
    snapshot_download(
        repo_id="ElnaggarLab/ankh-large",
        local_dir=target_dir,
        allow_patterns="*"
    )
    print("🎉 All models downloaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download pretrained weights for MDBind')
    parser.add_argument("-o", "--outpath", type=str, default='./../tools/', help='Saved path for weights')
    args = parser.parse_args()

    download_all_weights(outpath=args.outpath)