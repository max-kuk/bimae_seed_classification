"""
Export Outlier Detection datasets from tfrecords

How to run:
    nohup python -u export_sc_dataset.py > export.log &
"""

import argparse
import logging

from datasets import export_sc_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Export Supervised Classification Dataset from tfrecords"
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/mnt/data/processed_data/",
        required=False,
        help="Path to tfrecords",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="/mnt/data/sc/bimodal_24_192_300_bands",
        required=False,
        help="Path to save data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        required=False,
        help="Ratio of train/val split",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        required=False,
        help="Ratio of train/val split",
    )
    parser.add_argument(
        "--compression",
        default="GZIP",
        required=False,
        help="Compression type (None, GZIP)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=20,
        required=False,
        help="Number of shards per dataset",
    )

    args = parser.parse_args()
    # add timestamp to log
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting export...")

    export_sc_datasets(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        compression=args.compression,
        num_shards=args.num_shards,
    )
    logging.info("Export completed!")


if __name__ == "__main__":
    main()
