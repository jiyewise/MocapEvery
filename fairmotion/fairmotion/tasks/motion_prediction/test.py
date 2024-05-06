# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import numpy as np
import os
import torch
from multiprocessing import Pool

from fairmotion.data import amass_dip, bvh
from fairmotion.core import motion as motion_class
from fairmotion.tasks.motion_prediction import generate, metrics, utils
from fairmotion.ops import conversions, motion as motion_ops
from fairmotion.utils import utils as motion_utils

from IPython import embed

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def prepare_model(path, num_predictions, args, device):
    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def run_model(model, data_iter, max_len, device, mean, std):
    pred_seqs = []
    src_seqs, tgt_seqs = [], []
    for src_seq, tgt_seq in data_iter:
        max_len = max_len if max_len else tgt_seq.shape[1]
        # src_seq shape: [batch, src_len, 3J] (aa representation)] [64, 120, 72]
        # tgt_seq shape: [batch, tgt_len, 3J] (aa rep) [64, 24, 72] 
        src_seqs.extend(src_seq.to(device="cpu").numpy()) # src_seqs: list of length batch size(64) src_seqs[0].shape [120, 72]
        tgt_seqs.extend(tgt_seq.to(device="cpu").numpy()) # tgt_seqs: list of length batch size(64) tgt_seqs[0].shape [24, 72]
        pred_seq = (
            generate.generate(model, src_seq, max_len, device)
            .to(device="cpu")
            .numpy()
        ) # output shape: [batch, tgt_seq, 3J]
        pred_seqs.extend(pred_seq)  # pred_seqs: list of length batch size(64) pred_seqs[0].shape [24, 72]
    return [
        utils.unnormalize(np.array(l), mean, std)
        for l in [pred_seqs, src_seqs, tgt_seqs]
    ]


def save_seq(i, pred_seq, src_seq, tgt_seq, skel):
    # seq_T contains pred, src, tgt data in the same order
    motions = [
        motion_class.Motion.from_matrix(seq, skel)
        for seq in [pred_seq, src_seq, tgt_seq]
    ]
    ref_motion = motion_ops.append(motions[1], motions[2])
    pred_motion = motion_ops.append(motions[1], motions[0])
    bvh.save(
        ref_motion, os.path.join(args.save_output_path, "ref", f"{i}.bvh"),
    )
    bvh.save(
        pred_motion, os.path.join(args.save_output_path, "pred", f"{i}.bvh"),
    )


def convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep):
    ops = utils.convert_fn_to_R(rep)
    seqs_T = [
        conversions.R2T(utils.apply_ops(seqs, ops))
        for seqs in [pred_seqs, src_seqs, tgt_seqs]
    ]
    return seqs_T


def save_motion_files(seqs_T, args):
    idxs_to_save = [i for i in range(0, len(seqs_T[0]), len(seqs_T[0]) // 10)]
    # step by len/10
    # if len is 1713, idxs_to_save [0, 171, 342, 513, 684, 855, 1026, 1197, 1368, 1539, 1710]
    amass_dip_motion = amass_dip.load(
        file=None, load_skel=True, load_motion=False,
    ) # create only skel

    # utils imported differently. bug in original library
    motion_utils.create_dir_if_absent(os.path.join(args.save_output_path, "ref"))
    motion_utils.create_dir_if_absent(os.path.join(args.save_output_path, "pred"))

    pool = Pool(10)
    indices = range(len(seqs_T[0]))
    skels = [amass_dip_motion.skel for _ in indices]
    pool.starmap(
        save_seq, [list(zip(indices, *seqs_T, skels))[i] for i in idxs_to_save]
    )


def calculate_metrics(pred_seqs, tgt_seqs):
    metric_frames = [6, 12, 18, 24]
    R_pred, _ = conversions.T2Rp(pred_seqs)
    R_tgt, _ = conversions.T2Rp(tgt_seqs)
    euler_error = metrics.euler_diff(
        R_pred[:, :, amass_dip.SMPL_MAJOR_JOINTS],
        R_tgt[:, :, amass_dip.SMPL_MAJOR_JOINTS],
    )
    euler_error = np.mean(euler_error, axis=0)
    mae = {frame: np.sum(euler_error[:frame]) for frame in metric_frames}
    return mae


def test_model(model, dataset, rep, device, mean, std, max_len=None):
    pred_seqs, src_seqs, tgt_seqs = run_model(
        model, dataset, max_len, device, mean, std,
    )
    # shape of pred_seqs [1713(# of total frames in the test set), tgt_len, 3J] [1713, 24, 72]
    # shape of src_seqs [# of total frames, src_len, 3J] [1713, 120, 72]
    # shape of tgt_seqs [# of total frames, tgt_len, 3J] [1713, 24, 72]
    seqs_T = convert_to_T(pred_seqs, src_seqs, tgt_seqs, rep)
    # seqs_T: list of length 3 [pred_seqs, src_seqs, tgt_seqs]
    # seqs_T[0] shape: [# of frames, tgt_len, J, 4, 4]
    # seqs_T[1] shape: [# of frames, src_len, J, 4, 4]
    # seqs_T[2] shape: [# of frames, tgt_len, J, 4, 4]

    # Calculate metric only when generated sequence has same shape as reference
    # target sequence
    if len(pred_seqs) > 0 and pred_seqs[0].shape == tgt_seqs[0].shape:
        mae = calculate_metrics(seqs_T[0], seqs_T[2]) # compare pred and tgt
    return seqs_T, mae


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Preparing dataset")
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(args.preprocessed_path, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
    )
    # number of predictions per time step = num_joints * angle representation
    data_shape = next(iter(dataset["train"]))[0].shape # src shape: [64, 120, 72]
    num_predictions = data_shape[-1] # 72 [3J] for angle-axis representations

    logging.info("Preparing model")
    model = prepare_model(
        f"{args.save_model_path}/{args.epoch if args.epoch else 'best'}.model",
        num_predictions,
        args,
        device,
    )

    logging.info("Running model")
    _, rep = os.path.split(args.preprocessed_path.strip("/")) # if the path is /preprocess/aa, the _ is the front part and rep is aa
    seqs_T, mae = test_model(
        model, dataset["test"], rep, device, mean, std, args.max_len
    )
    logging.info(
        "Test MAE: "
        + " | ".join([f"{frame}: {mae[frame]}" for frame in mae.keys()])
    )

    if args.save_output_path:
        logging.info("Saving results")
        save_motion_files(seqs_T, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions and post process them"
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled" "files from dataset",
        required=True,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to saved models",
        required=True,
    )
    parser.add_argument(
        "--save-output-path",
        type=str,
        help="Path to store predicted motion",
        default=None,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
    )
    parser.add_argument(
        "--max-len", type=int, help="Length of seq to generate", default=None,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for testing", default=64
    )
    parser.add_argument(
        "--shuffle", action='store_true',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Model from epoch to test, will test on best"
        " model if not specified",
        default=None,
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq archtiecture to be used",
        default="seq2seq",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn",
        ],
    )

    args = parser.parse_args()
    main(args)
