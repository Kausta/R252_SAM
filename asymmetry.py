import copy
import torch
import sam
import argparse
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf

USE_CUDA = False


def strip_prefix(string, prefix):
    if string.startswith(prefix):
        return string[len(prefix) :]
    else:
        return string


def get_loss(batches, model, loss, label):
    loss_total = 0

    for x, y in tqdm(batches, desc="Calculating loss %s" % label, leave=False):
        if USE_CUDA:
            x, y = x.cuda(), y.cuda()

        output = model(x)
        loss_total += loss(output, y)

    return loss_total.cpu().numpy() / len(batches)


def get_random_unit_perturbation(model):
    param_shapes = [param.shape for param in model.parameters()]
    param_sizes = [np.prod(shape) for shape in param_shapes]
    model_size = sum(size for size in param_sizes)
    vector = torch.rand(model_size)

    if USE_CUDA:
        vector = vector.cuda()

    norm = torch.linalg.norm(vector)
    vector = vector / norm

    perturbation = []
    pointer = 0
    for size, shape in zip(param_sizes, param_shapes):
        param_permutation = vector[pointer : (pointer + size)].view(shape)
        perturbation.append(param_permutation)
        pointer += size

    return perturbation


def perturbate_models(model, scales):
    vector = get_random_unit_perturbation(model)
    models = [copy.deepcopy(model) for _ in scales]

    for model, scale in zip(models, scales):
        for param, element in zip(model.parameters(), vector):
            param.add_(scale * element)

    return models


def load_checkpoint(file_path):
    checkpoint = torch.load(
        file_path,
        map_location=torch.device("cuda" if USE_CUDA else "cpu"),
    )

    config = OmegaConf.create(checkpoint["hyper_parameters"])
    config.data.data_root = "/tmp"

    model = sam.models.MobileNetV2(config)
    state_dict = {
        strip_prefix(k, "model."): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    loss = torch.nn.CrossEntropyLoss(
        label_smoothing=0.0
        if config.loss.label_smoothing is None
        else config.loss.label_smoothing
    )

    data_loaders = sam.data.DataModule(
        config, getattr(sam.data, config.data.dataset_cls)
    )
    data_loaders.setup()
    test_batches = data_loaders.test_dataloader()

    if USE_CUDA:
        model = model.cuda()

    return model, loss, test_batches


def get_random_direction(model, loss, batches, intervals):
    models = perturbate_models(model, intervals)
    losses = [
        get_loss(batches, model, loss, i)
        for i, model in tqdm(
            enumerate(models), total=len(models), desc="Getting model losses"
        )
    ]

    return losses


def get_asymmetry(model, loss, batches, iterations, scale):
    total = 0
    zero_loss = get_loss(batches, model, loss, "initial")

    for i in tqdm(range(iterations), desc="Calculating asymmetry"):
        pos_model, neg_model = perturbate_models(model, [-1 * scale, 1 * scale])

        pos_loss = get_loss(batches, pos_model, loss, "pos_%d" % i)
        neg_loss = get_loss(batches, neg_model, loss, "neg_%d" % i)

        pos_delta = pos_loss - zero_loss
        neg_delta = neg_loss - zero_loss

        total += min(pos_delta, neg_delta) / max(pos_delta, neg_delta)

    return total / iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input-checkpoint-path", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--plot-random-direction", nargs="+", action="append", type=float)
    parser.add_argument("--get-asymmetry", nargs=2, action="append", type=int)

    args = parser.parse_args()

    if args.device == "cuda":
        USE_CUDA = True

    with torch.no_grad():
        model, loss, batches = load_checkpoint(args.input_checkpoint_path)

        if args.plot_random_direction is not None:
            args.plot_random_direction = np.concatenate(args.plot_random_direction)
            losses = get_random_direction(
                model, loss, batches, args.plot_random_direction
            )
            print("x:", args.plot_random_direction)
            print("loss:", losses)

        if args.get_asymmetry is not None:
            args.get_asymmetry = np.concatenate(args.get_asymmetry)
            asymmetry = get_asymmetry(
                model, loss, batches, args.get_asymmetry[0], args.get_asymmetry[1]
            )
            print("Asymmetry factor:", asymmetry)
