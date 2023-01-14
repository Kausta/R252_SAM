import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.backends.cuda
from torchvision import datasets

import sam.trainers as trainers
from analysis.deep_nets import utils, sam, losses, data, utils_train
from sam.util import load_config
import argparse

import analysis.deep_nets.utils_eval as utils_eval

import wandb
import json
from omegaconf import OmegaConf, DictConfig

"""
Could probably have uploaded the results to wandb instead, but I think pandas will be more predictable and manageable for now.
"""


def main():
    df = pd.read_csv("analysis/sharpness_collection.csv", index_col='run_name')

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", type=str, default="config/mobilenetv2.yaml", help="Main Config File")
    # parser.add_argument("--optimizer", type=str, help="128-sam, n-sam, sgd, average_sam")
    # parser.add_argument("-m", "--m", type=int, help="sam, average_sam")
    parser.add_argument("--epoch", type=int, default=199)
    parser.add_argument('--measures', nargs='*')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--no_config', action="store_true", default=False)
    parser.add_argument('--rho', type=float)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")

    if args.no_config:
        conf = load_config("config/mobilenetv2.yaml")
    else:
        api = wandb.Api()
        entity, project = "mvp_diffusion", "R252_SAM"
        runs = api.runs(entity + "/" + project, per_page=1000)
        # run this with debugger. dammit.
        run = [run for run in runs.objects if run.name == args.run_name][0]

        config_flat = json.loads(run.json_config)

        # Generated by chatGPT
        config = {}
        for k, v in config_flat.items():
            key_parts = k.split('/')
            if len(key_parts) == 3:
                if key_parts[0] not in config:
                    config[key_parts[0]] = {}
                if key_parts[1] not in config[key_parts[0]]:
                    config[key_parts[0]][key_parts[1]] = {}
                config[key_parts[0]][key_parts[1]][key_parts[2]] = v['value']
            elif len(key_parts) == 2:
                if key_parts[0] not in config:
                    config[key_parts[0]] = {}
                config[key_parts[0]][key_parts[1]] = v['value']
            else:
                config[k] = v['value']

        conf = OmegaConf.create(config)

    Model = getattr(trainers, conf.trainer.trainer)
    model = Model(conf)

    """
    Graphs to generate
    - Maybe first just create a pandas table of properties of all checkpoints. First just the last ones.
    - Train loss
    - Test loss
    - Test accuracy
    - m, 5000, 1-sharpness
    - average m, 1000, 1-sharpness.
    - shit, there's also rho. Let's just have it match the model atm.
    Definitely need to scale down ambitions.
    """

    checkpoint_path = 'checkpoints/' + args.run_name + '/checkpoint-' + str(args.epoch) + '.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    batches_128 = data.get_loaders('cifar10', 512, 128, 'train', True, False)
    batches_5000 = data.get_loaders('cifar10', 20000, 5000, 'train', True, False)

    rho = 0.1

    if 'adversarial-128' in args.measures:
        sharpness, obj, err, obj_orig, err_orig = eval_sharpness_wrapper(device, model, rho, batches_128)
        df.loc[args.run_name, 'sharpness-128'] = sharpness
        df.loc[args.run_name, 'obj-128'] = obj
        df.loc[args.run_name, 'err-128'] = err
        df.loc[args.run_name, 'obj_orig-128'] = obj_orig
        df.loc[args.run_name, 'err_orig-128'] = err_orig
        df.to_csv('analysis/sharpness_collection.csv')

    if 'adversarial-5000' in args.measures:
        sharpness, obj, err, obj_orig, err_orig = eval_sharpness_wrapper(device, model, rho, batches_5000)
        df.loc[args.run_name, 'sharpness-5000'] = sharpness
        df.loc[args.run_name, 'obj-5000'] = obj
        df.loc[args.run_name, 'err-5000'] = err
        df.loc[args.run_name, 'obj_orig-5000'] = obj_orig
        df.loc[args.run_name, 'err_orig-5000'] = err_orig
        df.to_csv('analysis/sharpness_collection.csv')

    if 'average-128' in args.measures:
        average_sharpness = eval_avg_sharpness(device, model, batches_128, noisy_examples='none', sigma=rho, n_repeat=1)
        df.loc[args.run_name, 'average-sharpness-128'] = average_sharpness
        df.to_csv('analysis/sharpness_collection.csv')

    if 'average-5000' in args.measures:
        average_sharpness = eval_avg_sharpness(device, model, batches_5000, noisy_examples='none', sigma=rho,
                                               n_repeat=1)
        df.loc[args.run_name, 'average-sharpness-5000'] = average_sharpness
        df.to_csv('analysis/sharpness_collection.csv')

    if 'custom' in args.measures:
        for rho in [0.05, 0.2]:
            sharpness, obj, err, obj_orig, err_orig = eval_sharpness_wrapper(device, model, rho, batches_128)
            df.loc[args.run_name, 'sharpness-128-' + str(rho)] = sharpness
            df.loc[args.run_name, 'obj-128-' + str(rho)] = obj
            df.loc[args.run_name, 'err-128-' + str(rho)] = err
            df.loc[args.run_name, 'obj_orig-128-' + str(rho)] = obj_orig
            df.loc[args.run_name, 'err_orig-128-' + str(rho)] = err_orig
            df.to_csv('analysis/sharpness_collection.csv')
            sharpness, obj, err, obj_orig, err_orig = eval_sharpness_wrapper(device, model, rho, batches_5000)
            df.loc[args.run_name, 'sharpness-5000-' + str(rho)] = sharpness
            df.loc[args.run_name, 'obj-5000-' + str(rho)] = obj
            df.loc[args.run_name, 'err-5000-' + str(rho)] = err
            df.loc[args.run_name, 'obj_orig-5000-' + str(rho)] = obj_orig
            df.loc[args.run_name, 'err_orig-5000-' + str(rho)] = err_orig
            df.to_csv('analysis/sharpness_collection.csv')

    df.loc[args.run_name, 'epoch'] = args.epoch

    if not args.no_config:
        for key, value in config_flat.items():
            df.loc[args.run_name, key] = value['value']

    df.to_csv('analysis/sharpness_collection.csv')


def eval_avg_sharpness(device, model, batches, noisy_examples, sigma, n_repeat=5):
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss_diff = 0
    for i in range(n_repeat):
        _, loss_before, _ = utils_eval.rob_err(device, batches, model, 0, 0, scaler, 0, 1, noisy_examples=noisy_examples, n_batches=1)
        weights_delta_dict = utils_train.perturb_weights(device, model, add_weight_perturb_scale=sigma, mul_weight_perturb_scale=0,
                                                         weight_perturb_distr='gauss')
        _, loss_after, _ = utils_eval.rob_err(device, batches, model, 0, 0, scaler, 0, 1, noisy_examples=noisy_examples, n_batches=1)
        utils_train.subtract_weight_delta(model, weights_delta_dict)

        loss_diff += (loss_after - loss_before) / n_repeat

    return loss_diff


def eval_sharpness_wrapper(device, model, rho, batches_sharpness):

    loss_dict = {
        'ce': losses.cross_entropy(),
        'ce_offset': losses.cross_entropy_with_offset(loss_offset=0.1),
        'gce': losses.generalized_cross_entropy(q=0.7),
        'smallest_k_ce': losses.smallest_k_cross_entropy(frac_rm_per_batch=0.0)
    }
    loss_f = loss_dict['ce']

    step_size = rho
    n_iters = 1
    n_restarts = 1
    apply_step_size_schedule = False
    no_grad_norm = False
    layer_name_pattern = 'all'
    random_targets = False
    batch_transfer = False

    # Maybe it retrains and does it for all checkpoints?
    return eval_sharpness(
        device, model, batches_sharpness, loss_f, rho, step_size, n_iters, n_restarts,
        apply_step_size_schedule, no_grad_norm,
        layer_name_pattern, random_targets, batch_transfer,
        verbose=True)


# Based on code from Towards understanding sharpness aware minimization paper
def eval_sharpness(device, model, batches, loss_f, rho, step_size, n_iters, n_restarts=1, apply_step_size_schedule=False,
                   no_grad_norm=False, layer_name_pattern='all', random_targets=False, batch_transfer=False,
                   rand_init=False, verbose=False):
    orig_model_state_dict = copy.deepcopy(model.state_dict())
    model.to(device)

    n_batches, best_obj_sum, final_err_sum, final_grad_norm_sum = 0, 0, 0, 0
    objs, errs, obj_origs, err_origs = [], [], [], []
    for i_batch, (x, _, y, _, _) in enumerate(batches):
        x, y = x.to(device), y.to(device)

        def f(model):
            obj = loss_f(model(x), y)
            return obj

        obj_orig = f(model).detach()
        err_orig = (model(x).max(1)[1] != y).float().mean().item()

        delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}
        orig_param_dict = {param: param.clone() for param in model.parameters()}
        best_obj, final_err, final_grad_norm = 0, 0, 0
        for restart in range(n_restarts):
            # random init on the sphere of radius `rho`
            if rand_init:
                delta_dict = sam.random_init_on_sphere_delta_dict(delta_dict, rho)
                for param in model.parameters():
                    param.data += delta_dict[param]
            else:
                delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}

            if rand_init:
                n_cls = 10
                y_target = torch.clone(y)
                for i in range(len(y_target)):
                    lst_classes = list(range(n_cls))
                    lst_classes.remove(y[i])
                    y_target[i] = np.random.choice(lst_classes)

            def f_opt(model):
                if not rand_init:
                    return f(model)
                else:
                    return -loss_f(model(x), y_target)

            for iter in range(n_iters):
                step_size_curr = utils_eval.step_size_schedule(step_size,
                                                               iter / n_iters) if apply_step_size_schedule else step_size
                delta_dict = sam.weight_ascent_step(model, f_opt, orig_param_dict, delta_dict, step_size_curr, rho,
                                                    layer_name_pattern, no_grad_norm=no_grad_norm, verbose=False)

            if batch_transfer:
                delta_dict_loaded = torch.load('deltas/gn_erm/batch{}.pth'.format(restart))
                delta_dict_loaded = {param: delta for param, delta in zip(model.parameters(),
                                                                          delta_dict_loaded.values())}  # otherwise `param` doesn't work directly as a key
                for param in model.parameters():
                    param.data = orig_param_dict[param] + delta_dict_loaded[param]

            obj, grad_norm = utils.eval_f_val_grad(model, f)
            err = (model(x).max(1)[1] != y).float().mean().item()

            # if obj > best_obj:
            if err > final_err:
                best_obj, final_err, final_grad_norm = obj, err, grad_norm
            model.load_state_dict(orig_model_state_dict)
            objs.append(obj)
            errs.append(err)
            obj_origs.append(obj_orig)
            err_origs.append(err_orig)
            if verbose:
                delta_norm_total = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
                print(
                    '[restart={}] Sharpness: obj={:.4f}, err={:.2%}, delta_norm={:.2f} (step={:.3f}, rho={}, n_iters={})'.format(
                        restart + 1, obj - obj_orig, err - err_orig, delta_norm_total, step_size, rho, n_iters))
                print(
                    '[restart={}] Sharpness: obj={:.4f}, err={:.2%}'.format(
                        restart + 1, obj, err))
                print(
                    '[restart={}] Sharpness: obj_orig={:.4f}, err_orig={:.2%}'.format(
                        restart + 1, obj_orig, err_orig))

        best_obj, final_err = best_obj - obj_orig, final_err - err_orig  # since we evaluate sharpness, i.e. the difference in the loss
        best_obj_sum, final_err_sum, final_grad_norm_sum = best_obj_sum + best_obj, final_err_sum + final_err, final_grad_norm_sum + final_grad_norm
        n_batches += 1

    print(type(objs[0]))
    print(type(errs[0]))
    print(type(obj_origs[0]))
    print(type(err_origs[0]))
    return np.mean(objs) - np.mean(obj_origs), np.mean(objs), np.mean(errs), np.mean(obj_origs), np.mean(err_origs)


if __name__ == '__main__':
    main()

