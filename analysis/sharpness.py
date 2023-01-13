import copy

import numpy as np
import torch
import torch.backends.cuda
from torchvision import datasets

import sam.trainers as trainers
from analysis.deep_nets import utils, sam, losses, data
from sam.util import load_config
import argparse

import analysis.deep_nets.utils_eval as utils_eval


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/mobilenetv2.yaml", help="Main Config File")
    args = parser.parse_args()
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu")
    config = load_config(args.config)
    Model = getattr(trainers, config.trainer.trainer)
    model = Model(config)

    sam_checkpoints = [
        'checkpoints/distinctive-disco-153/checkpoint-199.ckpt',
        'checkpoints/trim-water-150/checkpoint-199.ckpt',
        'checkpoints/treasured-bush-159/checkpoint-199.ckpt',
        'checkpoints/silver-haze-107/last.ckpt',
    ]

    sgd_checkpoints = [
        'checkpoints/stoic-jazz-100/last.ckpt',
        'checkpoints/kind-durian-155/last.ckpt',
    ]

    average_checkpoints = [

    ]

    """
    Graphs to generate
    - Maybe first just create a pandas table of properties of all checkpoints. First just the last ones.
    - Train loss
    - Test loss
    - Test accuracy
    - m, 5000, 1-sharpness
    - average m, 1000, 1-sharpness.
    """



    checkpoint_path = sam_checkpoints[3]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    batches_sharpness = data.get_loaders('cifar10', 512, 128, 'train', True, False)

    loss_dict = {
        'ce': losses.cross_entropy(),
        'ce_offset': losses.cross_entropy_with_offset(loss_offset=0.1),
        'gce': losses.generalized_cross_entropy(q=0.7),
        'smallest_k_ce': losses.smallest_k_cross_entropy(frac_rm_per_batch=0.0)
    }
    loss_f = loss_dict['ce']

    rho = 0.1
    step_size = rho
    n_iters = 1
    n_restarts = 1
    apply_step_size_schedule = False
    no_grad_norm = False
    layer_name_pattern = 'all'
    random_targets = False
    batch_transfer = False

    print(checkpoint_path)

    # Maybe it retrains and does it for all checkpoints?
    eval_sharpness(
        device, model, batches_sharpness, loss_f, rho, step_size, n_iters, n_restarts,
        apply_step_size_schedule, no_grad_norm,
        layer_name_pattern, random_targets, batch_transfer,
        verbose=True)


# Based on code from Towards understanding sharpness aware minimization paper
def eval_sharpness(device, model, batches, loss_f, rho, step_size, n_iters, n_restarts, apply_step_size_schedule=False,
                   no_grad_norm=False, layer_name_pattern='all', random_targets=False, batch_transfer=False,
                   rand_init=False, verbose=False):
    orig_model_state_dict = copy.deepcopy(model.state_dict())

    n_batches, best_obj_sum, final_err_sum, final_grad_norm_sum = 0, 0, 0, 0
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


if __name__ == '__main__':
    main()

