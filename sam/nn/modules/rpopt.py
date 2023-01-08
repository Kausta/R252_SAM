import torch

__all__ = ["RPOpt"]

# Based on https://github.com/davda54/sam/blob/main/sam.py

class RPOpt(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=1.0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(RPOpt, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def pre_forward(self):
        # Find all random perturbations
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                z = torch.randn_like(p)
                p._z = z
        # Take global norm
        z_norm = self._z_norm()
        # Perturb
        for group in self.param_groups:
            scale = group["rho"] / (z_norm + 1e-12)
            for p in group["params"]:
                e_w = p._z * scale.to(p)
                p.add_(e_w)  # go to the random perturbation "w + e(w)"
                del p._z

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def _z_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p._z.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups