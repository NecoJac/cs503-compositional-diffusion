import pickle
import torch as th
import dnnlib
import diff_collage as dc
import PIL.Image
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

def load_edm_model(network_pkl, device="cuda"):
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device).eval()
    return net

class EDMEpsWrapper:
    def __init__(self, net, class_idx=None):
        self.net = net
        self.class_idx = class_idx

    def __call__(self, xs, scalar_t, enable_grad=False):
        B = xs.shape[0]
        device = xs.device

        if self.class_idx is not None:
            class_labels = th.zeros(B, 1000, device=device)
            class_labels[:, self.class_idx] = 1.0
        else:
            class_labels = None

        if not th.is_tensor(scalar_t):
            scalar_t = th.tensor(scalar_t, device=device, dtype=xs.dtype)
        else:
            scalar_t = scalar_t.to(device)

        if scalar_t.ndim == 0:
            sigma = scalar_t.repeat(B)
        elif scalar_t.ndim == 1:
            sigma = scalar_t if scalar_t.shape[0] == B else scalar_t.repeat(B)
        else:
            raise ValueError("scalar_t must be scalar or 1D")

        x_in = xs.to(th.float32)
        sigma_in = sigma.to(th.float32)
        sigma_img = sigma_in.view(-1, 1, 1, 1)

        context = th.enable_grad() if enable_grad else th.no_grad()
        with context:
            x0_hat = self.net(x_in, sigma_in, class_labels)
            eps_hat = (x_in - x0_hat) / sigma_img

        return eps_hat.to(xs.dtype)



def save_tensor_image(x, path):
    x = (x.clamp(-1, 1) * 127.5 + 128).to(th.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    PIL.Image.fromarray(x).save(path)

def merge_xy_yz_take_xy_overlap(xy, yz, overlap_w):
    left = xy[:, :, :, :-overlap_w]   # x part
    overlap = xy[:, :, :, -overlap_w:]  # take y from xy
    right = yz[:, :, :, overlap_w:]   # z part
    return th.cat([left, overlap, right], dim=3)


if __name__ == "__main__":

    # Params. for generation
    img_shape=(3, 64, 64)
    img_half_w = img_shape[1] // 2
    index_imagenet = 970 # alp
    batch_size = 1
    n_iters = 5
    n_step=20
    ts_order=5
    s_churn=0.0
    lr = 1e-2

    device = "cuda"
    
    # Load NVDIA model
    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
    network_pkl = f"{model_root}/edm-imagenet-64x64-cond-adm.pkl"
    
    
    # Initialize model and wrappers
    net = load_edm_model(network_pkl, device=device)
    # Freeze model params, since we only optimize the initial noise
    for p in net.parameters():
        p.requires_grad_(False)
    # Wrappers 
    eps_xy = EDMEpsWrapper(net, class_idx=index_imagenet)
    eps_yz = EDMEpsWrapper(net, class_idx=index_imagenet)

    # Two workers, each samples one patch
    worker_xy = dc.BaseWorker(shape=img_shape, eps_scalar_t_fn=eps_xy)
    worker_yz = dc.BaseWorker(shape=img_shape, eps_scalar_t_fn=eps_yz)

    # Learnable source noises
    latents_xy = worker_xy.generate_xT(batch_size).detach().clone().requires_grad_(True)
    latents_yz = worker_yz.generate_xT(batch_size).detach().clone().requires_grad_(True)

    # Init. optimizers for initial noise
    optimizer = th.optim.Adam([latents_xy, latents_yz], lr=lr)


    # Iterate over noises with loss optimization
    
    for i in range(n_iters):

        optimizer.zero_grad()

        # Generate (x,y) from (eps_x, eps_y)
        xy = dc.sampling(
            x=latents_xy,
            rev_ts=worker_xy.rev_ts(n_step=n_step, ts_order=ts_order),
            noise_fn=worker_xy.noise,
            x0_pred_fn= lambda xt, t: worker_xy.x0_fn(xt, t, enable_grad=True),
            solver="euler",
            s_churn=s_churn,
            return_traj=False,
        )

        # Reconstruct (y,z) from (eps_y, eps_z)

        yz = dc.sampling(
        x=latents_yz,
        rev_ts=worker_yz.rev_ts(n_step=n_step, ts_order=ts_order),
        noise_fn=worker_yz.noise,
        x0_pred_fn=lambda xt, t: worker_yz.x0_fn(xt, t, enable_grad=True),
        solver="euler",
        s_churn=s_churn,
        return_traj=False,
        )

        # Build loss
        y_from_xy = xy[:, :, :, -img_half_w:]
        y_from_yz = yz[:, :, :, :img_half_w]

        loss = ((y_from_xy - y_from_yz) ** 2).mean()

        # Backpropagate to initial noise
        loss.backward()
        optimizer.step()
    
    # Report final loss and save image
    print("Final MSE loss on overlap region:", loss.item())
    final_img = merge_xy_yz_take_xy_overlap(xy, yz, overlap_w=img_half_w)
    save_tensor_image(final_img[0], "final_decoupled_gen.png")
    