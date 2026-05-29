import pickle
import torch as th
import dnnlib
import diff_collage as dc
import PIL.Image

def load_edm_model(network_pkl, device="cuda"):
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device).eval()
    return net

class EDMDualEpsWrapper:
    """
    Same EDM net used in two modes:
      - unconditional / null-label branch
      - conditional class-label branch

    Supports either:
      class_labels = int
    or
      class_labels = list[int] / tensor[num_patches]
    """

    def __init__(self, net, num_classes=1000):
        self.net = net
        self.num_classes = num_classes

    def _expand_sigma(self, scalar_t, B, device, dtype):
        if not th.is_tensor(scalar_t):
            scalar_t = th.tensor(scalar_t, device=device, dtype=dtype)
        else:
            scalar_t = scalar_t.to(device=device, dtype=dtype)

        if scalar_t.ndim == 0:
            sigma = scalar_t.repeat(B)
        elif scalar_t.ndim == 1:
            if scalar_t.shape[0] == B:
                sigma = scalar_t
            elif scalar_t.shape[0] == 1:
                sigma = scalar_t.repeat(B)
            else:
                raise ValueError(f"Bad sigma shape {scalar_t.shape} for batch {B}")
        else:
            raise ValueError("scalar_t must be scalar or 1D")

        return sigma

    def _build_class_labels(self, total_B, device, class_labels=None):
        """
        total_B = total number of inputs to the network in this call.

        class_labels:
          - None -> unconditional
          - int  -> same class for all inputs
          - list/tensor length K -> repeated over batch if total_B % K == 0
        """
        if class_labels is None:
            return None

        if isinstance(class_labels, int):
            idxs = th.full((total_B,), class_labels, device=device, dtype=th.long)

        elif isinstance(class_labels, (list, tuple)):
            idxs = th.tensor(class_labels, device=device, dtype=th.long)
            K = idxs.numel()
            if total_B % K != 0:
                raise ValueError(
                    f"Got {K} labels but network batch is {total_B}; "
                    "cannot tile labels over batch."
                )
            idxs = idxs.repeat(total_B // K)

        elif th.is_tensor(class_labels):
            idxs = class_labels.to(device=device, dtype=th.long).flatten()
            K = idxs.numel()
            if total_B % K != 0:
                raise ValueError(
                    f"Got {K} labels but network batch is {total_B}; "
                    "cannot tile labels over batch."
                )
            idxs = idxs.repeat(total_B // K)

        else:
            raise TypeError("class_labels must be None, int, list, tuple, or tensor")

        onehot = th.zeros(total_B, self.num_classes, device=device)
        onehot.scatter_(1, idxs.unsqueeze(1), 1.0)
        return onehot

    def _forward_eps(self, xs, scalar_t, class_labels=None, enable_grad=False):
        B = xs.shape[0]
        device = xs.device

        sigma = self._expand_sigma(scalar_t, B, device, xs.dtype)
        sigma_img = sigma.view(-1, 1, 1, 1)

        class_vec = self._build_class_labels(B, device, class_labels)

        x_in = xs.to(th.float32)
        sigma_in = sigma.to(th.float32)

        context = th.enable_grad() if enable_grad else th.no_grad()
        with context:
            x0_hat = self.net(x_in, sigma_in, class_vec)
            eps_hat = (x_in - x0_hat) / sigma_img

        return eps_hat.to(xs.dtype)

    def eps_uncond(self, xs, scalar_t, enable_grad=False):
        return self._forward_eps(xs, scalar_t, class_labels=None, enable_grad=enable_grad)

    def eps_cond(self, xs, scalar_t, class_labels, enable_grad=False):
        return self._forward_eps(xs, scalar_t, class_labels=class_labels, enable_grad=enable_grad)


def save_tensor_image(x, path):
    x = (x.clamp(-1, 1) * 127.5 + 128).to(th.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    PIL.Image.fromarray(x).save(path)


def generate_long_image_class_cfg(
    net,
    class_labels,          # int or list[int] length num_img
    device="cuda",
    batch_size=1,
    n_step=40,
    overlap_size=32,
    num_img=5,
    ts_order=5,
    img_shape=(3, 64, 64),
    s_churn=10.0,
    guidance_scale=3.0,
):
    edm_dual = EDMDualEpsWrapper(net)

    worker = dc.CondIndLongClassCFG(
        shape=img_shape,
        edm_dual_eps_wrapper=edm_dual,
        num_img=num_img,
        class_labels=class_labels,
        overlap_size=overlap_size,
        guidance_scale=guidance_scale,
    )

    sample = dc.sampling(
        x=worker.generate_xT(batch_size).to(device),
        noise_fn=worker.noise,
        rev_ts=worker.rev_ts(n_step, ts_order),
        x0_pred_fn=worker.x0_fn,
        s_churn=s_churn,
        return_traj=False,
    )

    return sample

if __name__ == "__main__":

    class_list = [
        ("lakeside", 975),
        ("volcano", 980),
        ("alp", 970),
        ("coral_reef", 973),
    ]

    guidance_scales = [1.0, 2.0, 3.0, 5.0]

    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
    network_pkl = f"{model_root}/edm-imagenet-64x64-cond-adm.pkl"

    device = "cuda"
    net = load_edm_model(network_pkl, device=device)

    # n generated patches -> n  factor  nodes
    var_class_labels = [980, 980, 980, 980, 975, 975, 975, 975]
    label_name = "volcano4_lakeside4"

    for guidance_scale in guidance_scales:
        print(f"Generating image with variable-node labels {var_class_labels}, gs={guidance_scale}")

        sample = generate_long_image_class_cfg(
            net=net,
            class_labels=var_class_labels,
            device=device,
            batch_size=1,
            n_step=40,
            overlap_size=32,
            num_img=8,
            img_shape=(3, 64, 64),
            s_churn=10.0,
            guidance_scale=guidance_scale,
        )

        save_path = f"image_outputs/sample_cond_class_factors_{label_name}_gs_{guidance_scale}.png"
        save_tensor_image(sample[0], save_path)



        '''

        class_list = [
        ("lakeside", 975),
        ("volcano", 980),
        ("alp", 970),
        ("coral_reef", 973),
    ]

        guidance_scales = [1.0, 1.1, 1.2, 1.3, 1.4]

        model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
        network_pkl = f"{model_root}/edm-imagenet-64x64-cond-adm.pkl"

        device = "cuda"
        net = load_edm_model(network_pkl, device=device)

        for name, idx in class_list:
            print(f"Generating images for class: {name} (idx={idx})")

            for guidance_scale in guidance_scales:
                sample = generate_long_image_class_cfg(
                    net=net,
                    class_labels=idx,   
                    device=device,
                    batch_size=1,
                    n_step=40,
                    overlap_size=32,
                    num_img=5,
                    img_shape=(3, 64, 64),
                    s_churn=10.0,
                    guidance_scale=guidance_scale,
                )

                save_path = f"image_outputs/sample_cond_class_all_{name}_gs_{guidance_scale}.png"
                save_tensor_image(sample[0], save_path)

        '''