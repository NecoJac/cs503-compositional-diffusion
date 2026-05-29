import pickle
import numpy as np
import torch as th
import dnnlib
import diff_collage as dc
import PIL.Image
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGENET_LANDSCAPES_ROOT = os.environ.get(
    "IMAGENET_LANDSCAPES_ROOT",
    os.path.join(PROJECT_ROOT, "data", "imagenet_landscapes"),
)


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



def load_fixed_image(path, device="cuda", image_size=64, overlap_size=32):
    img = PIL.Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), PIL.Image.LANCZOS) # Images from ImageNet so already 64*64, if higher resolution Lanczos downsampling

    img = th.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 # From PIL/np-> torch (c, h,w) ; pixel values: [0,255] to [0,1]
    img = img * 2.0 - 1.0   # to [-1, 1];  EDM pipeline works with images normalized to [-1,1]

    # Keep only the right half: [C, H, overlap_size] -> Variable node (x)
    img = img[:, :, -overlap_size:]

    return img.to(device)


def generate_long_image(
    net,
    fixed_images,
    device="cuda",
    batch_size=1,
    n_step=40,
    overlap_size=32,
    num_img=5,                 # number of generated patches only
    ts_order=5,
    img_shape=(3, 64, 64),
    s_churn=10.0,
    index_imagenet=None,
    guidance_scale=1.0,
):
    eps_fn = EDMEpsWrapper(net, index_imagenet)

    worker = dc.CondIndLongFixedEnd(
        shape=img_shape,
        eps_scalar_t_fn=eps_fn,
        num_img=num_img,
        fixed_images=fixed_images,
        overlap_size=overlap_size,
        guidance_scale=guidance_scale,
    )

    sample_gen = dc.sampling(
        x=worker.generate_xT(batch_size).to(device),
        noise_fn=worker.noise,
        rev_ts=worker.rev_ts(n_step, ts_order),
        x0_pred_fn=worker.x0_fn,
        s_churn=s_churn,
        return_traj=False,
    )

    sample_full = worker.attach_fixed_end(sample_gen)

    # Reset noise before new trajectory
    worker.reset_fixed_end_noise()


    return sample_full

if __name__ == "__main__":

    guidance_scales = [1,3,4,6]

    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
    network_pkl = f"{model_root}/edm-imagenet-64x64-cond-adm.pkl"

    device = "cuda"
    net = load_edm_model(network_pkl, device=device)
    
    for guidance_scale in guidance_scales:
        class_list = [
            ("lakeside", 975),
            ("volcano", 980),
            ("alp", 970),
            ("coral_reef", 973),
        ]

        image_root = IMAGENET_LANDSCAPES_ROOT
        first_image = "000000.jpg"  

        # Build dictionary
        fixed_image_paths = {
            name: os.path.join(image_root, f"class_{cid}_{name}", first_image)
            for name, cid in class_list
        }

        for name, idx in class_list:
            fixed_images = load_fixed_image(
                fixed_image_paths[name],
                device=device,
                image_size=64
            )

            sample = generate_long_image(
                net=net,
                fixed_images=fixed_images,
                device=device,
                batch_size=1,
                n_step=40,
                overlap_size=32,
                num_img=5,              # generate 5 patches, fixed one becomes the 6th
                img_shape=(3, 64, 64),
                s_churn=10.0,
                index_imagenet=idx,
                guidance_scale=guidance_scale
            )

            os.makedirs("image_outputs", exist_ok=True)
            save_tensor_image(sample[0], f"image_outputs/sample_fixed_end_{name}_gs_{guidance_scale}.png")
