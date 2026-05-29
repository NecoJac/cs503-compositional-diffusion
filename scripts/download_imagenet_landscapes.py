import argparse
import os


DEFAULT_SCRATCH_ROOT = "/scratch/izar/sjiang/diffusion"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "imagenet_landscapes")

CLASS_NAMES = {
    970: "alp",
    972: "cliff",
    973: "coral_reef",
    974: "geyser",
    975: "lakeside",
    976: "promontory",
    977: "sandbar",
    978: "seashore",
    979: "valley",
    980: "volcano",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download ImageNet landscape classes for fixed-end experiments.")
    parser.add_argument(
        "--scratch-root",
        default=os.environ.get("DIFFUSION_SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT),
        help="Scratch directory used for output, caches, and temporary files.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("IMAGENET_LANDSCAPES_ROOT", DEFAULT_OUTPUT_ROOT),
        help="Where class folders are written. Defaults to PROJECT_ROOT/data/imagenet_landscapes.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_DATASETS_CACHE"),
        help="HuggingFace datasets cache. Defaults to SCRATCH_ROOT/hf_cache/datasets.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap per class. 0 means save all matching examples.",
    )
    return parser.parse_args()


def configure_storage(args):
    scratch_root = os.path.abspath(args.scratch_root)
    output_root = args.output_root
    cache_dir = args.cache_dir or os.path.join(scratch_root, "hf_cache", "datasets")
    hf_home = os.environ.get("HF_HOME", os.path.join(scratch_root, "hf_cache", "home"))
    hf_hub_cache = os.environ.get("HF_HUB_CACHE", os.path.join(scratch_root, "hf_cache", "hub"))
    tmp_dir = os.environ.get("TMPDIR", os.path.join(scratch_root, "tmp"))

    for path in (scratch_root, output_root, cache_dir, hf_home, hf_hub_cache, tmp_dir):
        os.makedirs(path, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["TMPDIR"] = tmp_dir

    return output_root, cache_dir


def class_folder(output_root, class_id):
    return os.path.join(output_root, f"class_{class_id}_{CLASS_NAMES[class_id]}")


def main():
    args = parse_args()
    output_root, cache_dir = configure_storage(args)

    # Import after cache environment variables are set.
    from datasets import load_dataset
    from tqdm import tqdm

    for class_id in CLASS_NAMES:
        os.makedirs(class_folder(output_root, class_id), exist_ok=True)

    print(f"Writing ImageNet landscapes to: {output_root}")
    print(f"Using HuggingFace datasets cache: {cache_dir}")

    dataset = load_dataset(
        "benjamin-paine/imagenet-1k-64x64",
        split="train",
        cache_dir=cache_dir,
        streaming=True,
    )

    counters = {class_id: 0 for class_id in CLASS_NAMES}
    for sample in tqdm(dataset, desc="Filtering ImageNet landscape classes"):
        label = int(sample["label"])
        if label not in CLASS_NAMES:
            continue
        if args.max_per_class and counters[label] >= args.max_per_class:
            if all(count >= args.max_per_class for count in counters.values()):
                break
            continue

        img = sample["image"]
        filename = f"{counters[label]:06d}.jpg"
        img.save(os.path.join(class_folder(output_root, label), filename))
        counters[label] += 1

    print("Saved images per class:")
    for class_id, count in counters.items():
        print(f"  {class_id} {CLASS_NAMES[class_id]}: {count}")
    print("Done")


if __name__ == "__main__":
    main()
