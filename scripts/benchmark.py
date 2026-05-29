import argparse
import json
from utils import str2bool

from customguidance import benchmark

benchmark_parser = argparse.ArgumentParser(description='T2I Benchmark.')

benchmark_parser.add_argument("--model", type=str, default="SD3")
benchmark_parser.add_argument("--guidance_types", type=str, nargs="+", required=True)
benchmark_parser.add_argument("--model_path", type=str, required=True)
benchmark_parser.add_argument("--save_path", type=str, required=True)
benchmark_parser.add_argument("--data_annotations_path", type=str, required=True)
benchmark_parser.add_argument("--data_images_path", type=str, required=True)
benchmark_parser.add_argument("--height", type=int, default=512)
benchmark_parser.add_argument("--width", type=int, default=512)
benchmark_parser.add_argument("--num_inference_steps", type=int, default=28)
benchmark_parser.add_argument("--guidance_scale", type=float, default=7.0)
benchmark_parser.add_argument("--score_list", type=str, nargs="+", default=["FID"])
benchmark_parser.add_argument("--number_of_images", type=int, default=5000)
benchmark_parser.add_argument("--run_id", type=str, default="test_run")
benchmark_parser.add_argument("--clip_model_path", type=str, default=None)
benchmark_parser.add_argument("--blip_model_path", type=str, default=None)
benchmark_parser.add_argument("--seed", type=int, default=13)
benchmark_parser.add_argument("--keep_images", type=str2bool, default=False)
benchmark_parser.add_argument("--guidance_parameters", type=lambda x: json.loads(x) if x else None, nargs="+", default=None)

args = benchmark_parser.parse_args()

scores = benchmark(
    model=args.model,
    guidance_types=args.guidance_types,
    model_path=args.model_path,
    data_annotations_path=args.data_annotations_path,
    data_images_path=args.data_images_path,
    height=args.height,
    width=args.width,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    score_list=args.score_list,
    number_of_images=args.number_of_images,
    run_id=args.run_id,
    clip_model_path=args.clip_model_path,
    blip_model_path=args.blip_model_path,
    seed=args.seed,
    guidance_parameters=args.guidance_parameters,
    keep_images=args.keep_images,
    save_result_path=args.save_path
)

print(scores)