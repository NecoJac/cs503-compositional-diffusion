import argparse
import json
from pathlib import Path

from customguidance import run

inference_parser = argparse.ArgumentParser(description='T2I.')

inference_parser.add_argument("--model", type=str, default="SD3")
inference_parser.add_argument("--guidance_type", type=str, default="constant")
inference_parser.add_argument("--model_path", type=str, required=True)
inference_parser.add_argument("--prompt", type=str, required=True)
inference_parser.add_argument("--height", type=int, default=512)
inference_parser.add_argument("--width", type=int, default=512)
inference_parser.add_argument("--num_inference_steps", type=int, default=28)
inference_parser.add_argument("--guidance_scale", type=float, default=7.0)
inference_parser.add_argument("--output", type=str, default="output.png")
inference_parser.add_argument("--guidance_parameters", type=json.loads, default=None)

args = inference_parser.parse_args()

image = run(
    model=args.model,
    guidance_type=args.guidance_type,
    model_path=args.model_path,
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    guidance_params=args.guidance_parameters
    )

output_dir = Path("generated_images")
output_dir.mkdir(parents=True, exist_ok=True)
image.save(f"generated_images/{args.output}")

print(f"Image saved at: {args.output}.")