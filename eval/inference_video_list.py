import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import cv2

DEFAULT_NUM_FRAMES = 16


def infer_model_name(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name or model_path


def build_evaluator(video_list: str, model_path: str, model_name: str, save_dir: str):
    if "vLLM" in model_name:
        from models.Qwen2_5_VL_vLLM import vLLMModel

        return vLLMModel(index_json=video_list, model_path=model_path, model_name=model_name, save_dir=save_dir)
    if "BusterX" in model_name:
        from models.BusterX_vLLM import BusterXModel

        return BusterXModel(index_json=video_list, model_path=model_path, model_name=model_name, save_dir=save_dir)
    if "VideoLLaMA3" in model_name:
        from models.VideoLLaMA3 import VideoLLaMA3Model

        return VideoLLaMA3Model(index_json=video_list, model_path=model_path, model_name=model_name, save_dir=save_dir)
    if "Qwen2.5-VL" in model_name or "Skyra" in model_name:
        from models.Qwen2_5_VL import Qwen2Model

        return Qwen2Model(index_json=video_list, model_path=model_path, model_name=model_name, save_dir=save_dir)
    if "InternVL3" in model_name:
        from models.InternVL3 import InternVL3Model

        return InternVL3Model(index_json=video_list, model_path=model_path, model_name=model_name, save_dir=save_dir)
    if "gpt" in model_name.lower() or "gemini" in model_name.lower():
        from models.APIModel import APIModel

        return APIModel(index_json=video_list, model_name=model_name, dataset_set_name="video_list", save_dir=save_dir)

    raise ValueError(f"Unsupported model_name inferred from model_path: {model_name}")


def load_inputs(video_list_path: str) -> List[str]:
    path = Path(video_list_path)
    if not path.exists():
        raise FileNotFoundError(f"video list not found: {video_list_path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("JSON video list must be a list of paths")

    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def parse_answer(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not match:
        return "Error"
    answer = match.group(1).strip()
    if "fake" in answer.lower():
        return "Fake"
    if "real" in answer.lower():
        return "Real"
    return answer


def build_user_prompt(timestamps: List[float]) -> str:
    lines = ["Here are the video frames and their corresponding timestamps:"]
    for ts in timestamps:
        lines.append(f"[T={ts:.2f}s] <image>")
    lines.append("\nPlease analyze the video frames, determine if the video is real or fake, and provide your reasoning.")
    return "\n".join(lines)


def is_frame_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "timestamps.txt").exists()


def load_frame_dir(frame_dir: str) -> Tuple[List[str], List[float]]:
    frame_dir_path = Path(frame_dir)
    with open(frame_dir_path / "timestamps.txt", "r", encoding="utf-8") as f:
        timestamps = [float(x.strip()) for x in f if x.strip()]

    frame_paths: List[str] = []
    for idx in range(1, len(timestamps) + 1):
        frame_path = frame_dir_path / f"{idx}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing frame: {frame_path}")
        frame_paths.append(str(frame_path.resolve()))

    return frame_paths, timestamps


def extract_video_frames(video_path: str, cache_dir: str) -> Tuple[List[str], List[float], str]:
    video_abspath = str(Path(video_path).resolve())
    if not os.path.exists(video_abspath):
        raise FileNotFoundError(f"video not found: {video_abspath}")

    digest = hashlib.md5(video_abspath.encode("utf-8")).hexdigest()[:10]
    video_stem = Path(video_abspath).stem
    out_dir = Path(cache_dir) / f"{video_stem}_{digest}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_abspath)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_abspath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"video has no readable frames: {video_abspath}")
    if fps <= 0:
        fps = 1.0

    count = min(DEFAULT_NUM_FRAMES, total_frames)
    if count <= 1:
        indices = [0]
    else:
        indices = [round(i * (total_frames - 1) / (count - 1)) for i in range(count)]

    frame_paths: List[str] = []
    timestamps: List[float] = []

    for i, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_frame = out_dir / f"{i}.png"
        cv2.imwrite(str(out_frame), frame)
        frame_paths.append(str(out_frame.resolve()))
        timestamps.append(frame_idx / fps)

    cap.release()

    if not frame_paths:
        raise RuntimeError(f"failed to extract frames from: {video_abspath}")

    with open(out_dir / "timestamps.txt", "w", encoding="utf-8") as f:
        for ts in timestamps:
            f.write(f"{ts:.6f}\n")

    return frame_paths, timestamps, str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Skyra inference on a list of videos or frame directories.")
    parser.add_argument("--video_list", type=str, required=True, help="Path to txt/json with one video (or frame-dir) path per item.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint or API model name.")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory for output JSON.")
    args = parser.parse_args()

    model_name = infer_model_name(args.model_path)
    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, f"{model_name}_video_list.json")

    items = load_inputs(args.video_list)
    evaluator = build_evaluator(args.video_list, args.model_path, model_name, args.save_dir)

    results = []
    cache_root = tempfile.mkdtemp(prefix="skyra_video_frames_")
    extracted_dirs: List[str] = []

    for raw_item in items:
        try:
            if is_frame_dir(raw_item):
                frame_paths, timestamps = load_frame_dir(raw_item)
                source_type = "frame_dir"
                source = str(Path(raw_item).resolve())
            else:
                frame_paths, timestamps, out_dir = extract_video_frames(raw_item, cache_root)
                extracted_dirs.append(out_dir)
                source_type = "video"
                source = str(Path(raw_item).resolve())

            user_prompt = build_user_prompt(timestamps)
            response = evaluator.run_inference(frame_paths, user_prompt)
            answer = parse_answer(response)

            results.append(
                {
                    "source": source,
                    "source_type": source_type,
                    "mllm_model_name": model_name,
                    "response": response,
                    "answer": answer,
                    "num_frames": len(frame_paths),
                }
            )
            print(f"[OK] {source}")
        except Exception as exc:
            results.append(
                {
                    "source": raw_item,
                    "source_type": "unknown",
                    "mllm_model_name": model_name,
                    "response": f"Error: {exc}",
                    "answer": "Error",
                    "num_frames": 0,
                }
            )
            print(f"[ERROR] {raw_item}: {exc}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to: {output_path}")

    for d in extracted_dirs:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
