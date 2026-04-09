import torch
from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import ImageProcessor
from falcon_perception.paged_inference import (
    PagedInferenceEngine,
    SamplingParams,
    Sequence,
)
from falcon_perception.visualization_utils import decode_coco_rle

setup_torch_config()


def load_falcon_model(device):
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=PERCEPTION_MODEL_ID,
        device=device,
        dtype="float32",
        compile=True,
    )
    return model, tokenizer, model_args


def run_falcon_segmentation(model, tokenizer, model_args, image, query, device):
    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]

    engine = PagedInferenceEngine(
        model,
        tokenizer,
        image_processor,
        max_batch_size=2,
        max_seq_length=8192,
        n_pages=128,
        page_size=128,
        prefill_length_limit=8192,
        enable_hr_cache=False,
        capture_cudagraph=True,
    )

    prompt = build_prompt_for_task(query, "segmentation")
    sampling_params = SamplingParams(stop_token_ids=stop_token_ids)

    def _make_sequences():
        return [
            Sequence(
                text=prompt,
                image=image,
                min_image_size=256,
                max_image_size=1024,
                task="segmentation",
            )
        ]

    print("Warmup run ...", flush=True)
    warmup_seqs = _make_sequences()
    engine.generate(
        warmup_seqs, sampling_params=sampling_params, use_tqdm=False, print_stats=False
    )
    print("Warmup done", flush=True)

    print("Running inference ...", flush=True)
    sequences = _make_sequences()
    engine.generate(
        sequences, sampling_params=sampling_params, use_tqdm=True, print_stats=False
    )

    seq = sequences[0]
    masks_rle = seq.output_aux.masks_rle

    masks = []
    for rle in masks_rle:
        m = decode_coco_rle(rle)
        if m is not None and m.any():
            masks.append(torch.from_numpy(m).float())

    return {"masks": torch.stack(masks) if masks else torch.zeros(0)}
