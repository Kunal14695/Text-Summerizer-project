# TODO - Fix research/04_model_trainer.ipynb kernel crash

## Plan (approved)
1. Update `research/04_model_trainer.ipynb` to call the existing stage/pipeline code instead of inline training code.
2. Add a “fast mode” in the notebook: limit dataset size + reduce training steps/epochs to avoid OOM / long runs.
3. Make training arguments conservative: disable fp16 in notebook unless CUDA/AMP is available.
4. Keep environment variable cache paths, but avoid hardcoding conflicting paths.
5. Validate by running the notebook cell that launches training and ensure kernel no longer crashes.

## Progress
- [x] Step 1: Refactor notebook to use stage_04 pipeline.
- [ ] Step 2: Add dataset subsetting / fast mode.
- [ ] Step 3: Conservative TrainingArguments.
- [ ] Step 4: Ensure caches/env vars set once.
- [ ] Step 5: Run training (small sample) and verify stability.

