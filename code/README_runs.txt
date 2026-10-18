Run from the repository root.

Baselines under the shared stream protocol (results/stream/{task}/falcon/{method}_s{seed}.json)

Wiener filter, CPU (deterministic; computed once per task and copied across seeds):
CUDA_VISIBLE_DEVICES= ../.venv/bin/python code/falcon_baselines.py --tasks h1 m1 m2

Recurrent decoder, GPU (zero-shot, per-session oracle, sequential fine-tune), one process per task:
../.venv/bin/python code/falcon_baselines.py --tasks h1 --methods rnn_zeroshot rnn_oracle rnn_finetune
../.venv/bin/python code/falcon_baselines.py --tasks m1 --methods rnn_zeroshot rnn_oracle rnn_finetune
../.venv/bin/python code/falcon_baselines.py --tasks m2 --methods rnn_zeroshot rnn_oracle rnn_finetune

Base decoders are cached in results/checkpoints/falcon_rnn/{task}/s{seed}.pt and shared by the three regimes.
