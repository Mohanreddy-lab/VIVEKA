@'
set -e
apt-get update -qq && apt-get install -y -qq git
git clone https://github.com/Mohanreddy-lab/PrivacyOps-X.git /app
cd /app
pip install -q -e ".[train]" bitsandbytes huggingface_hub
python scripts/generate_sft_dataset.py --output outputs/train/privacyops_x_sft.jsonl
python scripts/evaluate_policies.py --policy teacher --output outputs/evals/teacher.json
python scripts/evaluate_policies.py --policy random --output outputs/evals/random.json
python scripts/train_trl_sft.py --dataset outputs/train/privacyops_x_sft.jsonl --model Qwen/Qwen3-1.7B --output-dir outputs/checkpoints/privacyops_x_sft_1_7b --use-lora --load-in-4bit --max-steps 150 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --gradient-checkpointing
python scripts/evaluate_policies.py --policy model --model-path outputs/checkpoints/privacyops_x_sft_1_7b --output outputs/evals/sft_checkpoint.json
python scripts/plot_eval_results.py --inputs outputs/evals/random.json outputs/evals/teacher.json outputs/evals/sft_checkpoint.json --output outputs/plots/policy_comparison.png
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="outputs/evals/sft_checkpoint.json", path_in_repo="outputs/evals/sft_checkpoint.json", repo_id="mohareddy1423/PrivacyOps-X-final", repo_type="space")
api.upload_file(path_or_fileobj="outputs/plots/policy_comparison.png", path_in_repo="outputs/plots/policy_comparison.png", repo_id="mohareddy1423/PrivacyOps-X-final", repo_type="space")
print("Uploaded sft_checkpoint.json and policy_comparison.png to the Space.")
PY
'@ | Set-Content -Path train_job.sh -NoNewline