# Marshmello → Hugging Face Hub

Upload **Marshmello-45M** to [`ostah-1010/Marshmello`](https://huggingface.co/ostah-1010/Marshmello).

## 1. Authenticate

Your cached HF token is invalid. Create a **write** token at  
https://huggingface.co/settings/tokens then:

```bash
hf auth login --token hf_xxxxxxxx
# or
export HF_TOKEN=hf_xxxxxxxx
```

Account must be **ostah-1010** (or have write access to that namespace).

## 2. Push Marshmello-45M

```bash
cd mini-transformer-from-scratch
source .venv/bin/activate
pip install huggingface_hub safetensors

python 13_gpt_pretraining/hub/push_to_hub.py \
  --config large_50m \
  --repo-id ostah-1010/Marshmello
```

## 3. Push both models (optional)

```bash
python 13_gpt_pretraining/hub/push_to_hub.py --all
```

| Config | Model | Default repo |
|--------|-------|--------------|
| `large_50m` | Marshmello-45M | `ostah-1010/Marshmello` |
| `default` | Marshmello-8M | `ostah-1010/Marshmello-8M` |

## 4. Dry run (no upload)

```bash
python 13_gpt_pretraining/hub/push_to_hub.py --config large_50m --dry-run
```

## Uploaded files

- `model.safetensors` — weights (~170 MB for 45M)
- `config.json` — architecture + param breakdown
- `tokenizer.json` — BPE tokenizer
- `generation_config.json` — default sampling settings
- `training_meta.json` — step, losses, full training config
- `README.md` — model card

## Download back into the project

```bash
python 13_gpt_pretraining/hub/download_from_hub.py --repo-id ostah-1010/Marshmello
python 13_gpt_pretraining/generate.py --config large_50m --prompt "Database systems"
```
