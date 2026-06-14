#!/usr/bin/env python3
"""Push Marshmello GPT checkpoints to Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(PHASE_ROOT / "hub"))

from export_model import export_model  # noqa: E402

DEFAULT_REPOS = {
    "default": "ostah-1010/Marshmello-8M",
    "large_50m": "ostah-1010/Marshmello",
}


def upload_readme_only(
    *,
    config_name: str,
    repo_id: str,
    checkpoint: Path | None = None,
) -> None:
    """Upload only README.md to Hub (fast model card update)."""
    import tempfile

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="marshmello_readme_") as tmp:
        export_dir = Path(tmp) / "export"
        export_model(
            config_name=config_name,
            output_dir=export_dir,
            checkpoint_path=checkpoint,
        )
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(export_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card with GitHub repo link",
        )
        print(f"README updated → https://huggingface.co/{repo_id}")


def push_to_hub(
    *,
    config_name: str,
    repo_id: str,
    checkpoint: Path | None,
    private: bool,
    dry_run: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix="marshmello_hub_") as tmp:
        export_dir = Path(tmp) / "export"
        info = export_model(
            config_name=config_name,
            output_dir=export_dir,
            checkpoint_path=checkpoint,
        )

        print(f"Exported {info['alias']} ({info['params']:,} params) → {export_dir}")
        print(f"Target repo: {repo_id}")

        if dry_run:
            print("Dry run — files prepared but not uploaded.")
            for path in sorted(export_dir.iterdir()):
                print(f"  {path.name} ({path.stat().st_size / 1024**2:.1f} MB)")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        try:
            user = api.whoami()["name"]
            print(f"Authenticated as: {user}")
        except Exception as exc:
            print("Hugging Face authentication failed.")
            print("Run:  hf auth login --token <your_write_token>")
            print("Or:   export HF_TOKEN=<your_write_token>")
            raise SystemExit(1) from exc

        api.create_repo(repo_id=repo_id, exist_ok=True, private=private, repo_type="model")
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {info['alias']} ({info['params']:,} params, step {info['step']})",
        )
        print(f"Uploaded → https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push Marshmello GPT to Hugging Face Hub.")
    parser.add_argument(
        "--config",
        type=str,
        default="large_50m",
        help="Model config: default (8M) | large_50m (45M)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hub repo (default: ostah-1010/Marshmello for 45M, ostah-1010/Marshmello-8M for 8M)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Export only, do not upload")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Push both Marshmello-8M and Marshmello-45M to their default repos",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Upload only README.md (model card update)",
    )
    args = parser.parse_args()

    if args.all:
        for config_name, repo_id in DEFAULT_REPOS.items():
            print(f"\n=== {config_name} → {repo_id} ===")
            if args.readme_only:
                upload_readme_only(config_name=config_name, repo_id=repo_id, checkpoint=None)
            else:
                push_to_hub(
                    config_name=config_name,
                    repo_id=repo_id,
                    checkpoint=None,
                    private=args.private,
                    dry_run=args.dry_run,
                )
        return

    config_name = args.config
    repo_id = args.repo_id or DEFAULT_REPOS.get(config_name, "ostah-1010/Marshmello")
    if args.readme_only:
        upload_readme_only(config_name=config_name, repo_id=repo_id, checkpoint=args.checkpoint)
        return
    push_to_hub(
        config_name=config_name,
        repo_id=repo_id,
        checkpoint=args.checkpoint,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
