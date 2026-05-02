"""Write Labels-scorevision.json next to each Labels-ball.json for subnet 44 training."""

from __future__ import annotations

import json
import os

import click

from dudek.data.bas_scorevision_mapping import ball_labels_json_to_scorevision


cli = click.Group()


def _resolve_existing_dataset_dir(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Accept a path to the SN-BAS-style root; suggest dudek\\dataset when ./dataset is wrong."""

    if not value:
        raise click.BadParameter("Path is empty.")
    resolved = os.path.abspath(os.path.expanduser(value))
    if os.path.isdir(resolved):
        return resolved
    fallback = os.path.abspath(os.path.join(os.getcwd(), "dudek", "dataset"))
    hint = ""
    if os.path.isdir(fallback):
        hint = f" Found a dataset tree at {fallback!r} — try --dataset_path=dudek\\dataset"
    raise click.BadParameter(f"Not a directory: {value!r} (resolved to {resolved!r}).{hint}")


@cli.command("migrate-from-ball")
@click.option(
    "--dataset_path",
    type=str,
    required=True,
    callback=_resolve_existing_dataset_dir,
    help=(
        "Root folder with league/season/match/Labels-ball.json (SN-BAS layout). "
        "In this repo labels are often under dudek\\dataset, not dataset\\."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing Labels-scorevision.json files.",
)
def migrate_from_ball(dataset_path: str, force: bool):
    """Convert SN-BAS Labels-ball.json to miner-style Labels-scorevision.json (recursive)."""

    dataset_path = os.path.abspath(dataset_path)
    written = 0
    skipped = 0
    for root, _, files in os.walk(dataset_path):
        if "Labels-ball.json" not in files:
            continue
        out_path = os.path.join(root, "Labels-scorevision.json")
        if os.path.exists(out_path) and not force:
            skipped += 1
            continue
        ball_path = os.path.join(root, "Labels-ball.json")
        with open(ball_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        converted = ball_labels_json_to_scorevision(data)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2)
        written += 1
    click.echo(
        f"Wrote Labels-scorevision.json in {written} match folder(s). "
        f"Skipped (already exists, use --force): {skipped}."
    )


if __name__ == "__main__":
    cli()
