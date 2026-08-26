from __future__ import annotations

import argparse
from pathlib import Path

from machine_bias_reproduction.config import OUTPUTS_ROOT, PROJECT_ROOT

from .cards import DATASET_REPO_ID, DATASET_URL
from .fetch import GROUPS

DEFAULT_STAGING = PROJECT_ROOT / "hub" / "datasets"
DEFAULT_DOWNLOAD = PROJECT_ROOT / "hub" / "downloads"
STATE_PATH = PROJECT_ROOT / "hub" / "upload_state.json"


def command_hub_status(arguments: argparse.Namespace) -> None:
    from .auth import describe_identity, read_token, resolve_api
    from .datasets import dataset_chunks

    print(f"repo      {arguments.repo}")
    print(f"url       https://huggingface.co/datasets/{arguments.repo}")
    print(f"outputs   {arguments.outputs}")
    print(f"staging   {arguments.staging}")

    if read_token() is None:
        print("token     absent")
    else:
        try:
            api = resolve_api()
        except SystemExit as error:
            print(f"token     rejected ({error})")
        else:
            print(f"token     {describe_identity(api)}")
            try:
                info = api.repo_info(arguments.repo, repo_type="dataset")
                print(f"remote    {len(info.siblings or [])} files at {info.sha}")
            except Exception as error:
                print(f"remote    unavailable ({error})")

    if arguments.staging.is_dir():
        chunks = dataset_chunks(arguments.staging)
        print(f"chunks    {len(chunks)} staged")
    else:
        print("chunks    staging not built")


def command_hub_build(arguments: argparse.Namespace) -> None:
    from .cards import write_dataset_card
    from .datasets import build_staging

    counts = build_staging(arguments.outputs, arguments.staging, force=arguments.force)
    write_dataset_card(arguments.staging, counts, arguments.repo)
    print(
        f"staged {counts['tables']} tables and {counts['parquet']} parquet files "
        f"from {counts['runs']} runs"
    )


def command_hub_push(arguments: argparse.Namespace) -> None:
    from .auth import resolve_api
    from .cards import write_dataset_card
    from .datasets import build_staging, dataset_chunks, dataset_files
    from .push import UploadState, upload_chunks, upload_file

    counts = build_staging(arguments.outputs, arguments.staging, force=arguments.force)
    write_dataset_card(arguments.staging, counts, arguments.repo)

    api = None if arguments.dry_run else resolve_api(write=True)
    state = UploadState(arguments.state)
    summary = upload_chunks(
        api,
        arguments.repo,
        arguments.staging,
        dataset_chunks(arguments.staging),
        state,
        dry_run=arguments.dry_run,
    )
    for path in dataset_files(arguments.staging):
        relative = path.relative_to(arguments.staging)
        print(f"upload {relative}")
        if not arguments.dry_run:
            upload_file(api, arguments.repo, path, str(relative))

    print(
        f"\nuploaded={summary['uploaded']} skipped={summary['skipped']} "
        f"of {summary['chunks']} chunks"
    )
    print(f"https://huggingface.co/datasets/{arguments.repo}")


def command_hub_pull(arguments: argparse.Namespace) -> None:
    from .fetch import download_dataset

    path = download_dataset(
        arguments.repo,
        arguments.destination,
        groups=arguments.groups,
        revision=arguments.revision,
        force=arguments.force,
    )
    print(f"downloaded {arguments.repo} to {path}")


def _add_repo_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo",
        default=DATASET_REPO_ID,
        help=f"dataset repo (default: {DATASET_URL})",
    )


def _add_staging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outputs", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("hub", help="publish the outputs to the Hugging Face Hub")
    actions = parser.add_subparsers(dest="action", required=True)

    status = actions.add_parser("status", help="auth, remote and staging state")
    _add_repo_argument(status)
    _add_staging_arguments(status)
    status.set_defaults(handler=command_hub_status)

    build = actions.add_parser("build", help="mirror tables and convert raw records to parquet")
    _add_repo_argument(build)
    _add_staging_arguments(build)
    build.add_argument("--force", action="store_true", help="rewrite parquet already up to date")
    build.set_defaults(handler=command_hub_build)

    push = actions.add_parser("push", help="build the staging tree and upload it")
    _add_repo_argument(push)
    _add_staging_arguments(push)
    push.add_argument("--state", type=Path, default=STATE_PATH)
    push.add_argument("--force", action="store_true", help="rewrite parquet already up to date")
    push.add_argument("--dry-run", action="store_true", help="report what would upload")
    push.set_defaults(handler=command_hub_push)

    pull = actions.add_parser("pull", help="download the published dataset")
    _add_repo_argument(pull)
    pull.add_argument("--destination", type=Path, default=DEFAULT_DOWNLOAD)
    pull.add_argument("--groups", nargs="+", choices=sorted(GROUPS), help="default: every group")
    pull.add_argument("--revision")
    pull.add_argument("--force", action="store_true", help="re-download cached files")
    pull.set_defaults(handler=command_hub_pull)
