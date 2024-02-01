# copied from https://github.com/Nordgaren/Github-Folder-Downloader/blob/master/gitdl.py
import os

import requests
from github import ContentFile, Github, Repository


def download(c: ContentFile, out: str):
    r = requests.get(c.download_url)
    output_path = f"{out}/{c.path}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        print(f"downloading {c.path} to {out}")
        f.write(r.content)


def download_folder(repo: Repository, folder: str, out: str, recursive: bool):
    contents = repo.get_contents(folder)
    for c in contents:
        if c.download_url is None:
            if recursive:
                download_folder(repo, c.path, out, recursive)
            continue
        download(c, out)


def download_file(repo: Repository, folder: str, out: str):
    c = repo.get_contents(folder)
    download(c, out)


def download_model_weights(target_folder):
    # dl
    g = Github()
    repo = g.get_repo("BrainLesion/AURORA")
    dl_folder = "brainles_aurora/model_weights"
    # get parent of parget folder to prevent /AURORA/brainles_aurora/brainles_aurora/model_weights
    if "brainles_aurora" in target_folder:
        target_folder = os.path.dirname(target_folder)

    download_folder(
        repo=repo,
        folder=dl_folder,
        out=target_folder,
        recursive=True,
    )
