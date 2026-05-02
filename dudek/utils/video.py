import os

from typing import List

from tqdm import tqdm

from dudek.data.scorevision_actions import ScorevisionAction
from dudek.data.team_bas import SoccerVideo


def load_action_spotting_videos(
    directory_path: str,
    resolution: int,
    load_as_bas: bool = False,
    random_team_when_no_team: bool = False,
) -> List["SoccerVideo"]:
    soccer_videos = []
    with tqdm(desc=f"Loading videos from {directory_path}") as pbar:
        for league_name in os.listdir(directory_path):
            for season_name in os.listdir(os.path.join(directory_path, league_name)):
                for match_label in os.listdir(
                    os.path.join(directory_path, league_name, season_name)
                ):
                    half1, half2 = SoccerVideo.action_spotting_video_from_path(
                        os.path.join(
                            directory_path,
                            league_name,
                            season_name,
                            match_label,
                        ),
                        resolution,
                        load_as_bas=load_as_bas,
                        random_team_when_no_team=random_team_when_no_team,
                    )

                    soccer_videos.append(half1)
                    soccer_videos.append(half2)

                    pbar.update(1)
    return soccer_videos


def load_bas_videos_scorevision_subnet44(
    directory_path: str, resolution: int
) -> List["SoccerVideo"]:
    """SN-BAS videos with Score Vision labels (from Labels-scorevision.json or BAS remap)."""

    videos = load_bas_videos(directory_path, resolution)
    return [
        v
        if v.labels_class == ScorevisionAction
        else v.with_scorevision_subnet44_annotations_from_bas()
        for v in videos
    ]


def load_bas_videos(directory_path: str, resolution: int) -> List["SoccerVideo"]:
    directory_path = os.path.abspath(os.path.normpath(directory_path))
    flat_mp4 = os.path.join(directory_path, f"{resolution}p.mp4")
    if os.path.isfile(flat_mp4):
        return [SoccerVideo.bas_video_from_flat_folder(directory_path, resolution)]

    videos = []
    for league_name in os.listdir(directory_path):
        league_path = os.path.join(directory_path, league_name)
        if not os.path.isdir(league_path):
            continue
        for season_name in os.listdir(league_path):
            season_path = os.path.join(league_path, season_name)
            if not os.path.isdir(season_path):
                continue
            for match_label in os.listdir(season_path):
                match_path = os.path.join(season_path, match_label)
                if not os.path.isdir(match_path):
                    continue
                soccer_video = SoccerVideo.bas_video_from_path(match_path, resolution)
                videos.append(soccer_video)
    return videos


import subprocess


def get_actual_video_length(video_file_path: str):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_file_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)
