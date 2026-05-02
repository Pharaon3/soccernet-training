"""Evaluation helpers for Bittensor subnet 44 (Score Vision) team-agnostic heads."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from typing import List, Optional, Type
import enum

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dudek.data.team_bas import PredictedAnnotation, SoccerVideo, Team
from dudek.ml.data.tdeed import (
    TeamTDeed2HeadsPrediction,
    TeamTDeedDataset,
    TdeedVideoClip,
)
from dudek.ml.model.tdeed.eval.base import TeamBASScoredVideo, TDeedMAPEvaluator
from dudek.ml.model.tdeed.modules.tdeed import TDeedModule
from dudek.utils.common import soft_non_maximum_suppression


@dataclasses.dataclass
class Subnet44ScoredVideo(TeamBASScoredVideo):
    """Per-frame scores/targets over Score Vision actions (no left/right split)."""

    annotations: List[PredictedAnnotation] = dataclasses.field(default=None, init=False)

    @classmethod
    def from_predictions(
        cls,
        video: SoccerVideo,
        predictions: List[TeamTDeed2HeadsPrediction],
        labels_enum: Type[enum.Enum],
        use_snms: bool = False,
        snms_params: Optional[dict] = None,
    ) -> "Subnet44ScoredVideo":
        scores = TeamTDeed2HeadsPrediction.compute_action_scores_matrix(
            video, predictions, labels_enum, no_background=True
        )
        targets = video.compute_action_labels_matrix(labels_enum, no_background=True)

        if use_snms:
            snms_params = snms_params or dict(class_window=12, threshold=0.01)
            scores = soft_non_maximum_suppression(scores, **snms_params)

        return cls(video=video, scores=scores, targets=targets)

    def annotate(self, labels_enum: Type[enum.Enum], use_true_fps: bool = False):
        predicted_annotations: List[PredictedAnnotation] = []
        int2labels_map = TdeedVideoClip.get_int2label_map(labels_enum, team_aware=False)
        fps = self.video.actual_fps if use_true_fps else self.video.metadata_fps
        for i, x in enumerate(self.scores):
            confidence = float(np.max(x))
            label_idx = int(np.argmax(x))
            label = int2labels_map.get(label_idx + 1)
            if label is None:
                continue
            position = int(i / fps * 1000)
            predicted_annotations.append(
                PredictedAnnotation(
                    label=label,
                    position=position,
                    team=Team.NOT_APPLICABLE,
                    confidence=confidence,
                    game_time="unset",
                    half=1 if ((position / 1000) / 60) > 45 else 0,
                )
            )

        self.annotations = predicted_annotations


class Subnet44TDeedEvaluator:
    """mAP@delta_frames over mapped Score Vision labels (optional SNMS)."""

    def __init__(
        self,
        model: TDeedModule,
        dataset: TeamTDeedDataset,
        delta_frames_tolerance: int = 25,
    ):
        self.model = model
        self.dataset = dataset
        self.delta_frames_tolerance = delta_frames_tolerance

    def eval(
        self,
        batch_size: int = 32,
        use_snms: Optional[bool] = True,
        snms_params: Optional[dict] = None,
    ):
        scored_videos = self.get_scored_videos(
            batch_size=batch_size,
            use_snms=use_snms,
            snms_params=snms_params,
        )
        map_score = TDeedMAPEvaluator.compute_map(
            scored_videos,
            self.delta_frames_tolerance,
            len(self.dataset.labels_enum),
        )
        return {"mAP": map_score}, map_score

    def get_scored_videos(
        self,
        batch_size: int = 32,
        use_snms: Optional[bool] = False,
        snms_params: Optional[dict] = None,
    ) -> List[Subnet44ScoredVideo]:
        video_dataset_map = self.dataset.group_by_videos()
        scored_videos: List[Subnet44ScoredVideo] = []
        for video, clips_dataset in video_dataset_map.items():
            video_predictions: List[TeamTDeed2HeadsPrediction] = []
            clips_loader = DataLoader(
                clips_dataset, batch_size=batch_size, collate_fn=lambda x: x
            )
            for batch_of_clips in tqdm(
                clips_loader,
                desc=f"Scoring video {video.absolute_path}",
            ):
                video_predictions += self.predict(batch_of_clips)

            scored_video = Subnet44ScoredVideo.from_predictions(
                video,
                video_predictions,
                use_snms=use_snms,
                labels_enum=self.dataset.labels_enum,
                snms_params=snms_params,
            )
            scored_videos.append(scored_video)
        return scored_videos

    def predict(self, clips: List[TdeedVideoClip], use_amp: bool = True, device: str = "cuda"):
        clips_tensor = torch.stack([c.clip_tensor for c in clips])
        if clips_tensor.device != device:
            clips_tensor = clips_tensor.to(device)
        clips_tensor = clips_tensor.float()
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda") if use_amp else nullcontext():
                predictions, _ = self.model(clips_tensor, inference=True)
                return [
                    TeamTDeed2HeadsPrediction(
                        labels_prediction=predictions["im_feat"][i].squeeze(),
                        label_displacement_prediction=predictions["displ_feat"][
                            i
                        ].squeeze(),
                        clip=clip.origin_video_clip,
                    )
                    for i, clip in enumerate(clips)
                ]
