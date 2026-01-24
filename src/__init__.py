"""
Visual Construction of the Enemy - Cultural Analytics Project

A quantitative analysis of "Us" vs. "Them" visual representation in Nazi-era films.
"""

__version__ = "0.1.0"

from .frame_extraction import (
    extract_frames_by_interval,
    extract_frames_uniform,
    get_video_info
)

from .annotation import (
    FrameAnnotator,
    create_annotation_template,
    show_frame
)

from .feature_extraction import (
    VisualFeatureExtractor,
    extract_features_from_directory
)

from .model import (
    FriendFoeClassifier,
    train_and_evaluate_pipeline
)

__all__ = [
    'extract_frames_by_interval',
    'extract_frames_uniform',
    'get_video_info',
    'FrameAnnotator',
    'create_annotation_template',
    'show_frame',
    'VisualFeatureExtractor',
    'extract_features_from_directory',
    'FriendFoeClassifier',
    'train_and_evaluate_pipeline',
]
