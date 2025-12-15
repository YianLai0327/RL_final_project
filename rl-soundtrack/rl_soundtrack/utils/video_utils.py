import gc
import os
import tempfile
from typing import List, Tuple

from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    afx,
    concatenate_videoclips,
)


def mix_soundtrack(video_clip, audio_clip) -> CompositeAudioClip:
    return CompositeAudioClip(
        [
            video_clip.audio,
            audio_clip.with_volume_scaled(0.5),
        ]
    )


def render_video_soundtrack(
    video_path: str,
    audio_paths: List[str],
    segment_times: List[Tuple[float, float]],
    audio_offsets: List[float],
    output_path: str,
):
    """
    Stitches video segments with selected audio tracks.

    Args:
        video_path: Path to the full raw video file.
        audio_paths: List of paths to audio files, one per segment.
        segment_times: List of (start, end) times for each segment.
        output_path: Path to save the final rendered video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    full_video = VideoFileClip(video_path)
    full_duration = full_video.duration
    assert full_duration is not None
    # full_video = full_video.with_effects([afx.AudioNormalize()])
    clips = []

    if len(audio_paths) != len(segment_times) or len(audio_offsets) != len(
        segment_times
    ):
        # Handle case where audio_paths might be shorter if stopped early?
        # Or mismatch. Assume length matches for now or truncate.
        min_len = min(len(audio_paths), len(segment_times), len(audio_offsets))
        audio_paths = audio_paths[:min_len]
        segment_times = segment_times[:min_len]
        audio_offsets = audio_offsets[:min_len]

    for i, (audio_path, (start, end), offset) in enumerate(
        zip(audio_paths, segment_times, audio_offsets)
    ):
        # 1. Extract video segment
        # Ensure start/end are within bounds and valid
        if start >= full_duration:
            break
        end = min(end, full_duration)

        video_segment = full_video.subclipped(start, end)
        assert isinstance(video_segment, VideoFileClip)
        duration = end - start

        audio_clip = None
        sub_audio_clip = None
        composite_audio = None

        # 2. Add Audio
        if audio_path and os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path).with_effects([afx.AudioNormalize()])

            # Handle audio duration
            audio_duration = audio_clip.duration
            assert audio_duration is not None
            sub_clip_duration = min(audio_duration - offset, duration)
            sub_audio_clip = audio_clip.subclipped(offset, offset + sub_clip_duration)
            assert isinstance(audio_clip, AudioFileClip)

            composite_audio = mix_soundtrack(video_segment, sub_audio_clip)
            video_segment = video_segment.with_audio(  # pyrefly: ignore[not-callable]
                composite_audio
            )

        else:
            print(f"Warning: Audio path invalid or missing: {audio_path}")

        clips.append(video_segment)
        # if audio_clip is not None:
        #     audio_clip.close()
        # if sub_audio_clip is not None:
        #     sub_audio_clip.close()
        # if composite_audio is not None:
        #     composite_audio.close()

    # 3. Concatenate
    final_video = concatenate_videoclips(clips)

    # 4. Write output
    final_video.write_videofile(  # pyrefly: ignore[not-callable]
        output_path,
        temp_audiofile_path=tempfile.gettempdir(),  # pyrefly: ignore[unexpected-keyword]
        logger=None,  # pyrefly: ignore[unexpected-keyword]
    )

    # Cleanup
    full_video.close()
    for clip in clips:
        clip.close()  # pyrefly: ignore[missing-attribute]
    gc.collect()
