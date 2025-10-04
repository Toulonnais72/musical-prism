from basic_pitch import ICASSP_2022_MODEL_PATH

CONFIG = {
    # Demucs
    "demucs_model": "htdemucs",
    "demucs_overlap": 0.25,
    "demucs_chunk_sec": 30,

    # Torchcrepe / CREPE compatibility
    "crepe_model_capacity": "large",
    "confidence_threshold": 0.8,
    "crepe_chunk_sec": 90,

    # Basic Pitch
    "basic_pitch_model_path": ICASSP_2022_MODEL_PATH,
    "basic_pitch_segment_sec": 40.0,
    "basic_pitch_segment_overlap_sec": 2.0,
    "basic_pitch_workers": 4,
    "basic_pitch_chorus_only": False,
    "basic_pitch_chorus_rms_db": -35.0,
    "basic_pitch_chorus_window_sec": 2.0,
    "basic_pitch_onset_threshold": 0.5,
    "basic_pitch_frame_threshold": 0.3,
    "basic_pitch_min_note_length_ms": 127.7,
    "basic_pitch_min_freq": None,
    "basic_pitch_max_freq": None,
    "basic_pitch_multiple_pitch_bends": False,
    "basic_pitch_melodia_trick": True,
    "basic_pitch_midi_tempo": 120.0,

    # Drum classification (base profile)
    "drum_classification": {
        "short_window": 0.045,
        "long_window": 0.32,
        "kick_long_window_ratio": 0.25,
        "kick_sub_ratio": 0.26,
        "kick_centroid_max": 900.0,
        "kick_peak_max": 150.0,
        "snare_mid_ratio": 0.18,
        "snare_broad_ratio": 0.45,
        "snare_centroid_min": 900.0,
        "snare_centroid_max": 3500.0,
        "crash_hi_ratio": 0.25,
        "crash_centroid_min": 5000.0,
        "crash_split_centroid": 6500.0,
        "hihat_hi_ratio": 0.30,
        "hihat_centroid_min": 3500.0,
        "hihat_open_gain": 0.05,
        "floor_sub_ratio": 0.18,
        "velocity_scale": 320.0,
        "max_double_kick_gap": 0.12,
        "cymbal_score_threshold": -41.0,
    },
    "drum_presets": {
        "metal": {
            "short_window": 0.032,
            "long_window": 0.28,
            "kick_sub_ratio": 0.28,
            "kick_long_window_ratio": 0.22,
            "max_double_kick_gap": 0.10,
            "cymbal_score_threshold": -39.0,
        }
    },

    # MIDI mapping
    "drum_map": {
        "kick": 36,
        "snare": 38,
        "floor": 41,
        "tom1": 47,
        "tom2": 48,
        "tom3": 50,
        "hihat": 42,
        "hihat_open": 46,
        "crash1": 49,
        "crash2": 57,
        "ride": 51,
        "china": 52,
        "unknown": 39,
    },

    # Audio processing
    "min_audio_length": 16000,
    "resample_rate": 16000,
    "normalize_audio": True,

    # Note grouping/smoothing
    "pitch_tol": 1.0,
    "max_gap": 0.15,
    "min_note_length": 0.05,
    "smooth_window_size": 5,

    # Post-processing
    "adjacency_threshold": 0.05,
    "enable_quantize": False,
    "bpm": 120,
    "steps_per_beat": 4,
    "quantize_velocity": True,
    "auto_bpm": False,
    "auto_bpm_sr": 22050,

    # Default genre (used to apply drum preset overrides)
    "genre": "metal",
}
