
from basic_pitch import ICASSP_2022_MODEL_PATH

###############################################################################
# CONFIG: All parameters
###############################################################################
CONFIG = {
    # DEMUCS
    "demucs_model": "htdemucs",
    "demucs_overlap": 0.25,
    "demucs_chunk_sec": 30,

    # CREPE
    "crepe_model_capacity": "large",
    "confidence_threshold": 0.8,
    "crepe_chunk_sec": 90,

    # Basic Pitch
    "basic_pitch_model_path": ICASSP_2022_MODEL_PATH,  # Use package constant

    # Drum processing
    "drum_classification": {
        "short_window": 0.045,  # For transient analysis (metal kicks)
        "long_window": 0.35,  # For cymbal tail analysis
        "frequency_bands": {
            'sub': (20, 80),
            'punch': (80, 200),
            'body': (200, 400),
            'snap': (400, 800),
            'bright': (800, 3000),
            'air': (3000, 12000)
        },
        "score_weights": {
            'kick': [0.6, 0.4, -0.2],  # [sub, punch, anti-snap]
            'snare': [0.5, 0.3, 0.2],  # [snap, body, brightness]
            'tom': [0.7, 0.3, -0.5],  # [body, sub, anti-snap]
            'cymbal': [0.6, -0.4]  # [bright+air, anti-body]
        },
        "thresholds": {
            'kick': -45,
            'snare': -50,
            'tom': -55,
            'cymbal': -40
        },
        "attack_time_threshold": 0.025,  # Max attack time for drums (seconds)
        "decay_rate_threshold": -0.5,  # Min decay rate for toms (dB/ms)
        "max_double_kick_gap": 0.15,  # Max time between double kicks
        "bpm_aware_gap_adjust": True,  # Auto-adjust gap based on BPM
        "tom_decay_threshold": 0.6,  # Min decay for tom classification
        "cymbal_score_threshold": -40,
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
    "unknown": 39  # hand clap par défaut
},

    # Audio processing
    "min_audio_length": 16000,
    "resample_rate": 16000,  # For CREPE processing
    "normalize_audio": True,  # Peak normalization before processing

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
    "quantize_velocity": True,  # Add velocity variation after quantization
    "auto_bpm": False,
    "auto_bpm_sr": 22050,

    # Metal-specific defaults
    "metal_preset": {
        "snap_weight_multiplier": 1.4,  # Boost snare wire sensitivity
        "high_attack_bias": 0.25,  # Prefer faster attacks
        "double_kick_boost": True  # Enhance double-kick detection
    }
}

CONFIG["drum_classification"] = {
    "short_window": 0.045,  # Shorter to catch fast transients like double kicks
    "long_window": 0.35,    # Slightly reduced to avoid false cymbal tails

    "frequency_bands": {
        'sub': (20, 90),       # Slightly extended sub range
        'punch': (90, 200),
        'body': (200, 500),    # Extended body for deep snares/toms
        'snap': (500, 950),    # Slightly extended snap
        'bright': (950, 4000), # Cymbals and hi-hats
        'air': (4000, 12000)
    },

    "score_weights": {
        "kick": [0.65, 0.45, -0.25],       # Stronger sub and punch, penalize snap
        "snare": [0.4, 0.4, 0.2],          # Balanced between snap/body/brightness
        "tom": [0.75, 0.25, -0.4],         # More weight on body
        "cymbal": [0.65, -0.3]             # Stronger weight for brightness + air
    },

    "thresholds": {
        "kick": -47,     # A bit more permissive
        "snare": -52,    # Allow more subtle snares
        "tom": -57,      # Allow toms to trigger more easily
        "cymbal": -42
    },

    "attack_time_threshold": 0.020,  # Catch ultra-fast kicks
    "decay_rate_threshold": -0.4,    # Allow more cymbal/tom decay

    "max_double_kick_gap": 0.12,     # Double kicks often under 120 ms apart

    "bpm_aware_gap_adjust": True,    # Yes – very useful at 160–220 BPM

    "tom_decay_threshold": 0.5,      # Let short toms through

    "cymbal_score_threshold": -41,   # Slightly more permissive

}

CONFIG["drum_classification"].update({
    "short_window": 0.04,           # raccourci pour capter les transitoires rapides
    "long_window": 0.3,             # plus court, car decay rapide dans le power metal
    "attack_time_threshold": 0.015, # frappe très brève
    "decay_rate_threshold": -0.4,   # plus permissif
    "max_double_kick_gap": 0.09,    # double kicks très rapides (~100 ms)
    "tom_decay_threshold": 0.4,     # tolère des toms plus secs
    "cymbal_score_threshold": -45,  # autorise des cymbales moins brillantes
    })