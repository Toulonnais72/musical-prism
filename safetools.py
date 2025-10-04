import numpy as np
import pretty_midi
import librosa
import os, time

def get_unique_output_path(audio_path, output_directory, ext):
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    candidate = os.path.join(output_directory, f"{base_filename}_basic_pitch.{ext}")
    if os.path.exists(candidate):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = os.path.join(output_directory, f"{base_filename}_basic_pitch_{timestamp}.{ext}")
    return candidate

def safe_float(x):
    """Force a Python scalar from anything (float32, array, etc.)"""
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return float(x.item())
        else:
            raise ValueError(f"Expected a single-element array, got an array with shape {x.shape}")
    return float(x)


def safe_int(x):
    # Convert numpy arrays to a scalar if they have only one element
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return int(x.item())
        else:
            raise ValueError("Expected a single-element array, got array with size {}".format(x.size))
    return int(x)

def safe_note(velocity, pitch, start, end):
    """Crée un pretty_midi.Note sûr même si les arguments sont des numpy types"""
    return pretty_midi.Note(
        velocity=safe_int(velocity),
        pitch=safe_int(pitch),
        start=safe_float(start),
        end=safe_float(end)
    )

def safe_rms(y, frame_length=512, hop_length=256):
    """Renvoie un vecteur RMS 1D (jamais un tableau shape (1, N))"""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    return rms.flatten() if rms.ndim > 1 else rms

def estimate_short_window(bpm, division=16, window_factor=0.5):
    note_duration_sec = (60 / bpm) * (4 / division)
    short_window = note_duration_sec * window_factor
    return short_window

def estimate_long_window(bpm, measure_fraction=0.25):
    measure_duration_sec = (60 / bpm) * 4
    long_window = measure_duration_sec * measure_fraction
    return long_window