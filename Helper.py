# Helper.py
# -*- coding: utf-8 -*-
"""
Utilitaires audio/MIDI pour le pipeline:
- Pitch tracking monophonique par chunks avec torchcrepe (GPU/CPU)
- Lissage fréquentiel + regroupement de frames en notes
- Fusion de notes adjacentes + quantification temporelle
- Vélocité MIDI dérivée de l'amplitude (RMS)
- Fonctions "safe" pour éviter les erreurs de types

Auteur: refactor torchcrepe (remplace crepe) – Python 3.11 / CUDA 11.8
"""

from __future__ import annotations
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import librosa
import torch
import pretty_midi
import torchcrepe


# ======================================================================
# Configuration runtime
# ======================================================================

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Epsilon num pour éviter les divisions par 0 et end==start
_EPS = 1e-9

# ======================================================================
# Utilitaires "safe"
# ======================================================================

def safe_float(x) -> float:
    """Convertit x en float Python."""
    try:
        # Gère numpy scalars, tensors .item(), etc.
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float(0.0)


def safe_int(x) -> int:
    """Convertit x en int Python."""
    try:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return int(0)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_note(velocity: int, pitch: int, start: float, end: float) -> pretty_midi.Note:
    """
    Construit une note PrettyMIDI avec clamp des bornes et correction end>start.
    """
    v = int(clamp(velocity, 1, 127))
    p = int(clamp(pitch, 0, 127))
    s = safe_float(start)
    e = safe_float(end)
    if not (e > s):
        e = s + 0.01
    return pretty_midi.Note(velocity=v, pitch=p, start=s, end=e)


# ======================================================================
# Audio → vélocité (RMS → 30..127)
# ======================================================================

def velocity_from_amplitude(segment: np.ndarray, floor_db: float = -40.0) -> int:
    """
    Convertit l'amplitude d'un segment en vélocité MIDI.
    - Calcule le RMS → dBFS, puis mappe dans [30..127].
    """
    if segment is None or len(segment) == 0:
        return 64
    seg = segment.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(seg * seg) + _EPS))
    # dBFS (si max=1)
    db = 20.0 * math.log10(max(rms, _EPS))
    # Normalise par un plancher (ex: -40 dBFS) → [0..1]
    norm = (db - floor_db) / (0.0 - floor_db)
    norm = clamp(norm, 0.0, 1.0)
    vel = int(round(30 + norm * (127 - 30)))
    return int(clamp(vel, 30, 127))


# ======================================================================
# Fréquences / lissage / groupage
# ======================================================================

def smooth_frequencies(freq_array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Moyenne glissante simple (padding 'edge')."""
    if window_size < 2 or len(freq_array) == 0:
        return freq_array
    pad = window_size // 2
    padded = np.pad(freq_array, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def hz_to_midi(f_hz: float) -> float:
    f = max(f_hz, _EPS)
    return 69.0 + 12.0 * math.log2(f / 440.0)


def group_frames_to_notes(times: np.ndarray,
                          freqs: np.ndarray,
                          pitch_tol: float = 1.0,
                          max_gap: float = 0.15,
                          min_note_length: float = 0.05) -> List[Tuple[float, float, int]]:
    """
    Regroupe des frames consécutives en notes si:
      - la variation de pitch (en demi-tons) reste < pitch_tol
      - le gap temporel entre frames < max_gap
      - la durée du groupe >= min_note_length
    Retour: liste de tuples (start_time, end_time, midi_pitch_int)
    """
    notes: List[Tuple[float, float, int]] = []
    if len(times) == 0 or len(freqs) == 0:
        return notes

    midi_pitches = np.array([hz_to_midi(f) for f in freqs], dtype=np.float32)
    start_idx = 0
    current_pitch = float(midi_pitches[0])

    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        dpitch = float(abs(midi_pitches[i] - current_pitch))

        if dpitch > pitch_tol or dt > max_gap:
            group_times = times[start_idx:i]
            group_pitches = midi_pitches[start_idx:i]

            if len(group_times) > 0:
                note_start = float(group_times[0])
                note_end = float(group_times[-1])
                duration = note_end - note_start
                if duration >= min_note_length and len(group_pitches) > 0:
                    # IMPORTANT: cast explicite → float() avant round(int)
                    med = float(np.median(group_pitches))
                    median_pitch = int(round(med))
                    notes.append((note_start, note_end, median_pitch))

            start_idx = i
            current_pitch = float(midi_pitches[i])
        else:
            current_pitch = float(0.5 * (current_pitch + float(midi_pitches[i])))

    # Dernier groupe
    if start_idx < len(times):
        group_times = times[start_idx:]
        group_pitches = midi_pitches[start_idx:]
        if len(group_times) > 0:
            note_start = float(group_times[0])
            note_end = float(group_times[-1])
            duration = note_end - note_start
            if duration >= min_note_length and len(group_pitches) > 0:
                med = float(np.median(group_pitches))
                median_pitch = int(round(med))
                notes.append((note_start, note_end, median_pitch))

    return notes


# ======================================================================
# Fusion + quantification
# ======================================================================

def merge_adjacent_notes(notes: List[pretty_midi.Note],
                         adjacency_threshold: float = 0.05) -> List[pretty_midi.Note]:
    """
    Fusionne les notes consécutives si:
      - même pitch
      - gap entre end courant et start suivant ≤ adjacency_threshold
    """
    if not notes:
        return []
    notes_sorted = sorted(notes, key=lambda n: n.start)
    merged: List[pretty_midi.Note] = []
    current = notes_sorted[0]

    for nxt in notes_sorted[1:]:
        if nxt.pitch == current.pitch and (nxt.start - current.end) <= adjacency_threshold:
            current.end = max(current.end, nxt.end)
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def quantize_time(t: float, bpm: float = 120.0, steps_per_beat: int = 4) -> float:
    beat = 60.0 / max(bpm, _EPS)
    steps = t / beat * steps_per_beat
    steps_rounded = round(steps)
    return (steps_rounded / float(steps_per_beat)) * beat


def snap_notes_to_tempo(notes: List[pretty_midi.Note],
                        bpm: float = 120.0,
                        steps_per_beat: int = 4) -> List[pretty_midi.Note]:
    snapped: List[pretty_midi.Note] = []
    for n in notes:
        s = quantize_time(n.start, bpm, steps_per_beat)
        e = quantize_time(n.end, bpm, steps_per_beat)
        if e <= s:
            e = s + 0.01
        snapped.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=s, end=e))
    return snapped


# ======================================================================
# Pitch tracking monophonique par chunks (torchcrepe)
# ======================================================================

def chunk_torchcrepe(audio: np.ndarray,
                     sr: int,
                     chunk_sec: float = 30.0,
                     model_capacity: str = "full",      # "tiny","small","medium","large","full"
                     confidence_threshold: float = 0.85, # périodicité min (voicing)
                     smooth_window: int = 5,
                     pitch_tol: float = 1.0,
                     max_gap: float = 0.15,
                     min_note_length: float = 0.05,
                     fmin: float = 50.0,
                     fmax: float = 1100.0,
                     hop_length: int = 160,             # 10 ms @ 16 kHz
                     batch_size: int = 2048) -> List[pretty_midi.Note]:
    """
    Extraction des notes monophoniques par chunks via torchcrepe.
    Retourne une liste de pretty_midi.Note (non fusionnées, non quantifiées).
    """
    # 1) Mono + normalisation
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return []

    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak

    # 2) Resample 16 kHz (torchcrepe)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # 3) Chunks
    chunk_size_samples = int(max(0.5, chunk_sec) * sr)
    total_len = len(audio)
    start_sample = 0

    notes_all: List[pretty_midi.Note] = []

    while start_sample < total_len:
        end_sample = min(start_sample + chunk_size_samples, total_len)
        chunk = audio[start_sample:end_sample]

        # Tensor (batch, time) sur DEVICE
        x = torch.tensor(chunk, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # torchcrepe.predict → f0 (Hz), periodicity
        with torch.no_grad():
            f0, periodicity = torchcrepe.predict(
                x,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                model=model_capacity,
                batch_size=batch_size,
                device=DEVICE,
                return_periodicity=True
            )

        # Convert en numpy 1D
        f0 = f0.squeeze(0).detach().cpu().numpy()
        per = periodicity.squeeze(0).detach().cpu().numpy()

        # Masque "voicing": périodicité >= seuil
        mask = per >= confidence_threshold
        if not np.any(mask):
            start_sample += chunk_size_samples
            continue

        # Timeline (en s) + offset global du chunk
        t = (np.arange(len(f0), dtype=np.float64) * hop_length) / float(sr)
        t += (start_sample / float(sr))

        t_keep = t[mask]
        f_keep = f0[mask]

        # Lissage
        f_smooth = smooth_frequencies(f_keep, window_size=smooth_window)

        # Groupage frames → notes
        groups = group_frames_to_notes(
            times=t_keep,
            freqs=f_smooth,
            pitch_tol=pitch_tol,
            max_gap=max_gap,
            min_note_length=min_note_length
        )

        # Notes + vélocité (amplitude locale)
        for (start_t, end_t, midi_pitch) in groups:
            s_idx = max(0, int(math.floor(start_t * sr)))
            e_idx = min(len(audio), int(math.ceil(end_t * sr)))
            seg = audio[s_idx:e_idx] if e_idx > s_idx else audio[s_idx:s_idx+1]
            vel = velocity_from_amplitude(seg)
            notes_all.append(safe_note(velocity=vel, pitch=int(midi_pitch),
                                       start=float(start_t), end=float(end_t)))

        start_sample += chunk_size_samples

    return notes_all


# Pour compatibilité avec l’ancien code qui appelait "chunk_crepe"
chunk_crepe = chunk_torchcrepe


# ======================================================================
# (Optionnel) Détection BPM robuste (scalaire) – évite les erreurs round(ndarray)
# ======================================================================

def detect_bpm_librosa_mono(y: np.ndarray, sr: int, resample_to: int = 22050) -> float:
    """
    Détecte un tempo scalaire via librosa.beat.beat_track.
    - Mono + resample pour stabilité
    - Retourne un float BPM (pas un ndarray)
    """
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = y.astype(np.float32, copy=False)
    if sr != resample_to:
        y = librosa.resample(y, orig_sr=sr, target_sr=resample_to)
        sr = resample_to
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


# ======================================================================
# (Optionnel) Détection d’onsets en chunks (utile pour drums)
# ======================================================================

def chunk_onset_detect(audio: np.ndarray, sr: int, chunk_sec: float = 30.0) -> List[float]:
    """
    Onset detection par chunks, retourne une liste de temps (s).
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32, copy=False)
    onsets_all: List[float] = []

    chunk_size = int(max(0.5, chunk_sec) * sr)
    total_len = len(audio)
    start = 0
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk = audio[start:end]
        times = librosa.onset.onset_detect(y=chunk, sr=sr, units='time', backtrack=True)
        offset = start / float(sr)
        onsets_all.extend([float(t) + offset for t in times])
        start += chunk_size

    onsets_all.sort()
    return onsets_all
