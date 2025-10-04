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
from typing import Iterable, Dict, List, Sequence, Tuple
import numpy as np
from scipy import signal as sig
import librosa
import torch
import torchaudio
import pretty_midi
import torchcrepe
import os

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


def velocity_from_amplitude(segment, floor_db: float = -40.0) -> int:
    """Convert segment amplitude to a MIDI velocity."""
    if segment is None:
        return 64
    if torch.is_tensor(segment):
        if segment.numel() == 0:
            return 64
        seg = segment.detach().to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(seg * seg) + _EPS)
        rms = torch.clamp(rms, min=1e-12)
        db = 20.0 * torch.log10(rms)
        norm = torch.clamp((db - floor_db) / (-floor_db), 0.0, 1.0)
        velocity = torch.round(30 + norm * (127 - 30))
        velocity = torch.clamp(velocity, 30, 127)
        if torch.isnan(velocity):
            return 64
        return int(velocity.item())
    if isinstance(segment, np.ndarray):
        if len(segment) == 0:
            return 64
        seg = segment.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(seg * seg) + _EPS))
        db = 20.0 * math.log10(max(rms, _EPS))
        norm = (db - floor_db) / (0.0 - floor_db)
        norm = clamp(norm, 0.0, 1.0)
        vel = int(round(30 + norm * (127 - 30)))
        return int(clamp(vel, 30, 127))
    segment_np = np.asarray(segment, dtype=np.float32)
    return velocity_from_amplitude(segment_np, floor_db=floor_db)

def band_energy(y, sr, fmin, fmax):
    """Énergie spectrale intégrée entre fmin & fmax (Hz)."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    idx = (freqs >= fmin) & (freqs < fmax)
    return S[idx].sum()

# ======================================================================
# Fréquences / lissage / groupage
# ======================================================================


def smooth_frequencies(freq_array, window_size: int = 5):
    """Simple moving average smoothing that works with NumPy or torch tensors."""
    if window_size < 2:
        return freq_array
    if torch.is_tensor(freq_array):
        if freq_array.numel() == 0:
            return freq_array
        pad = window_size // 2
        kernel = torch.ones(1, 1, window_size, dtype=freq_array.dtype, device=freq_array.device) / float(window_size)
        padded = torch.nn.functional.pad(freq_array.view(1, 1, -1), (pad, pad), mode='replicate')
        smoothed = torch.nn.functional.conv1d(padded, kernel)
        L = freq_array.numel()
        return smoothed.view(-1)[:L]
    freq_array = np.asarray(freq_array, dtype=np.float32)
    if freq_array.size == 0:
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

def quantize_measure(meas: Dict, steps_per_beat: int = 4) -> Dict:
    bpm        = meas["bpm"]
    bar_len    = 60.0 / bpm * 4            # 4/4
    step       = bar_len / (4 * steps_per_beat)
    q_notes    = []

    if not meas["notes"]:
        meas["q_notes"] = []
        return meas

    for t, lab, vel in meas["notes"]:
        local_t  = t - meas["start"]
        q_idx    = round(local_t / step)
        q_time   = meas["start"] + q_idx * step
        q_notes.append((q_time, lab, vel))

    meas["q_notes"] = q_notes
    return meas

def measures_to_midi(measures: List[Dict], drum_map: Dict[str, int]) -> pretty_midi.PrettyMIDI:
    pm  = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for m in measures:
        for t, lab, vel in m["q_notes"]:
            if lab not in drum_map:
                continue
            num  = drum_map[lab]
            note = pretty_midi.Note(velocity=int(vel),
                                    pitch=num,
                                    start=t,
                                    end=t + 0.05)
            inst.notes.append(note)

    pm.instruments.append(inst)
    return pm


def ensure_pattern(meas, drum_map, beats_per_bar=4, bpm=120,
                   default_vel=45):
    """
    Fills missing kick/snare/hihat/crash hits in a *single* measure dict
    (in‑place) so it is fully populated on the grid.
    """
    bar_len = 60.0 / bpm * beats_per_bar
    step    = bar_len / (beats_per_bar * 2)         # 8ths → fine enough
    grid    = np.arange(meas["start"], meas["start"] + bar_len, step)

    # Kick 1 & 3  ‑‑ Snare 2 & 4  ‑‑ Closed‑HH every 8th  ‑‑ Crash 1
    recipe = {
        0:  "kick",
        2:  "snare",
        4:  "kick",
        6:  "snare",
    }
    for idx in range(len(grid)):
        if idx in recipe:
            # is there already a note at this grid slot?
            slot = grid[idx]
            if not any(abs(t - slot) < 1e-3 for (t, _, _) in meas["notes"]):
                meas["notes"].append((slot, recipe[idx], default_vel))
        # regular closed hihat
        if not any(abs(t - grid[idx]) < 1e-3 and lab.startswith("hihat")
                   for (t, lab, _) in meas["notes"]):
            meas["notes"].append((grid[idx], "hihat", default_vel))

    # simple crash on the first down‑beat (if none)
    if not any(lab.startswith("crash") for (_, lab, _) in meas["notes"]):
        meas["notes"].append((meas["start"], "crash1", default_vel))

    meas["notes"].sort(key=lambda x: x[0])
    return meas


def quantize_pretty_midi(pm, bpm=120, steps_per_beat=4):
    """
    Snap *all* notes in a PrettyMIDI object to the global grid.
    """
    beat = 60.0 / bpm
    step = beat / steps_per_beat
    for inst in pm.instruments:
        for n in inst.notes:
            n.start = round(n.start / step) * step
            n.end   = max(n.start + 1e-2, round(n.end   / step) * step)
    return pm


def get_unique_output_path(src_path, out_dir, ext):
    """
    Utility used when saving stems/MIDI. Returns a path that doesn’t collide.
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    candidate = os.path.join(out_dir, f"{base}.{ext}")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate = os.path.join(out_dir, f"{base}_{i}.{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _estimate_beats(audio_mono: np.ndarray, sr: int, bpm_hint: float) -> np.ndarray:
    """Return beat times (seconds) using librosa.beat.beat_track but anchored
    to a *known* bpm_hint to stabilise detection across tempo‑changes.
    """
    hop = 512
    onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, start_bpm=bpm_hint, units="time", hop_length=hop
    )
    return beats

# ======================================================================
# Pitch tracking monophonique par chunks (torchcrepe)
# ======================================================================


def _normalize_torchcrepe_capacity(cap: str) -> str:
    """
    torchcrepe ne supporte que 'full' et 'tiny'.
    On mappe les valeurs héritées de CREPE ('small','medium','large','full','tiny')
    vers une valeur valide.
    """
    if not cap:
        return "full"
    cap = cap.lower().strip()
    if cap in ("full",):
        return "full"
    if cap in ("tiny",):
        return "tiny"
    # Mappings hérités de CREPE → torchcrepe
    if cap in ("large", "xlarge", "xl"):
        return "full"
    if cap in ("small", "sm", "medium", "md"):
        return "tiny"
    # fallback par défaut
    return "full"


def chunk_torchcrepe(audio: np.ndarray,
                     sr: int,
                     chunk_sec: float = 30.0,
                     model_capacity: str = "full",
                     confidence_threshold: float = 0.85,
                     smooth_window: int = 5,
                     pitch_tol: float = 1.0,
                     max_gap: float = 0.15,
                     min_note_length: float = 0.05,
                     fmin: float = 50.0,
                     fmax: float = 1100.0,
                     hop_length: int = 160,
                     batch_size: int = 2048) -> List[pretty_midi.Note]:
    """Torchcrepe chunked inference with minimal CPU/GPU transfers."""
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    device = torch.device(DEVICE)
    audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=device)
    if audio_tensor.numel() == 0:
        return []

    peak = audio_tensor.abs().max()
    if peak > 0:
        audio_tensor = audio_tensor / peak

    capacity = _normalize_torchcrepe_capacity(model_capacity)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        if device.type == "cuda":
            resampler = resampler.to(device)
        audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        sr = 16000

    audio_tensor = audio_tensor.contiguous()
    chunk_size_samples = int(max(0.5, chunk_sec) * sr)
    total_len = audio_tensor.shape[0]
    notes_all: List[pretty_midi.Note] = []
    start_sample = 0

    while start_sample < total_len:
        end_sample = min(start_sample + chunk_size_samples, total_len)
        chunk = audio_tensor[start_sample:end_sample]
        if chunk.numel() == 0:
            start_sample = end_sample
            continue

        x = chunk.unsqueeze(0)
        with torch.no_grad():
            try:
                f0, periodicity = torchcrepe.predict(
                    x,
                    sr,
                    hop_length,
                    fmin,
                    fmax,
                    model=capacity,
                    batch_size=batch_size,
                    device=DEVICE,
                    return_periodicity=True,
                )
            except TypeError:
                f0 = torchcrepe.predict(
                    x,
                    sr,
                    hop_length,
                    fmin,
                    fmax,
                    model=capacity,
                    batch_size=batch_size,
                    device=DEVICE,
                )
                periodicity = torch.ones_like(f0)

        f0 = f0.squeeze(0)
        periodicity = periodicity.squeeze(0)
        mask = periodicity >= confidence_threshold
        if not bool(mask.any()):
            start_sample = end_sample
            continue

        t = torch.arange(f0.shape[0], device=f0.device, dtype=torch.float32)
        t = t * hop_length / float(sr)
        t = t + (start_sample / float(sr))

        t_keep = t[mask]
        f_keep = f0[mask]
        f_smooth = smooth_frequencies(f_keep, window_size=smooth_window)

        t_np = t_keep.detach().cpu().numpy()
        f_np = f_smooth.detach().cpu().numpy()

        groups = group_frames_to_notes(
            times=t_np,
            freqs=f_np,
            pitch_tol=pitch_tol,
            max_gap=max_gap,
            min_note_length=min_note_length,
        )

        for (start_t, end_t, midi_pitch) in groups:
            s_idx = max(0, int(math.floor(start_t * sr)))
            e_idx = min(total_len, int(math.ceil(end_t * sr)))
            if e_idx <= s_idx:
                e_idx = min(total_len, s_idx + 1)
            seg = audio_tensor[s_idx:e_idx]
            vel = velocity_from_amplitude(seg)
            notes_all.append(
                safe_note(
                    velocity=vel,
                    pitch=int(midi_pitch),
                    start=float(start_t),
                    end=float(end_t),
                )
            )

        start_sample = end_sample

    return notes_all



def compute_spectrogram_features(y, sr: int, device: str | torch.device | None = None) -> Dict[str, torch.Tensor]:
    """Compute a reusable STFT representation plus helpful frequency masks."""
    if device is None:
        device = torch.device(DEVICE)
    else:
        device = torch.device(device)
    if torch.is_tensor(y):
        waveform = y.detach().to(device=device, dtype=torch.float32)
    else:
        waveform = torch.as_tensor(y, dtype=torch.float32, device=device)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    if waveform.numel() == 0:
        raise ValueError("Empty waveform provided to compute_spectrogram_features")

    window = torch.hann_window(1024, device=device)
    stft = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=128,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = stft.abs()
    power = magnitude.pow(2)

    mag_np = magnitude.detach().cpu().numpy()
    try:
        _, percussive = librosa.decompose.hpss(mag_np, kernel_size=(31, 31))
        percussive_power = torch.from_numpy((percussive ** 2).astype(np.float32)).to(device)
    except Exception:
        percussive_power = power.clone()

    freqs = torch.from_numpy(np.fft.rfftfreq(1024, d=1.0 / sr).astype(np.float32)).to(device)
    band_masks = {
        "sub": (freqs >= 20.0) & (freqs < 120.0),
        "mid": (freqs >= 90.0) & (freqs < 200.0),
        "hi": (freqs >= 2000.0) & (freqs <= 8000.0),
        "snare_broad": (freqs >= 400.0) & (freqs <= 5000.0),
        "tom_low": (freqs >= 110.0) & (freqs < 180.0),
        "tom_mid": (freqs >= 180.0) & (freqs < 260.0),
        "tom_high": (freqs >= 260.0) & (freqs < 400.0),
    }
    time_axis = torch.arange(power.shape[1], device=device, dtype=power.dtype) * (128.0 / sr)
    return {
        "power": power,
        "percussive_power": percussive_power,
        "freqs": freqs,
        "hop_length": 128,
        "sr": sr,
        "band_masks": band_masks,
        "time_axis": time_axis,
        "device": torch.device(device),
    }


def detect_double_kick_subband(y, sr: int, min_distance_ms: float = 6.0) -> List[float]:
    """Detect additional kick onsets in the 20-120 Hz band."""
    if torch.is_tensor(y):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim > 1:
        y_np = y_np.mean(axis=0)
    if y_np.size == 0:
        return []

    sub = butter_band(y_np, sr, 20, 120, btype="band")
    window = max(1, int(sr * 0.006))
    env = np.abs(sub)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    env = np.convolve(env, kernel, mode="same")
    median = np.median(env)
    mad = np.median(np.abs(env - median))
    threshold = median + 1.5 * (mad if mad > 1e-9 else 0.0)
    min_distance = max(1, int(sr * (min_distance_ms / 1000.0)))
    peaks, _ = sig.find_peaks(env, height=threshold, distance=min_distance)
    return (peaks / sr).tolist()


def classify_drum_hits_metal(audio,
                             sr: int,
                             onset_times: Sequence[float],
                             features: Dict[str, torch.Tensor],
                             config: Dict) -> Tuple[List[str], List[int]]:
    """Vectorised drum-hit classification using precomputed spectrogram features."""
    onset_times = list(onset_times)
    if len(onset_times) == 0:
        return [], []

    device = features.get("device", torch.device(DEVICE))
    hop_length = features["hop_length"]
    sr_float = float(sr)

    if torch.is_tensor(audio):
        audio_tensor = audio.detach().to(device=device, dtype=torch.float32)
    else:
        audio_tensor = torch.as_tensor(audio, dtype=torch.float32, device=device)
    if audio_tensor.ndim > 1:
        audio_tensor = audio_tensor.mean(dim=0)

    percussive_power = features["percussive_power"]
    power_device = percussive_power.device
    if power_device != device:
        percussive_power = percussive_power.to(device)
        features = {**features, "percussive_power": percussive_power}
        device = percussive_power.device
    total_per_frame = torch.clamp(percussive_power.sum(dim=0), min=1e-9)

    band_masks = features["band_masks"]
    def band_sum(mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return torch.sum(percussive_power[mask], dim=0)
        return torch.zeros_like(total_per_frame)

    sub_energy = band_sum(band_masks.get("sub", torch.zeros_like(features["freqs"], dtype=torch.bool)))
    mid_energy = band_sum(band_masks.get("mid", torch.zeros_like(features["freqs"], dtype=torch.bool)))
    hi_energy = band_sum(band_masks.get("hi", torch.zeros_like(features["freqs"], dtype=torch.bool)))
    snare_energy = band_sum(band_masks.get("snare_broad", torch.zeros_like(features["freqs"], dtype=torch.bool)))

    freqs = features["freqs"].to(device)
    framewise_centroid = torch.sum(percussive_power.transpose(0, 1) * freqs, dim=1) / total_per_frame
    peak_indices = torch.argmax(percussive_power, dim=0)
    framewise_peak = freqs[peak_indices]

    def cumulative(arr: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(torch.cat([arr.new_zeros(1), arr], dim=0), dim=0)

    total_cumsum = cumulative(total_per_frame)
    sub_cumsum = cumulative(sub_energy)
    mid_cumsum = cumulative(mid_energy)
    hi_cumsum = cumulative(hi_energy)
    snare_cumsum = cumulative(snare_energy)
    centroid_cumsum = cumulative(framewise_centroid)
    peak_cumsum = cumulative(framewise_peak)

    onset_times_t = torch.as_tensor(onset_times, dtype=torch.float32, device=device)
    num_onsets = onset_times_t.numel()
    frame_idx = torch.clamp((onset_times_t * sr_float / hop_length).floor().long(), min=0, max=percussive_power.shape[1] - 1)
    audio_duration = audio_tensor.shape[0] / sr_float
    next_times = torch.cat([onset_times_t[1:], onset_times_t[-1:].clone() + torch.tensor(config["drum_classification"].get("long_window", 0.3), device=device)])
    next_times = torch.clamp(next_times, max=audio_duration)
    available_times = torch.clamp(next_times - onset_times_t - 0.003, min=config["drum_classification"].get("short_window", 0.04))
    available_frames = torch.clamp((available_times * sr_float / hop_length).round().long(), min=1)

    short_window = config["drum_classification"].get("short_window", 0.04)
    long_window = config["drum_classification"].get("long_window", 0.3)
    short_frames = torch.tensor(max(1, int(round(short_window * sr_float / hop_length))), device=device, dtype=torch.long)
    long_frames = torch.tensor(max(1, int(round(long_window * sr_float / hop_length))), device=device, dtype=torch.long)
    short_frames_actual = torch.minimum(short_frames.expand_as(available_frames), available_frames)
    long_frames_actual = torch.minimum(long_frames.expand_as(available_frames), available_frames)

    short_end = torch.clamp(frame_idx + short_frames_actual, max=percussive_power.shape[1])
    long_end = torch.clamp(frame_idx + long_frames_actual, max=percussive_power.shape[1])

    def segment_sum(cumsum: torch.Tensor, end_indices: torch.Tensor) -> torch.Tensor:
        return cumsum[end_indices] - cumsum[frame_idx]

    total_short = torch.clamp(segment_sum(total_cumsum, short_end), min=1e-9)
    sub_short = segment_sum(sub_cumsum, short_end)
    hi_short = segment_sum(hi_cumsum, short_end)
    sub_ratio_short = sub_short / total_short
    hi_ratio_short = hi_short / total_short

    kick_long_ratio = config["drum_classification"].get("kick_long_window_ratio", 0.25)
    use_long = sub_ratio_short > kick_long_ratio
    end_indices = torch.where(use_long, long_end, short_end)
    frames_count = torch.clamp(end_indices - frame_idx, min=1)

    total_final = torch.clamp(segment_sum(total_cumsum, end_indices), min=1e-9)
    sub_final = segment_sum(sub_cumsum, end_indices)
    mid_final = segment_sum(mid_cumsum, end_indices)
    hi_final = segment_sum(hi_cumsum, end_indices)
    snare_final = segment_sum(snare_cumsum, end_indices)

    sub_ratio = sub_final / total_final
    mid_ratio = mid_final / total_final
    hi_ratio = hi_final / total_final
    snare_ratio = snare_final / total_final

    centroid = segment_sum(centroid_cumsum, end_indices) / frames_count.to(centroid_cumsum.dtype)
    peak_freq = segment_sum(peak_cumsum, end_indices) / frames_count.to(peak_cumsum.dtype)

    total_long = torch.clamp(segment_sum(total_cumsum, long_end), min=1e-9)
    hi_long = segment_sum(hi_cumsum, long_end)
    hi_ratio_long = hi_long / total_long
    hi_gain = hi_ratio_long - hi_ratio

    labels = torch.full((num_onsets,), fill_value=-1, dtype=torch.int64, device=device)
    cfg = config.get("drum_classification", {})

    label_ids = {
        "kick": 0,
        "snare": 1,
        "hihat": 2,
        "hihat_open": 3,
        "crash1": 4,
        "crash2": 5,
        "floor": 6,
        "tom2": 7,
        "tom1": 8,
    }
    label_names = ["kick", "snare", "hihat", "hihat_open", "crash1", "crash2", "floor", "tom2", "tom1"]

    def assign(mask: torch.Tensor, key: str) -> None:
        if key not in label_ids:
            return
        if mask.numel() == 0:
            return
        valid = mask & (labels == -1)
        labels[valid] = label_ids[key]

    kick_mask = (sub_ratio >= cfg.get("kick_sub_ratio", 0.28)) & (centroid < cfg.get("kick_centroid_max", 900.0)) & (peak_freq < cfg.get("kick_peak_max", 150.0))
    assign(kick_mask, "kick")

    snare_mask = (mid_ratio >= cfg.get("snare_mid_ratio", 0.18)) & (snare_ratio >= cfg.get("snare_broad_ratio", 0.45)) & (centroid >= cfg.get("snare_centroid_min", 900.0)) & (centroid <= cfg.get("snare_centroid_max", 3500.0))
    assign(snare_mask, "snare")

    crash_hi_base = cfg.get("crash_hi_ratio", 0.25)
    cymbal_db = cfg.get("cymbal_score_threshold")
    if cymbal_db is not None:
        crash_hi_base += max(0.0, (-cymbal_db - 40.0) / 200.0)
    crash_mask = (hi_ratio >= crash_hi_base) & (centroid >= cfg.get("crash_centroid_min", 5000.0))
    assign(crash_mask & (centroid > cfg.get("crash_split_centroid", 6500.0)), "crash2")
    assign(crash_mask & (centroid <= cfg.get("crash_split_centroid", 6500.0)), "crash1")

    hihat_mask = (hi_ratio >= cfg.get("hihat_hi_ratio", 0.30)) & (centroid >= cfg.get("hihat_centroid_min", 3500.0))
    assign(hihat_mask & (hi_gain > cfg.get("hihat_open_gain", 0.05)), "hihat_open")
    assign(hihat_mask, "hihat")

    floor_mask = (peak_freq >= 110.0) & (peak_freq < 180.0) & (sub_ratio >= cfg.get("floor_sub_ratio", 0.18))
    assign(floor_mask, "floor")

    tom2_mask = (peak_freq >= 180.0) & (peak_freq < 260.0)
    assign(tom2_mask, "tom2")

    tom1_mask = (peak_freq >= 260.0) & (peak_freq < 400.0)
    assign(tom1_mask, "tom1")

    start_samples = torch.clamp((onset_times_t * sr_float).floor().long(), min=0, max=audio_tensor.shape[0] - 1)
    window_samples = torch.clamp((frames_count * hop_length).long(), min=1)
    end_samples = torch.clamp(start_samples + window_samples, max=audio_tensor.shape[0])
    max_window = int(window_samples.max().item())
    sample_offsets = torch.arange(max_window, device=device)
    sample_indices = start_samples.unsqueeze(1) + sample_offsets.unsqueeze(0)
    mask_samples = sample_indices < end_samples.unsqueeze(1)
    sample_indices = torch.clamp(sample_indices, max=audio_tensor.shape[0] - 1)
    segments = audio_tensor[sample_indices]
    mask_f = mask_samples.to(segments.dtype)
    peak_amp = torch.max(torch.abs(segments) * mask_f, dim=1)[0]
    sum_sq = torch.sum((segments ** 2) * mask_f, dim=1)
    counts = mask_samples.sum(dim=1).clamp(min=1)
    rms = torch.sqrt(sum_sq / counts.to(segments.dtype))
    combined = 0.7 * peak_amp + 0.3 * rms
    velocities = torch.clamp(torch.round(combined * cfg.get("velocity_scale", 320.0)), min=30.0, max=127.0).to(torch.int64)

    label_list = []
    for value in labels.cpu().tolist():
        if value >= 0:
            label_list.append(label_names[value])
        else:
            label_list.append("unknown")
    velocity_list = [int(v) for v in velocities.cpu().tolist()]
    return label_list, velocity_list


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


def detect_time_signature(audio_mono: np.ndarray, sr: int, bpm: float) -> int:
    """Very lightweight meter estimation.

    Strategy (heuristic):
      • Extract beat times with librosa (anchored to bpm).
      • Compute beat‑to‑beat intervals and cluster them into *bars* by looking
        for regularly recurring down‑beats (higher onset strength on the first
        beat of each bar).
      • Count beats between successive down‑beats – choose the modal count.

    Returns *beats per bar* (default 4 if ambiguous).
    """
    if audio_mono.ndim > 1:
        audio_mono = audio_mono.mean(axis=0)

    beats = _estimate_beats(audio_mono, sr, bpm)
    if len(beats) < 8:
        return 4  # not enough data – assume common time

    # Onset strength for each beat frame (use spectral flux proxy)
    hop = 512
    onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sr, hop_length=hop)
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    beat_strengths = onset_env[np.clip(beat_frames, 0, len(onset_env)-1)]

    # Heuristic: local maxima in beat_strengths denote down‑beats
    peak_mask = beat_strengths > (np.median(beat_strengths) + np.std(beat_strengths))
    downbeat_indices = np.where(peak_mask)[0]
    if len(downbeat_indices) < 3:
        return 4

    # Count *in‑between* beats for successive down‑beats
    intervals = np.diff(downbeat_indices)
    if len(intervals) == 0:
        return 4
    # Choose the modal interval length as beats‑per‑bar
    counts, bins = np.histogram(intervals, bins=np.arange(1, 13))
    beats_per_bar = int(bins[np.argmax(counts)])
    beats_per_bar = max(2, min(beats_per_bar, 12))
    return beats_per_bar or 4


def butter_band(
    y: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float | None = None,
    order: int = 4,
    btype: str = "band",          # "band"/"bandpass", "high"/"highpass", "low"/"lowpass"
    zero_phase: bool = True       # filtfilt pour éviter le déphasage
) -> np.ndarray:
    """
    Filtre passe-bande / passe-haut / passe-bas Butterworth.
    - Supporte mono (T,) ou multi-canal (..., T) en filtrant sur l'axe temps (-1).
    - Normalise correctement les fréquences et vérifie les bornes.
    """
    y = np.asarray(y)
    if y.size == 0:
        return y

    ny = 0.5 * sr

    # normalisation des fréquences coupures (en [0, 1))
    def _clip01(x: float) -> float:
        return float(np.clip(x, 1e-6, 0.999999))

    # Harmonise les alias de btype vers ceux attendus par SciPy
    btype_norm = {
        "band": "bandpass",
        "bandpass": "bandpass",
        "high": "highpass",
        "highpass": "highpass",
        "low": "lowpass",
        "lowpass": "lowpass",
    }.get(btype.lower())
    if btype_norm is None:
        raise ValueError(f"btype inconnu: {btype}. Utilise 'band', 'high', ou 'low'.")

    if btype_norm == "bandpass":
        if fmax is None:
            raise ValueError("Pour un passe-bande, fmax est requis.")
        lo = _clip01(fmin / ny)
        hi = _clip01(fmax / ny)
        if not (hi > lo):
            raise ValueError(f"Bornes invalides: fmin={fmin}Hz, fmax={fmax}Hz (nyquist={ny}Hz).")
        sos = sig.butter(order, [lo, hi], btype="bandpass", output="sos")
    elif btype_norm == "highpass":
        w = _clip01(fmin / ny)
        sos = sig.butter(order, w, btype="highpass", output="sos")
    else:  # lowpass
        w = _clip01(fmin / ny)
        sos = sig.butter(order, w, btype="lowpass", output="sos")

    # Filtrage zero-phase (évite le déphasage). Sinon, sosfilt (causal).
    if zero_phase:
        y_filt = sig.sosfiltfilt(sos, y, axis=-1)
    else:
        y_filt = sig.sosfilt(sos, y, axis=-1)

    return y_filt



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



def preprocess_drum_stem(y: np.ndarray, sr: int, target_sr: int = 48000, apply_expander: bool = True,
                         apply_hpss: bool = True, apply_spectral_gate: bool = True, apply_rms_gate: bool = True,
                         apply_eq_bands: bool = False):
    """
    Renvoie y_proc, sr_target
    ° Mix très compressé	apply_expander=True, apply_hpss=True, apply_spectral_gate=False
    ° Bruit de fond / live	apply_spectral_gate=True, apply_rms_gate=True, expander modéré
    ° Stem “sec” de studio	peut se contenter de apply_hpss=True seul
    """
    print("Preprocessing Drum Stem")
    print("Expander:", apply_expander)
    print("HPSS:", apply_hpss)
    print("Spectral Gate", apply_spectral_gate)
    print("RMS Gate:", apply_rms_gate)
    print("3-band EQ:", apply_eq_bands)
    # 0. Stéréo → Mono
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = y.astype(np.float32)

    # 1. Upsample / Downsample vers target_sr
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2. High‑pass 20Hz (enlève DC)
    y = butter_band(y, sr, 20, None, btype="high")

    # 3. Expander dynamique léger (dé-compression)
    if apply_expander:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        env = (env - env.min()) / (env.max() - env.min() + 1e-8)
        gain = np.interp(np.arange(len(y)),
                         np.linspace(0, len(y), len(env)),
                         env)
        y = y * (1 + 1.5 * (1 - gain))  # facteur 1→2, réglable

    # 4. HPSS (isole le percussif)
    if apply_hpss:
        y, _ = librosa.effects.hpss(y, kernel_size=[3, 31])

    # 5. Spectral gate (soustrait le «fond»)
    if apply_spectral_gate:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
        S_filt = librosa.decompose.nn_filter(S,
                                             aggregate=np.median, metric="cosine", width=17)
        # Seuil : dB sous la moyenne locale
        S_mask = S / (S_filt + 1e-9) > 2
        y = librosa.istft(S * S_mask, hop_length=256)

    # 6. RMS gate (coupe < −60dB)
    if apply_rms_gate:
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        thr = 10 ** (-60 / 20)
        mask = np.repeat(rms, 512)[:len(y)]
        y = np.where(mask < thr, 0.0, y)

    # 7. Égalisation par bandes (optionnel: restitue 3flux puis les mixe)
    if apply_eq_bands:
        sub = butter_band(y, sr, 20, 120, btype="band") * 1.5
        body = butter_band(y, sr, 120, 800, btype="band") * 1.2
        high = butter_band(y, sr, 4500, None, btype="high") * 1.3
        y = sub + body + high
        # Normalise -3dBFS
        peak = np.max(np.abs(y)) + 1e-9
        y = y / peak * 0.7

    # Sécurité : remplace NaN/Inf (évite crash librosa)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y, sr


# -----------------------------------------------------------------------------
# 2) Drum‑grid completion (kick / snare / hihat / crash)
# -----------------------------------------------------------------------------

def _default_drum_vel(map_name: str) -> int:
    """Returns a reasonable velocity per drum piece."""
    return {
        "kick": 100,
        "snare": 110,
        "hihat": 75,
        "hihat_open": 75,
        "floor": 90,
        "tom1": 90,
        "tom2": 90,
        "crash1": 105,
        "crash2": 105,
    }.get(map_name, 90)


def fill_missing_drums(measure, drum_map, beats_per_bar=4, bpm=120, default_vel=45):
    bar_len = 60.0 / bpm * beats_per_bar
    step = bar_len / (4 * beats_per_bar)  # based on existing quantiser

    for bar_idx, m in enumerate(measure):
        # Build a quick lookup of existing hits by (rounded) start time and label
        existing = {(round(t, 5), lab) for (t, lab, _) in m["q_notes"]}

        def _add_if_missing(rel_beat: float, voice: str, vel: int = None):
            t_abs = m["start"] + rel_beat * 60.0 / bpm
            t_q   = m["start"] + round((t_abs - m["start"]) / step) * step
            key   = (round(t_q, 5), voice)
            if key not in existing:
                m["q_notes"].append((t_q, voice, vel or _default_drum_vel(voice)))
                existing.add(key)

        if beats_per_bar == 3:
            # simple waltz‑ish pattern
            _add_if_missing(0, "kick")
            _add_if_missing(1, "snare")
            _add_if_missing(2, "snare")
        else:  # assume >=4
            _add_if_missing(0, "kick")
            _add_if_missing(1, "snare")
            _add_if_missing(2, "kick")
            _add_if_missing(3, "snare")

        # Hi‑hat every beat (closed) – use half‑notes in double‑time if tempo < 90
        hh_rate = 2 if bpm >= 90 else 1  # 2 => 8th notes, 1 => quarters
        hh_steps = beats_per_bar * hh_rate
        for i in range(hh_steps):
            _add_if_missing(i / hh_rate, "hihat")

        # Crash on first down‑beat of first bar only (overdubbing allowed)
        if bar_idx == 0:
            _add_if_missing(0, "crash1", vel=115)

        # Re‑sort by time for consistency
        m["q_notes"].sort(key=lambda x: x[0])
    return ensure_pattern(measure, drum_map,
                          beats_per_bar=beats_per_bar,
                          bpm=bpm,
                          default_vel=default_vel)

# -----------------------------------------------------------------------------
# 3) Global PrettyMIDI quantiser
# -----------------------------------------------------------------------------
def _quantize_time_scalar(t: float, bpm: float, steps_per_beat: int) -> float:
    beat_dur = 60.0 / bpm
    step_dur = beat_dur / steps_per_beat
    return round(t / step_dur) * step_dur
