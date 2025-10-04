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


def butter_band(y, sr, fmin, fmax, order=4, btype='band'):
    ny = 0.5 * sr
    if btype == 'band':
        lo, hi = fmin/ny, fmax/ny
        b, a = sig.butter(order, [lo, hi], btype='band')
    else:                       # 'high'
        b, a = sig.butter(order, fmin/ny, btype='high')
    return sig.lfilter(b, a, y)


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