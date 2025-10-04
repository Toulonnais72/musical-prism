
from safetools import safe_float, safe_int, safe_note, safe_rms
import crepe
import librosa, numpy as np, pretty_midi, math
from typing import List, Dict, Optional
import os, time
import faulthandler, torchaudio, torch, librosa, pretty_midi
faulthandler.enable()
import scipy.signal as sig
print('Imports fine – versions:', torch.__version__, torchaudio.__version__)

###############################################################################
# HELPER: band_energy + Merge adjacent notes + (6.2) Quantize + snap_notes_to_tempo
# + estimate_local_bpm + precompute_bpm_curve + get_bpm_at_time
###############################################################################
def band_energy(y, sr, fmin, fmax):
    """Énergie spectrale intégrée entre fmin & fmax (Hz)."""
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    idx = (freqs >= fmin) & (freqs < fmax)
    return S[idx].sum()

def merge_adjacent_notes(notes, adjacency_threshold=0.05):
    if not notes:
        return []
    notes.sort(key=lambda n: n.start)
    merged = []
    current_note = notes[0]
    for i in range(1, len(notes)):
        next_note = notes[i]
        # Only merge if same pitch and gap is below threshold
        if (next_note.pitch == current_note.pitch and
                next_note.start - current_note.end <= adjacency_threshold):
            current_note.end = next_note.end
        else:
            merged.append(current_note)
            current_note = next_note
    merged.append(current_note)
    return merged

def quantize_time(t, bpm=120, steps_per_beat=4):
    beat_duration = 60.0 / bpm
    steps = (t / beat_duration) * steps_per_beat
    steps_rounded = round(steps)
    return (steps_rounded / steps_per_beat) * beat_duration

def snap_notes_to_tempo(notes, bpm=120, steps_per_beat=4):
    snapped = []
    for n in notes:
        start_q = quantize_time(n.start, bpm, steps_per_beat)
        end_q   = quantize_time(n.end,   bpm, steps_per_beat)
        if end_q <= start_q:
            end_q = start_q + 0.01
        note = safe_note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=start_q,
            end=end_q
        )
        snapped.append(note)
    return snapped

def estimate_local_bpm(audio_segment, sr):
    """
    Estime le BPM local sur un segment audio mono.
    - Si segment trop court (<1.5s), retourne 120 BPM.
    - Adapte la détection pour blasts rapides.
    - Lisse pour éviter les valeurs aberrantes.
    """
    duration_sec = len(audio_segment) / sr
    if duration_sec < 1.5:
        return 120  # Par défaut si trop court

    try:
        # 1. Calcul onset envelope
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr, hop_length=512)

        # 2. Beat track rapide
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512)

        # 3. Nettoyage tempo
        if isinstance(tempo, (list, np.ndarray)):
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)

        # 4. Clamp BPM dans une plage réaliste pour du métal progressif
        bpm = np.clip(bpm, 70, 280)  # on sait que c'est entre 70 et 280 BPM

        # 5. Optionnel : arrondir légèrement pour éviter trop d'oscillations
        bpm = round(bpm, 1)

        return bpm

    except Exception as e:
        print(f"[WARN] BPM estimation failed: {e}")
        return 120

def is_snare_like(feats):
    """Heuristique caisse‑claire : mid dominant + attaque franche."""
    return (
        feats["e_mid"] > feats["e_high"] * 0.8 and
        850  < feats["centroid_mean"] < 3300 and
        feats["attack_strength"]  > 0.05 and
        feats["decay_rate"]       > -0.55
    )

def is_floor_tom(r_sub, r_mid, peak_freq):
    """Floor tom = sub très fort, peu de médiums, pic < 110Hz."""
    return r_sub > 0.35 and r_mid < 0.12 and peak_freq < 110

def smooth_frequencies(freq_array, window_size=5):
    if window_size < 2:
        return freq_array
    pad_size = window_size // 2
    padded = np.pad(freq_array, (pad_size, pad_size), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='same')
    return smoothed[pad_size:-pad_size]

def hz_to_midi(freq_hz):
    return 69.0 + 12.0 * np.log2(freq_hz / 440.0)

def group_frames_to_notes(times, freqs, pitch_tol=1.0, max_gap=0.15, min_note_length=0.05):
    if len(times) == 0:
        return []
    midi_pitches = np.array([hz_to_midi(f) for f in freqs])
    notes = []
    start_idx = 0
    current_pitch = midi_pitches[0]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        dpitch = abs(midi_pitches[i] - current_pitch)
        if dpitch > pitch_tol or dt > max_gap:
            group_times = times[start_idx:i]
            group_pitches = midi_pitches[start_idx:i]
            if len(group_times) > 0:
                note_start = group_times[0]
                note_end = group_times[-1]
                duration = note_end - note_start
                if duration >= min_note_length:
                    median_pitch = int(round(np.median(group_pitches)))
                    notes.append((note_start, note_end, median_pitch))
            start_idx = i
            current_pitch = midi_pitches[i]
        else:
            current_pitch = (current_pitch + midi_pitches[i]) / 2.0
    # last group
    if start_idx < len(times):
        group_times = times[start_idx:]
        group_pitches = midi_pitches[start_idx:]
        if len(group_times) > 0:
            note_start = group_times[0]
            note_end   = group_times[-1]
            duration   = note_end - note_start
            if duration >= min_note_length:
                median_pitch = int(round(np.median(group_pitches)))
                notes.append((note_start, note_end, median_pitch))
    return notes

def velocity_from_amplitude(audio_segment, mode="default"):
    """
    Estime la vélocité MIDI depuis un segment audio :
    - mode = 'drum', 'bass', 'vocals', etc.
    - combine RMS + Peak + boost contextuel
    """
    if len(audio_segment) == 0:
        return 64  # valeur par défaut raisonnable

    # Calcul RMS
    rms = safe_rms(audio_segment)
    rms_val = float(np.mean(rms)) if len(rms) else 1e-8

    # Pic absolu
    peak = float(np.max(np.abs(audio_segment)))

    # --- Mix RMS + Peak ---
    raw = 0.7 * peak + 0.3 * rms_val

    # --- Calibration / Boost en fonction du type ---
    if mode == "drum":
        scale = 350
        minimum = 45
    elif mode == "bass":
        scale = 320
        minimum = 35
    elif mode == "vocals":
        scale = 280
        minimum = 30
    else:
        scale = 280
        minimum = 30

    velocity = int(np.clip(raw * scale, minimum, 127))
    return velocity

def chunk_crepe(audio, sr, chunk_sec=30, model_capacity="large", confidence_threshold=0.8,
                smooth_window=5, pitch_tol=1.0, max_gap=0.15, min_note_length=0.05):
    # Convert to mono float32 in [-1,1]
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val
    # Resample to 16k if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    chunk_size_samples = int(chunk_sec * sr)
    notes_all = []

    start_sample = 0
    length = len(audio)
    while start_sample < length:
        end_sample = min(start_sample + chunk_size_samples, length)
        chunk_data = audio[start_sample:end_sample]

        t, f, conf, _ = crepe.predict(
            chunk_data, sr, model_capacity=model_capacity, viterbi=True
        )
        # offset times
        offset_time_sec = start_sample / sr
        t_offset = t + offset_time_sec
        # confidence filter
        mask = conf >= confidence_threshold
        t_final = t_offset[mask]
        t_final = np.array([safe_float(x) for x in t_final])
        f_final = f[mask]

        if len(f_final) == 0:
            start_sample += chunk_size_samples
            continue

        # smoothing
        f_smooth = smooth_frequencies(f_final, window_size=smooth_window)

        # grouping

        groups = group_frames_to_notes(
            times=t_final,
            freqs=f_smooth,
            pitch_tol=pitch_tol,
            max_gap=max_gap,
            min_note_length=min_note_length
        )
        for (start_t, end_t, midi_pitch) in groups:
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            segment = audio[start_sample:end_sample]
            velocity = velocity_from_amplitude(segment)
            note_obj = safe_note(
                velocity=velocity,
                pitch=midi_pitch,
                start=safe_float(start_t),
                end=safe_float(end_t)
            )

            notes_all.append(note_obj)
        start_sample += chunk_size_samples

    return notes_all

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

