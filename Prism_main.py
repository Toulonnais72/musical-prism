import warnings
warnings.filterwarnings("ignore")
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pygame
from demucs.pretrained import get_model
from demucs.apply import apply_model
from Config import *
from Helper import *
from typing import List, Dict
from plot_drum_detection import *

# For polyphonic transcription (Basic Pitch) on the "other" stem
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import numpy as np, librosa, scipy.signal as sig
from typing import Optional

# correctifs NumPy↔librosa (Py3.9)
for attr, alias in [("complex", complex), ("float", float), ("int", int)]:
    if not hasattr(np, attr):
        setattr(np, attr, alias)

pygame.mixer.init()

# --------------------------------------------------------------------------------
# Detect GPU or CPU
# --------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
torch.backends.cudnn.benchmark = True

CLUSTER_THRESHOLD = 0.01  # Onsets under 10 ms apart are considered “simultaneous”


###############################################################################
# BASIC PITCH => single pass on "other" stem
###############################################################################

def transcribe_polyphonic_basic_pitch(audio_path, output_midi_path):
    """
    Calls Basic Pitch on 'audio_path', writes <basename>_basic_pitch.mid,
    renames to 'output_midi_path'. Returns a list of instruments from that file.
    """
    out_dir = os.path.dirname(output_midi_path)
    os.makedirs(out_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    midi_filename = f"{base_filename}_other_poly.mid"
    output_midi_path = os.path.join(out_dir, midi_filename)
    # Ensure the output MIDI path is unique
    if os.path.exists(output_midi_path):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_filename = f"{base_filename}_other_poly_{timestamp}.mid"
        output_midi_path = os.path.join(out_dir, unique_filename)

    mid_created = os.path.join(out_dir, f"{base_filename}_basic_pitch.mid")
    if os.path.exists(mid_created):
        os.remove(mid_created)
    print("Notes detection on 'other instruments' using Basic Pitch (R)")
    predict_and_save(
        audio_path_list=[audio_path],
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        output_directory=out_dir,
        save_midi=True,
        save_model_outputs=False,
        sonify_midi=False
    )

    # Check for the created MIDI file and rename it if necessary
    mid_created = os.path.join(out_dir, f"{base_filename}_basic_pitch.mid")
    if os.path.exists(mid_created):
        # Ensure the final MIDI path is unique
        if os.path.exists(output_midi_path):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            unique_filename = f"{base_filename}_other_full_basic_pitch_{timestamp}.mid"
            output_midi_path = os.path.join(out_dir, unique_filename)
        os.rename(mid_created, output_midi_path)

    instruments = []
    if os.path.exists(output_midi_path):
        pm_temp = pretty_midi.PrettyMIDI(output_midi_path)
        instruments = pm_temp.instruments
    return instruments

def split_polyphonic_by_pitch_range(instruments, split_pitch=60):
    """
    Split polyphonic notes into different instruments based on pitch range.

    Parameters:
    - instruments: List of PrettyMIDI instruments containing notes.
    - split_pitch: The pitch value to split the notes.

    Returns:
    - A list of PrettyMIDI instruments with notes split by pitch range.
    """
    all_notes = []
    for inst in instruments:
        all_notes.extend(inst.notes)
    all_notes.sort(key=lambda n: n.start)

    low_instrument = pretty_midi.Instrument(program=32)  # e.g., Acoustic Bass
    high_instrument = pretty_midi.Instrument(program=0)   # e.g., Piano
    for note in all_notes:
        if note.pitch < split_pitch:
            low_instrument.notes.append(note)
        else:
            high_instrument.notes.append(note)

    out = []
    if len(low_instrument.notes) > 0:
        out.append(low_instrument)
    if len(high_instrument.notes) > 0:
        out.append(high_instrument)
    return out


###############################################################################
# MAIN CLASS => chunk-based DEMUCS + advanced drum classification, etc.
###############################################################################

class WavToMidiConverter:
    def __init__(self, config, output_folder="."):
        self.config = config
        self.progress_callbacks = []
        self.measures = []
        self.drums_notes = []
        self.bass_notes  = []
        self.vocal_notes = []
        self.other_instruments = []  # poly instruments from Basic Pitch
        self.output_folder = output_folder  # store stems + final MIDI in user-chosen folder

    def _debug_onsets(self, audio, sr, onsets, title="Onset debug"):
        import matplotlib.pyplot as plt
        times = np.arange(len(audio)) / sr
        plt.figure(figsize=(12, 3))
        plt.plot(times, audio, label="Waveform", alpha=0.7)
        for onset in onsets:
            plt.axvline(onset, color='r', linestyle='--', alpha=0.6)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def register_progress_callback(self, callback):
        self.progress_callbacks.append(callback)

    def update_progress(self, stage, value):
        for cb in self.progress_callbacks:
            cb(stage, value)

    def get_bpm_at_time(self, t_sec):
        """
        Récupère le BPM pré-calculé au temps t_sec par interpolation.
        """
        if not hasattr(self, 'bpm_curve') or not self.bpm_curve:
            return 120  # défaut

        idx = np.searchsorted(self.bpm_times, t_sec) - 1
        idx = max(0, min(idx, len(self.bpm_curve) - 1))
        return self.bpm_curve[idx]

    def precompute_bpm_curve(self, audio, sr, window_sec=8, hop_sec=6):
        """
        Version optimisée du pré-calcul de la courbe BPM.
        """
        bpm_curve = []
        time_points = []

        # --- Downsample rapide pour BPM uniquement ---
        if sr > 8000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=8000)
            sr = 8000

        total_duration = len(audio) / sr
        hop_samples = int(hop_sec * sr)
        window_samples = int(window_sec * sr)

        for start_sample in range(0, len(audio) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            segment = audio[start_sample:end_sample]

            try:
                # Utiliser onset_strength rapide
                onset_env = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=256)

                # Calculer tempogram et maximum pour estimer BPM
                tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=256)
                ac = np.mean(tempogram, axis=1)
                best_period = np.argmax(ac[1:]) + 1
                bpm = 60.0 * sr / (256 * best_period)

                # Clamp BPM dans limites métal
                bpm = np.clip(bpm, 70, 280)
                bpm = round(bpm, 1)

            except Exception as e:
                print(f"[WARN] BPM segment estimation failed: {e}")
                bpm = 120

            start_time_sec = start_sample / sr
            bpm_curve.append(bpm)
            time_points.append(start_time_sec)

        self.bpm_curve = bpm_curve
        self.bpm_times = time_points

    def convert(self, file_path):

        self.base_filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            total_stages = 6  # Number of major stages in the conversion process
            current_stage = 0

            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            # If auto_bpm => measure BPM first
            if self.config["auto_bpm"]:
                self.auto_detect_bpm(file_path)
                current_stage += 1
                self.update_progress("bpm", current_stage / total_stages * 100)

            # 1) chunk-based demucs => reassemble stems
            stems_dict, sr_stems = self.run_demucs_in_chunks(file_path)
            current_stage += 1
            self.update_progress("demucs", current_stage / total_stages * 100)

            # 2) chunk-based approach for drums, bass, vocals
            # Drums
            if "drums" in stems_dict:
                self.process_drums_in_chunks(stems_dict["drums"], sr_stems)
            current_stage += 1
            self.update_progress("drums", current_stage / total_stages * 100)

            # Bass
            if "bass" in stems_dict:
                bass_notes = chunk_crepe(
                    stems_dict["bass"], sr_stems,
                    chunk_sec=self.config["crepe_chunk_sec"],
                    model_capacity=self.config["crepe_model_capacity"],
                    confidence_threshold=self.config["confidence_threshold"],
                    smooth_window=self.config["smooth_window_size"],
                    pitch_tol=self.config["pitch_tol"],
                    max_gap=self.config["max_gap"],
                    min_note_length=self.config["min_note_length"]
                )
                self.bass_notes.extend(bass_notes)
            current_stage += 1
            self.update_progress("bass", current_stage / total_stages * 100)

            # Vocals
            if "vocals" in stems_dict:
                vocal_notes = chunk_crepe(
                    stems_dict["vocals"], sr_stems,
                    chunk_sec=self.config["crepe_chunk_sec"],
                    model_capacity=self.config["crepe_model_capacity"],
                    confidence_threshold=self.config["confidence_threshold"],
                    smooth_window=self.config["smooth_window_size"],
                    pitch_tol=self.config["pitch_tol"],
                    max_gap=self.config["max_gap"],
                    min_note_length=self.config["min_note_length"]
                )
                self.vocal_notes.extend(vocal_notes)
            current_stage += 1
            self.update_progress("vocals", current_stage / total_stages * 100)

            # 3) Polyphonic "other" => single pass Basic Pitch
            if "other" in stems_dict:
                stems_dir = os.path.join(self.output_folder, "stems")
                os.makedirs(stems_dir, exist_ok=True)

                other_path = os.path.join(stems_dir, f"{base_filename}_other.wav")
                torchaudio.save(
                    other_path,
                    torch.tensor(stems_dict["other"]),
                    sr_stems
                )
                # Basic Pitch
                out_midi = get_unique_output_path(other_path, os.path.join(self.output_folder, "stems"), "mid")
                instruments_poly = transcribe_polyphonic_basic_pitch(other_path, out_midi)
                splitted = split_polyphonic_by_pitch_range(instruments_poly, split_pitch=60)
                self.other_instruments.extend(splitted)
                current_stage += 1
                self.update_progress("polyphonic", current_stage / total_stages * 100)

            # 4) Build final PrettyMIDI => drums, bass, vocals, other
            pm = pretty_midi.PrettyMIDI()

            # Drums
            '''drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
            drum_merged = merge_adjacent_notes(
                self.drums_notes, adjacency_threshold=self.config["adjacency_threshold"]
            )
            if self.config["enable_quantize"]:
                drum_snapped = snap_notes_to_tempo(
                    drum_merged, bpm=self.config["bpm"], steps_per_beat=self.config["steps_per_beat"]
                )
                drum_inst.notes = drum_snapped
            else:
                drum_inst.notes = drum_merged
            pm.instruments.append(drum_inst)'''
            # Drums : reconstruction à partir des mesures quantisées
            drum_pm = measures_to_midi(self.measures, self.config["drum_map"])
            # drum_pm est un PrettyMIDI complet : on récupère juste l’instrument
            pm.instruments.extend(drum_pm.instruments)

            # Bass
            bass_inst = pretty_midi.Instrument(program=32)
            bass_merged = merge_adjacent_notes(
                self.bass_notes, adjacency_threshold=self.config["adjacency_threshold"]
            )
            if self.config["enable_quantize"]:
                bass_snapped = snap_notes_to_tempo(
                    bass_merged, self.config["bpm"], self.config["steps_per_beat"]
                )
                bass_inst.notes = bass_snapped
            else:
                bass_inst.notes = bass_merged
            pm.instruments.append(bass_inst)

            # Vocals
            voc_inst = pretty_midi.Instrument(program=52)
            vocal_merged = merge_adjacent_notes(
                self.vocal_notes, adjacency_threshold=self.config["adjacency_threshold"]
            )
            if self.config["enable_quantize"]:
                voc_snapped = snap_notes_to_tempo(
                    vocal_merged, self.config["bpm"], self.config["steps_per_beat"]
                )
                voc_inst.notes = voc_snapped
            else:
                voc_inst.notes = vocal_merged
            pm.instruments.append(voc_inst)

            # Other => splitted from Basic Pitch
            for inst in self.other_instruments:
                merged_notes = merge_adjacent_notes(
                    inst.notes, adjacency_threshold=self.config["adjacency_threshold"]
                )
                snapped_notes = []

                for n in merged_notes:
                    snapped_notes.append(
                        safe_note(
                            velocity=n.velocity,
                            pitch=n.pitch,
                            start=n.start,
                            end=n.end
                        )
                    )

                inst.notes = snapped_notes
                pm.instruments.append(inst)

            # Write final MIDI in the user-chosen folder
            final_midi_path = os.path.join(self.output_folder, f"{base_filename}_output.mid")
            if os.path.exists(final_midi_path):
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                final_midi_path = os.path.join(self.output_folder, f"{base_filename}_output_{timestamp}.mid")
            pm.write(final_midi_path)
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    def auto_detect_bpm(self, file_path):
        """Auto-detect BPM from entire file in mono, using librosa.beat.beat_track."""
        wav, sr = torchaudio.load(file_path)
        # average to mono
        mono = wav.mean(dim=0).numpy()
        if sr != self.config["auto_bpm_sr"]:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=self.config["auto_bpm_sr"])
            sr = self.config["auto_bpm_sr"]
        tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr)
        beat_times = np.array([safe_float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)])
        if isinstance(tempo, np.ndarray):
            tempo_scalar = float(np.max(tempo)) #<------ provides max BPM detected
        else:
            tempo_scalar = float(tempo)

        self.config["bpm"] = int(round(tempo_scalar))
        print(f"Auto BPM => {self.config['bpm']}")

    def detect_bpm_curve(self, audio: np.ndarray, sr: int, win_sec: float = 8.0, hop_sec: float = 4.0) -> None:
        bpms, times = [], []
        hop = int(hop_sec * sr)
        win = int(win_sec * sr)

        for start in range(0, len(audio) - win, hop):
            y = audio[start:start + win]
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpms.append(float(tempo))
            times.append(start / sr)

        self.bpm_curve = bpms
        self.bpm_times = times
        self.bpm_window = win_sec

    def get_bpm_at(self, t: float) -> float:
        if not hasattr(self, "bpm_curve"):
            return 120.0
        idx = max(0, min(np.searchsorted(self.bpm_times, t) - 1,
                         len(self.bpm_curve) - 1))
        return self.bpm_curve[idx]

    def analyse_measures(self, drums_mono: np.ndarray, sr: int, beats_per_bar: int = 4, ) -> List[Dict]:
        """
        Découpe le stem batterie en mesures 4/4, détecte les coups via trois passes:
            P0 : onsets bruts
            P1 : passe‑haut >2kHz  → Hi‑Hat/Crash
            P2 : bande 110‑400Hz    → Toms
        Renvoie une liste:
            [{"start":t0, "bpm":…, "notes":[(t_abs, label, velocity)…]}, …]
        """
        print("Multi pass measures analyser")
        measures = []
        total_dur = len(drums_mono) / sr
        t0 = 0.0

        while t0 < total_dur:
            # ---- BPM & longueur mesure ----
            bpm = self.get_bpm_at(t0)
            bar_len = 60.0 / bpm * beats_per_bar
            t1 = min(t0 + bar_len, total_dur)

            # ---- Segment audio ----
            s0, s1 = int(t0 * sr), int(t1 * sr)
            y_bar = drums_mono[s0:s1]

            # RMS moyen de la mesure (dBFS)
            rms_db = 20 * np.log10(np.sqrt(np.mean(y_bar ** 2)) + 1e-12)

            # Nombre d’échantillons > -35dBFS (pics)
            peaks = np.sum(np.abs(y_bar) > (10 ** (-35 / 20)))

            if rms_db < -50 and peaks < 100:  # seuils à ajuster
                measures.append({"start": t0, "bpm": bpm, "notes": []})
                t0 = t1
                continue

            # =========================================================
            #  P0 : onsets “généraux” (Kick / Snare / tout)
            # =========================================================
            on0 = librosa.onset.onset_detect(
                y=y_bar, sr=sr, hop_length=128, backtrack=True,
                delta=0.05, pre_max=10, post_max=10)

            # =========================================================
            #  P1 : Hi‑Hat & Crash  (passe‑haut 2kHz)
            # =========================================================
            y_hi = butter_band(y_bar, sr, 2000, None, order=4, btype='high')
            on1 = librosa.onset.onset_detect(
                y=y_hi, sr=sr, hop_length=128, backtrack=True,
                delta=0.03, pre_max=8, post_max=8)

            # =========================================================
            #  P2 : Toms  (bande 110‑400Hz)
            # =========================================================
            y_tom = butter_band(y_bar, sr, 110, 400, order=4, btype='band')
            on2 = librosa.onset.onset_detect(
                y=y_tom, sr=sr, hop_length=256, backtrack=True,
                delta=0.04, pre_max=6, post_max=6)

            # ---- Fusion / dé‑doublonnage (tolérance 20ms) ----
            frames = np.concatenate([on0, on1, on2])
            times = librosa.frames_to_time(frames, sr=sr, hop_length=128) + t0
            # arrondi au 0.02s pour cluster
            times_unique = np.unique(np.round(times / 0.02) * 0.02)

            # ---- Classification coup par coup ----
            notes = []
            for idx, t_abs in enumerate(times_unique):
                nxt = times_unique[idx + 1] if idx + 1 < len(times_unique) else None
                label, vel = self.classify_drum_hit_metal(
                    drums_mono, sr, float(t_abs), next_onset_time=nxt)
                if label != "unknown":
                    notes.append((float(t_abs), label, vel))

            measures.append({"start": t0, "bpm": bpm, "notes": notes})
            t0 = t1
        return measures

    def run_demucs_in_chunks(self, file_path):
        """
        Splits input into demucs_chunk_sec slices,
        runs demucs on each chunk, reassembles stems.
        """
        print("Demucs in chunks to split the audio in Drums, Bass, Vocals, Other")
        stems_dir = os.path.join(self.output_folder, "stems")
        os.makedirs(stems_dir, exist_ok=True)

        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        wav, sr = torchaudio.load(file_path)
        # stereo if needed
        if wav.shape[0] == 1:
            wav = torch.cat([wav, wav], dim=0)
        model = get_model(self.config["demucs_model"]).to(device)
        model.eval()

        # If sr != model.samplerate => resample
        if sr != model.samplerate:
            resampler = torchaudio.transforms.Resample(sr, model.samplerate)
            wav = resampler(wav)
            sr = model.samplerate

        chunk_size_samples = int(sr * self.config["demucs_chunk_sec"])

        # We'll accumulate partial stems in a list of arrays
        stems_accum = {
            "drums": [],
            "bass": [],
            "other": [],
            "vocals": []
        }
        stem_order = ["drums", "bass", "other", "vocals"]

        total_len = wav.shape[-1]
        start = 0
        while start < total_len:
            end = min(start + chunk_size_samples, total_len)
            chunk_data = wav[..., start:end].unsqueeze(0).to(device)  # shape (1,2,chunk)
            with torch.no_grad():
                out = apply_model(model, chunk_data, overlap=self.config["demucs_overlap"])
                # out => shape (1,4,2,chunk_len)
                out = out.squeeze(0).cpu().numpy()  # => (4,2,chunk_len)
            for i, name in enumerate(stem_order):
                stems_accum[name].append(out[i])
            start += chunk_size_samples

            # Update progress for demucs
            progress_value = (start / total_len) * 100
            self.update_progress("demucs", progress_value)

        # reassemble
        final_stems = {}
        for name in stem_order:
            arrs = stems_accum[name]
            if len(arrs) > 0:
                cat_data = np.concatenate(arrs, axis=-1)  # shape => (2, total_samples)
                final_stems[name] = cat_data
            else:
                # Handle empty case by returning an empty array with the correct shape
                final_stems[name] = np.zeros((2, 0), dtype=np.float32)

        # Save each stem with a unique filename
        for name in stem_order:
            stem_path = os.path.join(stems_dir, f"{base_filename}_{name}.wav")
            torchaudio.save(
                stem_path,
                torch.tensor(final_stems[name]),
                sr
            )

        return final_stems, sr

    def extract_drum_features(self, audio, sr):
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # --- Spectrogram ---
        n_fft = 1024
        hop_length = 256
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        S_mag = np.abs(D)  # <-- ensures real-valued magnitude

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        S_log = np.log10(S_mag ** 2 + 1e-9)
        S_mean = np.mean(S_log, axis=1)

        # --- Band energies ---
        def band_energy(fmin, fmax):
            idx = (freqs >= fmin) & (freqs < fmax)
            return float(np.mean(S_mean[idx])) if np.any(idx) else 0.0

        e_sub = band_energy(20, 80)
        e_punch = band_energy(80, 200)
        e_mid = band_energy(200, 800)
        e_high = band_energy(2000, 8000)

        # --- Spectral features ---
        centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.90)[0]

        centroid_mean = float(np.mean(centroid))
        bandwidth_mean = float(np.mean(bandwidth))
        rolloff_mean = float(np.mean(rolloff))

        # --- Peak frequency bin ---
        peak_freq = freqs[np.argmax(S_mean)] if len(S_mean) > 0 else 0.0

        # --- Envelope & transient features ---
        envelope = librosa.onset.onset_strength(y=audio, sr=sr)
        if len(envelope) >= 5:
            attack_strength = np.max(envelope[:5]) - np.mean(envelope[5:10]) if len(envelope) >= 10 else np.max(
                envelope[:5]) - np.mean(envelope[5:])
            envelope_std = float(np.std(envelope))
            decay_rate = float((envelope[2] - envelope[-1]) / (len(envelope) - 2)) if len(envelope) > 2 else 0.0
        else:
            attack_strength = 0.0
            envelope_std = 0.0
            decay_rate = 0.0

        # --- Amplitude ---
        rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0
        peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        amplitude = 0.7 * peak + 0.3 * rms

        # --- Spectral Flux ---
        spectral_flux = float(np.mean(librosa.onset.onset_strength(S=S_mag, sr=sr))) if len(audio) > 0 else 0.0

        # --- Zero Crossing Rate ---
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio))) if len(audio) > 0 else 0.0

        # --- Return real-valued, safe dictionary ---
        return {
            "e_sub": e_sub,
            "e_punch": e_punch,
            "e_mid": e_mid,
            "e_high": e_high,
            "centroid_mean": centroid_mean,
            "bandwidth_mean": bandwidth_mean,
            "rolloff_mean": rolloff_mean,
            "peak_freq_bin": peak_freq,
            "attack_strength": attack_strength,
            "envelope_std": envelope_std,
            "decay_rate": decay_rate,
            "spectral_flux": spectral_flux,
            "zero_crossing_rate": zcr,
            "amplitude": amplitude
        }

    def compute_drum_velocity(self, audio_segment, sr=16000):

        # Safety check: If the segment is empty, return a default
        if len(audio_segment) == 0:
            return 64

        # Calculate RMS and Peak
        rms_val = float(np.sqrt(np.mean(audio_segment ** 2)))  # Root Mean Square
        peak_val = float(np.max(np.abs(audio_segment)))  # Absolute peak amplitude

        # Combine them into a single “raw” measure
        # You can tune these multipliers based on your own mixes:
        # e.g. weighting peaks more strongly than RMS, or vice-versa.
        raw_amplitude = 0.7 * peak_val + 0.3 * rms_val

        # Scale up to a typical drum “velocity” range
        # You can tweak the scale and minimum to taste:
        scale_factor = 320.0  # overall multiplier
        minimum_val = 30  # minimum velocity so small hits don’t vanish
        velocity = int(round(raw_amplitude * scale_factor))

        # Clamp into 1..127
        velocity = max(1, min(velocity, 127))
        if velocity < minimum_val:
            velocity = minimum_val

        return velocity

    def classify_drum_hit_metal(self, audio_mono: np.ndarray, sr: int, onset_time: float, next_onset_time: Optional[float] = None, debug: bool = False):
        """
        •Fenêtre adaptative (60ms par défaut/12ms si kick dominateur)
        •Multi‑bande: Sub, Mid, Hi+ bande Snare large
        •Scoring Kick/Snare→arbitrage prioritaire
        •Fallback : Hi‑Hat/ Crash/ Tom/ Floor
        """
        # ---------------------- 1) Fenêtre adaptative -------------------
        bpm = self.get_bpm_at(onset_time) if hasattr(self, "get_bpm_at") else 120.0
        base_len = 0.06  # 60ms
        max_len = 0.12  # 120ms si kick fort
        seg_len = base_len

        # petit “sondage” Sub pour décider
        pre = audio_mono[int(onset_time * sr): int((onset_time + base_len) * sr)]
        p_perc, _ = librosa.effects.hpss(pre)
        r_sub_check = band_energy(p_perc, sr, 20, 120) / (
                band_energy(p_perc, sr, 20, 12000) + 1e-8
        )
        if r_sub_check > 0.25:
            seg_len = max_len

        if next_onset_time:
            seg_len = min(seg_len, next_onset_time - onset_time - 0.003)
        seg_len = max(0.04, seg_len)

        s0, s1 = int(onset_time * sr), int((onset_time + seg_len) * sr)
        seg = audio_mono[s0:s1]
        if len(seg) < 32:
            return "unknown", 40

        # ---------------------- 2) HPSS + nettoyage ---------------------
        percussive, _ = librosa.effects.hpss(seg)
        percussive = np.nan_to_num(percussive)

        # ---------------------- 3) Énergies de bandes -------------------
        E_tot = band_energy(percussive, sr, 20, 12000) + 1e-8
        E_sub = band_energy(percussive, sr, 20, 120)
        E_mid = band_energy(percussive, sr, 200, 2000)
        E_hi = band_energy(percussive, sr, 4500, 12000)
        E_sn_broad = band_energy(percussive, sr, 400, 5000)

        r_sub, r_mid, r_hi = E_sub / E_tot, E_mid / E_tot, E_hi / E_tot
        r_sn_broad = E_sn_broad / E_tot

        centroid = librosa.feature.spectral_centroid(y=percussive, sr=sr)[0].mean()
        S = np.abs(librosa.stft(percussive, n_fft=512, hop_length=128)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        peak_freq = freqs[np.argmax(S.mean(axis=1))]
        bandwidth = librosa.feature.spectral_bandwidth(y=percussive, sr=sr)[0].mean()

        attack = np.max(percussive) - np.mean(np.abs(percussive))
        decay = (percussive[0] - percussive[-1]) / len(percussive)

        vel = self.compute_drum_velocity(seg, sr)

        # ---------------------- 4) Scoring Kick / Snare -----------------
        kick_score = (3 * (r_sub > 0.30) +
                      2 * (centroid < 900) +
                      2 * (peak_freq < 120) +
                      1 * (attack > 0.05))

        sn_score = (3 * (r_mid > 0.18) +
                    2 * (900 <= centroid <= 3300) +
                    2 * (180 <= peak_freq <= 350) +
                    1 * (attack > 0.04) +
                    1 * (decay > -0.55) +
                    1 * (r_sn_broad > 0.45))

        primary = "kick" if kick_score >= sn_score + 2 else "snare"

        if primary in ("kick", "snare"):
            if debug:
                print(f"{onset_time:7.3f}s  {primary.upper():5}  "
                      f"K:{kick_score} S:{sn_score} "
                      f"sub:{r_sub:.2f} mid:{r_mid:.2f} hi:{r_hi:.2f}")
            return primary, vel

        # ---------------------- 5) Hi‑Hat / Crash -----------------------
        if r_hi > 0.30 and centroid > 3500:
            # Hi‑Hat (closed / open)
            if decay > -0.30 and attack < 0.10:
                return "hihat_open", vel
            return "hihat", vel

        if (centroid > 5000 or bandwidth > 4000) and r_hi > 0.25:
            return ("crash2" if centroid > 6500 else "crash1"), vel

        # ---------------------- 6) Toms / Floor ------------------------
        if 110 <= peak_freq < 180 and r_sub > 0.20:
            return "floor", vel
        if 180 <= peak_freq < 260:
            return "tom2", vel
        if 260 <= peak_freq < 400:
            return "tom1", vel

        # ---------------------- 7) Inconnu -----------------------------
        if debug:
            print(f"{onset_time:7.3f}s  UNKNOWN  "
                  f"sub:{r_sub:.2f} mid:{r_mid:.2f} hi:{r_hi:.2f} "
                  f"cent:{centroid:.0f} pf:{peak_freq:.0f}")
        return "unknown", vel

    def process_drums_in_chunks(self, drums_stem: np.ndarray, sr: int) -> None:
        """
        1.  Mono + courbe BPM locale
        2.  Découpe en mesures 4/4 → multi‑pass détection / classification
        3.  Quantize sur la grille de la mesure
        4.  Stocke le résultat dans self.measures
        """
        # 1) Stéréo → Mono
        if drums_stem.shape[0] > 1:
            audio_mono = drums_stem.mean(axis=0)
        else:
            audio_mono = drums_stem[0]
        mono = np.asarray(audio_mono).flatten().astype(np.float32)

        # 2) PRE‑PROCESSING (options lues depuis self.config)
        audio_proc, sr_proc = preprocess_drum_stem(
            mono,
            sr,
            target_sr=48_000,  # ou sr pour garder la fréquence d’origine
            apply_expander=self.config.get("pre_expander", True),
            apply_hpss=self.config.get("pre_hpss", True),
            apply_spectral_gate=self.config.get("pre_spec_gate", False),
            apply_rms_gate=self.config.get("pre_rms_gate", True),
            apply_eq_bands=self.config.get("pre_eq_bands", False),
        )

        # ---------- 2) BPM /‑chunks ----------
        self.detect_bpm_curve(mono, sr)  # crée self.bpm_curve / self.bpm_times

        # ---------- 3) Analyse mesure par mesure ----------
        measures = self.analyse_measures(mono, sr)  # détecte & classe
        measures = [quantize_measure(m) for m in measures]  # quantize

        # ---------- 4) Stockage ----------
        self.measures = measures  # utilisé plus tard pour l’écriture MIDI

        # 6) Diagnostic graphique (facultatif)
        if self.config.get("debug_plot", False):
            plot_drum_detections(audio_proc, sr_proc,
                                 self.drums_notes,
                                 title=f"Debug – {self.base_filename}")


    def process_drums_in_chunks_00(self, drums_stem, sr):
        """
        •Applique le pré‑processing choisi par l’utilisateur
        •Pré‑calcule la courbe BPM
        •Détecte + classe les onsets (Kick, Snare, HH, Toms…)
        •Alimente self.drums_notes
        """
        # 1) Stéréo → Mono
        if drums_stem.shape[0] > 1:
            audio_mono = drums_stem.mean(axis=0)
        else:
            audio_mono = drums_stem[0]
        audio_mono = np.asarray(audio_mono).flatten().astype(np.float32)

        # 2) PRE‑PROCESSING (options lues depuis self.config)
        audio_proc, sr_proc = preprocess_drum_stem(
            audio_mono,
            sr,
            target_sr=48_000,  # ou sr pour garder la fréquence d’origine
            apply_expander=self.config.get("pre_expander", True),
            apply_hpss=self.config.get("pre_hpss", True),
            apply_spectral_gate=self.config.get("pre_spec_gate", True),
            apply_rms_gate=self.config.get("pre_rms_gate", True),
            apply_eq_bands=self.config.get("pre_eq_bands", False),
        )
        # Analyse des mesures
        measures = self.analyse_measures(audio_proc, sr_proc)
        self.measures = measures

        # 3) Pré‑calcul BPM local
        self.precompute_bpm_curve(audio_proc, sr_proc)

        # 4) Découpage en chunks +détection onsets
        chunk_len = int(sr_proc * self.config["crepe_chunk_sec"])
        all_onsets = []
        n_samples = len(audio_proc)

        for start in range(0, n_samples, chunk_len):
            end = min(start + chunk_len, n_samples)
            chunk = audio_proc[start:end]
            offset = start / sr_proc

            onset_env = librosa.onset.onset_strength(y=chunk, sr=sr_proc, hop_length=64)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr_proc,
                hop_length=64,
                units='time',
                backtrack=True,
                delta=0.005, wait=3, pre_max=5, post_max=5
            )
            all_onsets.extend([o + offset for o in onsets])

            # progress bar «drums»
            prog = min(100.0, (end / n_samples) * 100)
            self.update_progress("drums", prog)

        # 5) Clustering (≤10ms) puis classification
        all_onsets.sort()
        clusters, CLUSTER = [], 0.01
        if all_onsets:
            cur = [all_onsets[0]]
            for t in all_onsets[1:]:
                if t - cur[-1] <= CLUSTER:
                    cur.append(t)
                else:
                    clusters.append(cur);
                    cur = [t]
            clusters.append(cur)

        for cl in clusters:
            for i, t in enumerate(cl):
                nxt = cl[i + 1] if i + 1 < len(cl) else None
                label, vel = self.classify_drum_hit_metal(
                    audio_proc, sr_proc, t, next_onset_time=nxt)
                if label in self.config["drum_map"]:
                    note = safe_note(
                        velocity=vel,
                        pitch=self.config["drum_map"][label],
                        start=t,
                        end=t + 0.10  # 100ms note‑off
                    )
                    self.drums_notes.append(note)




###############################################################################
# TKINTER UI
###############################################################################
class WavToMidiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wav to MIDI Converter")
        self.root.geometry("900x760")                # +60px pour la nouvelle section

        self.output_folder = os.path.abspath(".")
        self.converter = None
        self._setup_ui()

    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#444', foreground='white')
        style.configure('TCheckbutton', background='#2b2b2b', foreground='white')
        style.configure('TCombobox', fieldbackground='#444', foreground='white')

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



        # Audio file
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(file_frame, width=60)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side=tk.LEFT)

        # Output folder
        out_frame = ttk.Frame(main_frame)
        out_frame.pack(fill=tk.X, pady=5)
        ttk.Label(out_frame, text="Output Folder:").pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(out_frame, width=60)
        self.output_entry.insert(0, self.output_folder)
        self.output_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(out_frame, text="Browse", command=self._browse_output).pack(side=tk.LEFT)

        # Device label (GPU vs CPU)
        device_frame = ttk.Frame(main_frame)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text=f"Using device: {device_str}").pack(side=tk.LEFT)

        # DEMUCS model
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Demucs Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=CONFIG["demucs_model"])
        model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["htdemucs","htdemucs_ft","mdx_extra","mdx_extra_q","mdx_q","mdx"]
        )
        model_dropdown.pack(side=tk.LEFT, padx=5)

        # CREPE model
        crepe_frame = ttk.Frame(main_frame)
        crepe_frame.pack(fill=tk.X, pady=5)
        ttk.Label(crepe_frame, text="CREPE Model:").pack(side=tk.LEFT)
        self.crepe_var = tk.StringVar(value=CONFIG["crepe_model_capacity"])
        crepe_dropdown = ttk.Combobox(
            crepe_frame,
            textvariable=self.crepe_var,
            values=["tiny","small","medium","large","full"]
        )
        crepe_dropdown.pack(side=tk.LEFT, padx=5)

                # Confidence threshold
        conf_frame = ttk.Frame(main_frame)
        conf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=CONFIG["confidence_threshold"])
        conf_spin = ttk.Spinbox(conf_frame, textvariable=self.conf_var, from_=0.5, to=1.0, increment=0.05)
        conf_spin.pack(side=tk.LEFT, padx=5)

        # Auto BPM
        auto_frame = ttk.Frame(main_frame)
        auto_frame.pack(fill=tk.X, pady=5)
        self.auto_bpm_var = tk.BooleanVar(value=CONFIG["auto_bpm"])
        auto_bpm_check = ttk.Checkbutton(auto_frame, text="Auto-Detect BPM", variable=self.auto_bpm_var)
        auto_bpm_check.pack(side=tk.LEFT, padx=5)

        # Snap to grid
        quant_frame = ttk.Frame(main_frame)
        quant_frame.pack(fill=tk.X, pady=5)
        self.quant_var = tk.BooleanVar(value=CONFIG["enable_quantize"])
        quant_check = ttk.Checkbutton(quant_frame, text="Snap to Grid", variable=self.quant_var)
        quant_check.pack(side=tk.LEFT)

        # BPM
        ttk.Label(quant_frame, text="BPM:").pack(side=tk.LEFT, padx=5)
        self.bpm_var = tk.IntVar(value=CONFIG["bpm"])
        bpm_spin = ttk.Spinbox(quant_frame, textvariable=self.bpm_var, from_=40, to=240, increment=5)
        bpm_spin.pack(side=tk.LEFT, padx=5)

        # Steps/Beat
        ttk.Label(quant_frame, text="Steps/Beat:").pack(side=tk.LEFT, padx=5)
        self.spb_var = tk.IntVar(value=CONFIG["steps_per_beat"])
        spb_spin = ttk.Spinbox(quant_frame, textvariable=self.spb_var, from_=1, to=32, increment=1)
        spb_spin.pack(side=tk.LEFT, padx=5)

        # 🔵  SECTION: Drum Pre‑processing  🔵
        pre_frame = ttk.LabelFrame(main_frame, text="Drum Pre‑processing")
        pre_frame.pack(fill=tk.X, pady=8, ipady=4)

        self.pre_expander_var = tk.BooleanVar(value=True)
        self.pre_hpss_var = tk.BooleanVar(value=True)
        self.pre_spec_gate_var = tk.BooleanVar(value=False)
        self.pre_rms_gate_var = tk.BooleanVar(value=True)
        self.pre_eq_bands_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(pre_frame, text="Dynamic Expander", variable=self.pre_expander_var).grid(row=0, column=0, sticky='w', padx=6, pady=2)
        ttk.Checkbutton(pre_frame, text="HPSS (transient focus)", variable=self.pre_hpss_var).grid(row=0, column=1,sticky='w', padx=6, pady=2)
        ttk.Checkbutton(pre_frame, text="Spectral Gate", variable=self.pre_spec_gate_var).grid(row=0, column=2, sticky='w', padx=6, pady=2)
        ttk.Checkbutton(pre_frame, text="RMSGate (<‑60dB)", variable=self.pre_rms_gate_var).grid(row=1, column=0, sticky='w', padx=6, pady=2)
        ttk.Checkbutton(pre_frame, text="3‑Band EQ boost", variable=self.pre_eq_bands_var).grid(row=1, column=1, sticky='w', padx=6, pady=2)

        # Progress bars
        self.progress_bars = {
            "bpm": ttk.Progressbar(main_frame, mode="determinate"),
            "demucs": ttk.Progressbar(main_frame, mode="determinate"),
            "drums": ttk.Progressbar(main_frame, mode="determinate"),
            "bass": ttk.Progressbar(main_frame, mode="determinate"),
            "vocals": ttk.Progressbar(main_frame, mode="determinate"),
            "polyphonic": ttk.Progressbar(main_frame, mode="determinate")
        }
        for name, bar in self.progress_bars.items():
            ttk.Label(main_frame, text=name.capitalize()).pack()
            bar.pack(fill=tk.X, pady=2)

        # Control
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)
        ttk.Button(control_frame, text="Convert", command=self._start_conversion).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Play MIDI", command=self._play_midi).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export MIDI", command=self._export_midi).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).pack()

    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")]
        )
        if file_path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)

    def _browse_output(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder = folder_path
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder_path)

    def _start_conversion(self):
        file_path = self.file_entry.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an audio file.")
            return

        # Update config from UI
        CONFIG["demucs_model"] = self.model_var.get()
        CONFIG["crepe_model_capacity"] = self.crepe_var.get()
        CONFIG["confidence_threshold"] = float(self.conf_var.get())
        CONFIG["auto_bpm"] = self.auto_bpm_var.get()
        CONFIG["enable_quantize"] = self.quant_var.get()
        CONFIG["bpm"] = int(self.bpm_var.get())
        CONFIG["steps_per_beat"] = int(self.spb_var.get())
        CONFIG["pre_expander"] = self.pre_expander_var.get()
        CONFIG["pre_hpss"] = self.pre_hpss_var.get()
        CONFIG["pre_spec_gate"] = self.pre_spec_gate_var.get()
        CONFIG["pre_rms_gate"] = self.pre_rms_gate_var.get()
        CONFIG["pre_eq_bands"] = self.pre_eq_bands_var.get()

        # Make a new converter each time
        self.output_folder = self.output_entry.get() or "."
        self.converter = WavToMidiConverter(CONFIG, output_folder=self.output_folder)
        self.converter.register_progress_callback(self.update_progress)

        self.status_var.set("Processing. Please wait...")

        def conversion_thread():
            try:
                self.converter.convert(file_path)
                self.status_var.set("Conversion complete!")
            except Exception as e:
                self.status_var.set("Error occurred")
                messagebox.showerror("Error", str(e))
            finally:
                self.root.event_generate("<<ConversionComplete>>")

        threading.Thread(target=conversion_thread, daemon=True).start()
        self.root.bind("<<ConversionComplete>>", lambda e: self._enable_controls())






    def update_progress(self, stage, value):
        """
        Currently, this is only stubbed out.
        If you want actual progress, you need to call update_progress(...)
        in the chunk loops of run_demucs_in_chunks, etc.
        We'll just reflect it here.
        """
        self.root.after(0, lambda: self.progress_bars[stage].configure(value=value))

    def _enable_controls(self):
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button):
                widget["state"] = tk.NORMAL

    def _play_midi(self):
        # The final MIDI is "output.mid" in self.output_folder
        final_midi = os.path.join(self.output_folder, "output.mid")
        if os.path.exists(final_midi):
            os.system(f'start "{final_midi}"')
        else:
            messagebox.showerror("Error", "No MIDI file found.")

    def _export_midi(self):
        # The final MIDI is "output.mid" in self.output_folder
        final_midi = os.path.join(self.output_folder, "output.mid")
        if not os.path.exists(final_midi):
            messagebox.showerror("Error", "No MIDI file found to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".mid")
        if file_path:
            try:
                os.replace(final_midi, file_path)
                messagebox.showinfo("Success", f"MIDI exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = WavToMidiApp(root)
    root.mainloop()
