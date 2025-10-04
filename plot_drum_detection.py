import matplotlib.pyplot as plt
import librosa
import numpy as np
from Config import *

def plot_drum_detections(audio, sr, detected_notes, title="Drum Hit Detection Debug", max_duration_sec=60):
    """
    Affiche la waveform + positions des coups de batterie + labels détectés,
    limité à max_duration_sec secondes.

    - audio : numpy array (mono)
    - sr : sample rate
    - detected_notes : liste d'objets note (avec .start, .pitch, .velocity)
    - max_duration_sec : durée maximale affichée (en secondes)
    """

    # Construction mapping pitch -> label
    inverse_drum_map = {v: k for k, v in CONFIG["drum_map"].items()}

    # Limiter l'audio aux premières secondes
    max_samples = int(max_duration_sec * sr)
    audio = audio[:max_samples]

    time_axis = np.arange(len(audio)) / sr

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(time_axis, audio, alpha=0.6, label="Audio waveform")

    # Couleurs par type d'instrument
    color_map = {
        "kick": "red",
        "snare": "blue",
        "hihat": "green",
        "hihat_open": "lightgreen",
        "crash 1": "orange",
        "crash 2": "gold",
        "ride": "purple",
        "china": "cyan",
        "floor": "brown",
        "tom 1": "pink",
        "tom 2": "violet",
        "tom 3": "magenta",
        "unknown": "gray"
    }

    # Afficher uniquement les notes dans la fenêtre 0..max_duration_sec
    for note in detected_notes:
        if note.start > max_duration_sec:
            continue  # Ignore notes après la limite de temps

        start_time = note.start
        pitch = note.pitch
        label = inverse_drum_map.get(pitch, "unknown")
        color = color_map.get(label, "black")

        ax.axvline(x=start_time, color=color, linestyle="--", alpha=0.8)
        ax.text(start_time, np.max(audio) * 0.9, label, rotation=90,
                color=color, verticalalignment='bottom', fontsize=8)

    ax.set_xlim(0, max_duration_sec)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_bpm_over_time(audio, sr, window_sec=15, hop_sec=5, title="BPM Evolution"):
    """
    Calcule et trace le BPM local au fil du temps.

    - audio : signal mono
    - sr : sample rate
    - window_sec : taille de la fenêtre pour estimer BPM (ex: 15s)
    - hop_sec : intervalle entre deux mesures (ex: 5s)
    """
    bpm_list = []
    time_list = []

    total_duration = len(audio) / sr
    hop_samples = int(hop_sec * sr)
    window_samples = int(window_sec * sr)

    for start_sample in range(0, len(audio) - window_samples, hop_samples):
        end_sample = start_sample + window_samples
        segment = audio[start_sample:end_sample]

        tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
        if isinstance(tempo, (list, np.ndarray)):
            bpm = float(tempo[0])
        else:
            bpm = float(tempo)

        start_time_sec = start_sample / sr
        bpm_list.append(bpm)
        time_list.append(start_time_sec)

    # Tracé
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_list, bpm_list, marker="o", linestyle="-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated BPM")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()