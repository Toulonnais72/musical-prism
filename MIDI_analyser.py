import pretty_midi
import pandas as pd

# Charger ton fichier MIDI
midi_path = "Symphony X - Champion Of Ithaca_output_20250421-104252.mid"
pm = pretty_midi.PrettyMIDI(midi_path)

# Dictionnaire General MIDI drums
drum_mapping = {
    36: "Kick",
    35: "Kick 2",
    38: "Snare",
    40: "Electric Snare",
    37: "Side Stick",
    39: "Hand Clap",
    42: "Closed Hi-Hat",
    44: "Pedal Hi-Hat",
    46: "Open Hi-Hat",
    49: "Crash 1",
    57: "Crash 2",
    51: "Ride",
    52: "China Cymbal",
    41: "Floor Tom",
    43: "Floor Tom 2",
    45: "Low Tom",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    50: "High Tom"
}

# Compter les notes
note_counts = {}
for instrument in pm.instruments:
    if instrument.is_drum:
        for note in instrument.notes:
            label = drum_mapping.get(note.pitch, f"Unknown ({note.pitch})")
            note_counts[label] = note_counts.get(label, 0) + 1

# Mettre dans un DataFrame
df_counts = pd.DataFrame.from_dict(note_counts, orient="index", columns=["Count"]).sort_values(by="Count", ascending=False)

print(df_counts)
