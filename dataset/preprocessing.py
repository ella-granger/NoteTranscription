import pretty_midi
from pathlib import Path
import pickle
import numpy as np
import torch
import torchaudio
from constants import *
from mido import MidiFile
from tqdm import tqdm


def convert_midi(midi_path):
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    mid = MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat

    print(len(mid.tracks))
    valid_list = ["Soprano", "Alto", "Tenor", "Bass"]
    info_track = mid.tracks[0]
    track_note_dict = {}
    for track in mid.tracks[1:]:
        if track.name not in valid_list:
            continue
        print(track, track.name)
        cur_time = 0
        note_dict = {}
        track_note_list = []
        for msg in track:
            cur_time = cur_time + msg.time
            if msg.type == "note_on":
                if msg.velocity > 0:
                    if note_dict.get(msg.note, None) is None:
                        note_dict[msg.note] = cur_time
                else:
                    if note_dict.get(msg.note, None) is not None:
                        start_tick = note_dict[msg.note]
                        note_dict[msg.note] = None
                        end_tick = cur_time

                        start_beat = start_tick / ticks_per_beat
                        end_beat = end_tick / ticks_per_beat
                        
                        track_note_list.append([msg.note, start_beat, end_beat - start_beat])
                        last_end = end_beat
        track_note_dict[track.name] = track_note_list

    
    for instrument in midi_data.instruments:
        if instrument.name not in valid_list:
            continue
        print(instrument.name)
        track_note_list = track_note_dict[instrument.name]
        pm_notes = instrument.notes
        assert len(pm_notes) == len(track_note_list)

        for n, note in zip(pm_notes, track_note_list):
            assert note[0] == n.pitch
            note.append(n.start)
            note.append(n.end)

        last_end_beat = 0
        last_end_time = 0
        for i, n in enumerate(track_note_list):
            if i > 0 and n[1] > last_end_beat:
                track_note_list.insert(i, [0, last_end_beat, n[1] - last_end_beat, last_end_time, n[3]])
            last_end_beat = track_note_list[i][2] + track_note_list[i][1]
            last_end_time = track_note_list[i][4]
            
    return track_note_dict

    # onset_list = torch.FloatTensor([x.start for x in note_list])
    # pitch_list = torch.ShortTensor([x.pitch for x in note_list])
    # offset_list = torch.FloatTensor([x.end for x in note_list])


if __name__ == "__main__":
    # content_dir = Path("/storageSSD/huiran/NoteTranscription/BachChorale/BachChorale")
    # target_dir = Path("/storageSSD/huiran/NoteTranscription")
    # content_dir = Path("/media/ella/Yu/UR/datasets/BachChorale/midi_align")
    content_dir = Path("/storageNVME/huiran/NoteTranscription/BachChorale_Rev/flac")
    target_dir = Path("/storageNVME/huiran/NoteTranscription/BachChorale_Rev")
    # mel_dir = target_dir / "BachChorale" / "mel"
    # note_dir = target_dir / "BachChorale" / "note"
    mel_dir = target_dir / "mel"
    note_dir = target_dir / "note"
    mel_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)
    
    flac_list = list(content_dir.glob("*.flac"))
    midi_list = list(content_dir.glob("*.mid"))

    for flac in tqdm(flac_list):
        # """
        wave, sr = torchaudio.load(flac)
        wave_mono = wave.mean(dim=0)
        if sr != SAMPLE_RATE:
            trans = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wave_mono = trans(wave_mono)
        trans_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                         n_fft=N_FFT,
                                                         win_length=WIN_LENGTH,
                                                         hop_length=HOP_LENGTH,
                                                         pad_mode=PAD_MODE,
                                                         n_mels=N_MELS,
                                                         norm="slaney")
        mel_spec = trans_mel(wave_mono)
        mel_spec = torch.log(torch.clamp(mel_spec, min=MEL_EPSILON))

        with open(mel_dir / ("%s.pkl" % flac.stem), 'wb') as fout:
            pickle.dump(mel_spec, fout, protocol=4)

        """
        midi_path = content_dir / ("%s.mid" % flac.stem)
        if not midi_path.exists():
            continue
        
        track_note_list = convert_midi(midi_path)

        with open(note_dir / ("%s.pkl" % flac.stem), 'wb') as fout:
            pickle.dump(track_note_list, fout, protocol=4)
        """

