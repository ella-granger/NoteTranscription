import pretty_midi
from pathlib import Path
import pickle
import numpy as np
import torch
import torchaudio
from constants import *


if __name__ == "__main__":
    content_dir = Path("/storageSSD/huiran/BachChorale/BachChorale")
    target_dir = Path("/storageSSD/huiran/NoteTranscription")
    mel_dir = target_dir / "BachChorale" / "mel"
    note_dir = target_dir / "BachChorale" / "note"
    mel_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)
    
    flac_list = list(content_dir.glob("*.flac"))

    for flac in flac_list:
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

        midi_path = content_dir / ("%s.mid" % flac.stem)
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))

        note_list = []
        for instrument in midi_data.instruments:
            note_list = note_list + instrument.notes

        note_list = sorted(note_list, key=lambda x:(x.start, x.pitch, x.end))

        onset_list = torch.FloatTensor([x.start for x in note_list])
        pitch_list = torch.ShortTensor([x.pitch for x in note_list])
        offset_list = torch.FloatTensor([x.end for x in note_list])

        with open(note_dir / ("%s_onset.pkl" % flac.stem), 'wb') as fout:
            pickle.dump(onset_list, fout, protocol=4)

        with open(note_dir / ("%s_pitch.pkl" % flac.stem), 'wb') as fout:
            pickle.dump(pitch_list, fout, protocol=4)

        with open(note_dir / ("%s_offset.pkl" % flac.stem), 'wb') as fout:
            pickle.dump(offset_list, fout, protocol=4)
