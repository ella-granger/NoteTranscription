import pretty_midi
from pathlib import Path
import pickle
from math import floor
import numpy as np
import torch
import torchaudio
from constants import *
from mido import MidiFile
from tqdm import tqdm
import json


start_time = {}
duration = {}


def convert_midi(midi_path):
    print(midi_path)
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    mid = MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat

    print(len(mid.tracks))
    # valid_list = ["Soprano", "Alto", "Tenor", "Bass"]
    valid_list = ["S", "A", "T", "B"]
    info_track = mid.tracks[0]
    cur_time = 0
    prev_bar = 0
    time_sigs = []
    for msg in info_track:
        cur_time += msg.time
        if msg.type == "time_signature":
            if len(time_sigs) > 0:
                prev_bar += (cur_time - time_sigs[-1][1]) / ticks_per_beat / time_sigs[-1][0]
            bar_length = 4 * msg.numerator / msg.denominator
            if bar_length > 6:
                return None
            if abs(bar_length - int(bar_length)) != 0:
                return None
            time_sigs.append((bar_length, cur_time, prev_bar))
    if len(time_sigs) > 1:
        print(time_sigs)
    print("---------------------------------")
    # Expand time signatures
    measure_sigs = []
    sig = time_sigs[0][0]
    m_start = int(time_sigs[0][1])
    for s in time_sigs[1:]:
        end_time = s[1]
        while m_start < end_time:
            m_end = int(min(end_time, m_start + sig * ticks_per_beat))
            # print(end_time, m_start, sig, ticks_per_beat)
            measure_sigs.append([m_start, m_end, sig])
            m_start = int(m_start + sig * ticks_per_beat)
        m_start = min(m_start, end_time)
        sig = s[0]

    # print("loop end")
    end_time = cur_time
    while m_start < end_time:
        m_end = int(min(end_time, m_start + sig * ticks_per_beat))
        measure_sigs.append([m_start, m_end, sig])
        m_start = int(m_start + sig * ticks_per_beat)
    # print(m_start, m_end, sig)
    # measure_sigs.append([m_start, cur_time, sig])
    print(measure_sigs)
    print(cur_time)
    # _ = input()
    track_note_dict = {}
    for track in mid.tracks[1:]:
        if track.name == "":
            continue
        if track.name[0] not in valid_list:
            continue
        # print(track, track.name)
        cur_time = 0
        last_time = 0
        measure_sig_p = 0
        note_dict = {}
        track_note_list = []
        bar_note_list = []
        for msg in track:
            cur_time = cur_time + msg.time
            # ts = time_sigs[time_sig_p]
            if msg.type == "note_on":
                if msg.velocity > 0:
                    if note_dict.get(msg.note, None) is None:
                        note_dict[msg.note] = cur_time
                else:
                    if note_dict.get(msg.note, None) is not None:
                        start_tick = note_dict[msg.note]
                        note_dict[msg.note] = None
                        end_tick = cur_time
                        
                        # print(start_tick, end_tick)
                        while measure_sig_p >= 0:
                            if measure_sigs[measure_sig_p][0] > start_tick:
                                measure_sig_p -= 1
                            else:
                                break
                        while measure_sig_p < len(measure_sigs):
                            # print(measure_sigs[measure_sig_p])
                            if measure_sigs[measure_sig_p][1] <= start_tick:
                                measure_sig_p += 1
                            else:
                                break
                        if measure_sig_p == len(measure_sigs):
                            measure_sig_p -= 1
                        ts = measure_sigs[measure_sig_p]
                        # print(msg.note, start_tick, end_tick)
                        # print(ts)

                        # print(ts)
                        # print(start_tick)
                        cur_bar = measure_sig_p
                        if measure_sig_p == len(measure_sigs) - 1:
                            cur_bar += floor((start_tick - ts[0]) / ticks_per_beat / ts[2])
                        # print(cur_bar)
                        # print(len(track_note_list))
                        # print(cur_bar)
                        while len(track_note_list) < cur_bar:
                            track_note_list.append(bar_note_list)
                            bar_note_list = []
                        # print(track_note_list)

                        # start_q = round(start_tick / ticks_per_beat * 4) / 4
                        # end_q = round(end_tick / ticks_per_beat * 4) / 4
                        # dur = end_q - start_q
                        # if dur == 0:
                        #     dur = 0.25
                        if measure_sig_p == 0:
                            start_beat = ts[2] - (ts[1] - start_tick) / ticks_per_beat
                            end_beat = ts[2] - (ts[1] - end_tick) / ticks_per_beat
                        else:
                            start_beat = ((start_tick - ts[0]) / ticks_per_beat)
                            end_beat = ((end_tick - ts[0]) / ticks_per_beat)
                        # print(start_beat, end_beat)
                        start_beat_q = round(start_beat * 4) / 4
                        dur = round(end_beat * 4) / 4 - start_beat_q
                        if dur == 0:
                            dur = 0.25
                        bar_note_list.append([msg.note, start_beat_q, dur])
                        # if start_beat_q < 0:
                        #     print(bar_note_list[-2:])
                        #     print(ts)
                        #     print(start_tick, end_tick)
                        #     exit()
                        # print(bar_note_list[-1])
                        # _ = input()

                        if start_beat_q not in start_time:
                            start_time[start_beat_q] = 0
                        if dur not in duration:
                            duration[dur] = 0
                        start_time[start_beat_q] += 1
                        duration[dur] += 1

                        # if abs(start_beat - 0.375) < 0.0000001:
                        #     print(msg.note, start_beat, dur, cur_bar)
                        #     _ = input()
        # print("fin")
        # _ = input()
        track_note_list.append(bar_note_list)
        track_note_dict[track.name] = track_note_list
        # print(track_note_list)
        # _ = input()


    def idt(n1, n2):
        if n2 == -1:
            return False
        if n1.pitch != n2.pitch:
            return False
        if n1.start < n2.start:
            return False
        if n1.end > n2.end:
            return False
        return True

    for instrument in midi_data.instruments:
        if instrument.name == "":
            continue
        if instrument.name[0] not in valid_list:
            continue
        track_note_list = track_note_dict[instrument.name]
        pm_notes = instrument.notes

        i = 0
        prev_n = -1
        for bar in track_note_list:
            for note in bar:
                while idt(pm_notes[i], prev_n):
                    i += 1
                n = pm_notes[i]
                # print(i, n, note, prev_n)
                assert note[0] == n.pitch
                note.append(n.start)
                note.append(n.end)
                i += 1
                prev_n = n

    final_bar_count = max([len(x) for _, x in track_note_dict.items()])
    final_measure_list = [dict() for _ in range(final_bar_count)]
    for t, bars in track_note_dict.items():
        for i, b in enumerate(bars):
            final_measure_list[i][t] = b

    print(len(final_measure_list))
    print(len(measure_sigs))
    assert len(final_measure_list) <= len(measure_sigs)
    for i, sig in enumerate(measure_sigs):
        if i == len(final_measure_list):
            break
        measure = final_measure_list[i]
        measure["measure"] = sig[2]
    print("i:", i)
        
            
    return final_measure_list

    # onset_list = torch.FloatTensor([x.start for x in note_list])
    # pitch_list = torch.ShortTensor([x.pitch for x in note_list])
    # offset_list = torch.FloatTensor([x.end for x in note_list])


if __name__ == "__main__":
    # content_dir = Path("/storageSSD/huiran/NoteTranscription/BachChorale")
    # target_dir = Path("/storageSSD/huiran/NoteTranscription/BachChorale")
    # content_dir = Path("/media/ella/Yu/UR/datasets/BachChorale")
    # content_dir = Path("/media/ella/Yu/UR/datasets/BachChorale/audio")
    # target_dir = Path("./test")
    # content_dir = Path("/media/ella/Yu/UR/datasets/WebChoralDataset/program_change_midi")
    content_dir = Path("/storageSSD/huiran/NoteTranscription/WebChorale")
    target_dir = Path("/storageSSD/huiran/NoteTranscription/WebChorale")

    # content_dir = Path("/media/ella/Yu/UR/datasets/WebChoralDataset")
    # target_dir = Path("./test/WebChorale")
    # target_dir = Path("./test/BachChorale")
    
    mel_dir = target_dir / "mel"
    note_dir = target_dir / "note"
    mel_dir.mkdir(parents=True, exist_ok=True)
    note_dir.mkdir(parents=True, exist_ok=True)
    
    # flac_list = list(content_dir.glob("*.WAV"))
    # midi_list = list((content_dir / "aligned_midi").glob("*/*/*.mid"))
    # with open(content_dir / "split" / "aligned_midi" / "test.json") as fin:
    #     test_list = json.load(fin)
    # midi_list = [x for x in midi_list if x.stem in test_list]

    # midi_list = list((content_dir / "midi_align").glob("*.mid")) # BachChorale
    midi_list = list((content_dir / "aligned_midi").glob("*.mid"))
    # midi_list = list((content_dir / "BachChorale").glob("*.mid"))

    valid = 0
    # for flac in tqdm(flac_list):
    for j, midi_path in tqdm(enumerate(midi_list)):
        if j < 0:
            continue
        print("j:", j)
        # midi_path = content_dir / ("%s.mid" % flac.stem)
        if not midi_path.exists():
            continue
        
        track_note_list = convert_midi(midi_path)
        # print(track_note_list)
        if track_note_list is None:
            continue

        valid += 1

        with open(note_dir / ("%s.pkl" % midi_path.stem), 'wb') as fout:
            pickle.dump(track_note_list, fout, protocol=4)

            
        """
        # flac = content_dir / "aligned_midi" / ("%s.flac" % midi_path.stem)
        # flac = content_dir / "audio_clean" / midi_path.parts[-3] / midi_path.parts[-2] / ("%s.mp4" % midi_path.stem)
        # flac = content_dir / "BachChorale" / ("%s.flac" % midi_path.stem)
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

        # break
        # _ = input()
        
    # print(start_time)
    start_time_list = [(k ,v) for k, v in start_time.items()]
    start_time_list.sort()
    print(start_time_list)
    print(sum([x[1] for x in start_time_list]))
    duration_list = [(k, v) for k, v in duration.items()]
    duration_list.sort()
    print(duration_list)
    print(sum([x[1] for x in duration_list]))
    print(valid)
