import os
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import json
from copy import deepcopy
from dataset.constants import *
from tqdm import tqdm
# from constants import *


def in_seg(note, start, end):
    return note[2] > start and note[1] < end


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, note_dir, id_json, train_mode,
                 seg_len=320, shuffle=True, device="cpu"):
        super().__init__()

        self.voice_list = ["S", "A", "T", "B"]
        self.train_mode = train_mode
        self.seg_len = seg_len
        self.shuffle = shuffle
        self.device = device

        mel_dir = Path(mel_dir)
        note_dir = Path(note_dir)
        with open(id_json) as fin:
            id_list = json.load(fin)
        # id_list = ["oudkfwwrZq0"]
        # id_list = ["qjtMJxtoooI"]
        # id_list = ["BC059"]
        # id_list = ["sRmOnHiXU0o"]
        # id_list = ["rcNyTVnpVe4"]
        
        mel_files = list(mel_dir.glob("*.pkl"))

        self.slices = []
        
        for f in tqdm(mel_files):
            if f.stem not in id_list:
                continue

            fid = f.stem
            with open(f, 'rb') as fin:
                mel = pickle.load(fin)
            with open(note_dir / ("%s.pkl" % f.stem), "rb") as fin:
                notes = pickle.load(fin)

            slices = self.build_slices(fid, mel, notes)
            self.slices += slices

        self.start_shift = 0

        
    def __len__(self):
        return len(self.slices)


    def __getitem__(self, index):
        # TODO: "S" mode support
        test = False

        segment = self.slices[index]

        begin_idx = self.start_shift
        end_idx = begin_idx + self.seg_len
        begin_time = begin_idx * HOP_LENGTH / SAMPLE_RATE
        end_time = end_idx * HOP_LENGTH / SAMPLE_RATE
        dur = end_time - begin_time

        notes = segment["notes"]
        notes = [x for x in notes if in_seg(x, begin_time, end_time)]

        notes = [[x[0], (max(x[1], begin_time) - begin_time) / dur, (min(x[2], end_time) - begin_time) / dur] for x in notes]
        notes = sorted(notes, key=lambda x:(x[1], -x[0]))
        notes.insert(0, [MAX_MIDI+1, 0.0, 0.0])
        notes.append([MAX_MIDI+2, 1.0, 1.0])

        pitch = torch.LongTensor([x[0] for x in notes])
        pitch[pitch>0] = pitch[pitch>0] - MIN_MIDI + 1
        start_t = torch.FloatTensor([x[1] for x in notes])
        end = torch.FloatTensor([x[2] for x in notes])

        end = end - start_t

        mel = segment["mel"][:, begin_idx:end_idx]

        return dict(mel=mel,
                    pitch=pitch,
                    start_t=start_t,
                    end=end,
                    begin_time=segment["begin_time"],
                    end_time=segment["end_time"],
                    fid=segment["fid"])


    def collate_fn(self, batch):

        pad_dict = {"pitch": PAD_IDX,
                    "start": MAX_START+1,
                    "dur": 0,
                    "start_t": 0.0,
                    "end": 0.0}

        def pad_and_stack(x_list, pad):
            max_len = max([len(x) for x in x_list])
            x_list = [F.pad(x, (0, max_len - len(x)), value=pad) for x in x_list]
            return torch.stack(x_list)

        result = {}

        for key in batch[0]:
            v = [x[key] for x in batch]
            if key in pad_dict:
                v = pad_and_stack(v, pad_dict[key])
            result[key] = v
        result["mel"] = torch.stack(result["mel"])

        return result
            

    def get_length(self, mel, note):
        mel_length = mel.size(1)
        note_length = 0
        for bar in note[::-1]:
            note_count = sum([len(v) for k, v in bar.items() if k[0] in self.voice_list])
            # print(bar)
            # print(note_count)
            if note_count > 0:
                for k, v in bar.items():
                    if k[0] not in self.voice_list:
                        continue
                    for note in v:
                        if note[-1] > note_length:
                            note_length = note[-1]
                break
        # print(note_length)
        note_length = int(note_length * SAMPLE_RATE / HOP_LENGTH)
        # print(mel_length)
        # print(note_length)
        # _ = input()

        return min(mel_length, note_length)


    def update_shift(self):
        self.start_shift = np.random.randint(self.seg_len)


    def build_slices(self, fid, mel, notes):
        debug = False
        # TODO: "S" mode support
        data_length = self.get_length(mel, notes)

        flat_notes = []
        for measure in notes:
            for k, v in measure.items():
                if k[0] not in self.voice_list:
                    continue
                m_notes = [[x[0], x[3], x[4]] for x in v]
                flat_notes += m_notes

        flat_notes = sorted(flat_notes, key=lambda x: (x[1], -x[0]))

        result = []
        candidates = []
        note_p = 0
        for i in range(0, data_length, self.seg_len):
            begin_idx = i - self.seg_len // 2
            end_idx = i + self.seg_len * 3 // 2
            begin_time = begin_idx * HOP_LENGTH / SAMPLE_RATE
            end_time = end_idx * HOP_LENGTH / SAMPLE_RATE

            if debug:
                print(begin_idx, end_idx, begin_time, end_time)

            c_p = 0
            while len(candidates) > 0:
                if not in_seg(candidates[c_p], begin_time, end_time):
                    candidates.pop(c_p)
                else:
                    c_p += 1
                if c_p >= len(candidates):
                    break

            while note_p < len(flat_notes):
                cur_note = flat_notes[note_p]
                if debug:
                    print(cur_note)
                if in_seg(cur_note, begin_time, end_time):
                    candidates.append(cur_note)
                elif cur_note[1] >= end_time:
                    break
                note_p += 1

            if debug:
                print(candidates)

            seg_notes = [[x[0], max(x[1], begin_time) - begin_time, min(x[2], end_time) - begin_time] for x in candidates]

            seg_mel = mel[:, max(0, begin_idx):min(data_length, end_idx)]
            if begin_idx < 0:
                seg_mel = F.pad(seg_mel, (-begin_idx, 0), "constant", 0)
            if end_idx > data_length:
                seg_mel = F.pad(seg_mel, (0, end_idx-data_length), "constant", 0)

            assert seg_mel.size(1) == self.seg_len * 2

            seg = dict(mel=seg_mel,
                       notes=seg_notes,
                       begin_time=begin_time,
                       end_time=end_time,
                       fid=fid)
            result.append(seg)

            if debug:
                _ = input()

        return result            


        
if __name__ == "__main__":
    mel_dir = Path("./test/BachChorale/mel")
    note_dir = Path("./test/BachChorale/note")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MelDataset(mel_dir, note_dir, "./test/BachChorale/test.json", "S", device=device)

    from torch.utils.data import DataLoader
    batch_size = 1
    test_loader = DataLoader(dataset,
                             batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             collate_fn=dataset.collate_fn)

    print(len(dataset))
    for x in test_loader:
        print(x)
        for k, v in x.items():
            print(k, v.size())
            if k == "pitch":
                print(v)
        _ = input()

    
