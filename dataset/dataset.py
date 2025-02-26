import os
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import json
from copy import deepcopy
from dataset.constants import *
from math import ceil
# from constants import *


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, note_dir, id_json,
                 seg_len=320, shuffle=True, device="cpu"):
        super().__init__()

        self.voice_list = ["S", "A", "T", "B"]
        self.seg_len = seg_len
        self.shuffle = shuffle
        self.device = device

        mel_dir = Path(mel_dir)
        note_dir = Path(note_dir)
        with open(id_json) as fin:
            id_list = json.load(fin)
        
        mel_files = list(mel_dir.glob("*.pkl"))

        self.mel_list = []
        self.note_list = []
        self.dataset_len = 0

        self.fid_list = []
        for f in mel_files:
            if f.stem not in id_list:
                continue
            self.fid_list.append(f.stem)
            with open(f, 'rb') as fin:
                mel = pickle.load(fin)
            self.mel_list.append(mel)
            
            with open(note_dir / ("%s.pkl" % f.stem), "rb") as fin:
                note = pickle.load(fin)
            note = self.convert_notelist(note)
            self.note_list.append(note)

            data_length = self.get_length(mel, note)
            self.dataset_len += data_length

        self.dataset_len = self.dataset_len // self.seg_len


    def __len__(self):
        return self.dataset_len


    def convert_notelist(self, note):
        notes = []

        for bar in note:
            for part, note_list in bar.items():
                if part[0] in self.voice_list:
                    for n in note_list:
                        new_n = deepcopy(n)
                        new_n.append(self.voice_list.index(part[0]))
                        notes.append(tuple(new_n))
        
        begin = [] # begin time index. The first note which ENDS AFTER the index second
        end = [0] * ceil(notes[-1][3]) # end time index. The last note which BEGINS BEFORE the index+1 second

        notes = sorted(notes, key=lambda x:(x[3], -x[0]))
        for i, n in enumerate(notes):
            b = n[3]
            e = n[4]

            while len(begin) < ceil(e):
                begin.append(i)

            for tmp in range(int(b), len(end)):
                end[tmp] = i

        while len(end) != len(begin):
            end.append(end[-1])

        return {"notes": notes, "begin": tuple(begin), "end": tuple(end)}
        

    def __getitem__(self, index):
        index = np.random.randint(len(self.mel_list))

        mel = self.mel_list[index]
        note = self.note_list[index]

        total_length = self.get_length(mel, note)

        if total_length < self.seg_len:
            # pad to seg_len
            pass
        else:
            begin_idx = np.random.randint(total_length - self.seg_len + 1)
            end_idx = begin_idx + self.seg_len
            mel = mel[:, begin_idx:end_idx]

            begin_time = begin_idx * HOP_LENGTH / SAMPLE_RATE
            end_time = end_idx * HOP_LENGTH / SAMPLE_RATE

            begin_tmp = note["begin"][int(begin_time)]
            end_tmp = note["end"][int(end_time)]

            notes_tmp = note["notes"][begin_tmp:end_tmp+1]

            note_list = []
            for n in notes_tmp:
                if n[3] < end_time and n[4] >= begin_time:
                    new_n = (n[0], max(n[3], begin_time), min(n[4], end_time), n[5])
                    note_list.append(new_n)

            note_list = sorted(note_list, key=lambda x:(x[1], -x[0]))

            tokens = [x[0] for x in note_list]
            start_t = [x[1] for x in note_list]
            end = [x[2] for x in note_list]

            token.insert(0, MAX_MIDI + 1)
            token.append(MAX_MIDI + 2)
            token = torch.LongTensor(token)
            token = token - MIN_MIDI

            start_t.insert(0, begin_time)
            end.insert(0, begin_time)
            if len(note_list) == 0:
                start_t.append(begin_time)
                end.append(begin_time)
            else:
                start_t.append(max(end))
                end.append(max(end))

            start_t = torch.FloatTensor(start_t)
            end = torch.FloatTensor(end)

            start_t = (start_t - begin_time) / (end_time - begin_time)
            end = (end - begin_time) / (end_time - begin_time)

            end = end - start_t

        return dict(mel=mel,
                    pitch=token,
                    start_t=start_t,
                    end=end,
                    begin_time=begin_time,
                    end_time=end_time,
                    fid=fid)

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

        last_sec = note["begin"][-1]
        note_length = max([x[4] for x in note["notes"][last_sec:]])
        note_length = int(note_length * SAMPLE_RATE / HOP_LENGTH)
        # print(mel_length)
        # print(note_length)
        # _ = input()

        return min(mel_length, note_length)


        
if __name__ == "__main__":
    mel_dir = Path("./test/BachChorale/mel")
    note_dir = Path("./test/BachChorale/note")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MelDataset(mel_dir, note_dir, "./test/BachChorale/test.json", device=device)

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

    
