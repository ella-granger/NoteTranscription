import os
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import json
# from dataset.constants import *
from constants import *


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, note_dir, id_json,
                 shuffle=True, device="cpu"):
        super().__init__()

        self.seg_len = SEG_LEN
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

        for f in mel_files:
            if f.stem not in id_list:
                continue
            with open(f, 'rb') as fin:
                mel = pickle.load(fin)
            with open(note_dir / ("%s.pkl" % f.stem), "rb") as fin:
                note = pickle.load(fin)
            self.mel_list.append(mel)
            self.note_list.append(note)

            data_length = self.get_length(mel, note)
            self.dataset_len += data_length

        self.dataset_len = self.dataset_len // self.seg_len


    def __len__(self):
        return self.dataset_len


    def __getitem__(self, index):
        index = np.random.randint(len(self.mel_list))

        mel = self.mel_list[index]
        notes = self.note_list[index]

        # print(mel.size())

        total_length = self.get_length(mel, notes)

        if total_length < self.seg_len:
            # pad to seg_len
            pass
        else:
            begin_idx = np.random.randint(total_length - self.seg_len + 1)
            end_idx = begin_idx + self.seg_len
            begin_time = begin_idx * HOP_LENGTH / SAMPLE_RATE
            end_time = end_idx * HOP_LENGTH / SAMPLE_RATE

            cur_bar_list = []
            start_flag = False
            for bar in notes:
                note_count = sum([len(v) for k, v in bar.items()])
                time_min = 1e6
                time_max = -1
                # print(note_count, bar)
                if note_count > 0 or start_flag:
                    if note_count == 0:
                        cur_bar_list.append(bar)
                        continue
                    for k, v in bar.items():
                        for note in v:
                            if note[-1] > time_max:
                                time_max = note[-1]
                            if note[-2] < time_min:
                                time_min = note[-2]
                    if time_min < end_time and time_max > begin_time:
                        cur_bar_list.append(bar)
                        start_flag = True
                    if time_min > end_time:
                        break
            # print(begin_time, end_time)
            # print(cur_bar_list)
            voice_list = ["S", "A", "T", "B"]
            token = []
            start = []
            dur = []
            for bar in cur_bar_list:
                full = True
                for part in voice_list:
                    part_list = []
                    last_list = []
                    for k, v in bar.items():
                        if part in k:
                            part_list += v
                            if len(v) > 0:
                                last_list.append(v[-1])
                    part_list = sorted(part_list, key=lambda x: (x[3], -x[0]))
                    # print("--------------------------------")
                    # print(part)
                    # print(part_list)
                    # print(last_list)

                    last_count = 0
                    for idx, note in enumerate(part_list):
                        if note[3] < end_time and note[4] >= begin_time:
                            token.append(note[0])
                            start.append(note[1])
                            dur.append(note[2])
                            if note in last_list:
                                last_count += 1
                    if last_count != len(last_list):
                        full = False
                if full:
                    token.append(0)
                    start.append(0.0)
                    dur.append(0.0)
            # print(token)
            # print(start)
            # print(dur)
            
            token.insert(0, MAX_MIDI+1)
            start.insert(0, 0.0)
            dur.insert(0, 0.0)
            token.append(MAX_MIDI+1)
            start.append(0.0)
            dur.append(0.0)

            start = np.array(start) / 0.25
            dur = np.array(dur) / 0.25

            start = start.astype(int)
            dur = dur.astype(int)
            
            token = torch.LongTensor(token)
            start = torch.LongTensor(start)
            dur = torch.LongTensor(dur)

            token[token>0] = token[token>0] - MIN_MIDI + 1
            dur = torch.clip(dur, 0, MAX_DUR)

            mel = mel[:, begin_idx:end_idx]
        
        
        data_point = dict(mel=mel,
                          pitch=token,
                          start=start,
                          dur=dur)
        return data_point


    def collate_fn(self, batch):

        def pad_and_stack(x_list, pad):
            max_len = max([len(x) for x in x_list])
            x_list = [F.pad(x, (0, max_len - len(x)), value=pad) for x in x_list]
            return torch.stack(x_list)
        
        mel = [x["mel"] for x in batch]
        pitch = [x["pitch"] for x in batch]
        start = [x["start"] for x in batch]
        dur = [x["dur"] for x in batch]

        mel = torch.stack(mel)

        pitch = pad_and_stack(pitch, PAD_IDX)
        start = pad_and_stack(start, MAX_START+1)
        dur = pad_and_stack(dur, 0)

        return dict(mel=mel,
                    pitch=pitch,
                    start=start,
                    dur=dur)
            

    def get_length(self, mel, note):
        mel_length = mel.size(1)
        note_length = 0
        for bar in note[::-1]:
            note_count = sum([len(v) for k, v in bar.items()])
            # print(bar)
            # print(note_count)
            if note_count > 0:
                for k, v in bar.items():
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


        
if __name__ == "__main__":
    mel_dir = Path("./test/WebChorale/mel")
    note_dir = Path("./test/WebChorale/note")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MelDataset(mel_dir, note_dir, "./test/WebChorale/test.json", device=device)

    from torch.utils.data import DataLoader
    batch_size = 1
    test_loader = DataLoader(dataset,
                             batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=1,
                             collate_fn=dataset.collate_fn)

    for x in test_loader:
        print(x)
        for k, v in x.items():
            print(k, v.size())
            if k == "pitch":
                print(v)
        _ = input()

    
