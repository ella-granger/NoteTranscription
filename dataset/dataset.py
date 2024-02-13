import os
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import json
from dataset.constants import *
# from constants import *


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, note_dir, id_json, train_mode,
                 shuffle=True, device="cpu"):
        super().__init__()

        self.voice_list = ["S", "A", "T", "B"]
        self.train_mode = train_mode
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
                note_count = sum([len(v) for k, v in bar.items() if k[0] in self.voice_list])
                time_min = 1e6
                time_max = -1
                # print(note_count, bar)
                if note_count > 0 or start_flag:
                    if note_count == 0:
                        cur_bar_list.append(bar)
                        continue
                    for k, v in bar.items():
                        if k[0] not in self.voice_list:
                            continue
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
            token = []
            if "S" in self.train_mode:
                start = []
                dur = []
            if "T" in self.train_mode:
                start_t = []
                end = []

            for bar in cur_bar_list:
                full = False
                sig = bar["measure"]
                max_time = 0
                note_list = []
                for part in self.voice_list:
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
                            note_list.append(note)
                            if note in last_list:
                                full = True

                # print(note_list)
                note_list = sorted(note_list, key=lambda x:(x[3], -x[0]))
                for note in note_list:
                    token.append(note[0])
                    if "S" in self.train_mode:
                        start.append(note[1])
                        dur.append(note[2])
                    if "T" in self.train_mode:
                        start_t.append(note[3])
                        end.append(note[4])
                        if note[4] > max_time:
                            max_time = note[4]
                if full:
                    token.append(0)
                    if "S" in self.train_mode:
                        start.append(sig)
                        dur.append(0.0)
                    if "T" in self.train_mode:
                        start_t.append(max_time)
                        end.append(max_time)
            # print(token)
            # print(start)
            # print(dur)
            
            mel = mel[:, begin_idx:end_idx]
            token.insert(0, MAX_MIDI+1)
            token.append(MAX_MIDI+1)
            token = torch.LongTensor(token)
            token[token>0] = token[token>0] - MIN_MIDI + 1

            if "S" in self.train_mode:
                start.insert(0, 0.0)
                dur.insert(0, 0.0)
                start.append(0.0)
                dur.append(0.0)

                start = np.array(start) / 0.25
                dur = np.array(dur) / 0.25

                start = start.astype(int)
                dur = dur.astype(int)
                
                start = torch.LongTensor(start)
                dur = torch.LongTensor(dur)
                dur = torch.clip(dur, 0, MAX_DUR)
                
            if "T" in self.train_mode:
                start_t.insert(0, begin_time)
                if len(end) == 0:
                    start_t.append(end_time)
                else:
                    start_t.append(max(end))
                end.insert(0, begin_time)
                if len(end) == 0:
                    end.append(end_time)
                else:
                    end.append(max(end))
                start_t = torch.FloatTensor(start_t)
                end = torch.FloatTensor(end)

                start_t = (start_t - begin_time) / (end_time - begin_time)
                end = (end - begin_time) / (end_time - begin_time)

                start_t = torch.clip(start_t, 0.0, 1.0)
                end = torch.clip(end, 0.0, 1.0)

        if self.train_mode == "S":
            data_point = dict(mel=mel,
                              pitch=token,
                              start=start,
                              dur=dur)
        elif self.train_mode == "T":
            data_point = dict(mel=mel,
                              pitch=token,
                              start_t=start_t,
                              end=end)
        else:
            data_point = dict(mel=mel,
                              pitch=token,
                              start=start,
                              dur=dur,
                              start_t=start_t,
                              end=end)
        return data_point


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
        result["mel"] = torch.stack([x["mel"] for x in batch])

        for key in pad_dict:
            if key in batch[0]:
                v = [x[key] for x in batch]
                v = pad_and_stack(v, pad_dict[key])
                result[key] = v

        # print(result)
        # _ = input()
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

    
