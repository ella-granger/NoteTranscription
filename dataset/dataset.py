import os
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import json
from copy import deepcopy
from dataset.constants import *
# from constants import *


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split, train_mode,
                 seg_len=320, shuffle=True, device="cpu", hop_len=HOP_LENGTH):
        super().__init__()

        self.voice_list = ["S", "A", "T", "B"]
        self.train_mode = train_mode
        self.seg_len = seg_len
        self.shuffle = shuffle
        self.device = device
        self.hop_len = hop_len

        mel_dir = data_path / "mel_320"
        note_dir = data_path / "note"
        whisper_dir = data_path / "whisper"
        id_json = data_path / ("%s.json" % split)

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

        self.mel_list = []
        self.note_list = []
        self.whisper_list = []
        self.dataset_len = 0

        self.fid_list = []
        for f in mel_files:
            if f.stem not in id_list:
                continue
            self.fid_list.append(f.stem)
            with open(f, 'rb') as fin:
                mel = pickle.load(fin)
            with open(whisper_dir / f.name, "rb") as fin:
                whisper = pickle.load(fin)
            with open(note_dir / ("%s.pkl" % f.stem), "rb") as fin:
                note = pickle.load(fin)
            self.mel_list.append(mel)
            self.whisper_list.append(whisper)
            self.note_list.append(note)

            data_length = self.get_length(mel, note)
            self.dataset_len += data_length

        self.dataset_len = self.dataset_len // self.seg_len


    def __len__(self):
        return self.dataset_len


    def __getitem__(self, index):
        index = np.random.randint(len(self.mel_list))

        # fid = "rv8B6tMNFJI"
        # index = self.fid_list.index(fid)

        mel = self.mel_list[index]
        whisper = self.whisper_list[index]
        notes = self.note_list[index]
        fid = self.fid_list[index]

        # print(mel.size())

        total_length = self.get_length(mel, notes)

        if total_length < self.seg_len:
            # pad to seg_len
            begin_idx = 0
            end_idx = self.seg_len
            begin_time = 0
            end_time = end_idx * self.hop_len / SAMPLE_RATE
            mel_length = mel.size(1)
            if mel_length < self.seg_len:
                mel = F.pad(mel, (0, self.seg_len - mel_length), value=0)
            else:
                mel = mel[:, :end_idx]
            whisper_length = whisper.size(1)
            if whisper_length < self.seg_len:
                whisper = F.pad(whisper, (0, self.seg_len - whisper_length), value=0)
            else:
                whisper = whisper[:, :end_idx]
        else:
            begin_idx = np.random.randint(total_length - self.seg_len + 1)
            end_idx = begin_idx + self.seg_len
            begin_time = begin_idx * self.hop_len / SAMPLE_RATE
            end_time = end_idx * self.hop_len / SAMPLE_RATE

            # begin_time = 159.344
            # end_time = 164.464
            # begin_idx = int(begin_time * SAMPLE_RATE / self.hop_len)
            # end_idx = int(end_time * SAMPLE_RATE / self.hop_len)

            mel = mel[:, begin_idx:end_idx]
            whisper = whisper[:, begin_idx:end_idx]
        
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
                        new_note = deepcopy(note)
                        new_note[3] = max(note[3], begin_time)
                        new_note[4] = min(note[4], end_time)
                        note_list.append(tuple(new_note))
                        if note in last_list:
                            full = True

            # print(note_list)
            note_list = list(set(note_list))
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
                if "S" in self.train_mode:
                    token.append(0)
                    start.append(sig)
                    dur.append(0.0)
                    if "T" in self.train_mode:
                        start_t.append(max_time)
                        end.append(max_time)
            # print(token)
            # print(start)
            # print(dur)
            
        token.insert(0, MAX_MIDI+1)
        token.append(MAX_MIDI+2)
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
            if len(end) == 1:
                end.append(end_time)
            else:
                end.append(max(end))
            start_t = torch.FloatTensor(start_t)
            end = torch.FloatTensor(end)

            start_t = (start_t - begin_time) / (end_time - begin_time)
            end = (end - begin_time) / (end_time - begin_time)

            start_t = torch.clip(start_t, 0.0, 1.0)
            end = torch.clip(end, 0.0, 1.0)

            # increment
            end = end - start_t
            # start_t[1:] = start_t[1:] - start_t[:-1]

        if self.train_mode == "S":
            data_point = dict(mel=mel,
                              whisper=whisper,
                              pitch=token,
                              start=start,
                              dur=dur,
                              begin_time=begin_time,
                              end_time=end_time,
                              fid=fid)
        elif self.train_mode == "T":
            data_point = dict(mel=mel,
                              whisper=whisper,
                              pitch=token,
                              start_t=start_t,
                              end=end,
                              begin_time=begin_time,
                              end_time=end_time,
                              fid=fid)
        else:
            data_point = dict(mel=mel,
                              whisper=whisper,
                              pitch=token,
                              start=start,
                              dur=dur,
                              start_t=start_t,
                              end=end,
                              begin_time=begin_time,
                              end_time=end_time,
                              fid=fid)
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

        for key in batch[0]:
            v = [x[key] for x in batch]
            if key in pad_dict:
                v = pad_and_stack(v, pad_dict[key])
            result[key] = v
        result["mel"] = torch.stack(result["mel"])
        result["whisper"] = torch.stack(result["whisper"])

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
        note_length = int(note_length * SAMPLE_RATE / self.hop_len)
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

    
