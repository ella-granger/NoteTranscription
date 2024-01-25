import os
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import json
from dataset.constants import *


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, note_dir, id_json,
                 shuffle=True, score_info=False, device="cpu"):
        super().__init__()

        self.seg_len = SEG_LEN
        self.shuffle = shuffle
        self.score_info = score_info
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

        total_length = self.get_length(mel, notes)

        if total_length < self.seg_len:
            # pad to seg_len
            pass
        else:
            begin_idx = np.random.randint(total_length - self.seg_len + 1)
            end_idx = begin_idx + self.seg_len
            begin_time = begin_idx * HOP_LENGTH / SAMPLE_RATE
            end_time = end_idx * HOP_LENGTH / SAMPLE_RATE

            cur_note_list = []
            for _, track_note_list in notes.items():
                for n in track_note_list:
                    if n[3] < end_time and n[4] >= begin_time:
                        cur_note_list.append(n)
            cur_note_list = sorted(cur_note_list, key=lambda x: (x[3], -x[0]))
            pitch = [x[0] for x in cur_note_list]
            start = [x[3] for x in cur_note_list]
            end = [x[4] for x in cur_note_list]
            pitch.insert(MAX_MIDI+1)
            start.insert(begin_time)
            end.insert(begin_time)
            pitch.append(MAX_MIDI+2)
            start.append(max(end))
            end.append(max(end))
            
            pitch = torch.LongTensor(pitch)
            start = torch.FloatTensor(start)
            end = torch.FloatTensor(end)

            pitch[pitch>0] = pitch[pitch>0] - MIN_MIDI + 1
            start = (start - begin_time) / (end_time - begin_time)
            end = (end - begin_time) / (end_time - begin_time)

            start = torch.clip(start, 0.0, 1.0)
            end = torch.clip(end, 0.0, 1.0)

            mel = mel[:, begin_idx:end_idx]
        
        if self.score_info:
            data_point = dict(mel=mel,
                              pitch=pitch,
                              beat_start=beat_start,
                              dur=dur,
                              start=start,
                              end=end)
        else:
            data_point = dict(mel=mel,
                              pitch=pitch,
                              start=start,
                              end=end)
        return data_point


    def collate_fn(self, batch):

        def pad_and_stack(x_list, pad):
            max_len = max([len(x) for x in x_list])
            x_list = [F.pad(x, (0, max_len - len(x)), value=pad) for x in x_list]
            return torch.stack(x_list)
        
        mel = [x["mel"] for x in batch]
        pitch = [x["pitch"] for x in batch]
        start = [x["start"] for x in batch]
        end = [x["end"] for x in batch]

        mel = torch.stack(mel)

        pitch = pad_and_stack(pitch, PAD_IDX)
        start = pad_and_stack(start, 0.0)
        end = pad_and_stack(end, 0.0)

        return dict(mel=mel,
                    pitch=pitch,
                    start=start,
                    end=end)
            

    def get_length(self, mel, note):
        mel_length = mel.size(1)
        note_length = 0
        for name, note_list in note.items():
            track_length = note_list[-1][-1]
            if track_length > note_length:
                note_length = track_length
        note_length = int(note_length * SAMPLE_RATE / HOP_LENGTH)

        return min(mel_length, note_length)


        
if __name__ == "__main__":
    mel_dir = Path("./test/BachChorale/mel")
    note_dir = Path("./test/BachChorale/note")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MelDataset(mel_dir, note_dir, "./test/BachChorale/test.json", device=device)

    from torch.utils.data import DataLoader
    batch_size = 2
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
        _ = input()

    
