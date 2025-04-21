import torch
from dataset import MelDataset

from pathlib import Path
import pickle
from tqdm import tqdm

from constants import *


if __name__ == "__main__":
    data_path = Path("YouChorale")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    dataset = MelDataset(data_path / "mel_focus",
                         data_path / "note",
                         data_path / "full.json",
                         device=device)

    print(len(dataset.fid_list))
    print(len(dataset.note_list))
    _ = input()

    for f, note in tqdm(zip(dataset.fid_list, dataset.note_list)):
        last_sec = note["begin"][-1]
        note_length = max([x[4] for x in note["notes"][last_sec:]])
        note_length = round(note_length * SAMPLE_RATE / HOP_LENGTH)

        notes = note["notes"]

        bm = torch.zeros((MRK_IDX, note_length, 2), dtype=int)
        for n in notes:
            # print(n)
            n_p = n[0] - MIN_MIDI
            n_s = round(n[3] * SAMPLE_RATE / HOP_LENGTH)
            n_e = round(n[4] * SAMPLE_RATE / HOP_LENGTH)
            n_v = n[5]
            # print(n_p, n_s, n_e, n_v)
            for i, v in enumerate(n_v):
                n_id = pow(2, i)
                # print(n_s, n_e)
                bm[n_p, n_s:n_e, 0] = torch.bitwise_or(bm[n_p, n_s:n_e, 0],
                                                       n_id * torch.ones(n_e-n_s, dtype=int))
                bm[n_p, n_s, 1] = 1
                if n_e < note_length:
                    bm[n_p, n_e, 1] = 1
                else:
                    bm[n_p, n_e-1, 1] = 1
        # print(bm[20:40, 0:100])
        # _ = input()
        with open(Path("YouChorale_flow") / "bm" / ("%s.pkl" % f), 'wb') as fout:
            pickle.dump(bm, fout, protocol=4)
