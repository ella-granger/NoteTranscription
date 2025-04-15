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
    dataset = MelDataset(data_path / "mel",
                         data_path / "note",
                         data_path / "full.json",
                         device=device)

    for f, note in tqdm(zip(dataset.fid_list, dataset.note_list)):
        last_sec = note["begin"][-1]
        note_length = max([x[4] for x in note["notes"][last_sec:]])
        note_length = int(note_length * SAMPLE_RATE / HOP_LENGTH)

        notes = note["notes"]

        bm = torch.zeros((MRK_IDX, note_length))
        for n in notes:
            # print(n)
            n_p = n[0] - MIN_MIDI
            n_s = round(n[3] * SAMPLE_RATE / HOP_LENGTH)
            n_e = round(n[4] * SAMPLE_RATE / HOP_LENGTH)
            n_v = n[5]
            # print(n_p, n_s, n_e, n_v)
            for i, v in enumerate(n_v):
                bm[n_p, n_s:n_e] = 1 # += v * pow(2, i)
        # print(bm[20:40, 0:100])
        # _ = input()
        with open(Path("YouChorale_merge") / "mel" / ("%s.pkl" % f), 'wb') as fout:
            pickle.dump(bm, fout, protocol=4)
