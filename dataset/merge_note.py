import torch
from dataset import MelDataset

from pathlib import Path
import pickle
from tqdm import tqdm

from collections import defaultdict
from math import ceil

from constants import *


def merge_time(t_list):
    # print(t_list)
    t_list = list(set(t_list))
    t_list = sorted(t_list, key=lambda x:x[0])
    result = []
    start = -1
    end = -1
    for t in t_list:
        s = t[0]
        e = t[1]

        if s > end:
            if start != -1:
                result.append((start, end))
            start = s
            end = e
        else:
            end = max(end, e)

    if start != -1:
        result.append((start, end))

    # print(result)
    # _ = input()
    return result



if __name__ == "__main__":
    data_path = Path("YouChorale")

    device = "cpu"
    dataset = MelDataset(data_path / "mel",
                         data_path / "note",
                         data_path / "full.json",
                         device=device)

    for f, note in tqdm(zip(dataset.fid_list, dataset.note_list)):
        notes = note["notes"]

        note_dict = defaultdict(list)
        for n in notes:
            note_dict[n[0]].append((n[3], n[4]))

        note_dict = dict(note_dict)
        note_list = []
        for k, v in note_dict.items():
            times = merge_time(v)
            for t in times:
                note_list.append((k, 1, 1, t[0], t[1], (0, 0, 0, 1)))

        notes = sorted(note_list, key=lambda x:(x[3], -x[0]))

        begin = [] # begin time index. The first note which ENDS AFTER the index second
        end = [0] * ceil(notes[-1][3]) # end time index. The last note which BEGINS BEFORE the index+1 second

        for i, n in enumerate(notes):
            b = n[3]
            e = n[4]

            while len(begin) < ceil(e):
                begin.append(i)

            for tmp in range(int(b), len(end)):
                end[tmp] = i

        while len(end) != len(begin):
            end.append(end[-1])

        result = {"notes": notes, "begin": tuple(begin), "end": tuple(end)}

        with open(Path("YouChorale_merge") / "note" / ("%s.pkl" % f), 'wb') as fout:
            pickle.dump(result, fout, protocol=4)
        
