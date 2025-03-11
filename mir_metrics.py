import numpy as np
from dataset.constants import *


from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz


def cal_mir_metrics(pitch, start_t, end, pitch_p, start_t_p, end_p, seg_len, tolerance=0.05):
    pitch = pitch.detach().cpu().numpy()[0]
    start_t = start_t.detach().cpu().numpy()[0]
    end = end.detach().cpu().numpy()[0]
    pitch_p = pitch_p.detach().cpu().numpy()[0]
    start_t_p = start_t_p.detach().cpu().numpy()[0]
    end_p = end_p.detach().cpu().numpy()[0]
    
    scaling = HOP_LENGTH / SAMPLE_RATE * seg_len
    p_est = np.array([midi_to_hz(m + MIN_MIDI - 1) for m in pitch_p])
    p_ref = np.array([midi_to_hz(m + MIN_MIDI - 1) for m in pitch])
    i_est = np.array([(s * scaling, (s+d) * scaling) for (s, d) in zip(start_t_p, end_p)]).reshape(-1, 2)
    i_ref = np.array([(s * scaling, (s+d) * scaling) for (s, d) in zip(start_t, end)]).reshape(-1, 2)

    metrics = dict()
    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance)
    metrics['mir_metric/note/precision'] = p
    metrics['mir_metric/note/recall'] = r
    metrics['mir_metric/note/f1'] = f
    metrics['mir_metric/note/overlap'] = o

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, onset_tolerance=tolerance)
    metrics['mir_metric/note-with-offsets/precision'] = p
    metrics['mir_metric/note-with-offsets/recall'] = r
    metrics['mir_metric/note-with-offsets/f1'] = f
    metrics['mir_metric/note-with-offsets/overlap'] = o

    # frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    # metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
    # for key, loss in frame_metrics.items():
    #     metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

    return metrics
