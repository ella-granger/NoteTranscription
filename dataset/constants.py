SAMPLE_RATE = 16000
N_FFT = 2048
WIN_LENGTH = 2048
HOP_LENGTH = 256
PAD_MODE = "constant"
N_MELS = 256
MEL_EPSILON = 1e-8
SEG_LEN = 320

MIN_MIDI = 36
MAX_MIDI = 89

INI_IDX = MAX_MIDI - MIN_MIDI + 1 + 1
END_IDX = MAX_MIDI - MIN_MIDI + 1 + 2
PAD_IDX = MAX_MIDI - MIN_MIDI + 1 + 3

N_HEAD = 4
