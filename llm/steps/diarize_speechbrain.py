#!/usr/bin/env python
"""
diarize.py – one-file SpeechBrain speaker diarizer
Usage:  python diarize.py PATH/TO/audio.wav
Produces: PATH/TO/audio.rttm (RTTM with time-stamped speaker turns)
"""
import sys, torchaudio, torch
import time
from pathlib import Path
from speechbrain.inference.diarization import SpeakerDiarization

# ---------------------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------------------
MODEL_HF_ID   = "speechbrain/diarization-ecapa-voxceleb"   # ⟵ pre-trained recipe
OUT_RTTM_EXT  = ".rttm"                                    # output suffix
BATCH_SIZE    = 32                                         # embedding batch size
USE_GPU       = torch.cuda.is_available()

# ---------------------------------------------------------------------
# 1.  Initialise pipeline (downloads weights to ~/.cache/speechbrain/)
# ---------------------------------------------------------------------
diarizer = SpeakerDiarization.from_hparams(
    source=MODEL_HF_ID,
    savedir="pretrained_models/diarization-ecapa",         # local cache
    run_opts={"device": "cuda" if USE_GPU else "cpu"},
    overrides={
        "batch_size": BATCH_SIZE,
        # Optional overrides for VAD, clustering thresholds, etc.:
        # "oracle_num_speakers": 2,
        # "clustering": {"max_num_speakers": 8},
    }
)

# ---------------------------------------------------------------------
# 2.  Diarize the given WAV (16 kHz mono recommended)
# ---------------------------------------------------------------------
def main(wav_path: str, start: int, end: int):
    wav_path   = Path(wav_path).expanduser().resolve()
    assert wav_path.is_file(), f"{wav_path} not found"

    # (a) optional: resample / convert to mono on the fly
    begin_time = time.time()
    signal, sr = torchaudio.load(wav_path)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
        sr = 16000
    if signal.shape[0] > 1:                                # stereo -> mono
        signal = signal.mean(dim=0, keepdim=True)
    signal = signal[..., start:end]

    # (b) diarize (returns list of speechbrain.data_io.dataio.TemporalSegment)
    begin_time = time.time()
    diar_hyp = diarizer.diarize_tensor(signal, sr)
    print(f'Diarize {wav_path} in {time.time() - begin_time}s')
    print(diar_hyp)

    # (c) write RTTM
    out_rttm = wav_path.with_suffix(OUT_RTTM_EXT)
    with open(out_rttm, "w") as f:
        for seg in diar_hyp:
            # RTTM line: SPEAKER <file> 1 start dur <NA> <NA> speaker_id <NA> <NA>
            f.write(
                "SPEAKER {file} 1 {st:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
                .format(
                    file=wav_path.stem,
                    st=seg.start,
                    dur=seg.duration,
                    spk=seg.speaker_label,
                )
            )

    print(f"Diarization finished. RTTM saved to {out_rttm}")

if __name__ == "__main__":
    wav_path = '/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/wavs_16khz/0-0127/0-0127_20061108/DVR_0_0127_110806_738.wav'
    main(wav_path, 0, 30*16000)

