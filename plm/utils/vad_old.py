import argparse
import collections
import contextlib
import pandas as pd
from pathlib import Path

import sys
from tqdm import tqdm
import wave

import webrtcvad


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/data/sls/d/fhs/np_dvoice/add_cog_data_(5586)_[11717]_20250131_14_27_9_0179_dvoice_and_npath_add_dr.csv')
    parser.add_argument('--wav_root', default='/data/sls/scratch/limingw/workplace/speaker_anonymization/data/fhs/wavs_16khz')
    parser.add_argument('--target_dir', default='../data/fhs_split_into_chunks')
    parser.add_argument('--agressiveness', type=int, default=1)
    return parser


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        print(path, num_channels)
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        print(sample_rate) # XXX
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames]), voiced_frames[0].timestamp, voiced_frames[-1].timestamp + voiced_frames[-1].duration
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames]), voiced_frames[0].timestamp, voiced_frames[-1].timestamp + voiced_frames[-1].duration


def main():
    parser = get_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    tgt_dir = Path(args.target_dir)
    wav_root = Path(args.wav_root)
    out_wav_dir = tgt_dir / 'wavs'
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    total_dur = 0
    bad_files = []
    with open(tgt_dir / 'all.tsv', 'w') as f_tsv:
        print(out_wav_dir, file=f_tsv)
        for fn in tqdm(df['audio_filepath']):
            print('fn:', fn)
            wav_id = Path(fn).stem
            id_date = Path(fn).parent.name
            wav_path = wav_root / fn
            if not wav_path.exists():
                fn = fn.replace('clean_dvrs/2025-01-21/', '').split('.')[0]+'.wav'
                wav_path = wav_root / fn

            sample_rate = 16e3

            if (out_wav_dir / id_date).exists():
                fns = [p for p in (out_wav_dir / id_date).iterdir()]
                if len(fns) > 0:
                    for fn in fns:
                        start, end = fn.stem.split('_')[-2:]
                        dur = int(end) - int(start)
                        total_dur += float(dur) / (sample_rate * 3600)
                        print('{}\t{}'.format(str(Path(id_date) / fn.name), dur), file=f_tsv)
                    continue

            try:
                audio, sample_rate = read_wave(str(wav_path))
            except:
                wav_id = wav_id.lower().replace('-', '_')
                wav_path = wav_path.parent / f'{wav_id}.wav'
                try:
                    audio, sample_rate = read_wave(str(wav_path))
                except:
                    bad_files.append(str(wav_path))
                    print(f'Warning: {wav_path} not found, skip')
                    continue

            total_dur += len(audio) / (sample_rate * 3600)
            vad = webrtcvad.Vad(args.agressiveness)
            frames = frame_generator(30, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, 30, 300, vad, frames) 
            for segment, start, end in segments:
                start = int(start * sample_rate)
                end = int(end * sample_rate)
                path = '%s/%s_%s_%s.wav' % (id_date, wav_id, start, end)
                
                print(f'{path}\t{end-start}', file=f_tsv)
                (out_wav_dir / id_date).mkdir(parents=True, exist_ok=True)
                write_wave(str(out_wav_dir / path), segment, sample_rate)
    
    print(f'Total duration: {total_dur} hours')
    with open(tgt_dir / 'bad_files.txt', 'w') as f_bad:
        print('\n'.join(bad_files), file=f_bad)

if __name__ == '__main__':
    main()
