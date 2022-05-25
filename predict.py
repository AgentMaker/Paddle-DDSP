import paddle
import argparse
import numpy as np
import librosa as li
import soundfile as sf

from ddsp import DDSP
from ddsp.core import extract_loudness, extract_pitch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', '-c', type=str,
                        default='./pretrained_models/violin/pretrained.pdparams')
    parser.add_argument('--input', '-i', type=str,
                        default='./audios/singing.wav')
    parser.add_argument('--output', '-o', type=str,
                        default='./audios/output.wav')

    args = parser.parse_known_args()[0]

    ckpt = args.ckpt
    audio_file = args.input
    output_file = args.output

    sampling_rate = 48000
    signal_length = 192000
    block_size = 512

    hidden_size = 512
    n_harmonic = 64
    n_bands = 65

    model = DDSP(
        hidden_size=hidden_size,
        n_harmonic=n_harmonic,
        n_bands=n_bands,
        sampling_rate=sampling_rate,
        block_size=block_size
    )
    params = paddle.load(ckpt)
    model.set_state_dict(params)
    model.eval()

    # Load wav float and padding to signal_length
    x, sr = li.load(audio_file, sampling_rate)
    N = (signal_length - len(x) % signal_length) % signal_length
    x = np.pad(x, (0, N))

    # get pitch data per block_size
    pitch = extract_pitch(x, sampling_rate, block_size)

    # get loudness data per block_size
    loudness = extract_loudness(x, sampling_rate, block_size)

    x = x.reshape(-1, signal_length).astype(np.float32)
    p = pitch.reshape(x.shape[0], -1).astype(np.float32)
    l = loudness.reshape(x.shape[0], -1).astype(np.float32)

    with paddle.no_grad():
        ps = paddle.to_tensor(p, dtype=paddle.float32)
        ls = paddle.to_tensor(l, dtype=paddle.float32)

        audios = []
        for p, l in zip(ps, ls):
            p = p.unsqueeze(-1)
            l = l.unsqueeze(-1)
            p = p.unsqueeze(0)
            l = l.unsqueeze(0)

            l = (l - ls.mean()) / ls.std()

            y = model(p, l).squeeze(-1)

            audios.append(y)

        audios = paddle.concat(audios, -1)
        audios = audios.reshape((-1,)).detach().numpy()

    sf.write(output_file, audios, sampling_rate)
