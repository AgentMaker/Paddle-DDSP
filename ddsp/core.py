import paddle
import paddle.nn as nn
import paddle.fft as fft
import numpy as np
import librosa as li
import crepe
import math
import scipy.signal


def safe_log(x):
    return paddle.log(x + 1e-7)


@paddle.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s in scales:
        S = paddle.signal.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            paddle.to_tensor(scipy.signal.get_window('hann', s)),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.transpose((0, 2, 1)).reshape((batch * channel, 1, frame))

    window = paddle.to_tensor(scipy.signal.get_window(
        'hann', factor * 2)).reshape((1, 1, -1))
    y = paddle.zeros((x.shape[0], x.shape[1], factor * x.shape[2]))
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = nn.functional.pad(y, [factor, factor])
    y = nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape((batch, channel, factor * frame)).transpose((0, 2, 1))

    return y


def upsample(signal, factor):
    signal = signal.transpose((0, 2, 1))
    signal = nn.functional.interpolate(
        signal.unsqueeze(-1), size=[signal.shape[-1] * factor, 1]).squeeze(-1)
    return signal.transpose((0, 2, 1))


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * paddle.arange(1, n_harm + 1)
    aa = (pitches < sampling_rate / 2) + 1e-4
    return amplitudes * aa


def scale_function(x):
    return 2 * nn.functional.sigmoid(x)**(math.log(10)) + 1e-7


def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sampling_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


def extract_pitch(signal, sampling_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, time_major=False)


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = paddle.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * paddle.arange(1, n_harmonic + 1)
    signal = (paddle.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def amp_to_impulse_response(amp, target_size):
    amp = paddle.stack([amp, paddle.zeros_like(amp)], -1)
    amp = paddle.as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = paddle.roll(amp, filter_size // 2, -1)
    win = paddle.to_tensor(scipy.signal.get_window('hann', filter_size))

    amp = amp * win

    amp = nn.functional.pad(
        amp, (0, int(target_size) - int(filter_size)), data_format='NCL')
    amp = paddle.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    if signal.ndim == 2:
        signal = signal.unsqueeze(1)
        signal = nn.functional.pad(
            signal, (0, signal.shape[-1]), data_format='NCL')
        signal = signal.squeeze(1)
    else:
        signal = nn.functional.pad(
            signal, (0, signal.shape[-1]), data_format='NCL')
    if kernel.ndim == 2:
        kernel = kernel.unsqueeze(1)
        kernel = nn.functional.pad(
            kernel, (kernel.shape[-1], 0), data_format='NCL')
        kernel = kernel.squeeze(1)
    else:
        kernel = nn.functional.pad(
            kernel, (kernel.shape[-1], 0), data_format='NCL')

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output
