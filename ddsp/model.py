import math
import paddle
import paddle.nn as nn
from paddle.nn.initializer import Assign
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve


class Reverb(nn.Layer):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = self.create_parameter((length, 1), default_initializer=Assign(
            (paddle.rand((length,)) * 2 - 1).unsqueeze(-1)))
        self.decay = self.create_parameter(
            (1, ), default_initializer=Assign(paddle.to_tensor(initial_decay)))
        self.wet = self.create_parameter(
            (1, ), default_initializer=Assign(paddle.to_tensor(initial_wet)))

        t = paddle.arange(self.length) / self.sampling_rate
        t = t.reshape((1, -1, 1))
        self.register_buffer("t", t)

    def build_impulse(self):
        t = paddle.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * nn.functional.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(
            impulse, (0, lenx - self.length), data_format='NLC')
        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP(nn.Layer):
    def __init__(self, hidden_size, n_harmonic, n_bands, sampling_rate,
                 block_size):
        super().__init__()
        self.register_buffer("sampling_rate", paddle.to_tensor(sampling_rate))
        self.register_buffer("block_size",  paddle.to_tensor(block_size))

        self.in_mlps = nn.LayerList([mlp(1, hidden_size, 3)] * 2)
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 2, hidden_size, 3)

        self.proj_matrices = nn.LayerList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", paddle.zeros(
            (1, 1, hidden_size), dtype='float32'))
        self.register_buffer("phase", paddle.zeros((1, )))

    def forward(self, pitch, loudness):
        hidden = paddle.concat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = paddle.concat([self.gru(hidden)[0], pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = paddle.rand((
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        )) * 2 - 1

        noise = fft_convolve(noise, impulse)
        noise = noise.reshape((noise.shape[0], -1, 1))

        signal = harmonic + noise

        # reverb part
        signal = self.reverb(signal)

        return signal

    def realtime_forward(self, pitch, loudness):
        hidden = paddle.concat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.set_value(cache)

        hidden = paddle.concat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)

        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = paddle.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)
        omega = omega + self.phase
        self.phase.set_value(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * paddle.arange(1, n_harmonic + 1)

        harmonic = (paddle.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = paddle.rand((
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        )) * 2 - 1

        noise = fft_convolve(noise, impulse)
        noise = noise.reshape((noise.shape[0], -1, 1))

        signal = harmonic + noise

        return signal
