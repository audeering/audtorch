import pytest
import numpy as np

from audtorch import datasets


# --- datasets/noise.py ---
@pytest.mark.parametrize('duration', [0.01, 0.1, 1])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('mean', [0, 1])
@pytest.mark.parametrize('stdev', [1, 0.5])
def test_whitenoise(duration, sampling_rate, mean, stdev):
    dataset = datasets.WhiteNoise(duration=duration,
                                  sampling_rate=sampling_rate,
                                  mean=mean,
                                  stdev=stdev)
    noise, label = next(iter(dataset))
    samples = int(np.ceil(duration * sampling_rate))
    assert noise.shape == (1, samples)
    assert label == 'white noise'
    assert -1 <= np.max(np.abs(noise)) <= 1
    assert len(dataset) == 1
