import fire
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from PIL import Image
from scipy.io import wavfile
from scipy.signal import hilbert
from typing import Dict, List, Optional

# width of image components after reconstruction
# see https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
COMPONENT_SIZES = {
    "sync_a": (0, 39),
    "space_a": (39, 86),
    "image_a": (86, 995),
    "telemetry_a": (995, 1040),
    "sync_b": (1040, 1079),
    "space_b": (1079, 1126),
    "image_b": (1126, 2035),
    "telemetry_b": (2035, 2080),
}

# sequence for alignment
# https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
SYNCHRONIZATION_SEQUENCE = np.array([0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 0, 0, 0,
                                     0, 0, 0]) - 128


def audio_to_hilbert(in_path: str, resample_rate: int = 20800) -> np.ndarray:
    """

    :param in_path:
    :param resample_rate:
    :return:
    """

    #
    rate, audio = scipy.io.wavfile.read(in_path)

    #
    coef = resample_rate / rate
    samples = int(coef * len(audio))
    audio = scipy.signal.resample(audio, samples)

    #
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    #
    hilbert_transformed = np.abs(hilbert(audio))

    # assuming there is mutual information between points nearby and noise is
    # fairly decorrelated, Gaussian smoothing should improve quality
    filtered = scipy.ndimage.gaussian_filter(np.abs(hilbert_transformed), 5)

    return filtered


def subsample(arr: np.ndarray, step: np.ndarray = 5) -> np.ndarray:
    """ Sampling from the hilbert transformed signal for desired images

    :param arr:
    :param step:
    :return:
    """

    return arr[::step]


def signal_to_noise(arr: np.ndarray, axis=0) -> np.ndarray:
    """ Signal to noise (SNR) calculation (previously included in scipy)

    :param arr: array to calculate signal to noise against
    :param axis: axis to perform SNR calculation over
    :return: signal to noise ratio calculated along axis
    """

    # mean and standard deviation along axis
    mean, std = arr.mean(axis), arr.std(axis=axis)

    # SNR calculation along axis
    return np.where(std == 0, 0, mean / std)


def quantize(arr, black_point=0, white_point=97):
    """

    :param arr:
    :param black_point:
    :param white_point:
    :return:
    """

    #
    low, high = np.percentile(arr, (black_point, white_point))

    #
    arr = np.round((255 * (arr - low)) / (high - low)).clip(0, 255)

    #
    return arr.astype(np.uint8)


def reshape(arr: np.ndarray,
            synchronization_sequence: np.ndarray = SYNCHRONIZATION_SEQUENCE,
            minimum_row_separation: int = 2000):


    # list of maximum correlations found: (index value)
    rows = []

    # need to shift the values down to get meaningful correlation values
    # signalshifted = [x - 128 for x in arr]

    previous_corr, previous_ind = -np.inf, 0

    for current_loc in range(len(arr) - len(synchronization_sequence)):

        row = [x - 128 for x in arr[current_loc: current_loc + len(synchronization_sequence)]]
        temp_corr = np.dot(synchronization_sequence, row)

        if current_loc - previous_ind > minimum_row_separation:
            previous_corr, previous_ind = -np.inf, current_loc
            rows.append(arr[current_loc: current_loc + 2080])

        elif temp_corr > previous_corr:
            previous_corr, previous_ind = temp_corr, current_loc

    #
    return np.vstack([row for row in rows if len(row) == 2080])


def pad_sequence(arr: np.ndarray, n_before: int, n_after: int,
                 value=np.nan) -> np.ndarray:
    """
    # TODO: replace with nearest neighbor padding, numpy function rather than
            this hack

    :param arr:
    :param n_before:
    :param n_after:
    :param value:
    :return:
    """

    #
    arr = np.squeeze(arr)
    assert len(arr.shape) == 1, "Padding for flat rows, 1-dimensional"

    #
    return np.array([value] * n_before +
                    arr.tolist() +
                    [value] * n_after).ravel()


def maximize_row_correlation(arr: np.ndarray,
                             shimmy_width: int = 100) -> np.ndarray:
    """

    :param arr:
    :param shimmy_width:
    :return:
    """

    #
    rows = []

    #
    for row in range(arr.shape[0]):

        #
        this_row = arr[row, :]
        padded_this_row = np.pad(this_row.flatten(), (shimmy_width, shimmy_width),
                                 mode="constant", constant_values=np.nan)

        #pad_sequence(this_row, n_before=shimmy_width,
        #             n_after=shimmy_width)

        #
        corr, next_row = -np.inf, None

        for offset in range(2 * shimmy_width):

            #
            padded_next_row = pad_sequence(this_row, n_before=offset,
                                           n_after=2 * shimmy_width - offset)

            #
            temp_corr = np.nansum(padded_this_row * padded_next_row)

            #
            if temp_corr > corr:
                corr, next_row = temp_corr, padded_next_row

        rows.append(next_row)

    #
    arrayed = np.array(rows)[:, shimmy_width:-shimmy_width]

    return arrayed


def filter_noisy_rows(arr: np.ndarray, ) -> np.ndarray:
    """

    :param arr:
    :return:
    """

    #
    snr = signal_to_noise(arr, axis=1)
    snr_diff = np.diff(snr, prepend=0)

    #
    arr = arr[(snr > 0.8) &
              (snr_diff < 0.05) &
              (snr_diff > -0.05) &
              (snr_diff != 0), :]

    return arr


def save_image(arr: np.ndarray, out_path: str) -> None:
    """

    :param arr:
    :param out_path:
    :return:
    """

    image = Image.fromarray(arr.astype(np.uint8))
    image.save(out_path)


def select_image_components(arr: np.ndarray, components: Optional[List[str]]) -> np.ndarray:
    """

    :param arr:
    :param components:
    :return:
    """

    #
    image_regions = []

    #
    if components is None:
        return arr

    #
    for component in components:

        component_start, component_end = COMPONENT_SIZES[component]
        image_regions.append(arr[:, component_start:component_end])

    return np.hstack(image_regions)


def decode_noaa_audio_to_image(*, in_path: str, out_path: str,
                               black_point: int = 5,
                               white_point: int = 95,
                               components: Optional[List[str]] = None) -> None:
    """

    :param in_path:
    :param out_path:
    :param black_point:
    :param white_point:
    :param components:
    :return:
    """

    #
    if components is not None:
        assert set(components) < set(COMPONENT_SIZES.keys()), \
            f"Default to all segments when 'components' is None, otherwise all " \
            f"list elements must be in {COMPONENT_SIZES.keys()}"

    #
    decoded = audio_to_hilbert(in_path)

    #
    subsampled = subsample(decoded)

    #
    quantized = quantize(subsampled, black_point=black_point,
                         white_point=white_point)

    #
    reshaped = reshape(quantized)

    #
    row_shifted = maximize_row_correlation(reshaped)

    #
    denoised = filter_noisy_rows(row_shifted)

    #
    image_components = select_image_components(denoised, components)

    #
    save_image(image_components, out_path=out_path)


if __name__ == "__main__":

    fire.Fire(decode_noaa_audio_to_image)
