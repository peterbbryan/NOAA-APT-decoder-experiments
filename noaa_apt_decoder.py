from typing import List, Optional

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from PIL import Image
from scipy.io import wavfile
from scipy.signal import hilbert, resample

# fmt: off
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

# fmt: on


def audio_to_hilbert(in_path: str, resample_rate: int = 20800) -> np.ndarray:
    """
    Load the audio and convert to Hilbert transformed amplitude info

    :param in_path: string to path of audio file, relative or absolute
    :param resample_rate: rate to resample audio at
    :return: amplitude information corresponding to pixel intensity
    """

    rate, audio = scipy.io.wavfile.read(in_path)

    # resample audio at appropriate rate (20800)
    coef = resample_rate / rate
    samples = int(coef * len(audio))
    audio = scipy.signal.resample(audio, samples)

    # if two-channel audio, average across channels
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # hilbert transform audio and extract envelope information
    hilbert_transformed = np.abs(hilbert(audio))

    return hilbert_transformed


def subsample(arr: np.ndarray, step: np.ndarray = 5) -> np.ndarray:
    """
    Sampling from the Hilbert transformed signal for desired images

    :param arr: signal after Hilbert transform
    :param step: signal after sampling for channels of interest
    :return: subsampled signal
    """

    return resample(arr, len(arr) // step)


def signal_to_noise(arr: np.ndarray, axis=0) -> np.ndarray:
    """
    Signal to noise (SNR) calculation (previously included in scipy)

    :param arr: array to calculate signal to noise against
    :param axis: axis to perform SNR calculation over
    :return: signal to noise ratio calculated along axis
    """

    # mean and standard deviation along axis
    mean, std = arr.mean(axis), arr.std(axis=axis)

    # SNR calculation along axis
    return np.where(std == 0, 0, mean / std)


def quantize(arr: np.ndarray, black_point: int, white_point: int) -> np.ndarray:
    """
    Digitize signal to valid pixel intensities in the uint8 range

    :param arr: numpy array of continuous signal
    :param black_point: dynamic range lower bound, percent
    :param white_point: dynamic range upper bound, percent
    :return: quantized numpy array to uint8
    """

    #  percent for upper and lower saturation
    low, high = np.percentile(arr, (black_point, white_point))

    # range adjustment and quantization
    arr = np.round((255 * (arr - low)) / (high - low)).clip(0, 255)

    # cast to 8-bit range
    return arr.astype(np.uint8)


def reshape(
    arr: np.ndarray,
    synchronization_sequence: np.ndarray = SYNCHRONIZATION_SEQUENCE,
    minimum_row_separation: int = 2000,
) -> np.ndarray:
    """
    Reshape the numpy array to a 2D image array

    :param arr: 1D numpy array to arrange into 2D image array
    :param synchronization_sequence: sequence to indicate row start
    :param minimum_row_separation: minimum columns of separation
           (a hair less than 2080)
    :return: a 2D reshaped image array
    """

    # initialize
    rows, previous_corr, previous_ind = [None], -np.inf, 0

    for current_loc in range(len(arr) - len(synchronization_sequence)):

        # proposed start of row, normalized to zero
        row = [
            x - 128
            for x in arr[current_loc : current_loc + len(synchronization_sequence)]
        ]

        # correlation between the row and the synchronization sequence
        temp_corr = np.dot(synchronization_sequence, row)

        # if you're past the minimum separation, start hunting for the next synch
        if current_loc - previous_ind > minimum_row_separation:
            previous_corr, previous_ind = -np.inf, current_loc
            rows.append(arr[current_loc : current_loc + 2080])

        # if the proposed region matches the sequence better, update
        elif temp_corr > previous_corr:
            previous_corr, previous_ind = temp_corr, current_loc
            rows[-1] = arr[current_loc : current_loc + 2080]

    # stack the row to form the image, drop the incomplete rows at the end
    return np.vstack([row for row in rows if len(row) == 2080])


def filter_noisy_rows(arr: np.ndarray) -> np.ndarray:
    """
    Some empirically based filters for noisy rows

    :param arr: 2D image array
    :return: image with filtered noisy rows
    """

    # calculate signal to noise and the row to row difference in SNR
    snr = signal_to_noise(arr, axis=1)
    snr_diff = np.diff(snr, prepend=0)

    # image filter for rows with high snr (pure noise) and minimal distance
    # in SNR between rows (no theoretical basis, just seems to work)
    arr = arr[(snr > 0.8) & (snr_diff < 0.05) & (snr_diff > -0.05) & (snr_diff != 0), :]

    return arr


def select_image_components(
    arr: np.ndarray, components: Optional[List[str]]
) -> np.ndarray:
    """
    Select the image components to include

    :param arr: 2D image array
    :param components: portions of the image to preserve/filter
    :return: image array with just the appropriate image components
    """

    # image array components
    image_regions = []

    # if there are no components, return the full image
    if components is None:
        return arr

    # image components to include, based on column down selection
    for component in components:
        component_start, component_end = COMPONENT_SIZES[component]
        image_regions.append(arr[:, component_start:component_end])

    return np.hstack(image_regions)


def save_image(arr: np.ndarray, out_path: str) -> None:
    """
    Write numpy array to image file

    :param arr: 2D image array to write to image
    :param out_path: path to save image
    """

    image = Image.fromarray(arr.astype(np.uint8))
    image.save(out_path)


def apply_colormap(
    arr: np.ndarray,
    cm: matplotlib.colors.LinearSegmentedColormap = plt.get_cmap("gist_earth"),
) -> np.ndarray:
    """
    False colorization based on greyscale intensity

    :param arr: 2D image array
    :param cm: map between greyscale and colorized
    :return: colorized array
    """

    # Get the color map by name:
    colorized = cm(arr)

    return colorized[:, :, :3] * 255


def decode_noaa_audio_to_image(
    *,
    in_path: str,
    out_path: str,
    black_point: int = 5,
    white_point: int = 95,
    components: Optional[List[str]] = None,
    colorize: bool = False,
) -> None:
    """
    Command line interface call for NOAA  apt processing

    :param in_path: path to the input audio file
    :param out_path: output path to png
    :param black_point: dynamic range lower bound, percent
    :param white_point: dynamic range upper bound, percent
    :param components: image components to include ('sync_a', 'space_a', 'image_a',
                       'telemetry_a', 'sync_b', 'space_b', 'image_b', 'telemetry_b')
    :param colorize: boolean for colorization
    """

    # components of the image to include
    if components is not None:
        assert set(components) < set(COMPONENT_SIZES.keys()), (
            f"Default to all segments when 'components' is None, otherwise all "
            f"list elements must be in {COMPONENT_SIZES.keys()}"
        )

    # load the audio and convert to Hilbert transformed amplitude info
    decoded = audio_to_hilbert(in_path)

    # sampling from the Hilbert transformed signal for desired images
    subsampled = subsample(decoded)

    # digitize signal to valid pixel intensities in the uint8 range
    quantized = quantize(subsampled, black_point=black_point, white_point=white_point)

    # reshape the numpy array to a 2D image array
    reshaped = reshape(quantized)

    # some empirically based filters for noisy rows
    denoised = filter_noisy_rows(reshaped)

    # select the image components to include
    image_components = select_image_components(denoised, components)

    # colorize greyscale image if selected
    if colorize:
        image_components = apply_colormap(image_components)

    # write numpy array to image file
    save_image(image_components, out_path=out_path)


if __name__ == "__main__":
    fire.Fire(decode_noaa_audio_to_image)
