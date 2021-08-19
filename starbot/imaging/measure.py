import io
import logging
import dataclasses
import sys
from typing import Optional, List, BinaryIO
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from colour_demosaicing.bayer import demosaicing_CFA_Bayer_bilinear
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ['MeasureResult', 'measure_image']


@dataclasses.dataclass
class MeasureResult:
    success: bool
    image_scale: float
    num_stars: int
    median_fwhm_arcsec: float
    plot: Optional[Image.Image]


def _fig2img(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')


def _plot_image(data: np.ndarray, positions: List[np.ndarray], fwhm_px: float) -> Image.Image:
    sizes = data.shape
    fig = plt.figure()
    fig.set_size_inches(12. * sizes[1] / sizes[0], 12, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data, cmap='gray', norm=LogNorm(), interpolation='nearest')
    apertures = CircularAperture(positions, r=2.5 * fwhm_px)
    apertures.plot(axes=ax, color='red', lw=1.0)
    img = _fig2img(fig)
    return img


def measure_image(f: BinaryIO) -> MeasureResult:
    try:
        with fits.open(f) as hdul:
            image = hdul[0].data
            header = hdul[0].header

            if 'SCALE' in header:
                image_scale = header['SCALE']
            elif 'PIXSCALE' in header:
                image_scale = header['SCALE']
            elif 'FOCALLEN' in header \
                    and 'XPIXSZ' in header:
                image_scale = header['XPIXSZ'] / header['FOCALLEN'] * 206.265
            else:
                image_scale = 1.0

            # Detect CFA and convert to grayscale
            is_cfa = 'BAYERPAT' in header
            if is_cfa:
                rgb_image = demosaicing_CFA_Bayer_bilinear(image, header['BAYERPAT'])
                image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])

            saturation_cutoff = 60000
            max_star_count = 50

            # Median filter to remove hot pixels for our peak search
            blurred = median_filter(image, size=5)
            blurred_mask = blurred > saturation_cutoff

            mean, median, std = sigma_clipped_stats(blurred, sigma=3.0)
            tbl = find_peaks(blurred - median,
                             5.0 * std,
                             box_size=25,
                             npeaks=500,
                             border_width=25,
                             mask=blurred_mask)
            if tbl is None:
                raise Exception('No peaks found')

            # Fit Gaussian2D on each candidate star
            fwhms = []
            positions = []
            existing = set()
            fit_w = LevMarLSQFitter()
            for row in tbl:
                cutout = Cutout2D(image, (row['x_peak'], row['y_peak']), size=21)
                y0, x0 = np.unravel_index(np.argmax(cutout.data), cutout.shape)
                sigma = 5.0  # initial guess
                amp = row['peak_value']
                data = cutout.data - median
                w = Gaussian2D(amp, x0, y0, sigma, sigma)
                w.x_stddev.bounds = (0.01, 15)
                w.y_stddev.bounds = (0.01, 15)
                yi, xi = np.indices(cutout.shape)
                g = fit_w(w, xi, yi, data)
                fwhm = np.average((g.x_fwhm, g.y_fwhm))
                if 1.0 < fwhm < 30.0 and g.amplitude < saturation_cutoff:
                    fwhm_arcsec = fwhm * image_scale
                    xc = g.x_mean.value
                    yc = g.y_mean.value

                    # Reject stars that are at the same position.
                    # Using find_peaks is pretty primitive and it can fin the same centroid.
                    position = np.asarray(cutout.to_original_position((xc, yc)))
                    position_key = tuple(position.astype(int))
                    if position_key not in existing:
                        existing.add(position_key)
                        fwhms.append(fwhm_arcsec)
                        positions.append(position)

                    if len(fwhms) >= max_star_count:
                        break

            if is_cfa:
                # Hot pixels on a CFA image turn into "plus signs" in the debayered grayscal image.
                # This is exactly what an undersampled gaussian would look like, so we can't remove it with a
                # median filter like we do for a mono image.
                # Instead, k-means clustering can segment the true fwhm from the hot pixel fwhm.
                # ASSUMPTION: the larger of the two values is the real fwhm
                km = KMeans(n_clusters=2)
                clusters = km.fit(np.asarray(fwhms).reshape(-1, 1))

                idx = np.argmax(clusters.cluster_centers_)

                # On an image with a large nebula, we can't assume the largest value is the stars.
                # Anything over 15 arcsec, we can assume is garbage so pick the other instead.
                if float(clusters.cluster_centers_[idx]) > 15.0:
                    idx = 0 if idx == 1 else 1

                new_fwhms = []
                new_positions = []
                for label, fwhm, position in zip(clusters.labels_, fwhms, positions):
                    if label == idx:
                        new_fwhms.append(fwhm)
                        new_positions.append(position)
                fwhms = new_fwhms
                positions = new_positions

            median_fwhm = float(np.median(fwhms))

            img = _plot_image(image, positions, median_fwhm / image_scale)

            return MeasureResult(success=True,
                                 image_scale=image_scale,
                                 num_stars=len(fwhms),
                                 median_fwhm_arcsec=median_fwhm,
                                 plot=img)
    except BaseException as e:
        logger.exception(str(e), exc_info=e)
        return MeasureResult(success=False,
                             image_scale=1.0,
                             num_stars=0,
                             median_fwhm_arcsec=0,
                             plot=None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs=1)
    args = parser.parse_args(sys.argv[1:])

    with open(str(args.filename), 'rb') as f:
        result = measure_image(f)

        print(f'Image Scale (arcsec/px) = {result.image_scale:.3f}')
        print(f'Num Stars               = {result.num_stars}')
        print(f'Median FWHM (arcsec)    = {result.median_fwhm_arcsec:.3f}')

        if result.plot:
            result.plot.show()
