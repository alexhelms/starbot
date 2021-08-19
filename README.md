# Starbot

A discord bot for analyzing fit/fits images FWHM.

## Example

Measure an image and report the FHWM. The bot will respond with the FWHM and a stretched image with red circles on the
stars that were used for the calculation.

```
@Starbot measure https://somewhere.com/my_image.fits
```

# Supported Image Sources

The following image sources are supported:

- Google drive
- Direct link
- Attachment

# Suported Image Types

- FIT/FITS with header information
    - `SCALE`, `PIXSCALE`
    - `FOCALLEN`
    - `XPIXSZ`
    - `BAYERPAT`