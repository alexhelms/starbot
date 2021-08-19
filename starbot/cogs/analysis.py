import io
import logging
import urllib.parse
from typing import Optional
from discord import File, Message, Attachment
from discord.ext import commands
from pathlib import Path
import os
import tempfile
import validators
import gdown
import requests
from starbot.util import ProcessPool
from starbot.imaging.measure import measure_image, MeasureResult

logger = logging.getLogger(__name__)

pool = ProcessPool()


def google_url_to_id(url: str) -> Optional[str]:
    if url.startswith('https://drive.google.com/file/d/'):
        return url.split('/')[5]
    return None


def google_download(gid: str, tmp: Path):
    filename = gdown.download(f'https://drive.google.com/uc?id={gid}', output=str(tmp) + os.sep)
    return Path(filename)


def url_download(url: str, tmp: Path):
    a = urllib.parse.urlparse(url)
    filename = tmp / os.path.basename(a.path)
    with open(filename, 'wb') as file:
        response = requests.get(url)
        file.write(response.content)
    return filename


async def download_image_from_google_drive(url: str, temp_dir: Path) -> Optional[Path]:
    google_id = google_url_to_id(url)
    if google_id is None:
        return None

    return await pool.submit(google_download, google_id, temp_dir)


async def download_image_direct(url: str, temp_dir: Path) -> Path:
    return await pool.submit(url_download, url, temp_dir)


async def download_user_image(ctx: commands.Context, temp_dir: Path) -> Optional[Path]:
    if ctx.message.attachments:
        attachment: Attachment = ctx.message.attachments[0]
        image_filename = temp_dir / attachment.filename
        await attachment.save(image_filename)
        return image_filename

    split_contents = ctx.message.content.split(' ')

    # [0] = bot mention
    # [1] = command
    # [2] = image url
    if len(split_contents) == 3:
        url = split_contents[2]
        if not validators.url(url):
            await ctx.send('Invalid URL')
            return None

        if 'drive.google.com' in url:
            image_filename = await download_image_from_google_drive(url, temp_dir)
        elif url.lower().endswith('.fit') or url.lower().endswith('.fits'):
            image_filename = await download_image_direct(url, temp_dir)
        else:
            await ctx.send('Your image must be an attachment or on google drive')
            return None

        if image_filename is None:
            await ctx.send('I wasn\'t able to download your image')
            return None

        if not image_filename.suffix.lower() in ['.fit', '.fits']:
            await ctx.send('I downloaded the attachment but it wasn\'t a .fit or .fits file.')
            return None

        return image_filename


def process_image(fits_filename: Path) -> MeasureResult:
    with open(fits_filename, 'rb') as f:
        result = measure_image(f)
        return result


class AnalysisCog(commands.Cog, name='Analysis'):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.pool = pool

    @commands.command(usage='[attachment|google drive|url]', brief='-- Measure an image\'s FWHM')
    async def measure(self, ctx: commands.Context):
        msg: Message = ctx.message
        split_contents = msg.content.split(' ')

        if len(split_contents) == 3:
            url = split_contents[2]
        elif msg.attachments:
            url = msg.attachments[0].url
        else:
            await ctx.send('Measure command must have a URL to your FITS image, or an attachment')
            return

        if not validators.url(url):
            await ctx.send('Invalid URL')
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                image_filename = await download_user_image(ctx, Path(tmpdir))
                if not image_filename:
                    await ctx.send('Something went wrong downloading your image :frowning:')

                result = await self.pool.submit(process_image, image_filename)
                if result.success:
                    plot_buf = io.BytesIO()
                    result.plot.save(plot_buf, format='JPEG', subsampling=0, quality=90)
                    plot_buf.seek(0)
                    file = File(plot_buf, filename='stars.jpg')
                    await ctx.send(
                        f'I measured {result.num_stars} stars. Your image scale is {result.image_scale:.3f} '
                        f'arcsec/px and has a FWHM of **{result.median_fwhm_arcsec:.3f} arcsec**.', file=file)
                    return
                else:
                    await ctx.send('I wasn\'t able to analyze your image :frowning:')
                    return
            except BaseException as e:
                logger.exception(str(e), exc_info=e)
                await ctx.send('Something went wrong when I was analyzing your image :frowning:')
                return
