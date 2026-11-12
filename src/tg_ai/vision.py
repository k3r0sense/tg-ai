import base64
from io import BytesIO

from PIL import Image, ImageOps


def image_attachment(message):
    if message.photo:
        return message.photo[-1]
    doc = message.document
    if doc and doc.mime_type in {'image/jpeg', 'image/png', 'image/webp'}:
        return doc
    return None


async def image_data_url(bot, attachment):
    if attachment.file_size and attachment.file_size > 10_000_000:
        raise ValueError('Image exceeds 10 MB')
    target = BytesIO()
    await bot.download(attachment.file_id, destination=target)
    if target.tell() > 10_000_000:
        raise ValueError('Image exceeds 10 MB')
    target.seek(0)
    with Image.open(target) as source:
        if source.width * source.height > 25_000_000:
            raise ValueError('Image exceeds 25 megapixels')
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((1280, 1280))
        output = BytesIO()
        image.save(output, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode('ascii')
