"""PNG encoding with mandatory, non-configurable AI disclosure metadata."""

import io

from PIL import Image
from PIL.PngImagePlugin import PngInfo

DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
XMP_KEY = "XML:com.adobe.xmp"
XMP = f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
    Iptc4xmpExt:DigitalSourceType="{DIGITAL_SOURCE_TYPE}"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="r"?>"""


def encode_png(image: Image.Image, *, model: str, revision: str, seed: int) -> bytes:
    info = PngInfo()
    info.add_itxt("AI-Generated", "true")
    info.add_itxt("Model", model)
    info.add_itxt("Model-Revision", revision)
    info.add_itxt("Seed", str(seed))
    info.add_itxt(
        "Description", f"AI-generated image (model {model}, revision {revision}, seed {seed})"
    )
    info.add_itxt(XMP_KEY, XMP)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()
