"""Properties 2 and 3: signed, expiring keyless downloads; disclosure metadata in every PNG."""

import io
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urlencode, urlsplit

from conftest import ALICE, BOB, FakeClock, running, submit, wait_for
from PIL import Image

TRAINED_ALGORITHMIC_MEDIA = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
IPTC_EXT = "{http://iptc.org/std/Iptc4xmpExt/2008-02-29/}"


def finished_job(client, **fields):
    body = wait_for(client, submit(client, **{"width": 32, "height": 16} | fields)["id"])
    assert body["status"] == "succeeded", body
    return body


def with_query(url, **changes):
    parts = urlsplit(url)
    query = {key: values[0] for key, values in parse_qs(parts.query).items()} | changes
    return parts._replace(query=urlencode(query)).geturl()


def test_download_needs_no_api_key_and_returns_the_png(client):
    image = finished_job(client, seed=9)["images"][0]

    response = client.get(image["url"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    png = Image.open(io.BytesIO(response.content))
    assert png.format == "PNG"
    assert png.size == (32, 16)


def test_every_png_carries_ai_disclosure_model_revision_and_seed(client):
    body = finished_job(client, image_count=2, seed=77)

    for image in body["images"]:
        png = Image.open(io.BytesIO(client.get(image["url"]).content))
        text = png.text
        assert text["AI-Generated"] == "true"
        assert text["Model"] == "chriek-fake-backend"
        assert text["Model-Revision"] == "fake-v1"
        assert text["Seed"] == str(image["seed"])
        assert "AI-generated" in text["Description"]
        xmp = ET.fromstring(text["XML:com.adobe.xmp"].split("?>", 1)[1].rsplit("<?", 1)[0])
        source_types = [
            element.get(f"{IPTC_EXT}DigitalSourceType")
            for element in xmp.iter()
            if element.get(f"{IPTC_EXT}DigitalSourceType")
        ]
        assert source_types == [TRAINED_ALGORITHMIC_MEDIA]


def test_fake_backend_is_deterministic_per_prompt_size_and_seed(client):
    first = client.get(finished_job(client, seed=3)["images"][0]["url"]).content
    again = client.get(finished_job(client, seed=3)["images"][0]["url"]).content
    other_seed = client.get(finished_job(client, seed=4)["images"][0]["url"]).content

    assert first == again
    assert Image.open(io.BytesIO(first)).tobytes() != Image.open(io.BytesIO(other_seed)).tobytes()


def test_tampered_or_unsigned_urls_are_rejected(client):
    body = finished_job(client, image_count=2)
    url = body["images"][0]["url"]
    other = body["images"][1]["url"]
    query = parse_qs(urlsplit(url).query)
    expires, sig = query["expires"][0], query["sig"][0]
    flipped = ("0" if sig[0] != "0" else "1") + sig[1:]

    tampered = [
        with_query(url, sig=flipped),
        with_query(url, sig=""),
        with_query(url, sig="ü" * 64),
        with_query(url, expires=str(int(expires) + 3600)),
        with_query(url, expires="-1"),
        with_query(url, expires="1e9"),
        with_query(url, expires="9" * 5000),
        urlsplit(url)._replace(query="").geturl(),
        # The signature of image 1 does not open image 0, and vice versa.
        url.replace("/0.png?", "/1.png?"),
        with_query(other, sig=sig, expires=expires),
    ]
    for candidate in tampered:
        response = client.get(candidate)
        assert response.status_code == 403, candidate
        assert response.json() == {"detail": "invalid signature"}


def test_signature_of_one_job_does_not_open_another_or_unknown_image(client):
    first = finished_job(client)["images"][0]["url"]
    second = finished_job(client)["images"][0]["url"]
    second_id = urlsplit(second).path.split("/")[2]
    first_id = urlsplit(first).path.split("/")[2]

    assert client.get(first.replace(first_id, second_id)).status_code == 403
    assert client.get(first.replace("/0.png?", "/7.png?")).status_code == 403


def test_malformed_image_paths_are_404(client):
    url = finished_job(client)["images"][0]["url"]
    job_id = urlsplit(url).path.split("/")[2]
    for path in (
        f"/images/not-a-uuid/0.png?{urlsplit(url).query}",
        f"/images/{job_id}/x.png?{urlsplit(url).query}",
        f"/images/{job_id}/..%2F..%2Fdb%2Fchriek.db?{urlsplit(url).query}",
        f"/images/{job_id}/0.jpg?{urlsplit(url).query}",
    ):
        assert client.get(path).status_code == 404, path


def test_links_expire_and_polling_mints_fresh_ones(tmp_path):
    clock = FakeClock()
    with running(tmp_path, clock=clock, CHRIEK_URL_TTL_SECONDS="60") as client:
        old = finished_job(client)["images"][0]
        job_id = urlsplit(old["url"]).path.split("/")[2]

        clock.advance(59)
        assert client.get(old["url"]).status_code == 200
        clock.advance(1)
        expired = client.get(old["url"])
        assert expired.status_code == 403
        assert expired.json() == {"detail": "link expired; poll the job again"}

        fresh = client.get(f"/jobs/{job_id}", headers=ALICE).json()["images"][0]
        assert fresh["url"] != old["url"]
        assert fresh["expires_at"] > old["expires_at"]
        assert client.get(fresh["url"]).status_code == 200
        assert client.get(old["url"]).status_code == 403


def test_download_url_works_for_anyone_holding_it_but_job_stays_private(client):
    body = finished_job(client)
    assert client.get(body["images"][0]["url"], headers=BOB).status_code == 200
    assert client.get(f"/jobs/{body['id']}", headers=BOB).status_code == 404
