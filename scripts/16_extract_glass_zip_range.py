"""Extract .wav entries from a PARTIAL (byte-ranged) zip download.

A normal zip reader needs the central directory, which a ranged download of the middle of
an archive does not contain. These archives are also written in streaming mode: the local
file header carries csize=usize=0 and the real sizes follow the compressed data in a
trailing data descriptor. So entry length cannot be read from the header either.

Instead: locate each local header, then inflate from the start of its data with a
streaming decompressor, which stops cleanly at the end of the deflate stream and reports
the trailing bytes as unused_data.

Usage: python 16_extract_glass_zip_range.py <file.part> <out_dir>
"""
import struct
import sys
import zlib
from pathlib import Path

SIG = b"PK\x03\x04"


def main():
    part, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = part.read_bytes()

    offsets = []
    i = blob.find(SIG)
    while i >= 0:
        offsets.append(i)
        i = blob.find(SIG, i + 4)
    offsets.append(len(blob))

    n_ok = n_fail = 0
    for a, b in zip(offsets, offsets[1:]):
        method, = struct.unpack("<H", blob[a + 8:a + 10])
        nlen, elen = struct.unpack("<HH", blob[a + 26:a + 30])
        name = blob[a + 30:a + 30 + nlen].decode("utf-8", "ignore")
        if not name.lower().endswith(".wav"):
            continue
        start = a + 30 + nlen + elen
        chunk = blob[start:b]
        try:
            data = zlib.decompressobj(-15).decompress(chunk) if method == 8 else chunk
            if len(data) < 1000:          # truncated or empty -> not a usable clip
                n_fail += 1
                continue
            (out_dir / Path(name).name).write_bytes(data)
            n_ok += 1
        except zlib.error:
            n_fail += 1

    print(f"extracted {n_ok} wav -> {out_dir}  ({n_fail} unusable/truncated)")


if __name__ == "__main__":
    main()
