"""Extract CODEBRIM original images zip to data/raw/CODEBRIM_original_extracted/"""
import struct, zipfile, zlib
from pathlib import Path

ZIP_PATH  = Path(__file__).resolve().parents[2] / "data" / "raw" / "CODEBRIM_original_images.zip"
OUT_DIR   = Path(__file__).resolve().parents[2] / "data" / "raw" / "CODEBRIM_original_extracted"
ZIP_SHIFT = 4_294_967_296

def read_entry(info: zipfile.ZipInfo) -> bytes:
    with open(ZIP_PATH, "rb") as f:
        for offset in [info.header_offset, info.header_offset - ZIP_SHIFT]:
            f.seek(offset)
            if f.read(4) == b"PK\x03\x04":
                f.seek(offset + 26)
                fname_len = struct.unpack("<H", f.read(2))[0]
                extra_len = struct.unpack("<H", f.read(2))[0]
                f.seek(fname_len + extra_len, 1)
                raw = f.read(info.compress_size)
                if info.compress_type == zipfile.ZIP_DEFLATED:
                    return zlib.decompress(raw, -15)
                if info.compress_type == zipfile.ZIP_STORED:
                    return raw
    raise ValueError(f"Cannot read: {info.filename}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH) as zf:
    entries = [e for e in zf.infolist()
               if not e.is_dir() and "__MACOSX" not in e.filename]

print(f"Extracting {len(entries)} files to {OUT_DIR} ...")
for i, entry in enumerate(entries, 1):
    dst = OUT_DIR / Path(entry.filename).name
    if dst.exists():
        continue
    try:
        dst.write_bytes(read_entry(entry))
    except Exception as e:
        print(f"  SKIP {entry.filename}: {e}")
    if i % 100 == 0:
        print(f"  {i}/{len(entries)}")

print("Done.")
