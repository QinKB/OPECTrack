"""Read a top-level JSON array without loading the whole table."""
import json
import re

_WS = re.compile(r"[\s,]*")


def iter_json_array(path, chunk=1 << 23):
    dec = json.JSONDecoder()
    with open(path, "r") as f:
        buf = f.read(chunk)
        pos = _WS.match(buf, 0).end()
        if buf[pos] != "[":
            raise ValueError(f"{path}: not a JSON array")
        pos += 1
        eof = False
        while True:
            pos = _WS.match(buf, pos).end()
            if pos >= len(buf):
                if eof:
                    raise ValueError(f"{path}: truncated array")
                more = f.read(chunk)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = f.read(chunk)
                eof = not more
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > chunk:
                buf, pos = buf[pos:], 0
