"""Bounded anonymous HTTP range access to a published concatenated ZIP.

Never downloads the full archive when a server ignores Range. zipfile verifies
member CRC; individual extracted media also receive SHA256 provenance.
"""
from __future__ import annotations

import io
from pathlib import Path
import subprocess

from utility_gate.common import fingerprint, write_json


class RemoteConcat(io.RawIOBase):
    def __init__(self, members, cache):
        self.members = members  # [(URL, declared byte size), ...]
        self.length = sum(size for _, size in members)
        self.position = 0
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.requests = []

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.length+offset
        if position < 0:
            raise ValueError('negative archive offset')
        self.position = position
        return position

    def fetch(self, url, start, size):
        if size > 32*1024*1024:
            raise ValueError('single range exceeds32MiB safety bound')
        path = self.cache/(fingerprint([url, start, size])+'.bin')
        if not path.exists():
            temporary, headers = path.with_suffix('.part'), path.with_suffix('.headers')
            ranged = url+('&' if '?' in url else '?')+f'range_start={start}&range_length={size}'
            command = ['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2',
                       '--retry-all-errors', '--max-time', '120', '--range', f'{start}-{start+size-1}', '--max-filesize', str(max(size, 8192)),
                       '--dump-header', str(headers), '--output', str(temporary), ranged]
            response = subprocess.run(command, capture_output=True, text=True)
            if response.returncode:
                raise RuntimeError('bounded range failed: '+response.stderr[-240:])
            content = headers.read_text(errors='replace').lower()
            if f'content-range: bytes {start}-{start+size-1}/' not in content or temporary.stat().st_size != size:
                temporary.unlink(missing_ok=True)
                raise ValueError('server did not honor exact range')
            temporary.replace(path)
            write_json(path.with_suffix('.json'), {'url': url, 'start': start, 'bytes': size, 'exact_range_verified': True})
        value = path.read_bytes()
        if len(value) != size:
            raise ValueError('range cache size mismatch')
        self.requests.append({'url': url, 'start': start, 'bytes': size})
        return value

    def read(self, size=-1):
        remaining = max(0, self.length-self.position)
        if size < 0:
            size = remaining
        size = min(size, remaining)
        output = []
        base = 0
        for url, length in self.members:
            if self.position >= base+length:
                base += length
                continue
            while size and self.position < base+length:
                start = self.position-base
                count = min(size, length-start, 16*1024*1024)
                output.append(self.fetch(url, start, count))
                self.position += count
                size -= count
            if not size:
                break
            base += length
        return b''.join(output)
