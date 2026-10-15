#!/usr/bin/env python3
"""Install DB-IP Country Lite locally. Downloads the database, never uploads visitor IPs.

Run with the release venv; the private dashboard shows the required attribution.
Example: sudo /opt/ai-lab/current/.venv/bin/python tools/update_country_database.py
"""
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import os
from pathlib import Path
import re
import tempfile
import subprocess
import maxminddb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month',default=datetime.now(timezone.utc).strftime('%Y-%m'))
    parser.add_argument('--destination',type=Path,default=Path('/opt/ai-lab/shared/geoip/country.mmdb'))
    args=parser.parse_args()
    if not re.fullmatch(r'20\d{2}-(0[1-9]|1[0-2])',args.month): parser.error('Use YYYY-MM.')
    url=f'https://download.db-ip.com/free/dbip-country-lite-{args.month}.mmdb.gz'
    args.destination.parent.mkdir(parents=True,exist_ok=True)
    compressed=subprocess.run(['curl','--fail','--location','--max-time','60',
        '--max-filesize','30000000','--silent','--show-error',url],capture_output=True,check=True).stdout
    import io
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as source:
        data=source.read(60_000_001)
    if len(data)>60_000_000: raise ValueError('Unexpectedly large decompressed database.')
    filename=None
    try:
        with tempfile.NamedTemporaryFile(dir=args.destination.parent,delete=False) as output:
            filename=Path(output.name)
            output.write(data)
        with maxminddb.open_database(str(filename)) as reader:
            if not reader.metadata().database_type.lower().startswith('dbip-country'):
                raise ValueError('Unexpected database type.')
            if not reader.get('8.8.8.8').get('country'):
                raise ValueError('Country lookup validation failed.')
            built=datetime.fromtimestamp(reader.metadata().build_epoch,timezone.utc).isoformat()
        filename.chmod(0o644)
        os.replace(filename,args.destination)
        print(f'Installed {args.destination}; build={built}; SHA256={hashlib.sha256(data).hexdigest()}')
        print('Source: https://db-ip.com/db/download/ip-to-country-lite · CC BY 4.0')
    finally:
        if filename and filename.exists(): filename.unlink()


if __name__=='__main__': main()
