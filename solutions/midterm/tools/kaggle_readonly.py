"""Read/download competition inputs; deliberately offers no submission operation."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
SLUG = 'sweettv-tv-program-recommender'

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['files', 'download'])
    args = parser.parse_args()
    token = (ROOT/'midterm/api-token').read_text().strip()
    if not token or '\n' in token:
        raise SystemExit('Expected one nonempty token line; token contents suppressed.')
    suffix = 'list' if args.operation == 'files' else 'download-all'
    req = urllib.request.Request(f'https://www.kaggle.com/api/v1/competitions/data/{suffix}/{SLUG}',
        headers={'Authorization': 'Bearer '+token, 'User-Agent': 'midterm-readonly-audit/1.0'})
    opener = urllib.request.build_opener(SafeRedirect())
    try:
        with opener.open(req, timeout=60) as response:
            if args.operation == 'files':
                data = json.load(response)
                output = ROOT/'solutions/midterm/evidence/kaggle_files.json'
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(data, indent=2))
                print(json.dumps(data, ensure_ascii=False)[:14000])
            else:
                output = ROOT/'midterm/data/competition.zip'
                output.parent.mkdir(parents=True, exist_ok=True)
                partial = output.with_suffix('.part')
                total = 0
                with partial.open('wb') as stream:
                    while chunk := response.read(1024*1024):
                        stream.write(chunk); total += len(chunk)
                        if total % (25*1024*1024) == 0:
                            print(f'Downloaded {total//(1024*1024)} MiB', flush=True)
                partial.rename(output)
                print(json.dumps({'download': str(output.relative_to(ROOT)), 'bytes': total}))
    except urllib.error.HTTPError as error:
        print(json.dumps({'http_status': error.code, 'operation': args.operation,
            'detail': 'Credentials not displayed. 403 may require accepting competition rules.'}))
        raise SystemExit(1)
    except urllib.error.URLError as error:
        print(json.dumps({'network_error': type(error.reason).__name__, 'operation': args.operation}))
        raise SystemExit(2)

if __name__ == '__main__':
    main()
