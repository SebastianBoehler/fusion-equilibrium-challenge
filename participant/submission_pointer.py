"""Pin an uploaded prediction dataset without sharing the local write credential."""

import argparse
import json
from pathlib import Path
import re
import zipfile

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url


def build(repo, revision, token_file, out):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A pinned 40-character Hugging Face commit is required')
    if out.exists():
        raise FileExistsError('Use a fresh pointer filename for each submission')
    token = token_file.read_text().strip()
    api = HfApi(token=token)
    access = api.whoami()['auth']['accessToken']
    scopes = access.get('fineGrained', {}).get('scoped', [])
    if (access['role'] != 'fineGrained' or len(scopes) != 1
            or scopes[0].get('entity', {}).get('name') != repo):
        raise ValueError('The read token must be scoped only to the predictions repository')
    permissions = scopes[0].get('permissions', [])
    if not permissions or any('write' in p or 'admin' in p for p in permissions):
        raise ValueError('Only a repository read token may be submitted')
    info = api.repo_info(repo, repo_type='dataset', revision=revision)
    files = {f.rfilename for f in info.siblings}
    if not info.private or info.sha != revision or 'diii_d_public_test.npz' not in files:
        raise ValueError('The pinned private dataset must contain DIII-D predictions')
    metadata = get_hf_file_metadata(hf_hub_url(repo, 'diii_d_public_test.npz',
                                             repo_type='dataset', revision=revision), token=token)
    if metadata.size is None or metadata.size <= 0:
        raise ValueError('The read token cannot access a nonempty predictions file')
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.dumps({'repo_id': repo, 'revision': revision, 'token': token})
    with out.open('xb') as handle:
        out.chmod(0o600)
        with zipfile.ZipFile(handle, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', manifest)
    print(f'Pointer validated: {repo}@{revision}; repository read scope only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.repo, args.revision, args.token_file, args.out)


if __name__ == '__main__':
    main()
