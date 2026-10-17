"""Frozen MAST003: export verified hybrid q95, copy all other arrays exactly."""
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
import zipfile
import numpy as np
from .artifacts import digest, write_json
from .mast_submit import validate

ROOT = Path('run_data/submission-mast-003')


def main():
    started = time.monotonic(); (ROOT/'running.lock').mkdir()
    plan = json.loads((ROOT/'predeclared-plan.json').read_text())
    freeze = json.loads((ROOT/'source-freeze.json').read_text())
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'shots_validated': 0, 'frames': 0, 'MAST_labels_demos_envelopes_read': False,
              'optimizer_steps': 0, 'GPU_count': 0, 'new_flux_betaN_or_DIII_D': False,
              'upload_or_promotion': False}

    def check():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == 'darwin' else rss*1024
        result['maximum_process_RSS_bytes'] = rss
        if time.monotonic()-started > plan['hard_seconds'] or rss > plan['max_process_RSS_bytes']:
            raise ValueError('003original resource cap exceeded')

    def identities():
        if digest(ROOT/'predeclared-plan.json') != freeze['plan_sha256']: raise ValueError('003plan changed')
        for path, expected in (plan['byte_sha256_pins']|freeze['source_sha256']).items():
            check()
            if digest(Path(path)) != expected: raise ValueError('003source/input changed: '+path)

    try:
        identities()
        review = json.loads(Path('run_data/mast-017/completion-gate-review-20261013.json').read_text())
        audit = json.loads(Path('run_data/mast-017/full-audit.json').read_text())
        if not review['composed_numerical_source_order_coverage_export_gates_passed'] or not audit['complete']:
            raise ValueError('003full numerical audit not passed')
        source = Path('run_data/submission-mast-002/mast_public_test.npz')
        corrected = Path('run_data/mast-017/q95-diagnostic.npz')
        target = ROOT/'mast_public_test.npz'
        with np.load(source, allow_pickle=False) as old_arrays, np.load(corrected, allow_pickle=False) as q_arrays, \
                zipfile.ZipFile(source) as old, zipfile.ZipFile(corrected) as corrected_zip, \
                zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as new:
            for index, shot in enumerate(audit['shots']):
                check(); prefix = f'shot_{index:04d}_'
                if shot['index'] != index: raise ValueError('003original shot order changed')
                psi, beta, q = old_arrays[prefix+'psirz'], old_arrays[prefix+'betaN'], q_arrays[prefix+'q95']
                validate({'psirz':psi, 'betaN':beta, 'q95':q}, shot['frames'])
                if np.any(q<=0): raise ValueError('003nonpositive q')
                for key in ('psirz', 'betaN', 'q95'):
                    name = prefix+key+'.npy'
                    new.writestr(name, corrected_zip.read(name) if key == 'q95' else old.read(name))
                result['shots_validated'] += 1; result['frames'] += len(q)
                del psi, beta, q
        with zipfile.ZipFile(source) as old, zipfile.ZipFile(corrected) as corrected_zip, \
                zipfile.ZipFile(target) as new, np.load(target, allow_pickle=False) as saved:
            if new.namelist() != old.namelist(): raise ValueError('003keys/order changed')
            for index, shot in enumerate(audit['shots']):
                check(); prefix = f'shot_{index:04d}_'
                validate({k:saved[prefix+k] for k in ('psirz','q95','betaN')}, shot['frames'])
                for key in ('psirz','betaN','q95'):
                    name = prefix+key+'.npy'; expected = corrected_zip.read(name) if key == 'q95' else old.read(name)
                    if new.read(name) != expected: raise ValueError('003frozen member/pinned q bytes changed')
        if result['frames'] != 69075 or result['shots_validated'] != 1206: raise ValueError('003full coverage changed')
        identities()
        d3d = Path('run_data/submission-010/diii_d_public_test.npz')
        provenance = {'method':'Original002 total flux/contours/betaN with directexternal coil derivatives plus splineplasma q95',
            'MAST_equilibrium_labels_or_envelopes_used':False, 'MAST_leaderboard_feedback_used':True,
            'feedback_use':plan['feedback_disclosure'], 'AI_assistance':'Codex implemented fixed numerical diagnostics and exact export',
            'dataset_revision':'1e280905b85f2a6fdde7e06fca8cf3a1edf447cb', 'MAST_frames':69075, 'MAST_shots':1206,
            'frozen_from_submission':976708, 'DIII_D_source_submission':973093,
            'DIII_D_predictions_sha256':digest(d3d), 'MAST_predictions_sha256':digest(target),
            'audit_export_sha256':review['diagnostic_export_sha256'], 'plan_sha256':freeze['plan_sha256'],
            'source_sha256':freeze['source_sha256'], 'limitations':plan['scientific_limit']}
        archive = ROOT/'fusion-mast-derivative-003.zip'
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_STORED) as new:
            new.write(d3d,d3d.name); new.write(target,target.name)
            new.writestr('manifest.json',json.dumps(provenance,indent=2))
        with zipfile.ZipFile(archive) as saved:
            if saved.namelist() != [d3d.name,target.name,'manifest.json'] or saved.testzip() is not None:
                raise ValueError('003full archive integrity failed')
            for path in (d3d,target):
                import hashlib
                with saved.open(path.name) as stream:
                    if hashlib.file_digest(stream,'sha256').hexdigest() != digest(path): raise ValueError('003archive payload changed')
        identities(); check()
        result.update(complete=True, validated_for_submission_format=True,
            frozen_psi_betaN_DIII_D_exact=True, q95_exact_frozen017=True, official_accuracy_or_promotion=False,
            archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
            mast_predictions_sha256=digest(target), provenance=provenance)
        check()
    except Exception as exc:
        result.update(failed=True,failure=type(exc).__name__+': '+str(exc)); raise
    finally:
        result['actual_wall_seconds'] = time.monotonic()-started
        write_json(ROOT/'build-receipt.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
