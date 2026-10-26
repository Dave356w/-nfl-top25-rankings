"""Extend the frozen current-era exposure replay to ten paired seeds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'backtest_output/exposure_2026'
jobs=[p for p in sorted(OUT.glob('*.input.json'))
      if json.loads(p.read_text())['audit']['cohort']=='sleeper']
assert len(jobs)==17

def replay(pair):
    path,seed=pair
    target=OUT/(path.name.replace('.input.json',f'.seed{seed}.result.json'))
    if target.exists():
        result=json.loads(target.read_text())
        assert result['seed']==seed and len(result['rows'])==4
        return target.name+' cached'
    run=subprocess.run(['node',str(ROOT/'tools/replay_exposures.js'),str(path),str(seed)],
                       text=True,capture_output=True,timeout=1200)
    if run.returncode:
        raise RuntimeError(target.name+' '+run.stderr[-1000:])
    result=json.loads(run.stdout)
    assert len(result['rows'])==4 and all(r['filled']==20 for r in result['rows'])
    temporary=target.with_suffix('.pending')
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(target)
    return target.name+' OK'

with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(replay,(path,seed)) for path in jobs for seed in range(356,366)]
    for future in as_completed(futures):
        print(future.result(),flush=True)
