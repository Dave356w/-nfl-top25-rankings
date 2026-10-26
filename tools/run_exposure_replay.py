"""Run paired browser-worker replays, checkpointing every game/seed."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'backtest_output/exposure_2026'
jobs=sorted(OUT.glob('*.input.json'))
def replay(pair):
    path,seed=pair
    target=OUT/(path.name.replace('.input.json',f'.seed{seed}.result.json'))
    if target.exists():
        return target.name+' cached'
    result=subprocess.run(['node',str(ROOT/'tools/replay_exposures.js'),str(path),str(seed)],
                          text=True,capture_output=True,timeout=1200)
    if result.returncode:
        (OUT/(target.name+'.error.txt')).write_text(result.stderr)
        return target.name+' FAILED '+result.stderr[-500:]
    data=json.loads(result.stdout)
    assert len(data['rows'])==4
    target.write_text(json.dumps(data,indent=2))
    return target.name+' OK '+str([(r['setting'],r['filled'],round(r['actual_best'],2)) for r in data['rows']])

with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(replay,(path,seed)) for path in jobs for seed in (356,357,358)]
    for future in as_completed(futures):
        print(future.result(),flush=True)
