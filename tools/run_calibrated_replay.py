"""Ten paired seeds for all four exposure settings under chronological calibration."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'backtest_output/calibration_2026'
jobs=sorted(OUT.glob('*.input.json'));assert len(jobs)==17
def run(pair):
    path,seed=pair;target=OUT/path.name.replace('.input.json',f'.seed{seed}.result.json')
    if target.exists():
        r=json.loads(target.read_text());assert r['seed']==seed
        return target.name+' cached'
    p=subprocess.run(['node',str(ROOT/'tools/replay_exposures.js'),str(path),str(seed)],capture_output=True,text=True,timeout=1200)
    if p.returncode:raise RuntimeError(target.name+' '+p.stderr[-1000:])
    r=json.loads(p.stdout);assert all(x['filled']==20 for x in r['rows'])
    temp=target.with_suffix('.pending');temp.write_text(json.dumps(r,indent=2));temp.replace(target)
    return target.name+' OK'
manifest=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    worker_sha256=hashlib.sha256((ROOT/'site/showdown-worker.js').read_bytes()).hexdigest(),
    calibration_sha256=hashlib.sha256((ROOT/'pipeline/simulation_calibration.py').read_bytes()).hexdigest(),
    seeds=list(range(356,366)),games=17,settings=['0.75/0.25','1/0.25','0.75/1','1/1'],entries=20,
    validation='Expanding chronological fits; minimum 24h after each training kickoff; no target game in fit.',
    unchanged='Archived player means, modeled player pool and contest salary data.')
(OUT/'replay_manifest.json').write_text(json.dumps(manifest,indent=2))
with ThreadPoolExecutor(max_workers=4) as pool:
    for f in as_completed([pool.submit(run,(p,s)) for p in jobs for s in range(356,366)]):print(f.result(),flush=True)
