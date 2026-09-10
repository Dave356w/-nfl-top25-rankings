#!/usr/bin/env python3
"""Grade archived pregame forecasts against explicitly identified actual results."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.projection_archive import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',default='site/data/projection_archive/snapshots')
    parser.add_argument('--actuals',required=True,help='CSV: game_id,player_key,actual_fp,realized_at_utc')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    snapshots=[json.loads(gzip.decompress(p.read_bytes())) for p in Path(args.archive).glob('*.json.gz')]
    if not snapshots: raise ValueError('No archived snapshots found')
    actuals=pd.read_csv(args.actuals,dtype={'game_id':str,'player_key':str})
    report=evaluate(snapshots,actuals)
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['overall'],indent=2))

if __name__=='__main__':main()

