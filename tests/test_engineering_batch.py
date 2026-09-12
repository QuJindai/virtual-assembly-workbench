"""Batch status reflects computations actually performed, using synthetic data."""
import csv
from pathlib import Path
import runpy
import pytest
from assembly_workbench.emma import REQUIRED


@pytest.mark.parametrize('category,expected',[('A','no_reference'),('MPT','nominal_only')])
def test_batch_does_not_claim_screening_or_replay_without_a_pair(tmp_path,category,expected):
    folder=tmp_path/'input';folder.mkdir()
    row=['task','plan','part','P1','PNT',0,0,0,category,'sample' if category=='A' else '','2026-01-01']
    with (folder/'one.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(REQUIRED);w.writerow(row)
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/validate_engineering.py'))
    result=namespace['validate_folder'](folder,tmp_path/'out')['files'][0]
    assert result['status']==expected
    assert result['recipe_replay'] is False
    assert result['replayed_recipes']==0
    assert result['screened_comparisons']==0
    assert result['project_roundtrip'] is True
