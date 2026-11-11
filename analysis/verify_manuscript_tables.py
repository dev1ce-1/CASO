"""Compare archived results against the numeric text of six manuscript tables."""
from pathlib import Path
from statistics import mean
from types import SimpleNamespace as NS
import json,re
ROOT=Path(__file__).resolve().parents[1]
tables=json.loads((ROOT/'evidence/paper_tables.json').read_text())['tables']
doc=NS(tables={int(k):NS(rows=[NS(cells=[NS(text=c) for c in row]) for row in rows]) for k,rows in tables.items()})
def read(path,ref=None):
    if 'planner_complete' in path: p=ROOT/'evidence/planner'/Path(path).name
    elif 'gate_c' in path: p=ROOT/'evidence/surroundocc'/Path(path).name
    elif 'temporal' in path: p=ROOT/'evidence/temporal'/Path(path).name
    else: p=ROOT/'evidence/native'/Path(path).name
    return json.loads(p.read_text())
planner=read('planner_complete/bootstrap_ci.json')
gate=read('gate_c/bootstrap_ci.json')
temporal=read('temporal/bootstrap_ci.json')
records=[]
def check(table,row,col,values,source):
    cell=doc.tables[table].rows[row].cells[col].text
    nums=re.findall(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)',cell.replace('−','-'))
    if not nums and cell.strip() in ('–','—','-',''):return
    values=values if isinstance(values,list) else [values]
    ok=len(nums)==len(values)
    comparisons=[]
    for s,v in zip(nums,values):
        digits=len(s.split('.')[1]) if '.' in s else 0
        tol=0.5*10**(-digits)+1e-10
        matched=abs(float(s)-float(v))<=tol
        comparisons.append({'paper':s,'source_value':v,'matches_rounding':matched})
        ok=ok and matched
    records.append({'table_xml_index':table,'row':row,'column':col,'paper_text':cell,'source':source,'matches':ok,'numbers':comparisons})
def delta(d,scale=1):
    return [d.get('point_delta',d.get('point'))*scale]+[v*scale for v in d['ci95']]

mapping={'CE':'E1','RiskOcc':'E6','CE+Tversky':'CE+Tv','Uniform RiskOcc':'CTRL-U','Focal loss':'CTRL-F'}
for i,row in enumerate(doc.tables[14].rows[1:],1):
    g=mapping[row.cells[0].text];point=planner['group_point_estimates'][g]
    check(14,i,1,point['near_dynamic_auprc_mean_of_seed'],'planner_complete/bootstrap_ci.json')
    check(14,i,2,point['nav_recall_at_fp_0p10'],'planner_complete/bootstrap_ci.json')
    if g!='E6':
        c=planner['comparisons']['E6_minus_'+g]['metrics']
        check(14,i,3,delta(c['near_dynamic_auprc_mean_of_seed']),'planner_complete/bootstrap_ci.json')
        check(14,i,4,delta(c['nav_recall_at_fp_0p10']),'planner_complete/bootstrap_ci.json')

groups=[['reports/formal/T0_detpre_done.json','reports/formal/E1_seed1.json','reports/formal/E1_seed2.json'],['reports/formal/E3_seed0.json','reports/formal/E3_seed1.json','riskocc_rev2/results/E3_seed2.json'],['reports/formal/CETVERSKY_seed0.json','reports/formal/CETVERSKY_seed1.json','riskocc_rev2/results/CETVERSKY_seed2.json'],['reports/formal/E5_seed0.json','riskocc_rev2/results/E5_seed1.json','riskocc_rev2/results/E5_seed2.json'],['reports/formal/E6_seed0.json','reports/formal/E6_seed1.json','reports/formal/E6_seed2.json'],['riskocc_rev2/results/CTRL_U_seed0.json','riskocc_rev2/results/CTRL_U_seed1.json'],['riskocc_rev2/results/CTRL_F_seed0.json','riskocc_rev2/results/CTRL_F_seed1.json']]
keys=['HighRiskFN','FreeSpaceBlocking','NearFP','NearOccRecall','DynamicFN']
for i,paths in enumerate(groups,1):
    ds=[read(p) for p in paths]
    vals=[mean(d['streaming_miou_percent'] for d in ds),mean(d['SC_IoU']*100 for d in ds)]+[mean(d['safety_metrics'][k] for d in ds) for k in keys]
    for j,v in enumerate(vals,1):check(15,i,j,v,';'.join(paths))

for i,k in enumerate(['HighRiskFN','DynamicFN','NearOccRecall','NearFP','FreeSpaceBlocking','OccupiedVolumeRatio'],1):
    for j,g in enumerate(['E1T','E6T'],1):
        v=mean(temporal['per_model'][g+'_seed'+str(seed)][k]['point'] for seed in [0,1])
        check(16,i,j,v,'3090 temporal/bootstrap_ci.json')
    check(16,i,3,delta(temporal['comparisons']['E6T_seed_mean_minus_E1T_seed_mean'][k]),'3090 temporal/bootstrap_ci.json')

pk=['unsafe_all_at_fp_0p10','unsafe_near_dynamic_at_fp_0p10','over_block_all_at_fp_0p10','over_block_near_dynamic_at_fp_0p10']
for i,row in enumerate(doc.tables[19].rows[1:],1):
    g=mapping[row.cells[0].text];c=planner['comparisons']['E6_minus_'+g]['metrics']
    for j,k in enumerate(pk,1):check(19,i,j,delta(c[k]),'planner_complete/bootstrap_ci.json')
for i,row in enumerate(doc.tables[20].rows[1:],1):
    g=mapping[row.cells[0].text];point=planner['group_point_estimates'][g]
    for j,k in enumerate(pk+['realized_blocking_at_fp_0p10'],1):check(20,i,j,point[k],'planner_complete/bootstrap_ci.json')
for i,k in enumerate(['binary_occupied_iou','miou_18','miou_occupied_17','occupied_auprc','occupied_recall_at_occ3d_fp_0p10','report_fp_at_occ3d_threshold'],1):
    scale=100 if i<=3 else 1
    for j,g in enumerate(['E1-S','E6-S'],1):check(21,i,j,gate['group_point_estimates'][g][k]*scale,'gate_c/bootstrap_ci.json')
    check(21,i,3,delta(gate['E6-S_minus_E1-S'][k],scale),'gate_c/bootstrap_ci.json')

summary={'tables_checked':['II','III','IV','VII','VIII','IX'],'checked_cells':len(records),'checked_numbers':sum(len(r['numbers']) for r in records),'mismatches':[r for r in records if not r['matches']],'scope':'Archived result consistency only; no training, inference or confidence-interval rerun.'}
print(json.dumps(summary,indent=2))
if summary['mismatches']: raise SystemExit(1)
