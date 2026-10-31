"""Independent artifact/data/output consistency checks; no model tuning."""
from pathlib import Path
import json, hashlib, re, zipfile
import nbformat
import numpy as np
import pandas as pd
import pdfplumber
from pypdf import PdfReader
root=Path(__file__).resolve().parent.parent
nb=nbformat.read(root/'DSCI552_Homework5.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert [c.execution_count for c in code]==list(range(1,len(code)+1))
assert not [o for c in code for o in c.outputs if o.output_type=='error']
for row in json.loads((root/'results/data_sha256.json').read_text()):
 assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest()==row['sha256']
original_archive=root/'Homework 5 Data.zip'
archive_checks=0
if original_archive.exists():
 with zipfile.ZipFile(original_archive) as z:
  for name in z.namelist():
   p=root/name
   if not p.exists(): p=root/'Homework 5 Data'/Path(name).name
   if p.suffix in ['.data','.names'] and p.exists() and not name.startswith('__MACOSX/'):
    assert z.read(name)==p.read_bytes(); archive_checks+=1
metrics=pd.read_csv(root/'results/model_comparison.csv')
assert len(metrics)==6
raw=pd.read_csv(root/'Homework 5 Data/communities.data',header=None,na_values='?')
assert raw.shape==(1994,128)
y=raw.iloc[1495:,-1].to_numpy(float)
for row in metrics.to_dict('records'):
 pred=pd.read_csv(root/'results'/f'predictions_{row["Model"]}.csv')
 assert np.array_equal(pred['Original row (1-based)'],np.arange(1496,1995))
 assert np.allclose(pred['Observed'],y,rtol=0,atol=1e-14)
 residual=y-pred['Predicted'].to_numpy()
 mse=np.mean(residual**2)
 assert np.isclose(mse,row['Test MSE'],rtol=1e-10)
 assert np.isclose(np.sqrt(mse),row['Test RMSE'],rtol=1e-10)
 assert np.isclose(1-sum(residual**2)/sum((y-y.mean())**2),row['Test R2'],rtol=1e-10)
for name in ['LASSO_unscaled','LASSO_scaled']:
 coeff=pd.read_csv(root/'results'/f'coefficients_{name}.csv')
 selected=pd.read_csv(root/'results'/f'selected_{name}.csv')
 assert coeff['Selected'].sum()==len(selected)
 assert set(coeff.loc[coeff.Selected,'Feature'])==set(selected.Feature)
for p in root.glob('results/search_audit_*.json'):
 audit=json.loads(p.read_text()); last=audit[-1]
 if p.stem!='search_audit_PCR':
  assert last['min']<last['best']<last['max'] or last['best']==last['min']==0
 assert not any(row['warnings'] for row in audit)
for name in ['rules_full','rules_pruned']:
 rules=json.loads((root/'results'/f'{name}.json').read_text())
 assert all('bladder inflammation' in r['text'] and 'renal pelvis nephritis' in r['text'] for r in rules)
cv=pd.read_csv(root/'results/coefficient_of_variation.csv')
assert len(cv)==122 and np.allclose(cv['Sample SD (ddof=1)']/cv.Mean,cv.CV)
assert len(pd.read_csv(root/'results/top11_cv.csv'))==11
names=re.findall(r'^@attribute\s+(\S+)\s+', (root/'Homework 5 Data/communities.names').read_text(), re.MULTILINE|re.IGNORECASE)
train_predictors=raw.iloc[:1495,5:127].copy(); train_predictors.columns=names[5:127]
manual=train_predictors.fillna(train_predictors.median())
by_feature=cv.set_index('Feature')
assert np.allclose(manual.mean().loc[by_feature.index],by_feature['Mean'])
assert np.allclose(manual.std(ddof=1).loc[by_feature.index],by_feature['Sample SD (ddof=1)'])
corr=pd.read_csv(root/'results/feature_correlations.csv',index_col=0)
assert corr.shape==(122,122) and 'ViolentCrimesPerPop' not in corr.columns
folds=np.load(root/'results/cv_splits.npz')
all_validation=[]
for i in range(5):
 tr,va=folds[f'train_{i}'],folds[f'valid_{i}']
 assert not set(tr)&set(va)
 assert sorted(np.r_[tr,va].tolist())==list(range(1495))
 all_validation.extend(va.tolist())
assert sorted(all_validation)==list(range(1495))
pdf=root/'DSCI552_Homework5_Report.pdf'
reader=PdfReader(pdf)
text='\n'.join(p.extract_text() for p in reader.pages)
for q in [f'1({x})' for x in 'abcd']+[f'2({x})' for x in 'abcdefghij']:
 assert q in text,q
for row in metrics.to_dict('records'):
 assert f'{row["Test MSE"]:.8f}' in text
assert 'I used AI tools to help me learn the relevant programming techniques' in re.sub(r'\s+',' ',text)
assert '\ufffd' not in text and '\u25a0' not in text
bounds=[]
with pdfplumber.open(pdf) as p:
 for i,page in enumerate(p.pages,1):
  assert len(page.extract_text() or '')>40, f'Blank page {i}'
  bad=[c for c in page.chars if c['x0']<0 or c['x1']>float(page.width)+1 or c['top']<0 or c['bottom']>float(page.height)+1]
  assert not bad, f'Out-of-page text on page {i}'
  bounds.append({'page':i,'width_points':float(page.width),'height_points':float(page.height),'chars':len(page.chars)})
(root/'results/pdf_page_audit.json').write_text(json.dumps(bounds,indent=2))
result={'notebook_code_cells':len(code),'sequential_execution':True,'notebook_errors':0,'models':len(metrics),'metric_recomputations_passed':True,'data_hashes_passed':True,'original_zip_files_checked':archive_checks,'all_question_headings_in_pdf':True,'pdf_pages':len(reader.pages),'pdf_text_bounds_passed':True,'no_replacement_or_black_square_glyphs':True,'lasso_support_checks_passed':True,'search_boundaries_and_warning_checks_passed':True}
(root/'results/final_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
