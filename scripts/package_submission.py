"""Package explicit reproducibility inputs/outputs, never environments or caches."""
from pathlib import Path
import hashlib, json, zipfile
root=Path(__file__).resolve().parent.parent
required=['DSCI552_Homework5.ipynb','DSCI552_Homework5_Report.pdf','requirements.txt','README.md','Homework5.pdf']
files=[root/p for p in required]
for directory in ['Homework 5 Data','scripts','results','figures']:
 files += [p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['.DS_Store','submission_manifest.json'] and p.suffix!='.pyc']
assert all(p.exists() for p in files)
validation=json.loads((root/'results/final_validation.json').read_text())
assert validation['notebook_errors']==0 and validation['models']==6
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/'results/submission_manifest.json').write_text(json.dumps(manifest,indent=2))
files.append(root/'results/submission_manifest.json')
output=root/'DSCI552_Homework5_reproducible.zip'
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
 for p in sorted(files): z.write(p,arcname='DSCI552_Homework5/'+str(p.relative_to(root)))
with zipfile.ZipFile(output) as z:
 assert z.testzip() is None
 names=z.namelist()
 assert not any('/.venv/' in n or '/tmp/' in n or '__pycache__' in n or '.DS_Store' in n for n in names)
 for path,digest in manifest.items():
  assert hashlib.sha256(z.read('DSCI552_Homework5/'+path)).hexdigest()==digest
print(f'Created and CRC/SHA256-verified {output}; {len(files)} files; {output.stat().st_size:,} bytes.')
