"""Execute every cell in a new kernel; resolve paths relative to this script."""
from pathlib import Path
import sys, time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parent.parent
path=root/'DSCI552_Homework5.ipynb'
nb=nbformat.read(path,as_version=4)
# Use the interpreter running this script, independent of user kernel registrations.
km=KernelManager(kernel_name='python3')
km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=km,timeout=3600,resources={'metadata':{'path':str(root)}},allow_errors=False)
start=time.monotonic()
def progress(cell_index,**kwargs):
    print(f'Executing cell {cell_index+1}/{len(nb.cells)}',flush=True)
client.on_cell_start=progress
try:
    client.execute()
finally:
    nbformat.write(nb,path)
errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors
codecells=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in codecells)
print(f'Fresh-kernel run complete: {len(codecells)} code cells; {time.monotonic()-start:.1f} seconds; zero errors.',flush=True)
(root/'results'/'notebook_validation.txt').write_text(f'Fresh kernel; {len(codecells)} code cells executed sequentially; zero errors. Elapsed seconds: {time.monotonic()-start:.1f}\nPython: {sys.version}\n')
