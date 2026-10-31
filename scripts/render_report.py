"""Render every PDF page for visual QA, plus compact contact sheets."""
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw
root=Path(__file__).resolve().parent.parent
out=root/'tmp/pdfs/report'; out.mkdir(parents=True,exist_ok=True)
for pattern in ['page-*.png','contact-*.png']:
 for old in out.glob(pattern): old.unlink()
pdf=pdfium.PdfDocument(root/'DSCI552_Homework5_Report.pdf')
thumbs=[]
for i,page in enumerate(pdf):
 w,h=page.get_size(); scale=min(1600/w,1800/h)
 im=page.render(scale=scale).to_pil().convert('RGB')
 im.save(out/f'page-{i+1:02d}.png')
 thumb=ImageOps.contain(im,(460,570))
 tile=Image.new('RGB',(480,610),'#dfe5ec'); tile.paste(thumb,((480-thumb.width)//2,24))
 ImageDraw.Draw(tile).text((12,590),f'Page {i+1}',fill='black')
 thumbs.append(tile)
for start in range(0,len(thumbs),6):
 sheet=Image.new('RGB',(1440,1220),'white')
 for j,tile in enumerate(thumbs[start:start+6]): sheet.paste(tile,((j%3)*480,(j//3)*610))
 sheet.save(out/f'contact-{start//6+1:02d}.png')
print(f'Rendered {len(thumbs)} pages to {out}')
