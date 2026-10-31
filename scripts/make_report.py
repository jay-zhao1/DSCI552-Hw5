"""Render recorded notebook narratives, tables, and figures into a verified PDF."""
from pathlib import Path
import json, re, html, csv
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, NextPageTemplate, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from PIL import Image as PILImage
root=Path(__file__).resolve().parent.parent
out=root/'DSCI552_Homework5_Report.pdf'
content=json.loads((root/'results/report_content.json').read_text())
# Put final held-out results beside the corresponding questions as well as in
# the common frozen-model comparison. Values come only from the executed CSV.
with (root/'results/model_comparison.csv').open() as stream:
    metrics={row['Model']:row for row in csv.DictReader(stream)}
question_models={'2(f)':['OLS'],'2(g)':['Ridge'],'2(h)':['LASSO_unscaled','LASSO_scaled'],'2(i)':['PCR']}
expanded=[]; active=None
for block in content:
    match=re.search(r'^## ([12]\([a-j]\))',block.get('text',''),re.MULTILINE)
    if match:
        if active in question_models:
            for model in question_models[active]:
                r=metrics[model]
                expanded.append({'kind':'text','text':f"**Final held-out result - {model}: MSE = {float(r['Test MSE']):.8f}; RMSE = {float(r['Test RMSE']):.8f}; R-squared = {float(r['Test R2']):.6f}.** This evaluation was performed after every model's training-only selection was complete."})
        active=match.group(1)
    expanded.append(block)
content=expanded

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.3,leading=13.3,spaceAfter=7))
styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=7.4,leading=9.3,splitLongWords=True))
styles.add(ParagraphStyle(name='CaptionCustom',fontName='Helvetica',fontSize=9,leading=12,textColor=colors.HexColor('#42526A'),spaceAfter=8))
for name,size,leading in [('Title',24,29),('Heading1',16,20),('Heading2',13,17)]:
 styles[name].fontName='Helvetica-Bold'; styles[name].fontSize=size; styles[name].leading=leading; styles[name].textColor=colors.HexColor('#173B60')
 styles[name].spaceBefore=12; styles[name].spaceAfter=8

def markup(s):
 s=html.escape(str(s))
 s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#145C9E">\1</link>',s)
 s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'`([^`]+)`',r'<font face="Courier">\1</font>',s)
 return s

def foot(canvas,doc):
 w,h=canvas._pagesize
 canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#CED7E0')); canvas.line(36,29,w-36,29)
 canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#42526A'))
 canvas.drawString(36,18,'Yang Zhao | DSCI 552 Homework 5'); canvas.drawRightString(w-36,18,f'{doc.page}')
 canvas.restoreState()

doc=BaseDocTemplate(str(out),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=40,title='DSCI 552 Homework 5',author='Yang Zhao')
templates=[]
def template(name,w,h):
 templates.append(PageTemplate(id=name,pagesize=(w,h),frames=[Frame(36,38,w-72,h-74,id=name,leftPadding=6,rightPadding=6,topPadding=6,bottomPadding=6)],onPage=foot))
template('portrait',612,792)
# Figures retain legible type rather than squeezing all 122 labels into a letter page.
figure_templates={}
for block in content:
 if block['kind']=='figure':
  p=block['path']; name=Path(p).stem
  with PILImage.open(root/p) as im: iw,ih=im.size
  if name.startswith('correlation'):
   width=28*72 if 'overview' in name else 27*72
  elif name.startswith('tree_'): width=15*72
  elif name.startswith('top11'): width=10.5*72
  elif name=='cv_curves': width=12*72
  else: width=8.8*72
  height=width*ih/iw
  tname='fig_'+name
  template(tname,width+84,height+125)
  figure_templates[p]=(tname,width,height)
doc.addPageTemplates(templates)
story=[]
for block in content:
 kind=block['kind']
 if kind=='text':
  normalized=re.sub(r'^(#{1,2} [^\n]+)\n',r'\1\n\n',block['text'],flags=re.MULTILINE)
  for para in normalized.split('\n\n'):
   para=para.strip()
   if not para: continue
   if para.startswith('# '): style=styles['Title']; para=para[2:]
   elif para.startswith('## '):
    style=styles['Heading1']; para=para[3:]
    if para=='References' and story and not isinstance(story[-1],PageBreak): story.append(PageBreak())
    # New main questions on a new page, with all subsections in original order.
    if re.match(r'[12]\([a-j]\)',para) and story and not isinstance(story[-1], PageBreak): story.append(PageBreak())
   else: style=styles['BodyCustom']
   if para.startswith('- '):
    for line in para.split('\n'):
     story.append(Paragraph(markup(line[2:] if line.startswith('- ') else line),style,bulletText='-'))
   else: story.append(Paragraph(markup(para).replace('\n',' '),style))
 elif kind=='table':
  cols=block['columns']; rows=block['rows']; n=len(cols)
  # Selected parameters and long names get enough width; all cells wrap.
  width=528
  if n==2: weights=[.66,.34]
  elif block['name']=='model_comparison': weights=[.16,.24,.12,.12,.12,.12,.12]
  elif block['name']=='completion_checklist': weights=[.10,.75,.15]
  elif block['name']=='tuning_summary': weights=[.17,.19,.12,.13,.10,.10,.10,.09]
  elif n==4 and block['name']=='label_mapping': weights=[.15,.30,.35,.20]
  elif n in [3,4,5]: weights=[.36]+[(.64)/(n-1)]*(n-1)
  else: weights=[1/n]*n
  weights=[w/sum(weights) for w in weights]
  def format_cell(s):
   # Preserve alpha precision without making numeric tables unwieldy.
   try:
    x=float(s)
    if not (x!=x):
     return f'{x:.3e}' if 0<abs(x)<1e-3 else f'{x:.6g}'
    return 'N/A'
   except (ValueError,TypeError): return str(s)
  data=[[Paragraph('<b>'+markup('Solver warnings' if c=='Convergence warnings' else c)+'</b>',styles['CellCustom']) for c in cols]]
  data += [[Paragraph(markup(format_cell(v)),styles['CellCustom']) for v in r] for r in rows]
  t=Table(data,colWidths=[width*w for w in weights],repeatRows=1,hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EFF7')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#6583A0')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
  t.spaceAfter=9
  story.append(t)
 elif kind=='figure':
  tname,w,h=figure_templates[block['path']]
  if story and isinstance(story[-1],PageBreak):
   story.pop()
   if story and isinstance(story[-1],NextPageTemplate): story.pop()
  story.extend([NextPageTemplate(tname),PageBreak(),Paragraph(markup(block['caption']),styles['CaptionCustom']),Image(str(root/block['path']),width=w,height=h),NextPageTemplate('portrait'),PageBreak()])
doc.build(story)
print(out)
