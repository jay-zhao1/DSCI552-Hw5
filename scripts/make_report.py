"""Create the <=20-page report from saved outputs; never fit any model.

The unabridged notebook/report_content.json remain the experimental record.
Only report layout and figure views are changed here. Full labelled matrices,
coefficient tables, CV searches and original figures remain in the project.
"""
from pathlib import Path
import csv, html, json, math, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, KeepTogether, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from PIL import Image as PILImage
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
WIDTH=528
blocks=json.loads((ROOT/'results/report_content.json').read_text())
with (ROOT/'results/model_comparison.csv').open() as stream:
    metrics={row['Model']:row for row in csv.DictReader(stream)}
full_rules=json.loads((ROOT/'results/rules_full.json').read_text())
pruned_rules=json.loads((ROOT/'results/rules_pruned.json').read_text())
reduced_rules=json.loads((ROOT/'results/rules_reduced_candidate.json').read_text())
same_selected_rules=full_rules==pruned_rules
layout_notes=[]
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9.5,'ytick.labelsize':9.5})

def save_view(fig,name):
    # Report-specific views, without overwriting the original experiment plots.
    path=ROOT/'figures'/f'{name}.png'
    fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return path

def rules_tree_view(rules,name,title):
    """Reconstruct the saved split topology and leaf labels, without fitting."""
    tree={}
    for rule in rules:
        node=tree
        for feature,cut,left in rule['conditions']:
            if 'split' in node: assert node['split']==(feature,cut)
            node['split']=(feature,cut)
            node=node.setdefault('children',{}).setdefault(bool(left),{})
        assert 'leaf' not in node
        node['leaf']=(int(rule['class']),int(re.search(r'\(n=(\d+)\)',rule['text']).group(1)))
    leaves=[]
    def position(node,depth):
        node['depth']=depth
        if 'leaf' in node:
            node['x']=len(leaves)+0.5;leaves.append(node)
        else:
            assert set(node['children'])=={True,False}
            position(node['children'][True],depth+1);position(node['children'][False],depth+1)
            node['x']=sum(c['x'] for c in node['children'].values())/2
    position(tree,0)
    max_depth=max(n['depth'] for n in leaves)
    fig,ax=plt.subplots(figsize=(7.4,1.2+0.76*max_depth))
    features=['Temperature (C)','Nausea','Lumbar pain','Urine pushing','Micturition pains','Burning of urethra']
    palette=['#FCE4CE','#DAF1D7','#D6EAF8','#F3D8F4']
    def draw(node):
        x=node['x']/len(leaves);y=1-node['depth']/(max_depth+0.65)
        if 'leaf' in node:
            cls,support=node['leaf'];label=f'd1: {"yes" if cls//2 else "no"}\nd2: {"yes" if cls%2 else "no"}\nn={support}'
            color=palette[cls]
        else:
            feature,cut=node['split'];label=features[feature] if feature else f'Temperature (C)\ncut: {cut:.8g}';color='#E7EFF7'
            for left,child in node['children'].items():
                cx=child['x']/len(leaves);cy=1-child['depth']/(max_depth+0.65)
                ax.annotate('',xy=(cx,cy+.052),xytext=(x,y-.035),arrowprops={'arrowstyle':'->','lw':1,'color':'#42526A'})
                edge=('no' if left else 'yes') if feature else ('<=' if left else '>')
                ax.text((x+cx)/2,(y+cy)/2,edge,fontsize=9.2,ha='center',va='center',bbox={'facecolor':'white','edgecolor':'none','pad':.8})
                draw(child)
        ax.text(x,y,label,fontsize=9.5,ha='center',va='center',bbox={'boxstyle':'round,pad=.28','facecolor':color,'edgecolor':'#B5C4D1','linewidth':.6})
    draw(tree)
    ax.set(xlim=(-.015,1.015),ylim=(-.015,1.08));ax.axis('off');ax.set_title(title,pad=12)
    return save_view(fig,name)

full_tree_view=rules_tree_view(full_rules,'report_tree_full','Full decision tree: both original diagnoses retained')
reduced_tree_view=rules_tree_view(reduced_rules,'report_tree_reduced','Reduced pruning-path candidate')

# The original matrix values are read verbatim. Index labels avoid shrinking
# 122 long names to unreadable type on an ordinary submission page.
corr=pd.read_csv(ROOT/'results/feature_correlations.csv',index_col=0)
pd.DataFrame({'Report index':np.arange(1,len(corr)+1),'Feature':corr.columns}).to_csv(ROOT/'results/correlation_feature_index.csv',index=False)
fig,ax=plt.subplots(figsize=(7.4,6.5))
im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
ticks=np.arange(0,len(corr),10)
ax.set_xticks(ticks,ticks+1);ax.set_yticks(ticks,ticks+1)
ax.set(xlabel='Predictor index (original column order)',ylabel='Predictor index (original column order)',title='Training predictor Pearson correlations: all 122 features')
fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Pearson r')
corr_view=save_view(fig,'report_correlation_overview')

# Re-layout the same original training points used by 2(e), without fitting a
# model or changing any preprocessing choice. Validate against the saved stats.
columns=re.findall(r'^@attribute\s+(\S+)\s+', (ROOT/'Homework 5 Data/communities.names').read_text(),re.MULTILINE|re.IGNORECASE)
raw=pd.read_csv(ROOT/'Homework 5 Data/communities.data',names=columns,header=None,na_values='?')
training=raw.iloc[:1495]
x=training[corr.columns].astype(float)
x=x.fillna(x.median())
cv_saved=pd.read_csv(ROOT/'results/coefficient_of_variation.csv').set_index('Feature')
assert np.allclose(x.mean().loc[cv_saved.index],cv_saved['Mean'])
assert np.allclose(x.std(ddof=1).loc[cv_saved.index],cv_saved['Sample SD (ddof=1)'])
top=pd.read_csv(ROOT/'results/top11_cv.csv')['Feature'].tolist()
y=training['ViolentCrimesPerPop'].astype(float)
scatter_views=[]
for start in range(0,len(top),6):
    selected=top[start:start+6]
    fig,axes=plt.subplots(3,2,figsize=(7.4,7.0),layout='constrained')
    for ax,feature in zip(axes.flat,selected):
        ax.scatter(x[feature],y,s=4,alpha=.30,color='#3C70A3')
        ax.set(xlabel=feature,ylabel='ViolentCrimesPerPop',xlim=(-.03,1.03),ylim=(-.03,1.03))
        ax.grid(alpha=.22)
    for ax in list(axes.flat)[len(selected):]:ax.axis('off')
    scatter_views.append(save_view(fig,f'report_top11_scatter_{start//6+1}'))
fig,ax=plt.subplots(figsize=(7.4,4.4),layout='constrained')
ax.boxplot([x[f].to_numpy() for f in top],orientation='horizontal',tick_labels=top,
           flierprops={'markersize':2.2,'alpha':.5},medianprops={'color':'#B74126'})
ax.invert_yaxis();ax.set(xlabel='Predictor value (imputed training data)',title='Highest-CV predictor distributions');ax.grid(axis='x',alpha=.2)
box_view=save_view(fig,'report_top11_boxplots')

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportBody',fontName='Helvetica',fontSize=10.3,leading=13.6,spaceAfter=6))
styles.add(ParagraphStyle(name='ReportRule',parent=styles['ReportBody'],fontSize=9.6,leading=12.6,spaceAfter=5))
styles.add(ParagraphStyle(name='ReportCell',fontName='Helvetica',fontSize=9.3,leading=11.6,splitLongWords=True))
styles.add(ParagraphStyle(name='ReportCaption',fontName='Helvetica',fontSize=9.3,leading=12,spaceAfter=7,textColor=colors.HexColor('#42526A')))
for name,size,leading in [('Title',25,30),('Heading1',14,18),('Heading2',12,15)]:
    styles[name].fontName='Helvetica-Bold';styles[name].fontSize=size;styles[name].leading=leading
    styles[name].textColor=colors.HexColor('#173B60');styles[name].spaceBefore=9;styles[name].spaceAfter=5
    styles[name].keepWithNext=True

def markup(text):
    text=html.escape(str(text))
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#145C9E">\1</link>',text)
    return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)

story=[]
def prose(text):
    text=re.sub(r'^(#{1,2} [^\n]+)\n',r'\1\n\n',text,flags=re.MULTILINE)
    for paragraph in text.split('\n\n'):
        paragraph=paragraph.strip()
        if not paragraph:continue
        if paragraph.startswith('# '):style=styles['Title'];paragraph=paragraph[2:]
        elif paragraph.startswith('## '):style=styles['Heading1'];paragraph=paragraph[3:]
        else:style=styles['ReportRule'] if re.match(r'\*\*(?:Rule |Pruned rule |Reduced-candidate rule )',paragraph) else styles['ReportBody']
        if paragraph.startswith('- '):
            for line in paragraph.splitlines():story.append(Paragraph(markup(line.removeprefix('- ')),style,bulletText='-'))
        else:story.append(Paragraph(markup(paragraph).replace('\n',' '),style))

def fmt(value):
    try:
        number=float(value)
        if not math.isfinite(number):return 'N/A'
        return f'{number:.4g}' if 0<abs(number)<.001 else f'{number:.7g}'
    except (ValueError,TypeError):return str(value)

def report_table(columns,rows,weights=None,format_numbers=True):
    n=len(columns);weights=weights or [1/n]*n;weights=[v/sum(weights) for v in weights]
    data=[[Paragraph('<b>'+markup(c)+'</b>',styles['ReportCell']) for c in columns]]
    data += [[Paragraph(markup(fmt(v) if format_numbers else v),styles['ReportCell']) for v in row] for row in rows]
    table=Table(data,colWidths=[WIDTH*w for w in weights],repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EFF7')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
    table.spaceAfter=8;story.append(table)

def feature_list(model):
    features=pd.read_csv(ROOT/'results'/f'selected_{model}.csv')['Feature'].tolist()
    rows=math.ceil(len(features)/4)
    grid=[[features[col*rows+row] if col*rows+row<len(features) else '' for col in range(4)] for row in range(rows)]
    table=Table([[Paragraph(markup(value),styles['ReportCell']) for value in row] for row in grid],colWidths=[WIDTH/4]*4,hAlign='LEFT')
    table.setStyle(TableStyle([('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),1.5),('BOTTOMPADDING',(0,0),(-1,-1),1.5),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    begin=len(story)
    prose(f'**{model}: all {len(features)} selected variables** (read each column from top to bottom).')
    table.spaceAfter=8;story.append(table)
    story[begin:]=[KeepTogether(story[begin:])]

def picture(path,caption,max_height=470):
    with PILImage.open(path) as image:w,h=image.size
    scale=min(WIDTH/w,max_height/h)
    obj=Image(str(path),width=w*scale,height=h*scale)
    story.append(KeepTogether([obj,Paragraph(markup(caption),styles['ReportCaption'])]))

def result_line(model):
    row=metrics[model]
    prose(f'**{model}: test MSE = {float(row["Test MSE"]):.8f}; RMSE = {float(row["Test RMSE"]):.8f}; R-squared = {float(row["Test R2"]):.6f}.**')

# The title/AI disclosure and original experimental protocol are retained verbatim.
prose(blocks[0]['text'])
prose('**Report guide.** This report covers all 14 subquestions within the instructor\'s inclusive 20-page limit (cover, references and any appendices included). Detailed tables and enlarged diagnostic plots are retained in the executed Notebook, results/ and figures/. The compact figure views below use the same saved values and original training records; no models are fitted by this report generator.')
versions=pd.read_csv(ROOT/'results/versions.csv')
core=['Python','numpy','pandas','scikit-learn','xgboost','matplotlib']
prose('**Execution environment:** '+ '; '.join(f'{row.Dependency} {row.Version}' for row in versions.itertuples() if row.Dependency in core)+'. Full versions: results/versions.csv and requirements.txt.')
story.append(PageBreak())

active=None
for index,block in enumerate(blocks[1:],1):
    kind=block['kind']
    if kind=='text':
        text=block['text']
        match=re.search(r'^## ([12]\([a-j]\))',text,re.MULTILINE)
        if match:
            # Final results are inserted beside the requested regression question,
            # with all tuning completed before the saved test evaluations.
            for model in {'2(f)':['OLS'],'2(g)':['Ridge'],'2(h)':['LASSO_unscaled','LASSO_scaled'],'2(i)':['PCR']}.get(active,[]):result_line(model)
            active=match.group(1)
        if same_selected_rules and text.startswith('**Pruned rule '):continue
        if text.startswith('**Reduced-candidate rule '):
            if text.startswith('**Reduced-candidate rule 1.'):
                begin=len(story)
                for rule_index,rule in enumerate(reduced_rules,1):prose(f'**Reduced-candidate rule {rule_index}.** '+rule['text'])
                story[begin:]=[KeepTogether(story[begin:])]
            continue
        if text.startswith('## 2(c)'):
            text='''## 2(c) Feature correlation matrix
Pearson correlations use all 122 median-imputed training predictors; the target is excluded. The complete matrix is shown with readable predictor indices in the original predictive-column order. The index-to-name key is results/correlation_feature_index.csv. The fully named matrix, four enlarged panels and exact values remain in the Notebook, figures/correlation_overview.*, figures/correlation_panel_1.* through correlation_panel_4.*, and results/feature_correlations.csv. Strong correlated blocks suggest redundancy; imputation may alter correlations, especially for sparse LEMAS fields. Association does not establish causation.'''
        if text.startswith('Undefined CVs:'):
            text=text.replace('All 122 coefficients are reported above and in results/coefficient_of_variation.csv.','All 122 coefficients remain in the Notebook and results/coefficient_of_variation.csv; the 11 highest are included below.')
        if re.match(r'\*\*LASSO_(?:unscaled|scaled) selects ',text):
            model='LASSO_unscaled' if text.startswith('**LASSO_unscaled') else 'LASSO_scaled'
            count=len(pd.read_csv(ROOT/'results'/f'selected_{model}.csv'))
            ending=text[text.index('Final solver iterations:'):]
            text=f'**{model} selects {count} variables**, all listed above. '+ending
        prose(text)
    elif kind=='table':
        name=block['name'];rows=block['rows'];columns=block['columns']
        if name=='versions':
            layout_notes.append({'source':name,'location':'results/versions.csv','report':'Core versions on cover'})
        elif name=='coefficient_of_variation':
            prose('**Complete CV table:** all 122 predictor means, sample SDs and CVs are retained in the Notebook under 2(d) and results/coefficient_of_variation.csv. The ratio calculation and edge-case treatment are unchanged.')
            layout_notes.append({'source':name,'location':'results/coefficient_of_variation.csv','report':'Method and complete-table pointer; top 11 retained'})
        elif name=='missingness':
            missing=pd.read_csv(ROOT/'results/missingness.csv')
            grouped=missing.groupby(['Train missing','Train missing fraction'],sort=True).size().reset_index(name='Predictors')
            report_table(['Predictors','Missing training rows','Fraction'],[[r['Predictors'],r['Train missing'],r['Train missing fraction']] for _,r in grouped.iterrows()],[.25,.38,.37])
            prose('Feature-by-feature missingness remains in results/missingness.csv and Notebook 2(b). The group counts above summarize the same saved table.')
        elif name.startswith('selected_LASSO_'):
            model=name.removeprefix('selected_');feature_list(model)
            prose(f'All coefficients, including original-unit equivalents, remain in results/coefficients_{model}.csv; the selected-variable table is results/{name}.csv and Notebook 2(h).')
            layout_notes.append({'source':name,'location':f'results/{name}.csv','report':'Complete variable names retained; coefficient columns external'})
        elif name in ['tree_comparison','tree_reduced_candidate_comparison']:
            if name=='tree_reduced_candidate_comparison':continue
            original=pd.read_csv(ROOT/'results/tree_comparison.csv');reduced=pd.read_csv(ROOT/'results/tree_reduced_candidate_comparison.csv')
            merged=[]
            for _,r in original.iterrows():merged.append([r.Tree,r.alpha,r.Leaves,r.Depth,r['Train accuracy'],r['CV accuracy']])
            r=reduced.loc[reduced.Tree=='Reduced path candidate'].iloc[0]
            merged.insert(2,[r.Tree,r.ccp_alpha,r.Leaves,r.Depth,r['Train exact accuracy'],'N/A'])
            report_table(['Tree','ccp_alpha','Leaves','Depth','Train acc.','CV acc.'],merged,[.28,.19,.10,.10,.16,.17])
            prose('N/A denotes the descriptive path candidate, which is not a replacement CV selection. Full Hamming-loss and error-count details remain in results/tree_comparison.csv and results/tree_reduced_candidate_comparison.csv.')
        elif name=='model_comparison':
            # Method sections retain the exact selected parameters; this table
            # concentrates on all requested test metrics without duplicated JSON.
            report_table(['Model','Test MSE','RMSE','R-squared','LASSO count'],[[r['Model'],f"{float(r['Test MSE']):.8f}",f"{float(r['Test RMSE']):.8f}",f"{float(r['Test R2']):.6f}",r['Selected variables']] for r in metrics.values()],[.28,.20,.19,.19,.14],format_numbers=False)
        elif name=='tuning_summary':
            report_table(['Model','Selected parameter','CV MSE'],[[r['Model'],str(r['Parameter']).replace('model__','').replace('pca__','')+' = '+fmt(r['Selected']),r['CV MSE']] for r in pd.read_csv(ROOT/'results/tuning_summary.csv').to_dict('records')],[.28,.47,.25])
            prose('Every selected setting is interior to its admissible search range; no convergence warnings occurred. Full grid bounds, candidates and fold scores: results/tuning_summary.csv, results/search_audit_*.json and results/cv_*.csv.')
        elif name=='completion_checklist':
            prose('All 14 subquestions are answered in order. The item-by-item verification checklist is retained in results/completion_checklist.csv and the Notebook. No assignment items remain incomplete under the stated XGBoost interpretation.')
        else:
            weights=[.40,.20,.20,.20] if name=='top11_cv' else ([.14,.30,.36,.20] if name=='label_mapping' else None)
            report_table(columns,rows,weights)
            if name=='top11_cv':
                picture(box_view,'2(e). One box per selected predictor, on a common numeric scale; medians, interquartile ranges, whiskers and outliers remain visible.',max_height=310)
    elif kind=='figure':
        name=Path(block['path']).stem
        if name=='tree_full':
            picture(full_tree_view,'1(b). Complete saved tree topology and leaf predictions. d1 = bladder inflammation; d2 = renal-pelvis nephritis; n = leaf support. Binary edges are symptom no/yes; temperature edges compare to the displayed cut. Exact thresholds and every path appear in 1(c); the original impurity/count diagram remains in Notebook 1(b).',max_height=315)
        elif name=='tree_pruned' and same_selected_rules:
            prose(f'**One-SE tree and rules:** the selected tree has exactly the same topology and all {len(pruned_rules)} rules as 1(b)-1(c). Thus Rules 1-{len(full_rules)} are also the complete one-SE rule set. Its separate original diagram and rule file remain in figures/tree_pruned.* and results/rules_pruned.json.')
        elif name=='tree_reduced_candidate':
            picture(reduced_tree_view,'1(d). Actually reduced candidate reconstructed from its saved rules. Both diagnosis predictions and every terminal leaf are shown; all rules follow.',max_height=230)
        elif name=='correlation_overview':
            picture(corr_view,'2(c). Full 122 x 122 correlation matrix. Indices correspond to results/correlation_feature_index.csv; color scale is shared across every cell.',max_height=465)
        elif name.startswith('correlation_panel_'):
            layout_notes.append({'source':block['path'],'location':block['path'],'report':'Detailed correlation panel retained in Notebook and figures/'})
            continue
        elif name=='top11_scatter':
            for j,path in enumerate(scatter_views,1):picture(path,f'2(e). Scatter plots, panel {j}/{len(scatter_views)}. Every predictor is on x; the crime target is on y. Same imputed training observations as the original plots.',max_height=535)
        elif name=='top11_boxplots':continue # Combined with its feature table above.
        elif name in ['pruning_selection','cv_curves','model_test_mse']:
            prose(f'The original diagnostic plot is retained in the Notebook and {block["path"]}; corresponding numeric tables remain in results/.')
            layout_notes.append({'source':block['path'],'report':'Numeric conclusions retained; diagnostic plot external'})
        else:picture(ROOT/block['path'],block['caption'])

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CED7E0'));canvas.line(42,29,570,29)
    canvas.setFont('Helvetica',8.5);canvas.setFillColor(colors.HexColor('#42526A'))
    canvas.drawString(42,18,'jay-zhao1 | DSCI 552 Homework 5');canvas.drawRightString(570,18,str(doc.page));canvas.restoreState()

out=ROOT/'DSCI552_Homework5_Report.pdf'
doc=SimpleDocTemplate(str(out),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=39,title='DSCI 552 Homework 5',author='jay-zhao1')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(out)
assert len(reader.pages)<=20, f'Report has {len(reader.pages)} pages, exceeding the inclusive 20-page limit.'
layout={'page_limit_including_cover_references_and_appendices':20,'actual_pages':len(reader.pages),
        'page_size_points':[612,792],'body_font_points':10.3,'table_font_points':9.3,'rule_font_points':9.6,
        'all_14_subquestions_retained':True,'complete_LASSO_variable_names_retained':True,
        'one_SE_duplicate_tree_and_rules_cross_referenced_only_if_identical':same_selected_rules,
        'full_and_reduced_tree_diagrams_reconstructed_from_saved_rules_without_fitting':True,
        'correlation_index_key':'results/correlation_feature_index.csv','external_details':layout_notes,
        'models_trained_by_report_generator':0,'notebook_or_original_experiment_results_modified':False}
(ROOT/'results/report_layout_map.json').write_text(json.dumps(layout,indent=2))
print(f'{out}: {len(reader.pages)} pages including cover and references; no model fitting.')
