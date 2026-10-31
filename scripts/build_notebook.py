from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parent.parent
nb=nbf.v4.new_notebook(); cells=[]
def md(s):
 cells.append(nbf.v4.new_markdown_cell(s))
 cells.append(nbf.v4.new_code_cell('record_text('+repr(s)+')'))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
code('''
from pathlib import Path
import os, sys, re, json, hashlib, warnings, platform, importlib.metadata as metadata
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, hamming_loss
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor
SEED=552
np.random.seed(SEED)
thread_limit = threadpool_limits(limits=1)
ROOT=Path.cwd()
DATA=ROOT/'Homework 5 Data'
assert (DATA/'diagnosis.data').exists(), 'Run from the project root.'
for folder in ['results','figures']: (ROOT/folder).mkdir(exist_ok=True)
report=[]
def record_text(text): report.append({'kind':'text','text':text})
def say(text):
    display(Markdown(text)); record_text(text)
def table(df, name, report_include=True):
    df.to_csv(ROOT/'results'/f'{name}.csv',index=False)
    display(df)
    if report_include:
        report.append({'kind':'table','name':name,'columns':list(df.columns),
                       'rows':[[str(v) for v in row] for row in df.to_numpy().tolist()]})
def figure(fig, name, caption):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.show(); plt.close(fig)
    report.append({'kind':'figure','path':f'figures/{name}.png','caption':caption})
sns.set_theme(style='whitegrid',context='notebook')
pd.set_option('display.max_rows',150)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.precision',8)
''')
md('''# DSCI 552 Homework 5
**jay-zhao1**

## AI Usage Disclosure
I used AI tools to help me learn the relevant programming techniques and machine learning concepts, understand code implementations, and assist with debugging.

## Requirements and experimental protocol
The two-page assignment and all five footnotes were read, as were both supplied .names files and the supplied KDnuggets article. Local data are used without downloading replacements. The original question order is retained: 1(a) data, 1(b) full tree, 1(c) all rules, 1(d) pruning; 2(a) ordered split, 2(b) imputation, 2(c) correlations, 2(d) CV, 2(e) 11 feature plots, 2(f) OLS, 2(g) Ridge, 2(h) both LASSOs, 2(i) PCR, 2(j) XGBoost.

Seed 552 is fixed. All regression models retain all 122 predictive variables as candidate inputs; the 11-variable CV ranking is descriptive only. Regression tuning uses the same shuffled five-fold KFold within the training portion, minimizing validation MSE. The mandated external split is never shuffled. Imputation, scaling and PCA are fitted separately in each training fold through Pipeline. The test set is evaluated once per final fitted model, after all six models are frozen. No test metric is used to revise a model.''')
code('''
packages=['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib','seaborn','nbformat','nbclient','ipykernel','reportlab','pdfplumber','pypdf','pypdfium2','threadpoolctl']
versions=pd.DataFrame({'Dependency':['Python']+packages,'Version':[platform.python_version()]+[metadata.version(p) for p in packages]})
table(versions,'versions')
checksums=[]
for p in sorted(DATA.iterdir()):
    if p.suffix in ['.data','.names']:
        checksums.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT/'results'/'data_sha256.json').write_text(json.dumps(checksums,indent=2))
''')
md('''## 1(a) Acute Inflammations: data and label powerset
The actual byte encoding is detected from the BOM (UTF-16 if present, otherwise UTF-8); this takes precedence over the ASCII statement in diagnosis.names. Tabs separate fields. Decimal commas in temperature are converted to decimal points. The five binary symptoms and two diagnoses map no to 0 and yes to 1. The label powerset code is 2*d1+d2, so both original diagnoses remain identifiable. d1 denotes inflammation of urinary bladder; d2 denotes nephritis of renal pelvis origin. These are expert-created illustrative records, not evidence of clinical performance.''')
code('''
raw=(DATA/'diagnosis.data').read_bytes()
encoding='utf-16' if raw.startswith((b'\\xff\\xfe',b'\\xfe\\xff')) else 'utf-8-sig'
feature_names=['Temperature (C)','Nausea','Lumbar pain','Urine pushing','Micturition pains','Burning of urethra']
diag=pd.read_csv(DATA/'diagnosis.data',sep='\\t',header=None,encoding=encoding, dtype=str)
assert diag.shape==(120,8)
diag.iloc[:,0]=diag.iloc[:,0].str.replace(',','.',regex=False)
for j in range(1,8):
    assert set(diag[j]) <= {'yes','no'}
    diag[j]=diag[j].map({'no':0,'yes':1})
diag=diag.astype(float)
Xd=diag.iloc[:,:6].copy(); Xd.columns=feature_names
Yd=diag.iloc[:,6:].to_numpy(dtype=int)
yd=2*Yd[:,0]+Yd[:,1]
assert not diag.isna().any().any()
class_names=[f'd1={i//2}, d2={i%2}' for i in range(4)]
table(pd.DataFrame([{'Powerset class':i,'Bladder inflammation (d1)':'yes' if i//2 else 'no','Renal pelvis nephritis (d2)':'yes' if i%2 else 'no','Records':int((yd==i).sum())} for i in range(4)]),'label_mapping')
say(f'Validated: {len(Xd)} rows, {Xd.shape[1]} predictors, {Yd.shape[1]} labels; encoding **{encoding}**; no missing values.')
''')
md('''## 1(b) Full-data decision tree
A CART classifier uses Gini impurity and the powerset target, with ccp_alpha=0 and otherwise unrestricted growth. As requested, the final tree is fitted to all 120 records. Tree class labels explicitly show both diagnoses (0=no, 1=yes). Training accuracy is a description of fit, not a held-out estimate.''')
code('''
full=DecisionTreeClassifier(random_state=SEED,ccp_alpha=0).fit(Xd,yd)
def draw_tree(model,name,title):
    fig,ax=plt.subplots(figsize=(16,8))
    plot_tree(model,feature_names=feature_names,class_names=class_names,filled=True,
              rounded=True,proportion=False,precision=3,fontsize=10,ax=ax)
    ax.set_title(title)
    figure(fig,name,title+' All nodes and both diagnosis labels are shown.')
draw_tree(full,'tree_full','1(b): Unpruned full-data decision tree')
''')
md('''## 1(c) All root-to-leaf IF-THEN rules
Each conjunction is one complete path. Binary branches are translated back to yes/no, and leaf classes back to the original two diagnoses. Temperature thresholds retain the model threshold to sufficient precision. n is the training support of the leaf. The executable path conditions are checked against tree predictions for every record, with exactly one matching rule per record.''')
code('''
def extract_rules(model):
    t=model.tree_; rules=[]
    def walk(node,terms,conditions):
        if t.feature[node] != _tree.TREE_UNDEFINED:
            j=int(t.feature[node]); threshold=float(t.threshold[node]); name=feature_names[j]
            for left,child in [(True,t.children_left[node]),(False,t.children_right[node])]:
                if j==0:
                    term=f'{name} {"<=" if left else ">"} {threshold:.8g}'
                else:
                    assert threshold==0.5
                    term=f'{name} = {"no" if left else "yes"}'
                walk(child,terms+[term],conditions+[(j,threshold,left)])
        else:
            cls=int(model.classes_[np.argmax(t.value[node][0])])
            text=('IF '+' AND '.join(terms) if terms else 'IF TRUE')+f' THEN bladder inflammation = {"yes" if cls//2 else "no"}; renal pelvis nephritis = {"yes" if cls%2 else "no"} (n={t.n_node_samples[node]}).'
            rules.append({'text':text,'conditions':conditions,'class':cls,'leaf':int(node)})
    walk(0,[],[])
    counts=np.zeros(len(Xd),int); prediction=np.full(len(Xd),-1)
    for r in rules:
        mask=np.ones(len(Xd),bool)
        for j,threshold,left in r['conditions']:
            mask &= (Xd.iloc[:,j].to_numpy()<=threshold) if left else (Xd.iloc[:,j].to_numpy()>threshold)
        assert mask.sum()==t.n_node_samples[r['leaf']]
        counts+=mask; prediction[mask]=r['class']
    assert np.all(counts==1) and np.array_equal(prediction,model.predict(Xd))
    return rules
full_rules=extract_rules(full)
for i,r in enumerate(full_rules,1): say(f'**Rule {i}.** '+r['text'])
(ROOT/'results'/'rules_full.json').write_text(json.dumps(full_rules,indent=2))
''')
md('''## 1(d) Cost-complexity pruning and interpretable minimum
CART minimizes training leaf impurity plus alpha times the number of leaves. To define a useful minimum, a predeclared grid of 401 alpha values from 0 through 0.4 is evaluated by shuffled stratified five-fold CV on exact-match powerset accuracy. This grid is independent of label-derived pruning breakpoints. Select the largest alpha whose mean accuracy is within one standard error of the best mean; ties therefore favor fewer leaves. The standard error is the sample standard deviation of five fold accuracies divided by sqrt(5). This is a practical simplicity rule, not a proof of globally minimal trees or statistical equivalence. Final candidates are then refitted on all 120 records. The full-data pruning path is reported descriptively, not used to construct the CV grid. A stump is explicitly compared to the chosen model.''')
code('''
dcv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
alpha_grid=np.linspace(0,0.4,401)
prune_rows=[]
for a in alpha_grid:
    scores=cross_val_score(DecisionTreeClassifier(random_state=SEED,ccp_alpha=float(a)),Xd,yd,cv=dcv,scoring='accuracy')
    prune_rows.append({'alpha':a,'CV accuracy':scores.mean(),'SE':scores.std(ddof=1)/np.sqrt(5)})
pruning=pd.DataFrame(prune_rows)
best_idx=int(pruning['CV accuracy'].idxmax())
threshold=pruning.loc[best_idx,'CV accuracy']-pruning.loc[best_idx,'SE']
chosen_alpha=float(pruning.loc[pruning['CV accuracy']>=threshold-1e-12,'alpha'].max())
pruned=DecisionTreeClassifier(random_state=SEED,ccp_alpha=chosen_alpha).fit(Xd,yd)
stump=DecisionTreeClassifier(random_state=SEED,ccp_alpha=1).fit(Xd,yd)
pruning.to_csv(ROOT/'results'/'pruning_grid.csv',index=False)
path=full.cost_complexity_pruning_path(Xd,yd)
path_rows=[]
for a,imp in zip(path.ccp_alphas,path.impurities):
    t=DecisionTreeClassifier(random_state=SEED,ccp_alpha=float(a)).fit(Xd,yd)
    path_rows.append({'alpha breakpoint':a,'Leaf impurity':imp,'Nodes':t.tree_.node_count,'Leaves':t.get_n_leaves(),'Depth':t.get_depth(),'Train exact accuracy':t.score(Xd,yd)})
table(pd.DataFrame(path_rows),'pruning_path')
comparison=[]
for name,t in [('Unpruned',full),('Selected',pruned),('Stump baseline',stump)]:
    scores=cross_val_score(t,Xd,yd,cv=dcv,scoring='accuracy')
    pred=t.predict(Xd).astype(int); recovered=np.column_stack((pred//2,pred%2))
    comparison.append({'Tree':name,'alpha':t.ccp_alpha,'Nodes':t.tree_.node_count,'Leaves':t.get_n_leaves(),'Depth':t.get_depth(),'Train accuracy':accuracy_score(yd,pred),'Train Hamming loss':hamming_loss(Yd,recovered),'CV accuracy':scores.mean()})
table(pd.DataFrame(comparison),'tree_comparison')
say(f'The best CV accuracy is **{pruning.loc[best_idx,"CV accuracy"]:.4f}**, SE **{pruning.loc[best_idx,"SE"]:.4f}**; the acceptance threshold is **{threshold:.4f}**. Selected alpha = **{chosen_alpha:.3f}**. The selected tree has **{pruned.get_n_leaves()} leaves** and depth **{pruned.get_depth()}** versus **{full.get_n_leaves()} leaves** and depth **{full.get_depth()}** before pruning. '+('No smaller subtree meets this criterion; keeping the original structure is preferable to sacrificing useful accuracy.' if full.get_n_leaves()==pruned.get_n_leaves() else 'This reduces the number of diagnostic paths while meeting the predeclared CV accuracy criterion.'))
assert pruned.get_n_leaves()>1, 'A meaningless stump must not be accepted.'
say('These CV scores informed alpha selection and are not an independent post-selection performance estimate. There is no separate test set in Question 1, in accordance with the full-data fitting requirement.')
fig,ax=plt.subplots(figsize=(9,4)); ax.plot(pruning['alpha'],pruning['CV accuracy']); ax.axhline(threshold,ls='--',color='gray',label='One-SE threshold'); ax.axvline(chosen_alpha,color='red',label='Selected alpha'); ax.set(xlabel='ccp_alpha',ylabel='Five-fold CV exact-match accuracy',title='1(d): Pruning selection'); ax.legend(); figure(fig,'pruning_selection','Alpha selection using only the acute-inflammations data.')
draw_tree(pruned,'tree_pruned','1(d): Selected cost-complexity-pruned tree')
pruned_rules=extract_rules(pruned)
for i,r in enumerate(pruned_rules,1): say(f'**Pruned rule {i}.** '+r['text'])
(ROOT/'results'/'rules_pruned.json').write_text(json.dumps(pruned_rules,indent=2))
''')
md('''## 2(a) Communities and Crime: data and ordered split
Read the 128 attribute names directly from @attribute lines in communities.names. Keep original file order: training is original rows 1-1495, test is rows 1496-1994. The target is ViolentCrimesPerPop. The provided data were already normalized approximately to [0,1]; this is different from standardization to zero mean and unit variance. The data combine the 1990 Census, 1990 LEMAS and 1995 FBI UCR; normalization parameters used by the original dataset creator cannot be undone. All additional preprocessing here is learned solely from training data.''')
code('''
names_text=(DATA/'communities.names').read_text()
columns=re.findall(r'^@attribute\\s+(\\S+)\\s+',names_text,re.MULTILINE|re.IGNORECASE)
assert len(columns)==128 and len(set(columns))==128
communities=pd.read_csv(DATA/'communities.data',names=columns,header=None,na_values='?')
assert communities.shape==(1994,128)
nonpredictive=['state','county','community','communityname','fold']
target='ViolentCrimesPerPop'
X=communities.drop(columns=nonpredictive+[target]).astype(float)
y=communities[target].astype(float)
Xtrain=X.iloc[:1495].copy(); Xtest=X.iloc[1495:].copy()
ytrain=y.iloc[:1495].copy(); ytest=y.iloc[1495:].copy()
assert Xtrain.shape==(1495,122) and Xtest.shape==(499,122)
assert not y.isna().any() and Xtrain.index[-1]==1494 and Xtest.index[0]==1495
cv=list(KFold(n_splits=5,shuffle=True,random_state=SEED).split(Xtrain))
np.savez(ROOT/'results'/'cv_splits.npz',**{f'{kind}_{i}':indices for i,pair in enumerate(cv) for kind,indices in zip(['train','valid'],pair)})
say(f'Validated raw shape: **{communities.shape}**. Training: **{Xtrain.shape}**. Test: **{Xtest.shape}**. Target excluded from predictors; all five nonpredictive fields removed.')
''')
md('''## 2(b) Missing values and imputation
Question marks are parsed as NaN. Median imputation is robust to skew and retains all 122 predictors, including sparsely observed LEMAS variables. A median cannot reconstruct missing information or remove selection bias: LEMAS coverage is highly nonrandom according to the supplied documentation. No rows or predictive columns are dropped. An all-missing column within a training fold is retained and filled with zero via keep_empty_features=True (the audit below checks whether this fallback is actually needed). The descriptive training matrix is median-imputed using training rows only. Predictive CV uses a separate SimpleImputer inside every Pipeline; the descriptive matrix is never passed to GridSearchCV.''')
code('''
missing=pd.DataFrame({'Feature':Xtrain.columns,'Train missing':Xtrain.isna().sum().values,'Train missing fraction':Xtrain.isna().mean().values})
table(missing.loc[missing['Train missing']>0].reset_index(drop=True),'missingness')
empty_folds=sum(int(Xtrain.iloc[tr].isna().all().sum()) for tr,va in cv)
say(f'All-missing training-fold columns across five folds: **{empty_folds}**. Number of predictive columns dropped: **0**.')
descriptive_imputer=SimpleImputer(strategy='median',keep_empty_features=True)
Xdesc=pd.DataFrame(descriptive_imputer.fit_transform(Xtrain),columns=Xtrain.columns,index=Xtrain.index)
assert Xdesc.shape==Xtrain.shape and np.isfinite(Xdesc).all().all()
def make_pipe(model,scale=False,pca=False):
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if scale: steps.append(('scale',StandardScaler()))
    if pca: steps.append(('pca',PCA(svd_solver='full')))
    steps.append(('model',model))
    return Pipeline(steps)
''')
md('''## 2(c) Feature correlation matrix
Pearson correlations below use the median-imputed 1495-row training matrix, before standardization. The target is excluded. All 122 feature labels are shown in a large overview and four enlarged row panels; the complete numeric matrix is also exported. Strong blocks of correlation indicate redundant predictors and motivate regularization and PCR. Imputation can alter correlations, especially for sparse LEMAS fields. These correlations describe association, not causation.''')
code('''
corr=Xdesc.corr(); corr.to_csv(ROOT/'results'/'feature_correlations.csv')
fig,ax=plt.subplots(figsize=(28,25))
sns.heatmap(corr,vmin=-1,vmax=1,cmap='vlag',xticklabels=True,yticklabels=True,ax=ax,cbar_kws={'shrink':0.5})
ax.tick_params(axis='both',labelsize=7); ax.set_title('2(c): Training predictor Pearson correlation matrix (122 x 122)')
figure(fig,'correlation_overview','Full training feature correlation matrix; use the following enlarged panels for labels.')
for start in range(0,122,31):
    end=min(start+31,122)
    fig,ax=plt.subplots(figsize=(27,10))
    sns.heatmap(corr.iloc[start:end],vmin=-1,vmax=1,cmap='vlag',xticklabels=True,yticklabels=True,ax=ax,cbar_kws={'shrink':0.7})
    ax.tick_params(axis='x',labelsize=8); ax.tick_params(axis='y',labelsize=10)
    ax.set_title(f'2(c): Correlation matrix rows {start+1}-{end}; all 122 predictor columns')
    figure(fig,f'correlation_panel_{start//31+1}',f'Enlarged matrix panel {start//31+1}: training predictors {start+1}-{end}.')
''')
md('''## 2(d) Coefficient of variation for every predictor
CV = s/m uses sample standard deviation with ddof=1 and sample mean over the median-imputed training rows only. CV is computed before standardization. A nonzero constant with nonzero mean has CV=0; any zero (or numerically near-zero, absolute tolerance 1e-12) mean has undefined CV and is excluded from ranking. All-zero columns are also undefined. This rule avoids replacing an undefined ratio with an arbitrary large number. The target and identifier fields are excluded. CV is sensitive to the dataset's normalization origin and does not measure predictive importance.''')
code('''
means=Xdesc.mean(); sds=Xdesc.std(ddof=1)
cv_values=sds/means.mask(np.isclose(means,0,atol=1e-12,rtol=0))
cv_table=pd.DataFrame({'Feature':Xdesc.columns,'Mean':means.values,'Sample SD (ddof=1)':sds.values,'CV':cv_values.values}).sort_values('CV',ascending=False,na_position='last').reset_index(drop=True)
table(cv_table,'coefficient_of_variation')
say(f'Undefined CVs: **{cv_values.isna().sum()}**; constant training predictors: **{(sds==0).sum()}**. All 122 coefficients are reported above and in results/coefficient_of_variation.csv.')
''')
md('''## 2(e) The 11 highest-CV predictors: scatter plots and box plots
The count is floor(sqrt(128))=11 as specified, even though only 122 columns are predictive. These plots use the same imputed training rows. Each scatter plot has a predictor on the horizontal axis and ViolentCrimesPerPop on the vertical axis; each box plot summarizes one predictor. The 11-feature selection does not restrict subsequent regression models.

Scatter patterns can suggest skew, outliers, nonlinear associations, and heteroskedasticity. Neither scatter plots nor box plots establish statistical significance. Formal inference would require a specified null, a suitable model and assumptions, uncertainty estimates and multiple-comparison handling. Large CV alone does not imply a significant relationship with the target, and these observational data cannot establish causal effects.''')
code('''
top=cv_table.loc[cv_table['CV'].notna(),'Feature'].head(int(np.floor(np.sqrt(128)))).tolist()
assert len(top)==11
table(cv_table.head(11),'top11_cv')
fig,axes=plt.subplots(4,3,figsize=(15,16),constrained_layout=True)
for ax,f in zip(axes.flat,top):
    ax.scatter(Xdesc[f],ytrain,s=8,alpha=.3); ax.set(xlabel=f,ylabel=target)
for ax in list(axes.flat)[len(top):]: ax.axis('off')
fig.suptitle('2(e): Highest-CV predictors versus target (training rows)')
figure(fig,'top11_scatter','Scatter plots for the 11 highest training CV predictors; y is always the crime target.')
fig,axes=plt.subplots(4,3,figsize=(15,14),constrained_layout=True)
for ax,f in zip(axes.flat,top):
    ax.boxplot(Xdesc[f],orientation='horizontal'); ax.set(xlabel=f,yticks=[])
for ax in list(axes.flat)[len(top):]: ax.axis('off')
fig.suptitle('2(e): Predictor distributions (training rows)')
figure(fig,'top11_boxplots','Individual horizontal box plots of the selected predictors.')
''')
md('''## 2(f) Ordinary least squares
OLS fits an intercept and all 122 imputed predictors by minimizing residual sum of squares. It has no tuned hyperparameter. Coefficients can be unstable under multicollinearity; the matrix rank and condition number below describe this concern. Main test error is MSE = mean((y-yhat)^2), with RMSE=sqrt(MSE) and R-squared also reported. All model test results are deliberately collected together after Question 2(j) has frozen every fitted model.''')
code('''
models={}; searches={}; selected={}; tuning_rows=[]
models['OLS']=make_pipe(LinearRegression()).fit(Xtrain,ytrain)
centered=Xdesc-Xdesc.mean()
singular=np.linalg.svd(centered.to_numpy(),compute_uv=False)
say(f'Centered training design rank = **{np.linalg.matrix_rank(centered)} / 122**; condition number = **{singular[0]/singular[-1]:.3g}**.')
def tune(name,pipe,param,values,expand=True):
    values=np.unique(np.asarray(values)).tolist()
    history=[]
    for attempt in range(5):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            search=GridSearchCV(pipe,{param:values},scoring='neg_mean_squared_error',cv=cv,n_jobs=1,error_score='raise',refit=True,return_train_score=False)
            search.fit(Xtrain,ytrain)
        convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
        if convergence:
            raise RuntimeError(f'{name}: convergence warnings; increase max_iter or revise solver before reporting results: '+convergence[0])
        other=[str(w.message) for w in caught]
        best=search.best_params_[param]
        history.append({'attempt':attempt+1,'min':min(values),'max':max(values),'count':len(values),'best':best,'warnings':other})
        at_low=best==min(values); at_high=best==max(values)
        if not expand or not (at_low or at_high): break
        if at_low and min(values)==0: break # natural nonnegative boundary
        if at_low:
            smallest=min(v for v in values if v>0)
            values=sorted(set(values+np.geomspace(smallest/100,smallest,9).tolist()))
        if at_high:
            largest=max(values)
            values=sorted(set(values+np.geomspace(largest,largest*100,9).tolist()))
    else: raise RuntimeError(f'{name}: unresolved grid boundary after expansion')
    results=pd.DataFrame(search.cv_results_)
    results['mean_CV_MSE']=-results['mean_test_score']
    results.to_csv(ROOT/'results'/f'cv_{name}.csv',index=False)
    models[name]=search.best_estimator_; searches[name]=search
    tuning_rows.append({'Model':name,'Parameter':param,'Selected':best,'CV MSE':-search.best_score_,'Grid min':min(values),'Grid max':max(values),'Candidates':len(values),'Convergence warnings':len(convergence)})
    (ROOT/'results'/f'search_audit_{name}.json').write_text(json.dumps(history,indent=2,default=float))
    say(f'**{name}:** {param} = **{best}**; five-fold CV MSE = **{-search.best_score_:.8f}**; final grid [{min(values)}, {max(values)}], {len(values)} candidates. Convergence warnings: **{len(convergence)}**. '+('The optimum is at the natural zero-penalty boundary, which cannot be extended downward.' if best==0 and min(values)==0 else 'The selected value is interior to the final search range.' if best not in [min(values),max(values)] else 'All admissible component counts are searched; a boundary here is a legitimate endpoint.'))
    return search
''')
md('''## 2(g) Ridge regression
A Pipeline applies median imputation and StandardScaler before Ridge. Scikit-learn Ridge minimizes ||y-Xb||^2 + alpha*||b||^2, with an unpenalized intercept. Thus its alpha is the assignment's lambda under this convention. A 49-point logarithmic grid from 1e-6 through 1e6 is searched using training-fold MSE; if an artificial endpoint wins, the range is expanded. Standardizing equalizes penalty treatment across features, despite their existing [0,1] normalization.''')
code('''
tune('Ridge',make_pipe(Ridge(solver='svd'),scale=True),'model__alpha',np.logspace(-6,6,49))
''')
md('''## 2(h) LASSO without and with standardization
Both models use median imputation inside CV. The first retains the original supplied normalized units; the second applies fold-fitted StandardScaler. Lasso minimizes ||y-Xb||^2/(2*n) + alpha*||b||_1 with an unpenalized intercept, so numerical alpha is not directly comparable with Ridge alpha. Each model searches 41 logarithmically spaced values from 1e-5 to 1e-1, expanding artificial boundaries if needed. A strict tolerance of 1e-7 and max_iter=200000 are used; convergence warnings cause execution to fail instead of being hidden. Selected variables have absolute fitted coefficient greater than 1e-10. Coefficients for the standardized fit are expressed per training standard deviation; original-unit equivalents are also exported.''')
code('''
for name,scale in [('LASSO_unscaled',False),('LASSO_scaled',True)]:
    tune(name,make_pipe(Lasso(max_iter=200000,tol=1e-7,selection='cyclic'),scale=scale),'model__alpha',np.logspace(-5,-1,41))
    coef=models[name].named_steps['model'].coef_
    mask=np.abs(coef)>1e-10
    selected[name]=Xtrain.columns[mask].tolist()
    original=coef/models[name].named_steps['scale'].scale_ if scale else coef
    coeffs=pd.DataFrame({'Feature':Xtrain.columns,'Fitted coefficient':coef,'Original-unit coefficient':original,'Selected':mask})
    coeffs.to_csv(ROOT/'results'/f'coefficients_{name}.csv',index=False)
    table(coeffs.loc[mask,['Feature','Fitted coefficient','Original-unit coefficient']].reset_index(drop=True),f'selected_{name}')
    fit=models[name].named_steps['model']
    say(f'**{name} selects {mask.sum()} variables:** '+', '.join(selected[name])+f'. Final solver iterations: {fit.n_iter_}; dual gap: {fit.dual_gap_:.3g}.')
say('Standardization changes the effective L1 penalty in original units: a unit change in a low-variance feature is treated differently after scaling. Hence the fitted supports and optimal penalties can differ. Correlated predictors may substitute for each other, so an excluded variable is not necessarily irrelevant. The test-error comparison follows in the frozen evaluation table.')
''')
md('''## 2(i) Principal component regression (PCR)
The Pipeline is median imputation -> standardization -> full-SVD PCA -> ordinary least squares. M is selected by the same five training folds and validation MSE. All integer values 1 through 122 are searched, so neither endpoint is an artificial truncated search boundary. PCA directions, imputation medians and scaling statistics are refitted within each fold. PCA maximizes predictor variance, not association with the response; a high explained-variance fraction alone does not determine the best M.''')
code('''
tune('PCR',make_pipe(LinearRegression(),scale=True,pca=True),'pca__n_components',np.arange(1,123),expand=False)
pcr=models['PCR'].named_steps['pca']
say(f'The selected {pcr.n_components_} components explain **{pcr.explained_variance_ratio_.sum():.4%}** of standardized training predictor variance.')
''')
md('''## 2(j) L1-penalized boosting with XGBoost
Implementation assumption: interpret the allowed XGBoost option as **booster="gbtree"** with a CV-selected **reg_alpha**, an L1 penalty on scalar leaf weights. Standard tree leaves predict constants; they do not fit LASSO regressions at nodes. This differs from a literal tree with local multivariate L1 regressions. The standard gbtree implementation fulfills the explicit XGBoost option under this stated interpretation; no unsupported local-linear-tree claim is made. gblinear would fit a linear booster and would not be a boosting tree.

For a fixed tree structure, the regularizer includes reg_alpha*sum(|w_j|) + (reg_lambda/2)*sum(w_j^2). The L1 term can shrink leaf weights to zero and affects split gains. It is not direct coefficient selection over the original 122 features. See the official [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html) and [boosted-tree introduction](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

The fixed, predeclared configuration uses squared-error loss, 400 trees, learning_rate=0.04, max_depth=3, min_child_weight=5, reg_lambda=1, subsample=1, colsample_bytree=1, gamma=0, hist trees, n_jobs=1 and seed 552. Only reg_alpha is tuned: zero plus 29 positive log-spaced candidates from 1e-5 through 1e2, expanding if necessary. Median imputation remains inside the Pipeline for consistent data handling. No test-set early stopping is used; other tree settings are fixed modeling assumptions, not claimed globally optimal.''')
code('''
xgb=XGBRegressor(booster='gbtree',objective='reg:squarederror',n_estimators=400,
    learning_rate=.04,max_depth=3,min_child_weight=5,reg_lambda=1,subsample=1,
    colsample_bytree=1,gamma=0,tree_method='hist',random_state=SEED,n_jobs=1,verbosity=0)
tune('XGBoost',make_pipe(xgb),'model__reg_alpha',np.r_[0,np.logspace(-5,2,29)])
table(pd.DataFrame(tuning_rows),'tuning_summary')
# Freeze the six final fitted pipelines before any test prediction.
assert len(models)==6
frozen_names=tuple(models)
rows=[]
for name in frozen_names:
    prediction=models[name].predict(Xtest)
    mse=mean_squared_error(ytest,prediction)
    params=searches[name].best_params_ if name in searches else {'method':'least squares'}
    rows.append({'Model':name,'Selected parameter':json.dumps(params),'CV MSE':-searches[name].best_score_ if name in searches else np.nan,'Test MSE':mse,'Test RMSE':np.sqrt(mse),'Test R2':r2_score(ytest,prediction),'Selected variables':len(selected[name]) if name in selected else 'N/A'})
    pd.DataFrame({'Original row (1-based)':Xtest.index+1,'Observed':ytest,'Predicted':prediction}).to_csv(ROOT/'results'/f'predictions_{name}.csv',index=False)
metrics=pd.DataFrame(rows)
table(metrics,'model_comparison')
for row in rows:
    say(f'**{row["Model"]}: test MSE = {row["Test MSE"]:.8f}; RMSE = {row["Test RMSE"]:.8f}; R-squared = {row["Test R2"]:.6f}.**')
a=metrics.set_index('Model')
delta=a.loc['LASSO_scaled','Test MSE']-a.loc['LASSO_unscaled','Test MSE']
say(f'Standardized LASSO selected **{len(selected["LASSO_scaled"])}** variables versus **{len(selected["LASSO_unscaled"])}** without standardization. Its test MSE is **{abs(delta):.8f} {"lower" if delta<0 else "higher"}**. This is a final held-out comparison, not a reason to retune. Lowest observed test MSE among the prespecified fits: **{metrics.loc[metrics["Test MSE"].idxmin(),"Model"]}**. This descriptive ranking on one fixed split does not establish a statistically reliable winner or estimate the performance of a newly selected model.')
fig,ax=plt.subplots(figsize=(10,5)); ax.bar(metrics['Model'],metrics['Test MSE'],color=sns.color_palette('deep',6)); ax.set(ylabel='Test mean squared error',title='Final held-out comparison: original rows 1496-1994'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); figure(fig,'model_test_mse','One final test evaluation for each frozen model. No test-based tuning.')
# CV plots are based on training folds, independent of held-out outcomes.
fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for ax,(name,search) in zip(axes.flat,searches.items()):
    param=list(search.best_params_)[0]
    xx=np.array([float(p[param]) for p in search.cv_results_['params']]); yy=-search.cv_results_['mean_test_score']
    order=np.argsort(xx); ax.plot(xx[order],yy[order],'.-',ms=3)
    if name!='PCR': ax.set_xscale('symlog',linthresh=1e-5)
    ax.axvline(float(search.best_params_[param]),color='red',ls='--'); ax.set(title=name,xlabel=param,ylabel='Five-fold CV MSE')
figure(fig,'cv_curves','Training-only hyperparameter selection curves. Dashed lines mark selected settings.')
''')
md('''## Reproducibility, validation and limitations
The notebook must run from the project root. All input and output paths are relative. Pinned dependencies, raw data hashes, fold assignments, full search results, LASSO coefficient tables, rules and test predictions are included. Every code cell is executed in order by a fresh Jupyter kernel using scripts/run_notebook.py. scripts/make_report.py creates the English PDF from the same recorded narrative, tables and figures, so the report is tied to computed outputs.

The original nonrandom external split may have distribution shifts; shuffled internal CV assumes exchangeability within the training portion. It does not account for possible geographic or social dependence. Median filling of heavily missing police variables is an information limitation, not a claim that missingness is ignorable. The 11 highest-CV features are not a formal inferential screen. Model rankings are specific to this split and stated configurations. Hyperparameter searches resolve grid endpoints, but do not guarantee a continuous global optimum.

No additional course submission naming, upload, repository, or packaging rules appear in the supplied homework PDF. Only the supplied instructions are asserted here.''')
code('''
checklist=[('1(a)','120 rows, six features, two diagnoses; local data parsed'),('1(b)','Full-data fitted tree, complete plot and label mapping'),('1(c)','Every full-tree root-to-leaf rule; exact prediction agreement'),('1(d)','Cost-complexity CV, complexity/performance comparison, selected tree and all rules'),('2(a)','1994 x 128; original first 1495/last 499 split'),('2(b)','Five identifiers removed; fold-local median imputation; 122 predictors retained'),('2(c)','Full predictor correlation matrix plus readable panels'),('2(d)','All predictor CVs; ddof=1; training-only scope and edge cases'),('2(e)','Exactly 11 features; scatter and box plots; significance answer'),('2(f)','OLS fitted; test MSE, RMSE, R2'),('2(g)','Ridge CV alpha; test metrics'),('2(h)','Both LASSOs; CV alpha, support names/counts, convergence, comparison'),('2(i)','PCA plus OLS; all M searched by fold-local CV; test metrics'),('2(j)','gbtree/reg_alpha CV; explicit interpretation; test metrics')]
assert all(np.isfinite(metrics[c]).all() for c in ['Test MSE','Test RMSE','Test R2'])
assert all(p.named_steps['impute'].n_features_in_==122 for p in models.values())
assert len(full_rules)==full.get_n_leaves() and len(pruned_rules)==pruned.get_n_leaves()
assert set(top).issubset(set(Xtrain.columns)) and target not in Xtrain.columns
assert all(len(set(tr)&set(va))==0 for tr,va in cv)
table(pd.DataFrame([{'Item':q,'Verified requirement':s,'Status':'Complete'} for q,s in checklist]),'completion_checklist')
say('**Outstanding assignment items: none.** A literal local-LASSO model tree is not claimed; the documented standard-XGBoost interpretation of 2(j) is used. Unprovided course submission rules cannot be verified. Final PDF layout and fresh-kernel execution are additionally checked by the packaging workflow.')
(ROOT/'results'/'report_content.json').write_text(json.dumps(report,indent=2,allow_nan=False))
(ROOT/'results'/'execution_checks.json').write_text(json.dumps({'all_assignment_items_complete':True,'rows_acute':120,'features_acute':6,'labels_acute':2,'communities_shape':list(communities.shape),'train_rows':len(Xtrain),'test_rows':len(Xtest),'predictors':122,'top_cv_features':top,'rule_agreement_checked':True,'test_evaluations_per_model':1,'convergence_warnings':0,'seed':SEED},indent=2))
print('All computational checks passed; report content and reproducibility artifacts written.')
''')
md('''## References
- Supplied Homework5.pdf, Questions 1-2 and footnotes 1-5.
- Supplied diagnosis.names and diagnosis.data. J. Czerniak and H. Zarzycki, Application of rough sets in the presumptive diagnosis of urinary system diseases, ACS 2002, 2003, pp. 41-51.
- Supplied communities.names and communities.data, UCI Machine Learning Repository; source data: US Census 1990, LEMAS 1990, FBI Crime in the United States 1995. M. A. Redmond and A. Baveja, A Data-Driven Software Tool for Enabling Cooperative Information Sharing Among Police Departments, European Journal of Operational Research 141 (2002), 660-678.
- Matthew Mayo, Simplifying Decision Tree Interpretability with Python & Scikit-learn, KDnuggets (2017), supplied PDF. The path traversal here is a fresh implementation with label decoding and prediction assertions.
- [Scikit-learn Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and [Lasso documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), for objective conventions.
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) and [boosted-tree introduction](https://xgboost.readthedocs.io/en/stable/tutorials/model.html), for booster and leaf-weight regularization semantics.''')
# Include final references in the report after their registration cell runs.
code("(ROOT/'results'/'report_content.json').write_text(json.dumps(report,indent=2,allow_nan=False))\nprint('Report content finalized.')")
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.write(nb,root/'DSCI552_Homework5.ipynb')
print('Notebook generated:',len(cells),'cells')
