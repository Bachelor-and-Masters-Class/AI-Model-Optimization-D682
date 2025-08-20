
import os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
RANDOM_SEED=42; np.random.seed(RANDOM_SEED)
DATA_CSV=os.environ.get("DATA_CSV","data/demo_classification.csv"); TARGET_COL=os.environ.get("TARGET_COL","target")
OUT=Path("reports"); OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA_CSV); X=df.drop(columns=[TARGET_COL]); y=df[TARGET_COL].astype(int)
X_trv,X_te,y_trv,y_te=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RANDOM_SEED)
baseline=Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(penalty='l2',solver='saga',max_iter=2000,random_state=RANDOM_SEED))])
baseline.fit(X_trv,y_trv); pb=baseline.predict_proba(X_te)[:,1]; yb=(pb>=0.5).astype(int)
def E(y_true,y_pred,y_proba): 
    return {'accuracy':float(accuracy_score(y_true,y_pred)),
            'precision':float(precision_score(y_true,y_pred,zero_division=0)),
            'recall':float(recall_score(y_true,y_pred,zero_division=0)),
            'f1':float(f1_score(y_true,y_pred,zero_division=0)),
            'roc_auc':float(roc_auc_score(y_true,y_proba))}
rows=[{'model':'baseline_logreg_l2',**E(y_te,yb,pb)}]
pipe=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(solver='saga',max_iter=2000,random_state=RANDOM_SEED))])
param={'lr__penalty':['l2','elasticnet'],'lr__C':np.logspace(-2,2,6),'lr__l1_ratio':[None,0.5],'lr__class_weight':[None,'balanced']}
cv=StratifiedKFold(n_splits=2,shuffle=True,random_state=RANDOM_SEED)
search=RandomizedSearchCV(pipe,param,n_iter=2,scoring='f1',n_jobs=1,cv=cv,random_state=RANDOM_SEED,verbose=0)
search.fit(X_trv,y_trv); best=search.best_estimator_
pl=best.predict_proba(X_te)[:,1]; yl=(pl>=0.5).astype(int)
rows.append({'model':'tuned_logreg',**E(y_te,yl,pl)})
from sklearn.model_selection import train_test_split as tts; from sklearn.metrics import f1_score
X_tr,X_va,y_tr,y_va=tts(X_trv,y_trv,test_size=0.2,random_state=RANDOM_SEED,stratify=y_trv)
best.fit(X_tr,y_tr); pv=best.predict_proba(X_va)[:,1]
ths=np.linspace(0.4,0.6,3); fs=[f1_score(y_va,(pv>=t).astype(int),zero_division=0) for t in ths]; t=float(ths[int(np.argmax(fs))])
pt=best.predict_proba(X_te)[:,1]; yt=(pt>=t).astype(int)
rows.append({'model':f'tuned_logreg@thr={t:.2f}',**E(y_te,yt,pt)})
rf=RandomForestClassifier(n_estimators=80,max_depth=6,min_samples_leaf=5,random_state=RANDOM_SEED,n_jobs=1)
rf.fit(X_trv,y_trv); pr=rf.predict_proba(X_te)[:,1]; yr=(pr>=0.5).astype(int)
rows.append({'model':'random_forest_regconstrained',**E(y_te,yr,pr)})
def make_bag_clf(base, **kwargs):
    try: return BaggingClassifier(estimator=base, **kwargs)
    except TypeError: return BaggingClassifier(base_estimator=base, **kwargs)
bag=make_bag_clf(best,n_estimators=6,max_samples=0.8,max_features=0.8,bootstrap=True,n_jobs=1,random_state=RANDOM_SEED)
bag.fit(X_trv,y_trv); pbg=bag.predict_proba(X_te)[:,1]; ybg=(pbg>=0.5).astype(int)
rows.append({'model':'bagging(tuned_logreg)',**E(y_te,ybg,pbg)})
gb=GradientBoostingClassifier(random_state=RANDOM_SEED)
stack=StackingClassifier(estimators=[('rf',rf),('gb',gb)],final_estimator=LogisticRegression(max_iter=2000))
stack.fit(X_trv,y_trv); ps=stack.predict_proba(X_te)[:,1]; ys=(ps>=0.5).astype(int)
rows.append({'model':'stacking(rf+gb)->logreg',**E(y_te,ys,ps)})
import pandas as pd
tbl=pd.DataFrame(rows).sort_values('f1',ascending=False); tbl.to_csv(OUT/'comparison_before_after.csv',index=False)
from sklearn.metrics import confusion_matrix
def save_cm(y_true,y_pred,name):
    cm=confusion_matrix(y_true,y_pred); pd.DataFrame(cm,index=['True 0','True 1'],columns=['Pred 0','Pred 1']).to_csv(OUT/f'cm_{name}.csv',index=True)
save_cm(y_te,yb,'baseline'); save_cm(y_te,yl,'tuned_lr_0.50'); save_cm(y_te,yt,f'tuned_lr_thr{t:.2f}'); save_cm(y_te,yt,'tuned_lr_thrXX'); save_cm(y_te,yr,'rf'); save_cm(y_te,ybg,'bag'); save_cm(y_te,ys,'stack')
with open(OUT/'summary.json','w') as f: json.dump({'best_search_params':search.best_params_,'best_threshold':t,'models_sorted_by_f1':tbl['model'].tolist()},f,indent=2)
print('classification done')
