
import os, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor
import warnings; from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
RANDOM_SEED=42; np.random.seed(RANDOM_SEED)
DATA_CSV=os.environ.get('DATA_CSV','data/demo_regression.csv'); TARGET_COL=os.environ.get('TARGET_COL','target')
OUT=Path('reports'); OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA_CSV); X=df.drop(columns=[TARGET_COL]); y=df[TARGET_COL].astype(float)
X_trv,X_te,y_trv,y_te=train_test_split(X,y,test_size=0.2,random_state=RANDOM_SEED)
baseline=Pipeline([('sc',StandardScaler()),('en',ElasticNet(random_state=RANDOM_SEED,max_iter=10000))])
baseline.fit(X_trv,y_trv); pb=baseline.predict(X_te)
def E(y_true,y_pred):
    import numpy as _np
    return {'mae':float(mean_absolute_error(y_true,y_pred)),'rmse':float(_np.sqrt(mean_squared_error(y_true,y_pred))),'r2':float(r2_score(y_true,y_pred))}
rows=[{'model':'baseline_elasticnet',**E(y_te,pb)}]
pipe=Pipeline([('sc',StandardScaler()),('en',ElasticNet(random_state=RANDOM_SEED,max_iter=10000))])
param={'en__alpha':np.logspace(-2,2,6),'en__l1_ratio':np.linspace(0.0,1.0,5)}
cv=KFold(n_splits=2,shuffle=True,random_state=RANDOM_SEED)
search=RandomizedSearchCV(pipe,param,n_iter=2,scoring='neg_mean_absolute_error',n_jobs=1,cv=cv,random_state=RANDOM_SEED,verbose=0)
search.fit(X_trv,y_trv); best=search.best_estimator_; pe=best.predict(X_te); rows.append({'model':'tuned_elasticnet',**E(y_te,pe)})
from sklearn.model_selection import train_test_split as tts
X_tr,X_va,y_tr,y_va=tts(X_trv,y_trv,test_size=0.2,random_state=RANDOM_SEED)
best.fit(X_tr,y_tr); pv=best.predict(X_va); A=np.vstack([pv,np.ones_like(pv)]).T; a,b=np.linalg.lstsq(A,y_va.values,rcond=None)[0]
pbc=a*best.predict(X_te)+b; rows.append({'model':'tuned_elasticnet_bias_corrected',**E(y_te,pbc)})
rf=RandomForestRegressor(n_estimators=80,max_depth=6,min_samples_leaf=5,random_state=RANDOM_SEED,n_jobs=1)
rf.fit(X_trv,y_trv); pr=rf.predict(X_te); rows.append({'model':'rf_regconstrained',**E(y_te,pr)})
gb=GradientBoostingRegressor(random_state=RANDOM_SEED)
def make_bag_reg(base, **kwargs):
    try: return BaggingRegressor(estimator=base, **kwargs)
    except TypeError: return BaggingRegressor(base_estimator=base, **kwargs)
bag=make_bag_reg(best,n_estimators=6,max_samples=0.8,max_features=0.8,bootstrap=True,random_state=RANDOM_SEED)
bag.fit(X_trv,y_trv); pbg=bag.predict(X_te); rows.append({'model':'bagging(tuned_en)',**E(y_te,pbg)})
stack=StackingRegressor(estimators=[('rf',rf),('gb',gb)],final_estimator=ElasticNet(random_state=RANDOM_SEED,max_iter=10000))
stack.fit(X_trv,y_trv); ps=stack.predict(X_te); rows.append({'model':'stacking(rf+gb)->en',**E(y_te,ps)})
import pandas as pd
tbl=pd.DataFrame(rows).sort_values('mae',ascending=True); tbl.to_csv(OUT/'comparison_before_after_regression.csv',index=False)
with open(OUT/'summary_regression.json','w') as f: json.dump({'best_search_params':search.best_params_,'bias_correction':{'a':float(a),'b':float(b)},'models_sorted_by_mae':tbl['model'].tolist()},f,indent=2)
print('regression done')
