import os
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, TimeSeriesSplit, learning_curve
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
from understatapi import UnderstatClient

# Ensure Figures directory exists (one level deeper)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'Figures')
if not os.path.isdir(FIGURES_DIR): os.makedirs(FIGURES_DIR)

current_gw = 34

def train_model(mode: str):
    print(f"\n=== Training {mode}-term model ===")
    # Fetch data
    bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    df = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])
    team_map = dict(zip(teams['id'], teams['name']))

    # Enrich
    df['web_name']   = df['web_name'].astype(str)
    df['team_name']  = df['team'].map(team_map)
    df['minutes']    = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
    df = df[df['minutes'] > 0]
    df['position']   = df['element_type'].map({1:'GK',2:'DEF',3:'MID',4:'FWD'})

    # Form and team_form
    df['form']       = pd.to_numeric(df['form'], errors='coerce').fillna(0)
    df['team_form']  = df.groupby('team')['form'].transform('mean')

    # Fixture difficulty
    fixtures         = pd.DataFrame(requests.get("https://fantasy.premierleague.com/api/fixtures/").json())
    curr             = fixtures['event'].dropna().min()
    up               = fixtures[(fixtures['event'] >= curr) & (fixtures['event'] < curr+3)]
    diff_map = {t: (up[(up['team_h']==t)|(up['team_a']==t)]
                     .apply(lambda r: r['team_h_difficulty'] if r['team_h']==t else r['team_a_difficulty'], axis=1)
                     .mean())
                for t in df['team'].unique()}
    df['fixture_difficulty'] = df['team'].map(diff_map).fillna(0)

    # Mode-specific
    if mode=='short':
        df['ep_next']  = pd.to_numeric(df.get('event_points',0), errors='coerce').fillna(0)
        df['form']    *= .5
        target = 'ep_next'
        features = ['form','ict_index','influence','creativity','threat',
                    'minutes','goals_scored','assists','clean_sheets',
                    'yellow_cards','red_cards','team_form','fixture_difficulty']
    elif mode=='long':
        df['total_points'] = pd.to_numeric(df.get('total_points',0), errors='coerce').fillna(0)
        df['ep_this']      = df['total_points']
        df['avg_per_gw']   = df['ep_this'] / current_gw
        df['ep_next']      = pd.to_numeric(df.get('event_points',0), errors='coerce').fillna(0)
        rem = max(1, 38-current_gw)
        df['future_points']= ((df['avg_per_gw']+df['ep_next'])/2)*rem
        target = 'future_points'
        features = ['goals_scored','assists','ict_index','influence','creativity',
                    'minutes','team_form','fixture_difficulty']
        # Understat merge
        us = UnderstatClient()
        xd=[]
        for v in us.league('EPL').get_team_data(season='2024').values():
            h=v.get('history',[])
            xd.append({'team':v['title'].lower(),
                       'xG':sum(float(g.get('xG',0)) for g in h),
                       'xGA':sum(float(g.get('xGA',0)) for g in h)})
        xdf=pd.DataFrame(xd)
        df['team_lower']=df['team_name'].str.lower()
        df=df.merge(xdf,left_on='team_lower',right_on='team',how='left')
        df['xG']=pd.to_numeric(df['xG'],errors='coerce').fillna(0)
        df['xGA']=pd.to_numeric(df['xGA'],errors='coerce').fillna(0)
        features+=['xG','xGA']
    else:
        raise ValueError("mode must be 'short' or 'long'")

    # Build matrix
    keep_cols = features+[target,'web_name'] + (['ep_this'] if mode=='long' else [])
    clean = df[keep_cols].dropna()
    X,y = clean[features], clean[target]

    # CV
    cv_main = KFold(5,shuffle=True,random_state=42) if mode=='short' else TimeSeriesSplit(5)
    scores = -cross_val_score(RandomForestRegressor(n_estimators=100,random_state=42), X,y,cv=cv_main, scoring='neg_mean_absolute_error')
    print(f"[{mode}] CV MAE: {scores.mean():.3f} ± {scores.std():.3f}")

    # Hold-out
    Xt,Xv,yt,yv = train_test_split(X,y,test_size=0.2,shuffle=False,random_state=42)
    mdl=RandomForestRegressor(n_estimators=100,random_state=42).fit(Xt,yt)
    yp=mdl.predict(Xv)
    if mode=='long':
        epv = clean.iloc[Xv.index]['ep_this'].values
        true, pred = epv+yv, epv+yp
    else:
        true, pred = yv, yp
    print(f"[{mode}] Hold-out MAE: {mean_absolute_error(true,pred):.3f}, RMSE: {np.sqrt(mean_squared_error(true,pred)):.3f}, R²: {r2_score(true,pred):.3f}")

    # Learning curve
    lc_cv = TimeSeriesSplit(5) if mode=='long' else KFold(5,shuffle=True,random_state=42)
    tsizes, tr_s, cv_s = learning_curve(mdl,X,y,cv=lc_cv,train_sizes=np.linspace(0.1,1,5),scoring='neg_mean_absolute_error')
    tr_mae = -tr_s.mean(axis=1); cv_mae=-cv_s.mean(axis=1)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(tsizes,tr_mae,'o-',label='Train');ax.plot(tsizes,cv_mae,'o-',label='CV');ax.set_title(f"Learning Curve ({mode.capitalize()})")
    ax.set_xlabel("Training Set Size");ax.set_ylabel("MAE");ax.legend();fig.savefig(os.path.join(FIGURES_DIR,f"{mode}_learning_curve.png"),dpi=300);plt.close(fig)

    # Feature importances
    imp=mdl.feature_importances_; idx=np.argsort(imp)
    fig,ax=plt.subplots(figsize=(10,6));ax.barh([features[i] for i in idx],imp[idx]);ax.set_title(f"Feature Importances ({mode.capitalize()})")
    ax.set_xlabel("Relative Importance");fig.savefig(os.path.join(FIGURES_DIR,f"{mode}_feature_importance.png"),dpi=300);plt.close(fig)

    # Predictions
    clean['predicted_points']=mdl.predict(X)
    if mode=='short': clean['predicted_next']=clean['predicted_points']
    return mdl, clean

# Entrypoint
if __name__=='__main__':
    train_model('short'); train_model('long')
