import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from .config import FEATURES, TARGET, RANDOM_STATE

def fit_full_lda(df):
    X=df[FEATURES]
    y=df[TARGET]
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    lda=LinearDiscriminantAnalysis(solver="svd", n_components=2)
    Z=lda.fit_transform(Xs,y)
    transformed=pd.DataFrame(Z, columns=["LD1","LD2"], index=df.index)
    transformed[TARGET]=y.values
    # coefficients in standardized feature space: one row per class
    coef=pd.DataFrame(lda.coef_, index=[f"class_{c}" for c in lda.classes_], columns=FEATURES)
    means=pd.DataFrame(lda.means_, index=[f"class_{c}" for c in lda.classes_], columns=FEATURES)
    scalings=pd.DataFrame(lda.scalings_[:, :2], index=FEATURES, columns=["LD1","LD2"])
    return scaler, lda, transformed, coef, means, scalings

def holdout_evaluation(df, test_size=0.25):
    X=df[FEATURES]; y=df[TARGET]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,stratify=y,random_state=RANDOM_STATE)
    pipe=Pipeline([("scale",StandardScaler()),("lda",LinearDiscriminantAnalysis())])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    metrics={
        "accuracy": accuracy_score(yte,pred),
        "balanced_accuracy": balanced_accuracy_score(yte,pred),
        "macro_f1": f1_score(yte,pred,average="macro"),
        "n_train": len(ytr), "n_test": len(yte)
    }
    cm=confusion_matrix(yte,pred,labels=sorted(y.unique()))
    report=pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T
    return pipe, metrics, cm, report, yte, pred

def cross_validation(df, folds=10):
    X=df[FEATURES]; y=df[TARGET]
    pipe=Pipeline([("scale",StandardScaler()),("lda",LinearDiscriminantAnalysis())])
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=RANDOM_STATE)
    scores=cross_val_score(pipe,X,y,cv=cv,scoring="accuracy")
    return scores

def separation_summary(transformed):
    return transformed.groupby(TARGET)[["LD1","LD2"]].agg(["mean","std"])
