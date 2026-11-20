from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import FIGURES, TARGET, FEATURES

def _save(fig,name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIGURES/name,dpi=180,bbox_inches="tight"); plt.close(fig)

def plot_class_balance(df):
    counts=df[TARGET].value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(6.5,4.2)); ax.bar(counts.index.astype(str),counts.values)
    ax.set(title="Cultivar class distribution",xlabel="Cultivar",ylabel="Observations")
    for i,v in enumerate(counts.values): ax.text(i,v+1,str(v),ha="center")
    _save(fig,"class_balance.png")

def plot_lda_projection(transformed, explained):
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for c,g in transformed.groupby(TARGET):
        ax.scatter(g.LD1,g.LD2,label=f"Cultivar {c}",alpha=.8,s=38)
    ax.set(xlabel=f"LD1 ({explained[0]*100:.1f}% discriminant variance)",ylabel=f"LD2 ({explained[1]*100:.1f}% discriminant variance)",title="Wine samples projected into the LDA space")
    ax.legend(); ax.axhline(0,lw=.6); ax.axvline(0,lw=.6)
    _save(fig,"lda_projection.png")

def plot_discriminant_weights(scalings):
    strength=(scalings**2).sum(axis=1).pow(.5).sort_values(ascending=True)
    fig,ax=plt.subplots(figsize=(7.2,5.4)); ax.barh(strength.index,strength.values)
    ax.set(title="Feature contribution to the discriminant space",xlabel="Euclidean magnitude across LD1 and LD2")
    _save(fig,"feature_discriminant_strength.png")

def plot_confusion_matrix(cm):
    fig,ax=plt.subplots(figsize=(5.2,4.6)); im=ax.imshow(cm)
    ax.set(title="Holdout confusion matrix",xlabel="Predicted cultivar",ylabel="True cultivar",xticks=range(3),yticks=range(3),xticklabels=[1,2,3],yticklabels=[1,2,3])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04); _save(fig,"confusion_matrix.png")
