import json
import pandas as pd
from .config import TABLES, FIGURES
from .data import load_raw, audit_data, clean_data, save_processed
from .modeling import fit_full_lda, holdout_evaluation, cross_validation, separation_summary
from .visualization import plot_class_balance, plot_lda_projection, plot_discriminant_weights, plot_confusion_matrix

def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    raw=load_raw(); audit=audit_data(raw); clean=clean_data(raw); save_processed(clean)
    scaler,lda,Z,coef,means,scalings=fit_full_lda(clean)
    _,holdout,cm,report,yte,pred=holdout_evaluation(clean)
    cv=cross_validation(clean,10)
    explained=lda.explained_variance_ratio_
    descriptive=clean.describe().T
    class_means=clean.groupby("cultivar").mean().T
    separation=separation_summary(Z)
    audit_df=pd.DataFrame([audit]); audit_df.to_csv(TABLES/'data_audit.csv',index=False)
    descriptive.to_csv(TABLES/'descriptive_statistics.csv')
    class_means.to_csv(TABLES/'class_feature_means.csv')
    Z.to_csv(TABLES/'lda_coordinates.csv',index=False)
    coef.to_csv(TABLES/'lda_class_coefficients_standardized.csv')
    means.to_csv(TABLES/'lda_class_means_standardized.csv')
    scalings.to_csv(TABLES/'lda_scalings.csv')
    separation.to_csv(TABLES/'lda_separation_summary.csv')
    pd.DataFrame(cm,index=['true_1','true_2','true_3'],columns=['pred_1','pred_2','pred_3']).to_csv(TABLES/'confusion_matrix.csv')
    report.to_csv(TABLES/'classification_report.csv')
    pd.DataFrame([{**holdout,"cv_accuracy_mean":cv.mean(),"cv_accuracy_std":cv.std(ddof=1),"ld1_ratio":explained[0],"ld2_ratio":explained[1],"cumulative_ratio":explained.sum()}]).to_csv(TABLES/'model_metrics.csv',index=False)
    summary={"audit":audit,"lda":{"original_dimensions":13,"reduced_dimensions":2,"explained_discriminant_variance":[float(x) for x in explained],"cumulative":float(explained.sum())},"holdout":{k:float(v) if hasattr(v,'item') or isinstance(v,float) else int(v) for k,v in holdout.items()},"cross_validation":{"folds":10,"accuracy_mean":float(cv.mean()),"accuracy_std":float(cv.std(ddof=1))}}
    (TABLES/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot_class_balance(clean); plot_lda_projection(Z,explained); plot_discriminant_weights(scalings); plot_confusion_matrix(cm)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
