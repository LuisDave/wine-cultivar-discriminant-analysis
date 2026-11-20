from src.data import load_raw, clean_data, audit_data
from src.modeling import fit_full_lda, cross_validation

def test_dataset_contract():
    df=clean_data(load_raw())
    a=audit_data(df)
    assert a["rows"] == 178
    assert a["predictors"] == 13
    assert a["classes"] == 3
    assert a["missing_cells"] == 0

def test_lda_reduces_to_two_dimensions():
    df=clean_data(load_raw())
    _,lda,z,*_=fit_full_lda(df)
    assert list(z.columns[:2]) == ["LD1","LD2"]
    assert len(lda.explained_variance_ratio_) == 2
    assert lda.explained_variance_ratio_.sum() > 0.99

def test_cross_validated_accuracy_is_strong():
    df=clean_data(load_raw())
    scores=cross_validation(df,5)
    assert scores.mean() > 0.90
