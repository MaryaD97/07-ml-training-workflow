from src.preprocess import preprocess_data

def test_preprocess():
    df = preprocess_data("data/sample.csv")
    
    assert not df.empty
    assert df.isnull().sum().sum() == 0
