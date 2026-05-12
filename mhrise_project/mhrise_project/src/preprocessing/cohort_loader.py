# load cohort year files
import pandas as pd

def load_cohort(year):
    return pd.read_parquet(f'data/raw/survey/baseline/baseline_intake_FY{year}_v3.parquet')
