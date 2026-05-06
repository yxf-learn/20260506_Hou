Flagged data issues and their resolutions
==========================================

2023-11-04  Cohort 2014, 14 records with duplicated student_id values.
            Root cause: twin-sibling SSN confusion in registrar export.
            Action: dropped in 02b_dedup_students.R (all 14 re-verified
            with registrar as distinct individuals; corrected IDs
            re-ingested 2024-01-09).

2023-11-12  EFC = 0 observed for 184 records. Verified: these are
            genuine zero-EFC students (Pell-eligible by rule).
            Retained with need_index = 0.0.

2024-01-09  Cohort 2014, 3 records with cohort field encoded as "FA14".
            Fixed in 02c_fix_cohort_encoding.R.

2024-02-22  2 FAFSA records with missing EFC fields (manual entry error
            2014-03-18 batch). Imputed via family-income regression
            for robustness checks only; excluded from main analysis.

2024-07-08  institution_type capitalisation inconsistent across years
            ("Research" vs "research"). Normalised to lower case.

2024-09-30  test_percentile of 0 for 47 records (did not take ACT/SAT;
            admitted via alternative pathway). Imputed with cohort-year
            median for the baseline specification; sensitivity to this
            choice reported in Appendix B.
