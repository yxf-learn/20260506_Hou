Codebook: master_analysis_frame
===============================

Generated: 2024-11-15
Script: 02_code/R/03_merge/03_merge_master.R
Unit of observation: student-cohort

Variable                      Type        Description
-------------------------------------------------------------------------------
student_id                    string      Anonymised identifier (prefix S, 12 char)
cohort                        int         Admission year, 2013-2018
first_gen                     0/1         1 if neither parent completed BA
gender                        F/M/NB      Self-reported
rural_origin                  0/1         ZIP code classified rural per USDA-ERS
race_ethnicity                string      8 categories; see codebook_race.md
home_state                    string      2-char US postal code
institution_id                INST_0N     N in {1..5}
institution_name              string      Full name (redacted in release)
institution_type              research/teaching
hs_gpa                        float       4.0 scale, unweighted
test_percentile               int 1-99    ACT/SAT composite percentile (converted)
birth_year                    int
starting_major                string      First declared major at matriculation
major_category                STEM/Business/Humanities/Social/Health/Other
efc_dollars                   int         Winsorised at 99th pct
need_index                    float 0-10  Running variable; high need = low value
above_cutoff                  0/1         need_index > 3.0
distance_from_cutoff          float       need_index - 3.0
aid_received                  0/1         Treatment variable D
annual_aid_amount             int         $0 if D=0
y1_gpa_mean                   float       Year-1 GPA, averaged over FA+SP terms
y1_credits                    int         Sum of credits earned in year 1
y2_retained                   0/1         Enrolled fall of year 2
y6_completed                  0/1         Bachelor's degree within 6 years
time_to_degree_years          int or NA   4-6 if completed, NA otherwise
cumulative_gpa_final          float or NA 0-4

Missingness notes
-----------------
- y1_gpa_mean missing for 0.3% of sample (never enrolled after matriculation).
- time_to_degree_years and cumulative_gpa_final missing by construction
  when y6_completed = 0.
- efc_dollars missing for 2 cases (manual entry error in 2014 FAFSA export,
  see flagged_issues/).

Provenance
----------
See 01_data/metadata/provenance_logs/ for hash-validated ingest records.
