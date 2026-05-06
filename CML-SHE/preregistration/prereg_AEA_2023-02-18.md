# AEA RCT/Observational Study Registry - Preregistration

Registration ID:       AEARCTR-0010927
Date registered:       2023-02-18
Title:                 CML-SHE: Causal ML for Need-Based Financial Aid
Primary outcome:       6-year bachelor's degree completion
Secondary outcomes:    1st-year GPA; Year-2 retention
Target sample:         Cohorts 2013-2018 at five institutions (n ~ 18,400)
Primary analysis:      Fuzzy RDD with CCT bias-corrected CIs
Secondary analysis:    Causal forest CATE; depth-4 policy tree EWM
Subgroups of interest: first-generation; rural origin; first-gen x rural
Power consideration:   MDE = 0.04 p.p. on 6-yr completion at alpha=0.05
                       power=0.80 assuming pi_c = 0.40.

Deviations from prereg
----------------------
  - Added DML cross-check (Chernozhukov et al. 2018) at referee's request
    (revision 2024-12-03). Pre-specified analyses unchanged.
  - Policy-tree depth originally set to 3; expanded to 4 after referee
    noted the depth-3 tree missed an interpretable first-gen-rural
    split. Depth-3 results reported in Appendix C for transparency.
