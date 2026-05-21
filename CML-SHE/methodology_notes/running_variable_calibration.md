# Calibrating the need-index running variable

The federal EFC formula returns a dollar figure. For RDD purposes we
want a scale-free running variable with mass around the programme
cutoff. We rescale EFC by a factor of 6800 and clip to [0.1, 9.9].

    need_index = clip(efc_dollars / 6800, 0.1, 9.9)

The factor 6800 is the 60th-percentile EFC in the 2014 cohort
(the programme's target eligibility fraction). After rescaling, a
programme with a 60th-percentile EFC cutoff places the cutoff at
need_index = 3.0. This is deterministic and invertible, so the RDD is
unaffected.

Alternatives considered:
  - Percentile-rank within cohort (rejected; breaks smoothness of
    density under cross-cohort pooling).
  - Standardised EFC (rejected; unit is counter-intuitive for
    policy-oriented readers).

Sensitivity: results are essentially identical under scaling factor
5000, 6000, 7500, 8000. See sensitivity_bandwidth.log.
