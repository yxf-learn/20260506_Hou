# Bandwidth selection notes

Main specification: MSE-optimal via Calonico, Cattaneo, and
Titiunik (2014). For each outcome separately:

  y1_gpa_mean    h* = 0.92   (h_l = 0.89, h_r = 0.95)
  y2_retained    h* = 0.87   (0.84, 0.90)
  y6_completed   h* = 0.94   (0.91, 0.97)

Robustness: also run at 0.5x, 1.5x, 2.0x MSE-optimal. See
sensitivity_bandwidth.log and Figure 7(a).

CER-optimal bandwidth (coverage-error-rate) is ~15% smaller and
produces qualitatively identical point estimates with slightly wider
CIs. Not reported in main text for brevity.
