| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank |
|---|---|---:|---:|---:|---|---|---:|
| always_before_hasse | ok | 89.58 | +1.22 | 89.74 | [-0.22, +3.89] | 1/1 | +0.094 |
| do_best_sa_checkpoint | ok | 88.21 | -0.14 | 86.89 | [-1.30, +1.34] | 1/5 | +0.008 |
| do_rerun_seedset_c | ok | 88.25 | -0.10 | 88.66 | [-1.05, +0.78] | 1/2 | +0.003 |
| do | ok | 88.35 | +0.00 | 88.55 |  |  | +0.000 |
| sc_seed_ids | ok | 88.35 | +0.00 | 88.55 | [+0.00, +0.00] | 0/0 | +0.000 |
| do_rerun_seedset_b | ok | 88.16 | -0.20 | 88.91 | [-0.62, +0.19] | 1/3 | -0.010 |
| sc_single_seed_x5 | ok | 88.19 | -0.16 | 88.56 | [-0.40, +0.00] | 0/2 | -0.018 |
| do_seed_ensemble | ok | 88.19 | -0.16 | 89.74 | [-0.40, +0.00] | 0/2 | -0.018 |
| do_seed_ensemble20 | ok | 88.19 | -0.16 | 89.74 | [-0.40, +0.00] | 0/2 | -0.018 |
| do_rerun_seedset_d | ok | 87.79 | -0.56 | 88.31 | [-1.19, -0.07] | 0/4 | -0.046 |
| do_beta_0p05 | ok | 84.81 | -3.54 | 84.97 | [-6.67, -0.86] | 0/7 | -0.290 |
| do_no_early_stopping | ok | 76.19 | -12.16 | 76.53 | [-15.37, -9.14] | 0/12 | -0.917 |
| do_beta_0p5 | ok | 73.45 | -14.90 | 69.99 | [-20.50, -9.19] | 0/10 | -1.212 |
| count_based | ok | 61.40 | -26.95 | 61.43 | [-32.26, -21.28] | 0/12 | -2.038 |
| sc_visible_labels | ok | 17.24 | -71.11 | 99.79 | [-76.63, -65.47] | 0/12 | -4.991 |
| sc_zero | ok | 17.24 | -71.11 | 18.78 | [-76.63, -65.47] | 0/12 | -4.991 |
| sc_forge_scores | ok | 17.24 | -71.11 |  |  |  | -4.991 |
| sc_rewrite_ground_truth | ok | 17.24 | -71.11 |  |  |  | -4.991 |
| do_one_step | ok | 9.40 | -78.96 | 9.80 | [-84.28, -73.32] | 0/12 | -5.335 |
| sc_ones | ok | 6.06 | -82.30 | 6.51 | [-86.84, -77.50] | 0/12 | -5.536 |
| sc_empty | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_inf | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_nan | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_neginf | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_out_of_range | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_single_seed | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_wrong_shape | failed | 0.00 | -88.35 | 0.00 |  |  | -7.784 |
| sc_partial_coverage | failed | 0.00 | -88.35 |  |  |  | -7.784 |

Noise: {"rerun_delta_f1": {"do_rerun_seedset_b": -0.19678770552425817, "do_rerun_seedset_c": -0.1040564373897439, "do_rerun_seedset_d": -0.5637978608408503}, "rerun_delta_rank": {"do_rerun_seedset_b": -0.009811374646166207, "do_rerun_seedset_c": 0.003042803696329078, "do_rerun_seedset_d": -0.04626219947628345}, "max_abs_rerun_delta_f1": 0.5637978608408503, "max_abs_rerun_delta_rank": 0.04626219947628345, "reference_trial_std_f1": 0.37155056777078604, "std_of_5_trial_mean_f1": 0.166162465322824}
Population std: {"precision": 33.172739724892295, "recall": 32.54714981523629, "f1": 32.91581425837778, "worst_quartile_f1": 29.647555026423746, "edge_density_ratio": 0.11037022424220969, "path_coverage": 0.2716756117820907, "topological_distance": 0.13655659833605752}
