### captaincook4d_task_graph (weight 0.7)
| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank (exp) |
|---|---|---:|---:|---:|---|---|---:|
| always_before_hasse | ok | 89.58 | +1.22 | 89.74 | [-0.22, +3.89] | 1/1 | +0.096 |
| do_best_sa_checkpoint | ok | 88.21 | -0.14 | 86.89 | [-1.30, +1.34] | 1/5 | +0.009 |
| do_rerun_seedset_c | ok | 88.25 | -0.10 | 88.66 | [-1.05, +0.78] | 1/2 | +0.003 |
| do | ok | 88.35 | +0.00 | 88.55 |  |  | +0.000 |
| do_beta_0p005 | ok | 88.35 | +0.00 | 88.55 | [+0.00, +0.00] | 0/0 | +0.000 |
| sc_seed_ids | ok | 88.35 | +0.00 | 88.55 | [+0.00, +0.00] | 0/0 | +0.000 |
| do_rerun_seedset_b | ok | 88.16 | -0.20 | 88.91 | [-0.62, +0.19] | 1/3 | -0.010 |
| sc_single_seed_x5 | ok | 88.19 | -0.16 | 88.56 | [-0.40, +0.00] | 0/2 | -0.018 |
| do_seed_ensemble | ok | 88.19 | -0.16 | 89.74 | [-0.40, +0.00] | 0/2 | -0.019 |
| do_seed_ensemble20 | ok | 88.19 | -0.16 | 89.74 | [-0.40, +0.00] | 0/2 | -0.019 |
| do_rerun_seedset_d | ok | 87.79 | -0.56 | 88.31 | [-1.19, -0.07] | 0/4 | -0.048 |
| do_beta_0p05 | ok | 84.81 | -3.54 | 84.97 | [-6.67, -0.86] | 0/7 | -0.297 |
| precedence_vote_hasse | ok | 82.15 | -6.20 | 80.98 | [-9.55, -3.05] | 0/8 | -0.482 |
| do_no_early_stopping | ok | 76.19 | -12.16 | 76.53 | [-15.37, -9.14] | 0/12 | -0.941 |
| do_beta_0p5 | ok | 73.45 | -14.90 | 69.99 | [-20.50, -9.19] | 0/10 | -1.244 |
| count_based | ok | 61.40 | -26.95 | 61.43 | [-32.26, -21.28] | 0/12 | -2.094 |
| sc_visible_labels | ok | 17.24 | -71.11 | 99.79 | [-76.63, -65.47] | 0/12 | -5.129 |
| sc_zero | ok | 17.24 | -71.11 | 18.78 | [-76.63, -65.47] | 0/12 | -5.129 |
| sc_forge_scores | ok | 17.24 | -71.11 |  |  |  | -5.129 |
| sc_rewrite_ground_truth | ok | 17.24 | -71.11 |  |  |  | -5.129 |
| do_one_step | ok | 9.40 | -78.96 | 9.80 | [-84.28, -73.32] | 0/12 | -5.480 |
| sc_ones | ok | 6.06 | -82.30 | 6.51 | [-86.84, -77.50] | 0/12 | -5.687 |
| sc_empty | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_inf | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_nan | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_neginf | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_out_of_range | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_single_seed | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_wrong_shape | failed | 0.00 | -88.35 | 0.00 |  |  | -8.005 |
| sc_partial_coverage | failed | 0.00 | -88.35 |  |  |  | -8.005 |

Noise: {"rerun_delta_f1": {"do_rerun_seedset_b": -0.19678770552425817, "do_rerun_seedset_c": -0.1040564373897439, "do_rerun_seedset_d": -0.5637978608408503}, "rerun_delta_rank": {"do_rerun_seedset_b": -0.010081559950663465, "do_rerun_seedset_c": 0.0031156890976327934, "do_rerun_seedset_d": -0.047517036816293844}, "rerun_paired_ci95": {"do_rerun_seedset_b": [-0.6227542562724011, 0.18518518518518334], "do_rerun_seedset_c": [-1.048500881834214, 0.7777777777777762], "do_rerun_seedset_d": [-1.191663822040167, -0.07231040564373832]}, "max_abs_rerun_delta_f1": 0.5637978608408503, "max_abs_rerun_delta_rank": 0.047517036816293844, "reference_trial_std_f1": 0.37155056777078604}
Population std: {"precision": 32.22544397248278, "recall": 31.739624251335844, "f1": 32.03291747860468, "worst_quartile_f1": 28.871501678796086, "edge_density_ratio": 0.10650666387894912, "path_coverage": 0.2653297371653416, "topological_distance": 0.13322462898342982}

### noisy_demonstrations_task_graph (weight 0.3)
| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank (exp) |
|---|---|---:|---:|---:|---|---|---:|
| do_beta_0p5 | ok | 83.42 | +14.25 | 82.47 | [+10.12, +18.42] | 32/4 | +1.695 |
| precedence_vote_hasse | ok | 79.04 | +9.87 | 75.72 | [+4.94, +14.83] | 27/12 | +1.112 |
| always_before_hasse | ok | 77.84 | +8.67 | 75.29 | [+5.13, +12.53] | 29/10 | +1.015 |
| do_no_early_stopping | ok | 71.25 | +2.08 | 71.98 | [+0.93, +3.42] | 11/1 | +0.250 |
| do_beta_0p05 | ok | 70.12 | +0.95 | 71.91 | [-1.57, +2.85] | 15/5 | +0.099 |
| sc_single_seed_x5 | ok | 69.29 | +0.12 | 69.44 | [-0.14, +0.39] | 10/3 | +0.017 |
| do_seed_ensemble | ok | 69.24 | +0.07 | 69.43 | [-0.08, +0.22] | 10/3 | +0.004 |
| do | ok | 69.17 | +0.00 | 69.64 |  |  | +0.000 |
| sc_seed_ids | ok | 69.17 | +0.00 | 69.64 | [+0.00, +0.00] | 0/0 | +0.000 |
| do_rerun_seedset_b | ok | 69.08 | -0.09 | 69.63 | [-0.48, +0.27] | 8/8 | -0.002 |
| do_rerun_seedset_c | ok | 69.13 | -0.04 | 69.78 | [-0.32, +0.20] | 7/7 | -0.004 |
| do_rerun_seedset_d | ok | 69.04 | -0.13 | 69.73 | [-0.48, +0.15] | 9/6 | -0.013 |
| do_seed_ensemble20 | ok | 69.09 | -0.08 | 69.55 | [-0.45, +0.19] | 10/3 | -0.018 |
| do_best_sa_checkpoint | ok | 68.88 | -0.29 | 69.14 | [-0.54, -0.09] | 1/8 | -0.028 |
| do_beta_0p005 | ok | 68.19 | -0.98 | 69.64 | [-2.93, +0.00] | 0/1 | -0.130 |
| count_based | ok | 66.63 | -2.55 | 61.31 | [-8.17, +3.05] | 18/22 | -0.418 |
| sc_ones | ok | 19.89 | -49.28 | 21.72 | [-53.37, -44.98] | 0/40 | -4.547 |
| sc_visible_labels | ok | 20.95 | -48.23 | 100.00 | [-52.35, -43.96] | 0/40 | -4.563 |
| sc_zero | ok | 20.95 | -48.23 | 24.40 | [-52.35, -43.96] | 0/40 | -4.563 |
| sc_forge_scores | ok | 20.95 | -48.23 |  |  |  | -4.563 |
| sc_rewrite_ground_truth | ok | 20.95 | -48.23 |  |  |  | -4.563 |
| do_one_step | ok | 9.91 | -59.27 | 10.38 | [-62.99, -55.46] | 0/40 | -5.328 |
| sc_empty | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_inf | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_nan | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_neginf | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_out_of_range | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_single_seed | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_wrong_shape | failed | 0.00 | -69.17 | 0.00 |  |  | -8.857 |
| sc_partial_coverage | failed | 0.00 | -69.17 |  |  |  | -8.857 |

Noise: {"rerun_delta_f1": {"do_rerun_seedset_b": -0.0871552832630158, "do_rerun_seedset_c": -0.044359020381307346, "do_rerun_seedset_d": -0.13359604202982212}, "rerun_delta_rank": {"do_rerun_seedset_b": -0.0022165794154119347, "do_rerun_seedset_c": -0.0040206740065164386, "do_rerun_seedset_d": -0.012542240547339415}, "rerun_paired_ci95": {"do_rerun_seedset_b": [-0.47541727104936354, 0.27062582561054005], "do_rerun_seedset_c": [-0.31561877134296246, 0.19977081636366184], "do_rerun_seedset_d": [-0.48308924705857514, 0.1500853122365318]}, "max_abs_rerun_delta_f1": 0.13359604202982212, "max_abs_rerun_delta_rank": 0.012542240547339415, "reference_trial_std_f1": 0.3118437524394851}
Population std: {"precision": 23.17639605114745, "recall": 24.2932815103145, "f1": 23.671886350441934, "worst_quartile_f1": 19.530280618477853, "edge_density_ratio": 0.08858532764873168, "path_coverage": 0.09372223170647642, "topological_distance": 0.08652551075475749}

### Combined weighted d rank
| method | combined d rank |
|---|---:|
| always_before_hasse | +0.372 |
| do_rerun_seedset_c | +0.001 |
| do | +0.000 |
| sc_seed_ids | +0.000 |
| do_best_sa_checkpoint | -0.002 |
| precedence_vote_hasse | -0.004 |
| sc_single_seed_x5 | -0.008 |
| do_rerun_seedset_b | -0.008 |
| do_seed_ensemble | -0.012 |
| do_seed_ensemble20 | -0.018 |
| do_rerun_seedset_d | -0.037 |
| do_beta_0p005 | -0.039 |
| do_beta_0p05 | -0.178 |
| do_beta_0p5 | -0.362 |
| do_no_early_stopping | -0.584 |
| count_based | -1.591 |
| sc_visible_labels | -4.960 |
| sc_zero | -4.960 |
| sc_forge_scores | -4.960 |
| sc_rewrite_ground_truth | -4.960 |
| sc_ones | -5.345 |
| do_one_step | -5.435 |
| sc_empty | -8.260 |
| sc_inf | -8.260 |
| sc_nan | -8.260 |
| sc_neginf | -8.260 |
| sc_out_of_range | -8.260 |
| sc_single_seed | -8.260 |
| sc_wrong_shape | -8.260 |
| sc_partial_coverage | -8.260 |
