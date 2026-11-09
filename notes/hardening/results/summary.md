### captaincook4d_task_graph (weight 0.7)
| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank (exp) |
|---|---|---:|---:|---:|---|---|---:|
| always_before_hasse | ok | 89.58 | +1.22 | 89.74 | [-0.22, +3.89] | 1/1 | +0.097 |
| do_best_sa_checkpoint | ok | 88.21 | -0.14 | 86.89 | [-1.30, +1.34] | 1/5 | +0.009 |
| do_rerun_seedset_c | ok | 88.25 | -0.10 | 88.66 | [-1.05, +0.78] | 1/2 | +0.003 |
| do | ok | 88.35 | +0.00 | 88.55 |  |  | +0.000 |
| do_beta_selected_by_sa | ok | 88.35 | +0.00 | 88.55 | [+0.00, +0.00] | 0/0 | +0.000 |
| sc_seed_ids | ok | 88.35 | +0.00 | 88.55 | [+0.00, +0.00] | 0/0 | +0.000 |
| do_beta_selected_on_public_split | ok | 88.35 | +0.00 |  | [+0.00, +0.00] | 0/0 | +0.000 |
| do_rerun_seedset_b | ok | 88.16 | -0.20 | 88.91 | [-0.62, +0.19] | 1/3 | -0.010 |
| sc_single_seed_x5 | ok | 88.19 | -0.16 | 88.56 | [-0.40, +0.00] | 0/2 | -0.018 |
| do_seed_ensemble | ok | 88.19 | -0.16 | 89.74 | [-0.40, +0.00] | 0/2 | -0.019 |
| do_seed_ensemble20 | ok | 88.19 | -0.16 | 89.74 | [-0.40, +0.00] | 0/2 | -0.019 |
| do_rerun_seedset_d | ok | 87.79 | -0.56 | 88.31 | [-1.19, -0.07] | 0/4 | -0.048 |
| do_beta_0p05 | ok | 84.81 | -3.54 | 84.97 | [-6.67, -0.86] | 0/7 | -0.300 |
| precedence_vote_hasse | ok | 82.15 | -6.20 | 80.98 | [-9.55, -3.05] | 0/8 | -0.487 |
| do_no_early_stopping | ok | 76.19 | -12.16 | 76.53 | [-15.37, -9.14] | 0/12 | -0.950 |
| do_beta_0p5 | ok | 73.45 | -14.90 | 69.99 | [-20.50, -9.19] | 0/10 | -1.257 |
| count_based | ok | 61.40 | -26.95 | 61.43 | [-32.26, -21.28] | 0/12 | -2.116 |
| sc_visible_labels | ok | 17.24 | -71.11 | 99.79 | [-76.63, -65.47] | 0/12 | -5.186 |
| sc_zero | ok | 17.24 | -71.11 | 18.78 | [-76.63, -65.47] | 0/12 | -5.186 |
| sc_forge_scores | ok | 17.24 | -71.11 |  | [-76.63, -65.47] | 0/12 | -5.186 |
| sc_rewrite_ground_truth | ok | 17.24 | -71.11 |  | [-76.63, -65.47] | 0/12 | -5.186 |
| do_one_step | ok | 9.40 | -78.96 | 9.80 | [-84.28, -73.32] | 0/12 | -5.540 |
| sc_ones | ok | 6.06 | -82.30 | 6.51 | [-86.84, -77.50] | 0/12 | -5.749 |
| sc_empty | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_inf | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_nan | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_neginf | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_out_of_range | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_single_seed | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_wrong_shape | failed | 0.00 | -88.35 | 0.00 |  |  | -8.096 |
| sc_partial_coverage | failed | 0.00 | -88.35 |  |  |  | -8.096 |

Noise: {"rerun_delta_f1": {"do_rerun_seedset_b": -0.19678770552425817, "do_rerun_seedset_c": -0.1040564373897439, "do_rerun_seedset_d": -0.5637978608408503}, "rerun_delta_rank": {"do_rerun_seedset_b": -0.010185617255585566, "do_rerun_seedset_c": 0.0031553246563215537, "do_rerun_seedset_d": -0.04802256164930276}, "rerun_paired_ci95": {"do_rerun_seedset_b": [-0.6227542562724011, 0.18518518518518334], "do_rerun_seedset_c": [-1.048500881834214, 0.7777777777777762], "do_rerun_seedset_d": [-1.191663822040167, -0.07231040564373832]}, "max_abs_rerun_delta_f1": 0.5637978608408503, "max_abs_rerun_delta_rank": 0.04802256164930276, "reference_trial_std_f1": 0.37155056777078604}
Population std: {"precision": 31.85731877074783, "recall": 31.406952613496287, "f1": 31.683127606679342, "worst_quartile_f1": 28.562363650386608, "edge_density_ratio": 0.10502943463824625, "path_coverage": 0.26412488814648444, "topological_distance": 0.13173125611596734}

### noisy_demonstrations_task_graph (weight 0.3)
| method | status | test F1 | dF1 | train F1 | paired dF1 95% CI | better/worse procs | d rank (exp) |
|---|---|---:|---:|---:|---|---|---:|
| do_beta_0p5 | ok | 81.16 | +13.11 | 79.75 | [+10.68, +15.56] | 80/18 | +1.511 |
| do_beta_selected_on_public_split | ok | 81.16 | +13.11 |  | [+10.68, +15.56] | 80/18 | +1.511 |
| precedence_vote_hasse | ok | 78.53 | +10.48 | 78.17 | [+7.63, +13.39] | 75/20 | +1.223 |
| always_before_hasse | ok | 77.48 | +9.43 | 76.14 | [+7.27, +11.64] | 78/16 | +1.119 |
| do_beta_0p05 | ok | 69.11 | +1.07 | 67.84 | [+0.26, +1.78] | 31/11 | +0.132 |
| do_no_early_stopping | ok | 68.62 | +0.57 | 67.90 | [+0.04, +1.09] | 19/5 | +0.088 |
| do_beta_selected_by_sa | ok | 68.32 | +0.27 | 67.35 | [-0.44, +0.83] | 14/6 | +0.044 |
| do_rerun_seedset_d | ok | 68.14 | +0.09 | 67.04 | [-0.13, +0.36] | 8/13 | +0.014 |
| do_rerun_seedset_c | ok | 68.13 | +0.08 | 67.01 | [-0.11, +0.29] | 13/11 | +0.012 |
| do_seed_ensemble20 | ok | 68.14 | +0.09 | 66.95 | [-0.06, +0.26] | 12/13 | +0.006 |
| sc_single_seed_x5 | ok | 68.08 | +0.03 | 66.85 | [-0.21, +0.30] | 11/11 | +0.005 |
| do_rerun_seedset_b | ok | 68.06 | +0.01 | 66.97 | [-0.13, +0.15] | 8/11 | +0.002 |
| do | ok | 68.05 | +0.00 | 66.98 |  |  | +0.000 |
| sc_seed_ids | ok | 68.05 | +0.00 | 66.98 | [+0.00, +0.00] | 0/0 | +0.000 |
| do_seed_ensemble | ok | 67.98 | -0.07 | 67.08 | [-0.19, +0.05] | 8/14 | -0.012 |
| do_best_sa_checkpoint | ok | 67.90 | -0.15 | 66.33 | [-0.25, -0.06] | 1/14 | -0.016 |
| count_based | ok | 63.36 | -4.69 | 62.65 | [-8.41, -0.93] | 43/54 | -0.628 |
| sc_visible_labels | ok | 21.29 | -46.76 | 100.00 | [-49.41, -44.13] | 0/100 | -4.519 |
| sc_zero | ok | 21.29 | -46.76 | 22.18 | [-49.41, -44.13] | 0/100 | -4.519 |
| sc_forge_scores | ok | 21.29 | -46.76 |  | [-49.41, -44.13] | 0/100 | -4.519 |
| sc_rewrite_ground_truth | ok | 21.29 | -46.76 |  | [-49.41, -44.13] | 0/100 | -4.519 |
| sc_ones | ok | 19.52 | -48.53 | 19.67 | [-51.14, -45.92] | 0/100 | -4.589 |
| do_one_step | ok | 9.88 | -58.17 | 11.02 | [-60.79, -55.55] | 0/100 | -5.372 |
| sc_empty | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_inf | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_nan | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_neginf | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_out_of_range | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_single_seed | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_wrong_shape | failed | 0.00 | -68.05 | 0.00 |  |  | -8.848 |
| sc_partial_coverage | failed | 0.00 | -68.05 |  |  |  | -8.848 |

Noise: {"rerun_delta_f1": {"do_rerun_seedset_b": 0.007847195272333352, "do_rerun_seedset_c": 0.0811058519735326, "do_rerun_seedset_d": 0.09326563361155138}, "rerun_delta_rank": {"do_rerun_seedset_b": 0.002360892568766355, "do_rerun_seedset_c": 0.012128998241418457, "do_rerun_seedset_d": 0.013718218535410161}, "rerun_paired_ci95": {"do_rerun_seedset_b": [-0.12584221072583596, 0.1472311146654807], "do_rerun_seedset_c": [-0.10734407100525249, 0.2939495673247182], "do_rerun_seedset_d": [-0.1292596524455377, 0.36186624257951416]}, "max_abs_rerun_delta_f1": 0.09326563361155138, "max_abs_rerun_delta_rank": 0.013718218535410161, "reference_trial_std_f1": 0.08629663584433493}
Population std: {"precision": 22.908832684919798, "recall": 23.35835036174662, "f1": 23.150060318751972, "worst_quartile_f1": 19.610741804176044, "edge_density_ratio": 0.09251570390366715, "path_coverage": 0.0808274728921295, "topological_distance": 0.08549409442591668}

### Combined weighted d rank
| method | combined d rank |
|---|---:|
| do_beta_selected_on_public_split | +0.453 |
| always_before_hasse | +0.404 |
| precedence_vote_hasse | +0.026 |
| do_beta_selected_by_sa | +0.013 |
| do_rerun_seedset_c | +0.006 |
| do_best_sa_checkpoint | +0.001 |
| do | +0.000 |
| sc_seed_ids | +0.000 |
| do_rerun_seedset_b | -0.006 |
| do_seed_ensemble20 | -0.011 |
| sc_single_seed_x5 | -0.011 |
| do_seed_ensemble | -0.017 |
| do_rerun_seedset_d | -0.030 |
| do_beta_0p05 | -0.170 |
| do_beta_0p5 | -0.426 |
| do_no_early_stopping | -0.639 |
| count_based | -1.669 |
| sc_visible_labels | -4.986 |
| sc_zero | -4.986 |
| sc_forge_scores | -4.986 |
| sc_rewrite_ground_truth | -4.986 |
| sc_ones | -5.401 |
| do_one_step | -5.489 |
| sc_empty | -8.322 |
| sc_inf | -8.322 |
| sc_nan | -8.322 |
| sc_neginf | -8.322 |
| sc_out_of_range | -8.322 |
| sc_single_seed | -8.322 |
| sc_wrong_shape | -8.322 |
| sc_partial_coverage | -8.322 |
