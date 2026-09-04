# Final TIDE-DiT Configurations

Every configuration in this list uses the final TIDE-DiT protocol:

- Training: the TIDE-DiT primitive-aware objective with `tdsm_triplet_weight=2.5` and `tdsm_triplet_margin=0.1`.
- Inference: frozen feature-graph refinement with `k=100`, `tau=0.10`, `alpha=1.0`, and five iterations.
- The YAML files default to direct TIDE-DiT inference (`eval_feature_graph: false`). Feature-graph refinement is enabled only by the inference command below.

Set the three variables for one row, then run the training and feature-graph inference commands. The inference command never trains or changes the checkpoint.

```powershell
$Python = 'D:/anaconda/envs/tide/python.exe'
$Config = 'config/tide_shiftgcn_ntu60_unseen5.yaml'
$TrainDir = 'result/shiftgcn/ntu60/unseen5'

# TIDE-DiT training
& $Python scripts/train_feat_c2u_zsl.py `
  --config $Config `
  --seed 2026 `
  --tdsm-triplet-weight 2.5 `
  --tdsm-triplet-margin 0.1 `
  --early-stop-patience 10 `
  --work-dir $TrainDir

# Feature-graph inference from the TIDE-DiT checkpoint
& $Python scripts/train_feat_c2u_zsl.py `
  --config $Config `
  --seed 2026 `
  --eval-only `
  --resume-from-checkpoint "$TrainDir/best" `
  --enable-feature-graph `
  --eval-tag feature_graph `
  --work-dir "${TrainDir}_feature_graph"
```

| Backbone | Protocol | Config | `TrainDir` |
| --- | --- | --- | --- |
| Shift-GCN | NTU60 55/5 | `config/tide_shiftgcn_ntu60_unseen5.yaml` | `result/shiftgcn/ntu60/unseen5` |
| Shift-GCN | NTU60 48/12 | `config/tide_shiftgcn_ntu60_unseen12.yaml` | `result/shiftgcn/ntu60/unseen12` |
| Shift-GCN | NTU60 40/20 | `config/tide_shiftgcn_ntu60_unseen20.yaml` | `result/shiftgcn/ntu60/unseen20` |
| Shift-GCN | NTU60 30/30 | `config/tide_shiftgcn_ntu60_unseen30.yaml` | `result/shiftgcn/ntu60/unseen30` |
| Shift-GCN | NTU120 110/10 | `config/tide_shiftgcn_ntu120_unseen10.yaml` | `result/shiftgcn/ntu120/unseen10` |
| Shift-GCN | NTU120 96/24 | `config/tide_shiftgcn_ntu120_unseen24.yaml` | `result/shiftgcn/ntu120/unseen24` |
| Shift-GCN | NTU120 80/40 | `config/tide_shiftgcn_ntu120_unseen40.yaml` | `result/shiftgcn/ntu120/unseen40` |
| Shift-GCN | NTU120 60/60 | `config/tide_shiftgcn_ntu120_unseen60.yaml` | `result/shiftgcn/ntu120/unseen60` |
| ST-GCN | NTU60 55/5, split 2 | `config/tide_stgcn_ntu60_split2_unseen5.yaml` | `result/stgcn/ntu60/split2_unseen5` |
| ST-GCN | NTU60 55/5, split 3 | `config/tide_stgcn_ntu60_split3_unseen5.yaml` | `result/stgcn/ntu60/split3_unseen5` |
| ST-GCN | NTU60 55/5, split 4 | `config/tide_stgcn_ntu60_split4_unseen5.yaml` | `result/stgcn/ntu60/split4_unseen5` |
| ST-GCN | NTU120 110/10, split 2 | `config/tide_stgcn_ntu120_split2_unseen10.yaml` | `result/stgcn/ntu120/split2_unseen10` |
| ST-GCN | NTU120 110/10, split 3 | `config/tide_stgcn_ntu120_split3_unseen10.yaml` | `result/stgcn/ntu120/split3_unseen10` |
| ST-GCN | NTU120 110/10, split 4 | `config/tide_stgcn_ntu120_split4_unseen10.yaml` | `result/stgcn/ntu120/split4_unseen10` |
| ST-GCN | PKU-MMD 46/5, split 1 | `config/tide_stgcn_pku51_split1_unseen5.yaml` | `result/stgcn/pku51/split1_unseen5` |
| ST-GCN | PKU-MMD 46/5, split 2 | `config/tide_stgcn_pku51_split2_unseen5.yaml` | `result/stgcn/pku51/split2_unseen5` |
| ST-GCN | PKU-MMD 46/5, split 3 | `config/tide_stgcn_pku51_split3_unseen5.yaml` | `result/stgcn/pku51/split3_unseen5` |

Training selects the best checkpoint using direct TIDE-DiT inference. The feature-graph directory contains only the separate final evaluation artifacts.

