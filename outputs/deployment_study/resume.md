# Resuming the frozen stage 5 run

Run ID: `6a030fd903e8e581`. Do not start a second copy while the original training
process is still running. The progress reader reports checkpoints, not process liveness.

```bash
python scripts/deployment_progress.py
python scripts/deployment_progress.py --metrics
```

The run caches 118 folds per configuration and cutoff. Completed folds are reused;
unfinished folds are retrained. Aggregation and prediction-draw sensitivity follow fitting.

A driver-query compatibility fix and stricter feature-contract cache validation were added
after the fitting source was frozen. They do not change the model operations or this run's
unchanged inputs, but they do change the conservative source fingerprint. To continue this
exact local cache, use its saved source:

```bash
PYTHONPATH=work/history_expansion/deployment_study/6a030fd903e8e581/source_snapshot/src \
  .venv/bin/python -m f1predict.cli evaluate deployment-study --workers 4
```

The frozen source reproduces the original run ID. Its archive is
`work/history_expansion/deployment_study/6a030fd903e8e581/source_at_run.tar.gz`;
`source_fingerprints.json` records the source hashes. These local checkpoint/source files
are intentionally not uploaded as model artifacts. A fresh clone without this local cache
uses the ordinary `f1predict evaluate deployment-study` command to start a new reproducible
run from its installed source and data.

Once the study is complete:

```bash
python scripts/export_deployment_checkpoint.py
f1predict model replay-deployment --mode post_quali
f1predict model replay-deployment --mode pre_quali --driver Verstappen
```

Replay uses the saved model trained strictly before the final studied race. It does not
replace production models or claim a prospective forecast.
