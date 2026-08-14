# Experiment Registry

An `experiment_registry` record answers why one model was evaluated and whether it should be considered further. Required identifiers are `experiment_id` and `feature_set_version`; optional `model_version` links the tested artifact. It records train, validation, and test period boundaries, parameters JSON, metrics JSON, a written conclusion, and planned/running/completed/rejected status.

Feature definitions live in `feature_sets`. A model and every prediction run reference a `feature_set_version`, so results remain interpretable after feature code changes.

For a future fair comparison, create one experiment record before training, use a fixed time-series test definition, write probability and betting metrics to `metric_records`, then record an explicit conclusion. A high short-period ROI alone is not a promotion criterion.

Leakage requirement for all future feature sets: compute every aggregate with records whose event time is strictly earlier than the run's `data_cutoff_time`/`prediction_time`. `build_stats_cache()` must eventually become an as-of statistic keyed by the target race/time; `lookup_stats()` must receive that cutoff. Phase 2 documents this rule but does not modify ML behavior.
