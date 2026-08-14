# Model Registry and Production Policy

`model_registry` makes a model artifact identifiable by an immutable `model_version` (for example `v1.0.0`) rather than a mutable fixed path. It records model type, feature set version, training/validation periods, parameters, metrics, artifact path and SHA-256, creation/approval time, notes, and status.

Allowed lifecycle:

`candidate -> validated -> production -> retired`

- Training may only create a **candidate** artifact at a versioned path.
- Evaluation records metrics and may mark a candidate **validated**; it does not promote it.
- Only the explicit `ResearchRegistry.approve_production(model_version)` operation can promote a validated model. It retires the former production model in one transaction.
- Result registration must not call this operation and must not overwrite a production artifact.

Current state: `api/main.py:/result/register` still calls `retrain_with_new_data()` and writes fixed model paths. Phase 2 intentionally does not wire or alter that live behavior. Before any integration, create a compatibility plan that removes automatic retraining/promotion, versions all artifacts, preserves the old artifact, and requires a human approval record.
