# 共有ストレージ設計（無料運用）

調査日: 2026-08-14 JST。これは設計文書のみであり、Google Drive、Google Cloud Storage（GCS）、Colab、Render、Vercelへの接続・設定変更・ファイル操作は行っていない。

## 結論

現時点で採用する構成は **共有Google Drive + Google Colab中心** とする。Google Driveの共有フォルダを実データ・versioned artifactの正本とし、Colabを計画的な単一writer、各PCをread-only consumerとする。GitHubはコードと開発状態の正本、Render/Vercelは共有artifactの正本にしない。

無料Googleアカウントには最大15 GBの共有ストレージがある（Drive/Gmail/Photosで共有）。[Google Drive公式](https://support.google.com/drive/answer/2375123?hl=ja) GCS Always Freeは5 GB-month、操作数・地域制約があり、上限を超えると課金対象になるため、完全無料を絶対条件とする現段階の正本には採用しない。[GCS公式料金](https://cloud.google.com/storage/pricing?hl=ja)

## 現在の参照と依存関係

| 対象 | 現状 | 設計上の扱い |
|---|---|---|
| `project_paths.py` | `KEIBA_DATA_DIR`、`KEIBA_MODEL_DIR`、個別DB pathを受ける | 各PCのread-only同期/一時コピーを指定する |
| `colab_training.py` | `drive.mount('/content/drive')`、`KEIBA_SHARED_ROOT` | Driveの共有rootを指定する唯一のwriter候補 |
| `training.ipynb` | `/content/drive/MyDrive/...`の旧defaultが残る | Phase 3前の別taskで`KEIBA_SHARED_ROOT`へ統一する |
| `ml/train.py` | DBを読み、model/cache/metaを書き出す | versioned Drive artifact directoryへだけ出力する（将来承認時） |
| `ml/predict.py` | model + stats cacheが必須 | PC/Colab上の検証済みread-only copyから読む |
| `api/main.py` | model meta、model/cache、DB（result登録時）を参照 | 現状は共有storageへ直接接続しない |
| `render.yaml` | Python APIを起動するだけ | local filesystemはartifact正本にしない |
| Vercel frontend | `NEXT_PUBLIC_API_URL`だけを参照 | Drive/GCS credentialを持たない |

PC固有のWindows絶対pathは実行コードにない。Colabの`/content/drive`はColab runtimeの標準mount pathであり、PC pathではない。ただし共有先folder ID/URIは未決定である。

## 必要ファイルと依存関係

| 作業 | 必要な共有artifact | 出力 |
|---|---|---|
| 学習 | 固定したSQLite DB snapshot、データmanifest、feature set定義 | model、stats cache、metadata、experiment結果 |
| 推論 | 承認済みmodel、stats cache、metadata、feature set定義、出馬表入力snapshot | prediction record、入力snapshot、結果照合予定 |
| バックテスト | 固定したDB snapshot、as-of feature仕様、experiment定義、対象model | metrics、予測entry、betting結果、report |

DB snapshotとmodelは必ずmanifest経由で結ぶ。model manifestには`model_version`、DB snapshot version/SHA-256、feature_set_version、experiment_id、作成日時、各artifactのSHA-256を記録する。modelとstats cacheだけを単独で更新してはならない。

## 比較

| 案 | 無料維持 | 複数PC / Colab / Python | Render / Vercel | 主なリスク・評価 |
|---|---|---|---|---|
| A. Google Drive中心 | 15 GB内なら可。Gmail/Photosと容量共有 | Drive for desktop同期または手動download、Colab mount、Python local file accessが容易 | Renderはmount不可、Vercelは不要 | SQLiteを複数PCで同期中に書くと破損/競合し得る。単一writer・immutable snapshotなら適する |
| B. GCS中心 | Always Freeは5 GB-month・操作/地域制約。超過で課金可能性 | Python SDK/Colabは適する | Renderから取得可能だがcredential/egress/課金管理が必要。Vercelにはcredentialを置けない | API、IAM、billing account、secret運用が増える。無料絶対条件には不適 |
| C. Google Drive + Colab中心 | Aと同じ15 GB枠内で可 | ColabのDrive mountは自然。PCはread-only copy | Render/VercelからDriveを直接使わない | Colab Drive I/Oは多数ファイルで遅延/失敗し得る。versioned bundleと少数ファイル化で軽減 |
| D. GitHub Releases/LFS等 | 大型data/modelやquota面で不適 | cloneに重く、artifact正本と混ぜる | deploy artifactに不向き | GitHubをコード正本に保つ方針と衝突。採用しない |

Colab公式FAQは、Drive mountで多数のtop-level itemや頻繁なI/Oが失敗/遅延し得ることを説明している。[Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) そのためSQLiteと大きなartifactは階層化し、実行中に何度も同期しない。

Render無料Web Serviceはfilesystemがephemeralで、idle停止・redeploy時にlocal SQLite等が失われる。無料instanceにはpersistent diskを付けられない。[Render Free公式](https://render.com/docs/free) よってRenderをDB/modelの永続保存先やDrive同期先にしてはならない。Vercelは環境変数を扱えるが、`NEXT_PUBLIC_`値はクライアント公開となるため、Drive/GCS認証情報を置かない。[Vercel公式](https://vercel.com/docs/environment-variables)

## 採用構成

```
GitHub main ── code / tests / docs / migrations / AI handoff

Shared Google Drive (single authoritative artifact root)
  datasets/<dataset_version>/keiba.sqlite + manifest.json
  models/<model_version>/{model,stats_cache,metadata,manifest}.json
  experiments/<experiment_id>/{config,metrics,report}.json
  snapshots/<prediction_id>.json[.gz]
  manifests/{latest_dataset,production_model}.json

Colab (single writer, approved jobs only) ── publish versioned files + manifest last
PCs (read-only consumers) ── verify SHA-256 -> local cache -> set KEIBA_* paths
Render API ── stateless; no direct Drive/GCS access in current free design
Vercel ── UI -> Render API URL only; no artifact credentials
```

### publication protocol

1. writer creates a new immutable versioned directory/file name; never overwrites `latest` artifact.
2. writer computes SHA-256 and writes an artifact manifest.
3. writer verifies the manifest and files from a fresh read.
4. writer updates a small pointer manifest (for example `production_model.json`) **last**.
5. consumers download/sync the named version, verify SHA-256, then set `KEIBA_*` paths to their local read-only copy.

No two PC/Colab jobs may write the same SQLite file. SQLite is transferred as a closed snapshot, never as a live DB with WAL/journal files. Prediction/experiment writes require a future serialized writer design; this Phase does not introduce it.

## 役割

| Component | Role |
|---|---|
| GitHub | source, tests, docs, task/handoff, registry schema; no data/model/token |
| shared Google Drive | authoritative immutable DB snapshots, model bundles, manifests, experiment/prediction artifacts, backups |
| Colab | manually approved collection/training/evaluation writer; no direct model GitHub push |
| each PC | Git clone, local development, verified read-only artifact cache; never sole source of truth |
| Render | stateless API hosting only. Until a separately approved delivery/auth design exists, do not rely on it for shared DB/model access |
| Vercel | frontend hosting only; expose API URL, never storage credential |

## 未解決・Phase 3前の判断

1. Drive共有folderの所有者、共有範囲、容量使用量、folder ID/URI、backup担当を決める。
2. `manifest.json`の正式schema、SHA-256作成/検証CLI、version命名規約を承認する。
3. `training.ipynb`を`KEIBA_SHARED_ROOT`へ統一する。
4. 完全無料のRenderではmodel artifactを安全に永続利用できない。Web予測を再開するなら、公開/認証済みartifact deliveryまたは別の無料ホストを別taskで評価する。GCSを採用する場合は、課金上限・billing/IAM・credential rotationを先に承認する。
5. Drive 15 GBを超える実データにはこの構成を適用できない。取得前に推定容量と削減/分割方針を確認する。
