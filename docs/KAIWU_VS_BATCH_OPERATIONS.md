# Kaiwu VS batch operations

## Read-only progress and batch preview

Run the status command before a batch:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.kaiwu_vs_batch_status `
  --protocol configs\paper\fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json `
  --preparation export\opsd_sa_negative_kaiwu_preparation_10read_v1 `
  --batch-size 1
```

It verifies all existing VS responses through the normal receipt, latent
diagnostic, and prediction validation path, then prints the frozen task order,
platform short names, verified/pending/blocked counts, reads, and the next
contiguous runnable batch. `--batch-size N` is a task cap; under this protocol
the cap is `10 * N` reads. Credits per read have not been confirmed, so no
credit estimate is produced.

The submission driver still requires `--limit N`, `--resume`,
`--confirm-real-submission`, and `FA_BM_VAE_RUN_REAL_KAIWU=1`. A single-task
canary also requires `--canary`, its exact frozen instance/gain/matrix flags,
and `--confirm-canary`. A non-canary batch requires `--confirm-batch` and the
frozen quality policy. Use `--limit 1` while the installed SDK omits a
verifiable task ID. If a call saves samples without an ID, it pauses. Do not
remove the intent or marker; provide a platform record and reviewed screenshot,
then resume after the receipt and full response validation pass.

## Batch receipt manifest

The batch receipt tool creates only missing receipt sidecars and never edits
responses, platform records, screenshots, matrices, or intents. It first
preflights every item in a temporary directory. It rejects duplicate
conditions/task IDs, stale protocol/preparation hashes, mismatched platform
records, invalid samples, and evidence whose bytes do not match a declared
`screenshot_sha256`.

After manually checking each screenshot against its platform export, create a
manifest like this. `human_review_attested` is a human assertion; its presence
does not mean the program has interpreted the image. The tool fixes
`verified_by` to `wsy`.

```json
{
  "schema_version": 1,
  "protocol_sha256": "<from the status report>",
  "preparation_manifest_sha256": "<from the status report>",
  "human_review_attested": true,
  "items": [
    {
      "instance_id": "opsd-wind_opsd-wind_QBMVAE_2_anchor_sa_neg_sa_0_vs_000",
      "gain": 150,
      "platform_record": "platform_export/task.json",
      "evidence_file": "platform_export/task.png"
    }
  ]
}
```

Then run:

```powershell
D:\anaconda\envs\wsy\python.exe -m GEFcom2014.models.QBM_VAE.kaiwu_platform_receipt_batch `
  --manifest platform_export\receipt_manifest.json `
  --protocol configs\paper\fa_bm_vae_opsd_sa_negative_kaiwu_protocol_10read_v1.json `
  --preparation export\opsd_sa_negative_kaiwu_preparation_10read_v1 `
  --verified-by wsy
```

The receipt records the source for each platform, local response, matrix, and
screenshot hash field. A screenshot hash proves which image bytes were bound;
visual content still requires human review.

## Recovery and TEST gate

On `--resume`, an existing response is skipped only after platform receipt,
sample identity/read validation, latent diagnostic recomputation, and VS
prediction validation. A pending intent is changed to `completed` only after
those checks pass. If an identity is unresolved or a validation fails, later
submissions stop and the task requires manual review; the driver does not
retry it.

TEST preparation requires all 30 physical VS responses, the frozen VS-only
gain selection bound to this protocol/model/data/matrix identity, and a
separate `--confirm-test`. TEST outcomes are never used to choose the gain.
