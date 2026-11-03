# Kaiwu canary task-name correction — 2026-11-02

The first OPSD v8 canary attempt reached the SDK's task-creation endpoint,
which reported `(1406, "Data too long for column 'name' at row 1")`.
The frozen task name is 148 ASCII characters. No samples or platform task ID
were saved by this attempt. Platform task existence and billing still require
inspection of the account; the error alone is not a balance audit.

The adapter now sends a stable name of at most 32 ASCII bytes. Long names map
to `fbv_` plus the first 28 hexadecimal characters of SHA-256 of the complete
frozen task name. This is a conservative local naming policy, not a verified
vendor maximum. This canary's platform name is
`fbv_d2f9be1aa975a63bdbee9035c1fc`.

`task_name` remains the frozen experiment identity; `platform_task_name` records
the actual SDK name in new raw responses, audits, and failure evidence. Manual
platform verification checks the short name against the platform record and
recomputes its mapping to the frozen identity. Historical raw responses without
this field retain their original full-name semantics. SDK cache subdirectories
also use the short name; nonempty caches under either name still block reuse.

No protocol, model, matrix, preparation manifest, or frozen VS execution order
was changed. Existing failure evidence and pending intent remain in place.
The driver still refuses another real submission from this state. Do not delete
the failure or pending intent, use a new directory to bypass them, or rerun with
`--resume` merely to clear this failure. First inspect the platform task list
and balance, then explicitly review recovery of this one attempt. No paid retry
was executed during this correction.

Validation: 121 passed (14 dependency warnings), using the `wsy` Python and
`tests/test_kaiwu_adapter.py`, `tests/test_kaiwu_platform_receipt.py`,
`tests/test_submit_kaiwu_safety.py`, `tests/test_kaiwu_batch_driver.py`, and
`tests/test_kaiwu_response_recovery_validation.py`. Tests use local fixtures and
mocked submission calls; they do not establish platform acceptance of the new
name. They cover bounded SDK names, distinct identities with identical prefixes,
raw/audit persistence, old and new cache guards, receipt mismatch rejection,
formal import, and interrupted-response recovery.

## Reviewed recovery

The user subsequently confirmed: “没有新任务，余额未变” (no new task and
unchanged balance). Local inspection found exactly the reviewed creation-failure
JSON and its matching pending intent, with no sample files or populated SDK
cache. These two files were moved to
`export/opsd_kaiwu_recovery_20261102_name_length/`; their original SHA-256 hashes
were verified after the move. `recovery_record.json` records the user testimony,
original paths, archived paths, hashes, and old/new task names. No platform
query or real submission was performed by this recovery. The prior local
failure lock is now cleared for this one reviewed canary; protocol, preparation,
response and checkpoint locations remain unchanged.
