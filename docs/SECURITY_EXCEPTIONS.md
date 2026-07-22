# Accepted `pip-audit` exceptions

This file records vulnerability IDs that CI ignores after review. Each entry
must include rationale and an expiry date. Do not add ignores without updating
`docs/pip-audit-ignores.txt` (the machine-readable list consumed by CI).

## Active exceptions

### Torch advisories under `torch>=2.4.0,<2.6`

| Field | Value |
|-------|--------|
| Package | `torch` (resolved 2.5.1 in a clean `[dev]` install on 2026-07-22) |
| Constraint | `torch>=2.4.0,<2.6` (Phase C / T3.1; characterization suite proven on 2.5.1) |
| Expiry | **2026-10-22** |
| Rationale | Published fixes require torch 2.6–2.13, which is outside the current upper bound. Raising the bound needs a dedicated oracle re-proof, not a silent CI ignore expansion. |

Ignored IDs (also listed in `docs/pip-audit-ignores.txt`):

- `CVE-2025-2148`, `CVE-2025-2149`, `CVE-2025-2998`, `CVE-2025-2999`, `CVE-2025-3001`
- `PYSEC-2025-41`, `PYSEC-2025-191`, `PYSEC-2025-194`, `PYSEC-2025-198`
- `PYSEC-2025-203`, `PYSEC-2025-204`, `PYSEC-2025-205`, `PYSEC-2025-206`
- `PYSEC-2025-207`, `PYSEC-2025-208`, `PYSEC-2025-209`
- `PYSEC-2026-139`, `PYSEC-2026-1970`, `PYSEC-2026-2286`

**Follow-up:** Before expiry, re-run the characterization suite against a wider
torch range (for example `>=2.4,<2.8` or higher as wheels allow), then remove
these ignores and tighten or drop this exception.

## Local audit

```bash
pip install -e ".[dev]"
ignore_args=()
while IFS= read -r id; do
  [[ -z "$id" || "$id" =~ ^# ]] && continue
  ignore_args+=(--ignore-vuln "$id")
done < docs/pip-audit-ignores.txt
pip-audit "${ignore_args[@]}"
```
