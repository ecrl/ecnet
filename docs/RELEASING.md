# Releasing ECNet

Maintainers cut compatibility releases with a GitHub Release, which runs
`.github/workflows/release.yml` and uploads to PyPI via **trusted publishing**
(OIDC). No long-lived PyPI API token is stored in GitHub secrets.

## One-time portal setup (human click-through)

Complete these steps once before the first OIDC publish (or after changing
the workflow filename / environment name).

### 1. GitHub Environment

1. Open the repository **Settings → Environments**.
2. Create an environment named exactly `pypi`.
3. Optional but recommended: require reviewers or restrict which branches may
   deploy to `pypi`.
4. Do **not** add a `PYPI_API_TOKEN` secret for this flow.

### 2. PyPI trusted publisher

1. Sign in at [pypi.org](https://pypi.org/) as a project owner/maintainer for
   `ecnet`.
2. Open **Publishing** for the project (or **Pending publisher** if configuring
   before the next upload).
3. Add a GitHub publisher with:
   - **Owner:** `ecrl`
   - **Repository:** `ecnet`
   - **Workflow:** `release.yml`
   - **Environment:** `pypi`
4. Save. PyPI documents the full UI at
   [Trusted publishers](https://docs.pypi.org/trusted-publishers/).

After the first successful OIDC upload, remove any unused legacy
`PYPI_API_TOKEN` from GitHub secrets if one remains.

## Cut a release

1. Ensure `main` (or the release branch) is green on CI and version / CHANGELOG
   match the intended tag (see blueprint Phase 5 / Design E).
2. Push the release commit, then create an annotated git tag
   (for example `4.1.5`) and a GitHub Release for that tag.
3. Publishing the GitHub Release triggers `release.yml`. You may also run the
   workflow manually via **Actions → Release to PyPI → Run workflow**
   (`workflow_dispatch`) on the tagged commit if needed.
4. Confirm the new files appear on [PyPI: ecnet](https://pypi.org/project/ecnet/).
5. Smoke-install in a clean virtualenv:

   ```bash
   python -m venv /tmp/ecnet-smoke && source /tmp/ecnet-smoke/bin/activate
   pip install ecnet==<version>
   python -c "import ecnet; print(ecnet.__version__)"
   ```

## Troubleshooting

- **OIDC / publisher mismatch:** Workflow filename, environment name, owner,
  and repository must match the PyPI trusted-publisher row exactly
  (`release.yml`, environment `pypi`, `ecrl/ecnet`).
- **Environment protection:** If the `pypi` environment requires reviewers,
  approve the deployment in the GitHub Actions UI after the job starts.
- **Do not** reintroduce `password: ${{ secrets.PYPI_API_TOKEN }}` into the
  release workflow.
