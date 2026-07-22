# Security policy

## Supported versions

Security fixes are considered for the current published release line on PyPI
(the latest `4.1.x` / compatibility series under active maintenance).

## Reporting a vulnerability

Please report security vulnerabilities **privately** by email to
travis.j.kessler@gmail.com.

Do **not** open a public GitHub issue for security bugs. Include a clear
description of the issue, steps to reproduce when possible, and any known
impact.

You should expect an acknowledgment within a reasonable time. We will discuss
next steps, including disclosure timing, with the reporter.

## Dependency advisories

Known accepted dependency audit exceptions for CI `pip-audit` (for example
torch advisories under the current upper bound) are documented in
`docs/SECURITY_EXCEPTIONS.md`. That file is for dependency policy tracking; it
is not the channel for reporting new project vulnerabilities.
