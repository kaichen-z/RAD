# Security policy

Please report suspected vulnerabilities or accidentally exposed credentials privately to the
maintainers through GitHub's security advisory feature. Do not open a public issue containing a
secret, personal dataset path, or private sample.

The repository never requires credentials in source files. Supply model-service credentials only
through the provider's documented environment variables and keep `.env` files local. If a secret
is committed, revoke it immediately; deleting the current file does not remove the secret from Git
history.

Only the latest `main` branch is supported. Third-party methods, checkpoints, hosted dataset files,
and external model services follow their own security and support policies.
