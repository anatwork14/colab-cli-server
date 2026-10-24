# Contributing

Contributions are welcome. Keep this a small community companion to Google's official [Colab CLI](https://github.com/googlecolab/google-colab-cli), not a duplicate cloud API.

- Use a focused feature branch and PR.
- Follow `AGENTS.md` and `docs/SECURITY.md`.
- Never commit tokens, OAuth codes, private datasets, session proxy URLs or identifying infrastructure details.
- All automated tests must be offline, with no VM provisioning and no paid compute consumption.
- Check `python3 -m unittest discover -s tests -v` and `bash -n scripts/*.sh`.
- Disclose live test scope/usage and obtain separate approval if live testing is truly needed.
- Document upstream version assumptions and provide links to official documentation.

This is not a Google-sponsored project. Do not imply affiliation or endorsement.
