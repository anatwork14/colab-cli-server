# Authentication and account security

**CLI-to-Colab authentication** is distinct from **Google Cloud credentials inside the VM**. `colab auth -s SESSION` configures the *remote runtime* for GCP/Drive services, and does **not** fix CLI-side 401/403 errors.

## Recommended: remote browser OAuth2

```bash
umask 077
bash scripts/authenticate.sh
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```

Use the **Google account holding the Colab subscription**. The upstream OAuth2 flow prints a Google authorization URL to your server terminal. Open that URL locally, complete consent in a browser, and paste the resulting code back into the terminal. This works with a headless server; no browser needs to run on the server itself.

The upstream CLI caches credentials at `~/.config/colab-cli/token.json`; do not commit it, display it, forward it to an AI model, or share it with other users. For an already-created token:

```bash
chmod 700 "$HOME/.config/colab-cli"
chmod 600 "$HOME/.config/colab-cli/token.json"
```

The wrapper always explicitly uses `--auth=oauth2` by default because older upstream documents have differed on the default.

## Alternative: Google Application Default Credentials (ADC)

For advanced environments with gcloud installed, authenticate the actual Colab user:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
colab --auth=adc sessions
```

ADC stored under a user does not automatically grant a service account access to a paid Colab subscription. Never hand a user refresh token to an autonomous agent.

## Human vs. agent boundary

Agents may validate package presence and provide sign-in instructions. The human should complete browser consent. A 401/403 typically requires correcting identity or scopes, not allocating more runtimes. Keep OAuth codes, refresh tokens and session proxy tokens out of public issues, agent transcripts, build logs and Git.
