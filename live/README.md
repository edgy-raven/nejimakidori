# Deployed live controllers

This directory contains ranked/friendly controllers and their vision worker.
Replay review and live display are a separate `../review/` product.

The ranked entry point is `web/mahjongsoul_live.mjs`; friendly accounts use
`web/mahjongsoul_friendly_live.mjs`. They share verified gameplay targeting.
Ranked automation includes Manila queue hours, scheduled browser hibernation,
rank/copper room eligibility, One More Match confirmation, and Discord reports.

Install JavaScript dependencies with `npm ci` in `web/`, and provide Python
vision dependencies from `../vision/requirements.txt`. Account profiles,
credentials, control PINs and notification ledgers are runtime state and are
not included. Configure paths and identities through the environment; see
[the live web documentation](web/README.md).

Verification:

```sh
cd web
npm run test:controller
cd ..
python -m unittest discover -s live/tests/vision
```
