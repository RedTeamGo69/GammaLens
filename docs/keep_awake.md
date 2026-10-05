# Gamma Lens availability

`Keep Gamma Lens Awake` opens a short Chromium browser session every four
hours, including weekends, at minute 17 UTC. It can click Community Cloud's
"Yes, get this app back up!" button and waits for the Forward Test results
view and a receiving Streamlit websocket. A sleep page or login page is not
a successful check. Visits retry up to three times and the job is capped at
15 minutes.

The Forward Test route reads stored results. The visit does not click
refresh, refit models, fetch option chains, or run capture/scoring jobs.
Browser tooling is isolated in `.github/keep-awake`; the app's Python
dependencies and existing data workflows do not change.

## Activation and control

1. For the private deployment, run `npm ci`, `npx playwright install chromium`,
   and `npm run sign-in` from `.github/keep-awake` on Windows. Sign in in the
   temporary browser. The helper waits for the actual results view, then saves
   only exact-app-host cookies to
   `%LOCALAPPDATA%\GammaLens\keep-awake\viewer-session.json`, outside the repo.
   It discards identity-provider/workspace cookies and all local storage.
2. With the owner's approval, upload that JSON as the GitHub Actions secret
   `STREAMLIT_VIEWER_STATE`. Set the Actions variable `STREAMLIT_APP_URL` to
   `https://gex-q-dashboard.streamlit.app/`. Never commit the session file.
   Before activation, verify that these app-only cookies work in a fresh
   browser. If they do not, stop; do not expand cookie scope automatically.
3. Publish this workflow on `main`, then run **Actions → Keep Gamma Lens
   Awake → Run workflow** and confirm the job succeeds.

An unset URL skips the job. Remove the variable or disable the workflow in
Actions to stop visits. Private-app sessions may expire or be revoked. If a
run fails at the login screen, repeat the local sign-in and replace the secret.
The secret grants app access to Actions and to anyone able to modify workflows
in this repository; keep repository write access limited. App privacy is not
changed. If the deployment later becomes public, delete the viewer secret.

This is a best-effort availability workaround, not an always-on hosting
guarantee. GitHub may delay or drop scheduled runs and disables public-repo
schedules after 60 days without repository activity. Standard GitHub failed
workflow notifications depend on the account's notification settings.
No separate notification service is installed.

Sources:
- [Streamlit app hibernation](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation)
- [GitHub scheduled workflow limitations](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
- [Playwright on CI](https://playwright.dev/docs/ci)
- [Playwright saved authentication state](https://playwright.dev/docs/auth)
