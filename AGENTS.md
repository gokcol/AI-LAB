# Release notes

For every user-visible update, prepend a dated entry to `gui/content/releases.json`
with a higher version, a short title, and concrete learner-facing changes. Include
lesson links where useful. Preserve previously published entries. This file drives
the homepage's What's new section, release history, and displayed app version.
When the owner explicitly requests keeping the public version unchanged, prepend
an entry with that version and a higher positive integer `revision` (default 1).
Preserve the previous entries exactly; the deployment check orders version/revision pairs.

Public release notes should describe learner-facing changes. Exclude SSH tunnels,
server access instructions, and private owner-console features. If the owner explicitly
requests removal of a published entry, use `--redact-version VERSION` when staging and
activating that release; preserve all other history and prepend a new version as usual.

Before deployment, run `python tools/check_release_notes.py`. The existing-server
release script checks against the live release during staging and activation and
rejects a deployment without a new entry. Deploy through `tools/release_update.sh`
and verify the homepage after activation.

When adding or removing public page routes, run `python tools/build_discovery.py`.
Deployment checks that `public/robots.txt` and `public/sitemap.xml` match the app routes.
The AI Lab nginx server includes `ops/nginx-ai-lab-discovery.conf` (installed in
`/etc/nginx/snippets/ai-lab-discovery.conf`) to serve these files from the active release.
