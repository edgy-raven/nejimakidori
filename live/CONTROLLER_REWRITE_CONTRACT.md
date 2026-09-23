# Controller rewrite contract

Scope: replace `live/web` only. The deployed controller release and the
separate production model server are not part of this change.

Preserved source: `/tmp/nejimakidori-live-controller-purge.u8qlwp/web`

Restore command (while the backup exists):

```sh
cp -a /tmp/nejimakidori-live-controller-purge.u8qlwp/web live/web
```

The replacement must keep these observable contracts:

- `npm run live`, `npm run friendly`, and `npm run control` remain the
  controller entrypoints and retain their existing environment configuration.
- Ranked play uses the configured ranked-room and schedule data, observes a
  result for two five-second intervals before treating it as failed, pauses on
  unsafe/unknown UI, records evidence, and needs explicit resume.
- Friendly play retains identity verification, room lifecycle, safe hibernation,
  and never uses ranked autoqueue rules.
- Manual control remains available while automation is paused; automatic input
  is never issued after a stale frame or failed action verification.
- The controller consumes the model through its existing external endpoint
  contract only. It does not start, configure, or modify any model server.

Acceptance tests import public controller modules and use fakes at browser,
clock, storage, notifier, and model boundaries. They must never read controller
source text, slice source strings, or execute extracted code in a VM.
