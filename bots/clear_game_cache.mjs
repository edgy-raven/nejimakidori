import {spawn} from 'node:child_process';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {BrowserTransport} from '../live/web/browser_transport.mjs';

export async function clearGameCache(chromePath, profilePath,
  origin = 'https://mahjongsoul.game.yo-star.com') {
  const failed = Promise.withResolvers();
  const closed = Promise.withResolvers();
  const loaded = Promise.withResolvers();
  const browser = spawn(chromePath, [
    '--headless=new', '--remote-debugging-pipe',
    `--user-data-dir=${path.resolve(profilePath)}`,
    `--password-store=${process.env.MAHJONG_SOUL_PASSWORD_STORE || 'basic'}`,
    '--disable-background-networking', '--disable-component-update',
    '--no-first-run', 'about:blank',
  ], {stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe']});
  let stderr = '';
  let closing = false;
  browser.stderr.on('data', bytes => { stderr = (stderr + bytes).slice(-4000); });
  const transport = new BrowserTransport(message => {
    if (message.method === 'Fetch.requestPaused') {
      transport.send('Fetch.fulfillRequest', {
        requestId: message.params.requestId,
        responseCode: 200,
        responseHeaders: [{name: 'Content-Type', value: 'text/html'}],
        body: Buffer.from('<!doctype html><title>Cache maintenance</title>').toString('base64'),
      }, message.sessionId).catch(failed.reject);
    }
    if (message.method === 'Page.loadEventFired') loaded.resolve();
  });
  transport.attach(browser);
  browser.on('error', error => { failed.reject(error); closed.resolve(); });
  browser.once('exit', () => {
    closed.resolve();
    transport.close();
    if (!closing) failed.reject(new Error(`Maintenance browser exited: ${stderr}`));
  });
  const timeout = setTimeout(() => failed.reject(
    new Error('Cache maintenance timed out')), 70000);
  try {
    await Promise.race([failed.promise, (async () => {
      const {targetId} = await transport.send('Target.createTarget', {url: 'about:blank'});
      const {sessionId} = await transport.send('Target.attachToTarget', {targetId, flatten: true});
      // Interception is installed before navigation; no game or login request
      // reaches the network while the profile is open for maintenance.
      await transport.send('Fetch.enable', {patterns: [{urlPattern: '*', requestStage: 'Request'}]}, sessionId);
      await transport.send('Page.enable', {}, sessionId);
      await transport.send('Page.navigate', {url: origin}, sessionId);
      await loaded.promise;
      const result = await transport.send('Runtime.evaluate', {
        awaitPromise: true, returnByValue: true,
        expression: `(async () => {
          for (const name of ['/idbfs', 'UnityCache']) {
            await new Promise((resolve, reject) => {
              const request = indexedDB.deleteDatabase(name);
              request.onsuccess = resolve;
              request.onerror = () => reject(request.error);
              request.onblocked = () => reject(new Error('Database remains open: ' + name));
            });
          }
          return true;
        })()`,
      }, sessionId);
      if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    })()]);
  } finally {
    clearTimeout(timeout);
    closing = true;
    const killTimer = setTimeout(() => browser.kill('SIGKILL'), 3000);
    if (browser.exitCode === null && browser.signalCode === null) {
      await transport.send('Browser.close');
    }
    await closed.promise;
    transport.close();
    clearTimeout(killTimer);
    browser.stdio[3].destroy();
    browser.stdio[4].destroy();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const [, , chromePath, profilePath] = process.argv;
  if (!chromePath || !profilePath) throw new Error('Usage: clear_game_cache.mjs CHROME PROFILE');
  await clearGameCache(chromePath, profilePath);
}
