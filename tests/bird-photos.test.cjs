const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createLookup, createRecovery, selectPhoto } = require('../public/js/bird-photos.js');
const photo = { id: 4608133, medium_url: 'https://inaturalist-open-data.s3.amazonaws.com/photos/4608133/medium.jpg', attribution: '(c) Ximo Galarza, CC BY-NC-SA', license_code: 'cc-by-nc-sa' };
const result = { name: 'Passer domesticus', default_photo: photo };
test('requires an exact scientific-name match and a credited reusable photo', () => {
  assert.equal(selectPhoto([result], ' passer DOMESTICUS ').url, photo.medium_url);
  assert.equal(selectPhoto([result], 'Passer montanus'), null);
  assert.equal(selectPhoto([{ ...result, default_photo: { ...photo, license_code: null } }], result.name), null);
  assert.equal(selectPhoto([{ ...result, default_photo: { ...photo, attribution: '' } }], result.name), null);
  assert.equal(selectPhoto([{ ...result, default_photo: { ...photo, medium_url: 'https://example.com/photo.jpg' } }], result.name), null);
});
test('deduplicates concurrent lookups and omits credentials', async () => {
  let requests = 0;
  const lookup = createLookup(async (url, options) => {
    requests++;
    assert.equal(new URL(url).searchParams.get('q'), result.name);
    assert.equal(options.credentials, 'omit');
    return { ok: true, json: async () => ({ results: [result] }) };
  });
  const [a, b] = await Promise.all([lookup(result.name), lookup('passer domesticus')]);
  assert.equal(requests, 1);
  assert.deepEqual(a, b);
});
test('limits parallel requests and retries transient failures', async () => {
  let active = 0, peak = 0, requests = 0;
  const lookup = createLookup(async () => {
    requests++; active++; peak = Math.max(peak, active);
    await new Promise(resolve => setTimeout(resolve, 5));
    active--;
    return { ok: false };
  }, { concurrency: 2 });
  await Promise.all(['a', 'b', 'c', 'd'].map(lookup));
  assert.equal(peak, 2);
  await lookup('a');
  assert.equal(requests, 5);
});
test('aborts a stalled lookup', async () => {
  const lookup = createLookup((url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('aborted')));
  }), { timeout: 5 });
  assert.equal(await lookup('Passer domesticus'), null);
});
function fixture() {
  const content = { children: [], querySelector() { return this.children[0]; }, appendChild(node) { this.children.push(node); } };
  const document = { createTextNode: text => ({ text }), createElement: () => ({ children: [], appendChild(node) { this.children.push(node); } }) };
  const img = { dataset: { birdSpecies: 'Passer%20domesticus' }, isConnected: true, closest: () => ({ querySelector: () => content }) };
  return { img, document, content };
}
test('renders attribution only after replacement loads; handles a failed replacement', async () => {
  const { img, document, content } = fixture();
  await createRecovery(async () => selectPhoto([result], result.name), document, () => '/real-time-pwa/')(img);
  assert.equal(img.src, photo.medium_url);
  assert.equal(content.children.length, 0);
  img.onload();
  assert.equal(content.children[0].children[1].href, 'https://www.inaturalist.org/photos/4608133');
  img.onerror();
  assert.equal(img.src, '/real-time-pwa/img/dummy.webp');
  img.onerror();
  assert.equal(img.hidden, true);
});
test('uses the app root placeholder on Explore and ignores detached cards', async () => {
  const { img, document } = fixture();
  await createRecovery(async () => null, document, () => '/real-time-pwa/')(img);
  assert.equal(img.src, '/real-time-pwa/img/dummy.webp');
  img.src = ''; img.isConnected = false;
  await createRecovery(async () => selectPhoto([result], result.name), document, () => '/real-time-pwa/')(img);
  assert.equal(img.src, '');
});
