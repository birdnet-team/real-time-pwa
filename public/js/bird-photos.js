/* Optional iNaturalist fallback for unavailable Cornell species photos. */
(function (root) {
  'use strict';
  const normalize = value => String(value || '').trim().toLowerCase();
  const licenses = new Set(['cc0', 'cc-by', 'cc-by-sa', 'cc-by-nc', 'cc-by-nc-sa', 'cc-by-nd', 'cc-by-nc-nd']);

  function selectPhoto(results, scientificName) {
    const taxon = results.find(item => normalize(item.name) === normalize(scientificName));
    const photo = taxon && taxon.default_photo;
    if (!photo || !licenses.has(photo.license_code) || !photo.attribution || !Number.isSafeInteger(photo.id)) return null;
    try {
      const url = new URL(photo.medium_url || photo.url);
      const host = url.hostname;
      if (url.protocol !== 'https:' || !(host === 'inaturalist-open-data.s3.amazonaws.com' || host === 'static.inaturalist.org' || host === 'inaturalist.org' || host.endsWith('.inaturalist.org'))) return null;
      return { url: url.href, attribution: photo.attribution, source: 'https://www.inaturalist.org/photos/' + photo.id };
    } catch (_) { return null; }
  }

  function createLookup(fetcher, { concurrency = 3, timeout = 10000 } = {}) {
    const cache = new Map();
    const queue = [];
    let active = 0;
    function pump() {
      while (active < concurrency && queue.length) {
        const { name, resolve } = queue.shift();
        active++;
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeout);
        (async () => {
          const query = new URLSearchParams({ q: name, taxon_id: '3', per_page: '5' });
          const response = await fetcher('https://api.inaturalist.org/v1/taxa?' + query, { credentials: 'omit', signal: controller.signal });
          if (!response.ok) throw new Error('Photo lookup unavailable');
          const data = await response.json();
          return selectPhoto(Array.isArray(data.results) ? data.results : [], name);
        })().catch(() => { cache.delete(normalize(name)); return null; }).then(resolve).finally(() => {
          clearTimeout(timer);
          active--;
          pump();
        });
      }
    }
    return scientificName => {
      const name = String(scientificName || '').trim();
      if (!name) return Promise.resolve(null);
      const key = normalize(name);
      if (!cache.has(key)) {
        cache.set(key, new Promise(resolve => queue.push({ name, resolve })));
        pump();
      }
      return cache.get(key);
    };
  }

  function createRecovery(lookup, document, getPrefix) {
    return async img => {
      // Disable the original error handler before changing src, avoiding loops.
      img.onerror = null;
      const placeholder = () => {
        img.onload = null;
        img.onerror = () => { img.hidden = true; };
        img.src = getPrefix() + 'img/dummy.webp';
      };
      let name;
      try { name = decodeURIComponent(img.dataset.birdSpecies || ''); } catch (_) { name = ''; }
      try {
        const photo = await lookup(name);
        if (!img.isConnected) return;
        if (!photo) { placeholder(); return; }
        img.onload = () => {
          const content = img.closest('.card')?.querySelector('.card-body');
          if (!content || content.querySelector('.bird-photo-credit')) return;
          const credit = document.createElement('p');
          credit.className = 'bird-photo-credit small text-muted mt-2 mb-0';
          credit.appendChild(document.createTextNode('Photo: ' + photo.attribution + ' '));
          const source = document.createElement('a');
          source.href = photo.source;
          source.target = '_blank';
          source.rel = 'noopener noreferrer';
          source.textContent = 'iNaturalist';
          credit.appendChild(source);
          content.appendChild(credit);
        };
        img.onerror = placeholder;
        img.src = photo.url;
      } catch (_) { placeholder(); }
    };
  }

  const api = { createLookup, createRecovery, selectPhoto };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root && root.document) {
    root.BirdNETPhotos = { recover: createRecovery(createLookup(root.fetch.bind(root)), root.document, () => root.PATH_PREFIX || '/') };
  }
})(typeof window !== 'undefined' ? window : null);
