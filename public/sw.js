const CACHE_NAME = "birdnet-rt-v2";
const CORE_URLS = [
  "/",
  "/css/main.css",
  "/js/app.js",
  "/manifest.webmanifest"
];

// Model files to cache
const MODEL_URLS = [
  "/public/models/birdnet/group1-shard1of13.bin",
  "/public/models/birdnet/group1-shard2of13.bin",
  "/public/models/birdnet/group1-shard3of13.bin",
  "/public/models/birdnet/group1-shard4of13.bin",
  "/public/models/birdnet/group1-shard5of13.bin",
  "/public/models/birdnet/group1-shard6of13.bin",
  "/public/models/birdnet/group1-shard7of13.bin",
  "/public/models/birdnet/group1-shard8of13.bin",
  "/public/models/birdnet/group1-shard9of13.bin",
  "/public/models/birdnet/group1-shard10of13.bin",
  "/public/models/birdnet/group1-shard11of13.bin",
  "/public/models/birdnet/group1-shard12of13.bin",
  "/public/models/birdnet/group1-shard13of13.bin",
  "/public/models/birdnet/model.json",
  "/public/models/birdnet/area-model/group1-shard1of2.bin",
  "/public/models/birdnet/area-model/group1-shard2of2.bin",
  "/public/models/birdnet/area-model/model.json",
  "/public/models/birdnet/labels/en_us.txt",
  "/public/models/birdnet/labels/en_uk.txt",
  "/public/models/birdnet/labels/de.txt",
  "/public/models/birdnet/labels/fr.txt",
  "/public/models/birdnet/labels/es.txt",
  "/public/models/birdnet/labels/it.txt"
];

const DISABLE_CACHE = false;

// Precache core + model (model fetched once then reused)
self.addEventListener("install", (event) => {
  self.skipWaiting();
  if (DISABLE_CACHE) return;
  event.waitUntil(
    (async () => {
      const coreCache = await caches.open(CACHE_NAME);
      await coreCache.addAll(CORE_URLS);
      // Fetch & cache model files individually to handle failures gracefully
      const modelCache = await caches.open(MODEL_CACHE_NAME);
      await Promise.all(
        MODEL_URLS.map(async (url) => {
          try {
            const resp = await fetch(url, { integrity: "", cache: "no-cache" });
            if (resp.ok) await modelCache.put(url, resp);
          } catch (e) {
            // swallow; will be cached later on first successful fetch
            console.warn(`Model file failed to cache during install: ${url}`, e);
          }
        })
      );
    })()
  );
});

// Clean old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keep = new Set([CACHE_NAME, MODEL_CACHE_NAME]);
      const keys = await caches.keys();
      await Promise.all(keys.filter(k => !keep.has(k)).map(k => caches.delete(k)));
      clients.claim();
    })()
  );
});

// Runtime caching: serve model from cache after first successful fetch
self.addEventListener("fetch", (event) => {
  if (DISABLE_CACHE) {
    event.respondWith(fetch(event.request));
    return;
  }
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  const isModel = MODEL_URLS.includes(url.pathname);

  if (isModel) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(MODEL_CACHE_NAME);
        const cached = await cache.match(url.pathname);
        if (cached) {
          // Stale-while-revalidate in background
            fetch(request).then(r => r.ok && cache.put(url.pathname, r)).catch(() => {});
          return cached;
        }
        // First load: fetch then cache
        const network = await fetch(request);
        if (network.ok) cache.put(url.pathname, network.clone());
        return network;
      })()
    );
    return;
  }

  // Default: cache-first for core assets, network fallback
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;
      try {
        const network = await fetch(request);
        if (network.ok && url.origin === location.origin) {
          cache.put(request, network.clone());
        }
        return network;
      } catch (e) {
        if (request.mode === "navigate") {
          const fallback = await cache.match("/");
          if (fallback) return fallback;
        }
        throw e;
      }
    })()
  );
});
