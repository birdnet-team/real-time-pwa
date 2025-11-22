const CACHE_NAME = "birdnet-rt-v0.1.0";
const MODEL_CACHE = CACHE_NAME + "-model";

// Dev flags
const ENABLE_CACHING = true;        // set false to disable all caching
const FORCE_CLEAR_ON_ACTIVATE = false; // set true to nuke old caches every reload

// All core local assets
const CORE_URLS = [
  "./",
  "favicon.png",
  "manifest.webmanifest",
  "css/main.css",
  "vendor/bootstrap/bootstrap.min.css",
  "vendor/bootstrap-icons/bootstrap-icons.css",
  "vendor/bootstrap-icons/fonts/bootstrap-icons.woff2",
  "vendor/bootstrap-icons/fonts/bootstrap-icons.woff",
  "vendor/d3/d3.min.js",
  "vendor/bootstrap/bootstrap.bundle.min.js",
  "js/app.js",
  "js/birdnet-worker.js",
  "js/tfjs-4.14.0.min.js"
];

// Model + label assets
const MODEL_URLS = [
  "models/birdnet/group1-shard1of13.bin",
  "models/birdnet/group1-shard2of13.bin",
  "models/birdnet/group1-shard3of13.bin",
  "models/birdnet/group1-shard4of13.bin",
  "models/birdnet/group1-shard5of13.bin",
  "models/birdnet/group1-shard6of13.bin",
  "models/birdnet/group1-shard7of13.bin",
  "models/birdnet/group1-shard8of13.bin",
  "models/birdnet/group1-shard9of13.bin",
  "models/birdnet/group1-shard10of13.bin",
  "models/birdnet/group1-shard11of13.bin",
  "models/birdnet/group1-shard12of13.bin",
  "models/birdnet/group1-shard13of13.bin",
  "models/birdnet/model.json",
  "models/birdnet/area-model/group1-shard1of2.bin",
  "models/birdnet/area-model/group1-shard2of2.bin",
  "models/birdnet/area-model/model.json",
  "models/birdnet/labels/en_us.txt",
  "models/birdnet/labels/en_uk.txt",
  "models/birdnet/labels/de.txt",
  "models/birdnet/labels/fr.txt",
  "models/birdnet/labels/es.txt",
  "models/birdnet/labels/it.txt"
];

async function purgeAllCaches() {
  const keys = await caches.keys();
  await Promise.all(keys.map(k => caches.delete(k)));
  console.log("[SW] Caches purged.");
}

self.addEventListener("install", (event) => {
  self.skipWaiting();
  if (!ENABLE_CACHING) return;
  event.waitUntil((async () => {
    const core = await caches.open(CACHE_NAME);
    await core.addAll(CORE_URLS);
    const modelCache = await caches.open(MODEL_CACHE);
    for (const url of MODEL_URLS) {
      try {
        const resp = await fetch(url, { cache: "no-cache" });
        if (resp.ok) await modelCache.put(url, resp);
      } catch (e) {
        console.warn("[SW] Model precache skipped:", url);
      }
    }
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    if (!ENABLE_CACHING || FORCE_CLEAR_ON_ACTIVATE) {
      await purgeAllCaches();
    } else {
      const keep = new Set([CACHE_NAME, MODEL_CACHE]);
      const keys = await caches.keys();
      await Promise.all(keys.filter(k => !keep.has(k)).map(k => caches.delete(k)));
    }
    clients.claim();
  })());
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "PURGE_CACHES") {
    purgeAllCaches().then(() => {
      event.source && event.source.postMessage({ type: "PURGE_DONE" });
    });
  }
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  if (!ENABLE_CACHING) {
    event.respondWith(fetch(request).catch(() => new Response("Offline (caching disabled)", { status: 503 })));
    return;
  }

  const url = new URL(request.url);
  const scopePath = new URL(self.registration.scope).pathname;
  let rel = url.pathname.startsWith(scopePath)
    ? url.pathname.slice(scopePath.length)
    : url.pathname.replace(/^\/+/, "");

  if (MODEL_URLS.includes(rel)) {
    event.respondWith((async () => {
      const cache = await caches.open(MODEL_CACHE);
      const cached = await cache.match(rel);
      if (cached) {
        fetch(request).then(r => r.ok && cache.put(rel, r)).catch(()=>{});
        return cached;
      }
      const net = await fetch(request);
      if (net.ok) cache.put(rel, net.clone());
      return net;
    })());
    return;
  }

  if (CORE_URLS.includes(rel)) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(rel);
      if (cached) {
        fetch(request).then(r => r.ok && cache.put(rel, r)).catch(()=>{});
        return cached;
      }
      const net = await fetch(request);
      if (net.ok) cache.put(rel, net.clone());
      return net;
    })());
    return;
  }

  event.respondWith((async () => {
    try {
      return await fetch(request);
    } catch {
      const cache = await caches.open(CACHE_NAME);
      const fallback = await cache.match("./");
      return fallback || Response.error();
    }
  })());
});
