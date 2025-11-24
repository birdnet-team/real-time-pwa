// Versioning
const APP_VERSION = "v0.1.10"; // Increment on any app code change
const MODEL_VERSION = "v2.4"; // Increment only when model files change

const APP_CACHE_NAME = `birdnet-app-${APP_VERSION}`;
const MODEL_CACHE_NAME = `birdnet-model-${MODEL_VERSION}`;
const IMAGE_CACHE_NAME = "birdnet-images-v1";

// Dev flags
const ENABLE_CACHING = true;
const FORCE_CLEAR_ON_ACTIVATE = false;

// All core local assets
const CORE_URLS = [
  "./",
  "explore/",
  "about/",
  "legal/",
  "share/",
  "favicon.png",
  "manifest.webmanifest",
  "img/birdnet-logo-circle.png",
  "img/qrcode.png",
  "img/dummy.webp",
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
    // Cache App Core
    const appCache = await caches.open(APP_CACHE_NAME);
    await appCache.addAll(CORE_URLS);
    
    // Cache Model (separately)
    const modelCache = await caches.open(MODEL_CACHE_NAME);
    for (const url of MODEL_URLS) {
      // Check if already cached to avoid re-fetching large files
      const match = await modelCache.match(url);
      if (!match) {
        try {
          const resp = await fetch(url, { cache: "no-cache" });
          if (resp.ok) await modelCache.put(url, resp);
        } catch (e) {
          console.warn("[SW] Model precache skipped:", url);
        }
      }
    }
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    if (!ENABLE_CACHING || FORCE_CLEAR_ON_ACTIVATE) {
      await purgeAllCaches();
    } else {
      // Keep only current app version and current model version
      const keep = new Set([APP_CACHE_NAME, MODEL_CACHE_NAME]);
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

  const url = new URL(request.url);

  // Strategy: Cache First for BirdNET API Images
  if (url.href.startsWith("https://birdnet.cornell.edu/api2/bird/")) {
    event.respondWith((async () => {
      const cache = await caches.open(IMAGE_CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;
      
      try {
        const net = await fetch(request);
        if (net.ok) cache.put(request, net.clone());
        return net;
      } catch {
        // Return transparent pixel or nothing on offline fail
        return new Response("", { status: 404 });
      }
    })());
    return;
  }

  if (!ENABLE_CACHING) {
    event.respondWith(fetch(request).catch(() => new Response("Offline (caching disabled)", { status: 503 })));
    return;
  }

  const scopePath = new URL(self.registration.scope).pathname;
  let rel = url.pathname.startsWith(scopePath)
    ? url.pathname.slice(scopePath.length)
    : url.pathname.replace(/^\/+/, "");

  // Strategy: Cache First for Model
  if (MODEL_URLS.includes(rel)) {
    event.respondWith((async () => {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const cached = await cache.match(rel);
      if (cached) return cached;
      
      const net = await fetch(request);
      if (net.ok) cache.put(rel, net.clone());
      return net;
    })());
    return;
  }

  // Strategy: Stale-While-Revalidate for App Core
  if (CORE_URLS.includes(rel)) {
    event.respondWith((async () => {
      const cache = await caches.open(APP_CACHE_NAME);
      const cached = await cache.match(rel);
      
      const fetchPromise = fetch(request).then(networkResponse => {
        if (networkResponse.ok) {
          cache.put(rel, networkResponse.clone());
        }
        return networkResponse;
      }).catch(() => { /* eat errors if cached exists */ });

      return cached || fetchPromise;
    })());
    return;
  }

  // Strategy: Network First (fallback to App Cache)
  event.respondWith((async () => {
    try {
      return await fetch(request);
    } catch {
      const cache = await caches.open(APP_CACHE_NAME);
      const fallback = await cache.match("./");
      return fallback || Response.error();
    }
  })());
});
