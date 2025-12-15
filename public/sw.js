/**
 * BirdNET Live - Service Worker
 * Handles offline caching of app assets, model files, and species images.
 * Implements different caching strategies based on resource type.
 */

/* =========================================================================
   1. CONFIGURATION & VERSIONING
   ========================================================================== */

const APP_VERSION = "v0.1.20";   // Increment on app code changes
const MODEL_VERSION = "v2.4";    // Increment only when model files change

const APP_CACHE_NAME = `birdnet-app-${APP_VERSION}`;
const MODEL_CACHE_NAME = `birdnet-model-${MODEL_VERSION}`;
const IMAGE_CACHE_NAME = "birdnet-images-v1";

const ENABLE_CACHING = true;
const FORCE_CLEAR_ON_ACTIVATE = false;

/* =========================================================================
   2. ASSET MANIFESTS
   ========================================================================== */

// Core Application Assets (UI, Logic, Styles)
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
  "js/tfjs-4.14.0.min.js",
  "locales/en.json",
  "locales/de.json",
  "locales/fr.json",
  "locales/it.json",
  "locales/es.json",
  "locales/pt.json",
  "locales/nl.json",
  "locales/ru.json"
];

// Model Files & Labels (Large, rarely changed)
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

/* =========================================================================
   3. LIFECYCLE EVENTS
   ========================================================================== */

self.addEventListener("install", (event) => {
  self.skipWaiting(); // Activate immediately
  if (!ENABLE_CACHING) return;

  event.waitUntil((async () => {
    // 1. Cache App Core
    const appCache = await caches.open(APP_CACHE_NAME);
    await appCache.addAll(CORE_URLS);
    
    // 2. Cache Model (separately to avoid re-downloading on app updates)
    const modelCache = await caches.open(MODEL_CACHE_NAME);
    for (const url of MODEL_URLS) {
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
      // Cleanup old caches
      const keep = new Set([APP_CACHE_NAME, MODEL_CACHE_NAME, IMAGE_CACHE_NAME]);
      const keys = await caches.keys();
      await Promise.all(keys.filter(k => !keep.has(k)).map(k => caches.delete(k)));
    }
    clients.claim(); // Take control of clients immediately
  })());
});

/* =========================================================================
   4. FETCH STRATEGIES
   ========================================================================== */

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // A. External API Images (BirdNET API) -> Cache First
  if (url.href.startsWith("https://birdnet.cornell.edu/api2/bird/")) {
    event.respondWith(handleImageFetch(request));
    return;
  }

  if (!ENABLE_CACHING) {
    event.respondWith(fetch(request).catch(() => new Response("Offline", { status: 503 })));
    return;
  }

  // Determine relative path for local assets
  const scopePath = new URL(self.registration.scope).pathname;
  let rel = url.pathname.startsWith(scopePath)
    ? url.pathname.slice(scopePath.length)
    : url.pathname.replace(/^\/+/, "");

  // B. Model Files -> Cache First
  if (MODEL_URLS.includes(rel)) {
    event.respondWith(handleCacheFirst(request, MODEL_CACHE_NAME, rel));
    return;
  }

  // C. Core App Assets -> Stale-While-Revalidate
  if (CORE_URLS.includes(rel)) {
    event.respondWith(handleStaleWhileRevalidate(request, APP_CACHE_NAME, rel));
    return;
  }

  // D. Navigation/Others -> Network First with Fallback
  event.respondWith(handleNetworkFirst(request));
});

/* =========================================================================
   5. STRATEGY IMPLEMENTATIONS
   ========================================================================== */

async function handleImageFetch(request) {
  const cache = await caches.open(IMAGE_CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  
  try {
    const net = await fetch(request);
    if (net.ok) cache.put(request, net.clone());
    return net;
  } catch {
    return new Response("", { status: 404 }); // Fail silently
  }
}

async function handleCacheFirst(request, cacheName, relPath) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(relPath || request);
  if (cached) return cached;
  
  const net = await fetch(request);
  if (net.ok) cache.put(relPath || request, net.clone());
  return net;
}

async function handleStaleWhileRevalidate(request, cacheName, relPath) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(relPath || request);
  
  const fetchPromise = fetch(request).then(networkResponse => {
    if (networkResponse.ok) {
      cache.put(relPath || request, networkResponse.clone());
    }
    return networkResponse;
  }).catch(() => { /* ignore network errors if cached */ });

  return cached || fetchPromise;
}

async function handleNetworkFirst(request) {
  try {
    return await fetch(request);
  } catch {
    const cache = await caches.open(APP_CACHE_NAME);
    // Fallback to root for navigation
    const fallback = await cache.match("./");
    return fallback || Response.error();
  }
}

/* =========================================================================
   6. UTILITIES
   ========================================================================== */

async function purgeAllCaches() {
  const keys = await caches.keys();
  await Promise.all(keys.map(k => caches.delete(k)));
  console.log("[SW] Caches purged.");
}

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "PURGE_CACHES") {
    purgeAllCaches().then(() => {
      event.source && event.source.postMessage({ type: "PURGE_DONE" });
    });
  }
});
