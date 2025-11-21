const CACHE_NAME = "birdnet-rt-v1";
const OFFLINE_URLS = [
  "/",
  "/css/main.css",
  "/js/app.js",
  "/manifest.webmanifest"
];
const DISABLE_CACHE = true;

self.addEventListener("install", (event) => {
  self.skipWaiting();
  if (DISABLE_CACHE) return;
  event.waitUntil(
    caches.open(CACHE_NAME).then((c) => c.addAll(OFFLINE_URLS))
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      if (DISABLE_CACHE) {
        const keys = await caches.keys();
        await Promise.all(keys.map((k) => caches.delete(k)));
      } else {
        const keys = await caches.keys();
        await Promise.all(
          keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
        );
      }
      clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  if (DISABLE_CACHE) {
    event.respondWith(fetch(event.request));
    return;
  }
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request).then(
      (cached) =>
        cached ||
        fetch(event.request).catch(() => {
          if (event.request.mode === "navigate") return caches.match("/");
        })
    )
  );
});
