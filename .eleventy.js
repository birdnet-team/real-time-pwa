module.exports = function (eleventyConfig) {
  // Static (public)
  eleventyConfig.addPassthroughCopy({ "public": "/" });

  // Prevent Jekyll on GitHub Pages
  eleventyConfig.addPassthroughCopy({ "src/.nojekyll": ".nojekyll" });

  // Vendor assets
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap/dist/css/bootstrap.min.css": "vendor/bootstrap/bootstrap.min.css",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap/dist/js/bootstrap.bundle.min.js": "vendor/bootstrap/bootstrap.bundle.min.js",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap-icons/font/bootstrap-icons.css": "vendor/bootstrap-icons/bootstrap-icons.css",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap-icons/font/fonts": "vendor/bootstrap-icons/fonts",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/d3/dist/d3.min.js": "vendor/d3/d3.min.js",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/@tensorflow/tfjs/dist/tf.min.js": "js/tfjs-4.14.0.min.js",
  });

  // Self-hosted ONNX Runtime Web (MIT), WASM backend only, for BirdBox
  for (const file of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
    eleventyConfig.addPassthroughCopy({
      [`node_modules/onnxruntime-web/dist/${file}`]: `vendor/onnxruntime-web/${file}`,
    });
  }

  // BirdBox and BirdNET V3.0 models, fetched and verified by scripts/fetch-models.mjs (npm run fetch-models)
  eleventyConfig.addPassthroughCopy({ "_models/dist": "models" });

  return {
    pathPrefix: "/real-time-pwa/",
    dir: { input: "src", includes: "_includes", output: "_site" },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    templateFormats: ["md", "njk", "html"]
  };
};
