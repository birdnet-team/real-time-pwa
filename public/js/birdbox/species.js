/**
 * BirdNET V3.0 species: scientific names, classes and common names in the
 * app's label language. Index order follows the V3.0 model's output.
 * Adapted from the BirdNET website (birdnet-team/birdnet-website, dev branch):
 * public/js/demo/species.js.
 */
import { fetchJson, getManifest } from './model-store.js';

// The app's label languages (BirdNET V2.4 label files) -> V3.0 name lists, the first one shipped
const LABEL_TO_NAMES = {
  en_us: ['en'], en_uk: ['en'], pt: ['pt_PT', 'pt'], zh: ['zh-CN'], no: ['no'],
};

class Catalog {
  constructor() {
    this.ready = null;
    this.lang = 'en';
    this.names = null;
  }

  /** Load the label table once; resolves to this catalog. */
  load() {
    if (!this.ready) {
      this.ready = (async () => {
        const manifest = await getManifest();
        const table = await fetchJson(manifest.v3.labels);
        this.manifest = manifest;
        this.classes = table.classes;
        this.aves = table.classes.indexOf('Aves');
        this.count = table.species.length;
        this.sci = table.species.map(s => s[0]);
        this.en = table.species.map(s => s[1]);
        this.cls = Uint8Array.from(table.species, s => s[2]);
        this.names = this.en;
        return this;
      })();
      this.ready.catch(() => { this.ready = null; });
    }
    return this.ready;
  }

  /** Common names in the app's label language (e.g. "de", "en_us"); English where there is none. */
  async setLabelLang(labelLang) {
    await this.load();
    const shipped = this.manifest.v3.names || {};
    const lang = (LABEL_TO_NAMES[labelLang] || [labelLang]).find(l => l === 'en' || shipped[l]) || 'en';
    if (lang === this.lang && this.names) return;
    let names = this.en;
    if (lang !== 'en') {
      const list = await fetchJson(shipped[lang]);
      names = list.map((n, i) => n || this.en[i]);
    }
    this.lang = lang;
    this.names = names;
  }

  name(i) { return this.names[i]; }
}

export const catalog = new Catalog();
