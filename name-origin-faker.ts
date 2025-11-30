import { allLocales } from "@faker-js/faker";

import * as fs from "fs";
import latinize from "latinize";
import CyrillicToTranslit from "cyrillic-to-translit-js";

const cyrillic = CyrillicToTranslit();

// ---------- Name normalization ----------
function normalizeName(name: string) {
  let s = cyrillic.transform(name); // Cyrillic → Latin
  s = latinize(s); // strip diacritics
  return s.toLowerCase(); // lowercase
}

// ---------- Utilities ----------
function ngrams(str: string, n = 3) {
  const s = `_${str}_`;
  const out: string[] = [];
  for (let i = 0; i <= s.length - n; i++) out.push(s.slice(i, i + n));
  return out;
}
function uniq<T>(arr: T[]) {
  return Array.from(new Set(arr));
}

// ---------- Vectorizer ----------
class Vectorizer {
  n: number;
  vocab: Map<string, number>;
  constructor(n = 3) {
    this.n = n;
    this.vocab = new Map();
  }
  fit(names: string[]) {
    const grams = new Set<string>();
    for (const nm of names) for (const g of ngrams(nm, this.n)) grams.add(g);
    Array.from(grams)
      .sort()
      .forEach((g, i) => this.vocab.set(g, i));
  }
  transform(name: string) {
    const vec = new Array(this.vocab.size).fill(0);
    for (const g of ngrams(name, this.n)) {
      const idx = this.vocab.get(g);
      if (idx !== undefined) vec[idx]++;
    }
    return vec;
  }
  toJSON() {
    return { n: this.n, vocab: Array.from(this.vocab.entries()) };
  }
  static fromJSON(obj: any) {
    const v = new Vectorizer(obj.n);
    v.vocab = new Map(obj.vocab);
    return v;
  }
}

// ---------- Multinomial Naive Bayes ----------
class MultinomialNB {
  classes: string[] = [];
  classCount: Record<string, number> = {};
  classLogPrior: Record<string, number> = {};
  featureCount: Record<string, number[]> = {};
  featureLogProb: Record<string, number[]> = {};
  vocabSize = 0;
  alpha = 1;

  fit(X: number[][], y: string[]) {
    this.vocabSize = X[0]!.length;
    this.classes = uniq(y);

    for (const cls of this.classes) {
      this.classCount[cls] = 0;
      this.featureCount[cls] = new Array(this.vocabSize).fill(0);
    }

    for (let i = 0; i < X.length; i++) {
      const cls = y[i]!;
      this.classCount[cls]!++;
      const row = X[i]!;
      const fc = this.featureCount[cls]!;
      for (let j = 0; j < row!.length; j++) fc[j]! += row[j]!;
    }

    const total = X.length;
    for (const cls of this.classes) {
      this.classLogPrior[cls] = Math.log(
        (this.classCount[cls]! + 1) / (total + this.classes.length)
      );
      const fc = this.featureCount[cls]!;
      const denom = fc.reduce((a, b) => a + b, 0) + this.alpha * this.vocabSize;
      this.featureLogProb[cls] = fc.map((c) =>
        Math.log((c + this.alpha) / denom)
      );
    }
  }

  _jointLogLikelihood(vec: number[]) {
    const out: Record<string, number> = {};
    for (const cls of this.classes) {
      let s = this.classLogPrior[cls]!;
      const flp = this.featureLogProb[cls]!;
      for (let i = 0; i < vec.length; i++)
        if (vec[i] !== 0) s += vec[i]! * flp[i]!;
      out[cls] = s;
    }
    return out;
  }

  predictProba(vec: number[]) {
    const logps = this._jointLogLikelihood(vec);
    const entries = Object.entries(logps);
    const maxLog = Math.max(...entries.map(([, v]) => v));
    const exps = entries.map(
      ([cls, v]) => [cls, Math.exp(v - maxLog)] as [string, number]
    );
    const sum = exps.reduce((a, b) => a + b[1], 0);

    const probs: Record<string, number> = {};
    for (const [cls, val] of exps) probs[cls] = val / sum;
    return probs;
  }

  toJSON() {
    return {
      classes: this.classes,
      classCount: this.classCount,
      classLogPrior: this.classLogPrior,
      featureCount: this.featureCount,
      featureLogProb: this.featureLogProb,
      vocabSize: this.vocabSize,
      alpha: this.alpha,
    };
  }
  static fromJSON(obj: any) {
    const m = new MultinomialNB();
    Object.assign(m, obj);
    return m;
  }
}

// ---------- Static Faker Name Sources ----------
const NAME_SOURCES = Object.fromEntries(
  Object.entries(allLocales)
    .filter(([lang, locale]) => 
      // Only include base European languages with sufficient name data
      ![
        "af_ZA",
        "fr_BE",
        "fr_CH",
        "fr_SN",
        "nl_BE",
        "id_ID",
        "es_MX",
        "en_ZA",
        "en_AU",
        "en_GH",
        "en_IN",
        "de_AT",
        "de_CH",
        "ro_MD",
        "pt_BR",
        "uz_UZ_latin",
        "yo_NG",
        "zh_CN",
        "he",
        "ja",
        "th",
        "vi",
        "ko"
      ].includes(lang)
      && (locale.person?.first_name?.male?.length ?? 0) > 75
      && (locale.person?.first_name?.female?.length ?? 0) > 75
  )
    .map(([lang, locale]) => [lang, locale.person?.first_name])
 ) as Record<string, { male: string[]; female: string[] }>;

// ---------- Deterministic dataset ----------
type Gender = "male" | "female";
const LANGS = Object.keys(NAME_SOURCES);

function generateNames(gender: Gender) {
  const data: { name: string; language: string; gender: string }[] = [];

  for (const lang of LANGS) {
    const src = NAME_SOURCES[lang];
    if (!src) continue;

    const rawList = src[gender] || [];
    for (const raw of rawList) {
      data.push({
        name: normalizeName(raw),
        language: lang,
        gender,
      });
    }
  }

  return data;
}

// ---------- Train & Save ----------
function trainAndSave(
  data: { name: string; language: string; gender: string }[],
  gender: Gender
) {
  const filtered = data.filter((d) => d.gender === gender);
  const names = filtered.map((d) => d.name);
  const labels = filtered.map((d) => d.language);

  const vectorizer = new Vectorizer(3);
  vectorizer.fit(names);

  const X = names.map((n) => vectorizer.transform(n));

  const nb = new MultinomialNB();
  nb.fit(X, labels);

  const model = {
    vectorizer: vectorizer.toJSON(),
    nb: nb.toJSON(),
    meta: { gender },
  };

  fs.writeFileSync(`model-${gender}.json`, JSON.stringify(model, null, 2));
  console.log(
    `Trained ${gender} model on ${filtered.length} names, saved to model-${gender}.json`
  );
}

// ---------- Load model ----------
function loadModel(file: string) {
  const obj = JSON.parse(fs.readFileSync(file, "utf8"));
  return {
    vectorizer: Vectorizer.fromJSON(obj.vectorizer),
    nb: MultinomialNB.fromJSON(obj.nb),
    meta: obj.meta,
  };
}

// ---------- Predict ----------
function predict(model: ReturnType<typeof loadModel>, name: string) {
  const normalized = normalizeName(name);
  const vec = model.vectorizer.transform(normalized);
  const probs = model.nb.predictProba(vec);
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  console.log(`Prediction for "${name}" (${model.meta.gender}):`);
  for (const [lang, p] of sorted) console.log(`${lang}: ${p.toFixed(3)}`);
}

// ---------- CLI ----------
const argv = process.argv.slice(2);
const cmd = argv[0];
if (!cmd) {
  console.log("Usage: train | predict <name>");
  process.exit(0);
}

if (cmd === "train") {
  const maleData = generateNames("male");
  const femaleData = generateNames("female");

  trainAndSave(maleData, "male");
  trainAndSave(femaleData, "female");

  process.exit(0);
}

if (cmd === "predict") {
  const name = argv[1];
  if (!name) {
    console.error("Please provide a name");
    process.exit(1);
  }

  const modelMale = loadModel("model-male.json");
  const modelFemale = loadModel("model-female.json");

  console.log("Male model:");
  predict(modelMale, name);

  console.log("\nFemale model:");
  predict(modelFemale, name);

  process.exit(0);
}
