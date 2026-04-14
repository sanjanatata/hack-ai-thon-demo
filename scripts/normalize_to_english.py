import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect, detect_langs


DetectorFactory.seed = 42


ENGLISHISH_RE = re.compile(r"^[\s\d\W_]*$")


def _is_blankish(s: Any) -> bool:
    if s is None:
        return True
    if isinstance(s, float) and pd.isna(s):
        return True
    if not isinstance(s, str):
        return False
    return s.strip() == ""


def _looks_like_english_or_nonlinguistic(s: str) -> bool:
    # Fast path: numbers/punctuation only, or very short tokens.
    if ENGLISHISH_RE.match(s):
        return True
    if len(s.strip()) < 3:
        return True
    return False


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Translator:
    cache_path: Path
    installed: bool = False
    _cache: Optional[Dict[str, str]] = None
    _translate_fn: Optional[Any] = None

    def load(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self._cache = {}

        try:
            from argostranslate import package, translate  # type: ignore

            installed_languages = translate.get_installed_languages()
            # We don't know which source langs are present; treat as installed if English target exists.
            self.installed = any(l.code == "en" for l in installed_languages)
            self._package = package
            self._translate_mod = translate
        except Exception:
            self.installed = False
            self._package = None
            self._translate_mod = None

    def save(self) -> None:
        if self._cache is None:
            return
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def ensure_models(self, verbose: bool = True) -> None:
        """
        Install Argos language packages that translate *to English* for common travel-review languages.
        Uses Argos package index; downloads models if not installed.
        """
        if self._package is None or self._translate_mod is None:
            raise RuntimeError(
                "Argos Translate is not available. Did you install requirements.txt?"
            )

        from argostranslate import translate  # type: ignore

        installed = translate.get_installed_languages()
        installed_pairs = set()
        for src in installed:
            for t in src.translations:
                installed_pairs.add((src.code, t.to_lang.code))

        # Common langs seen in your Reviews sample: de, fr. Include other likely ones.
        desired_sources = ["de", "fr", "es", "it", "pt", "nl"]
        to_lang = "en"

        # Update index and find packages
        self._package.update_package_index()
        available = self._package.get_available_packages()

        to_install = []
        for pkg in available:
            if getattr(pkg, "to_code", None) != to_lang:
                continue
            if getattr(pkg, "from_code", None) not in desired_sources:
                continue
            if (pkg.from_code, pkg.to_code) in installed_pairs:
                continue
            to_install.append(pkg)

        if verbose:
            print(f"[normalize] Argos packages to install: {len(to_install)}", file=sys.stderr)
            if to_install:
                print(
                    "[normalize] Installing: "
                    + ", ".join(f"{p.from_code}->{p.to_code}" for p in to_install),
                    file=sys.stderr,
                )

        for pkg in to_install:
            pkg_path = pkg.download()
            self._package.install_from_path(pkg_path)

        # refresh installed flag
        installed_languages = translate.get_installed_languages()
        self.installed = any(l.code == "en" for l in installed_languages)

    def _argos_translate(self, text: str, src_lang: str) -> Optional[str]:
        if self._translate_mod is None:
            return None
        try:
            from argostranslate import translate  # type: ignore

            installed_languages = translate.get_installed_languages()
            from_lang = next((l for l in installed_languages if l.code == src_lang), None)
            to_lang = next((l for l in installed_languages if l.code == "en"), None)
            if from_lang is None or to_lang is None:
                return None
            translation = from_lang.get_translation(to_lang)
            return translation.translate(text)
        except Exception:
            return None

    def translate_to_english(self, text: str) -> str:
        if self._cache is None:
            raise RuntimeError("Translator not loaded")

        if _looks_like_english_or_nonlinguistic(text):
            return text

        key = _stable_hash(text)
        if key in self._cache:
            return self._cache[key]

        # Detect language(s); langdetect can be brittle on short text, so try top candidates.
        candidates: list[str] = []
        top_lang: Optional[str] = None
        top_prob: float = 0.0
        try:
            langs = detect_langs(text)
            if langs:
                top_lang = langs[0].lang
                top_prob = float(langs[0].prob)
            candidates = [lp.lang for lp in langs[:3]]
        except LangDetectException:
            candidates = []

        if not candidates:
            try:
                candidates = [detect(text)]
            except LangDetectException:
                self._cache[key] = text
                return text

        # Only treat as English when detection is confident.
        if top_lang == "en" and top_prob >= 0.80:
            self._cache[key] = text
            return text

        translated: Optional[str] = None
        if self.installed:
            # First try detected candidates.
            for lang in candidates:
                translated = self._argos_translate(text, src_lang=lang)
                if translated and translated.strip() and translated != text:
                    break

            # Fallback: try all installed source languages (helps when detect() picks a close-but-wrong code).
            if translated is None or translated == text:
                try:
                    from argostranslate import translate  # type: ignore

                    installed_langs = [l.code for l in translate.get_installed_languages()]
                    for lang in installed_langs:
                        if lang in ("en",):
                            continue
                        translated = self._argos_translate(text, src_lang=lang)
                        if translated and translated.strip() and translated != text:
                            break
                except Exception:
                    translated = None

        if translated is None:
            # Can't translate: keep original (still normalized via caching)
            self._cache[key] = text
            return text

        self._cache[key] = translated
        return translated


def normalize_csv(
    input_path: Path,
    output_path: Path,
    text_columns: Iterable[str],
    translator: Translator,
    flush_every: int = 200,
) -> None:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)

    changed = 0
    processed = 0
    for col in text_columns:
        if col not in df.columns:
            continue

        new_values = []
        for v in df[col].tolist():
            if _is_blankish(v):
                new_values.append(v)
                continue
            processed += 1
            translated = translator.translate_to_english(str(v))
            if translated != v:
                changed += 1
            new_values.append(translated)

            if processed % flush_every == 0:
                translator.save()

        df[col] = new_values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    translator.save()
    print(
        f"[normalize] Wrote {output_path.name}. processed_text_cells={processed} changed={changed}",
        file=sys.stderr,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data_hackathon", help="Directory containing input CSVs")
    p.add_argument("--install-models", action="store_true", help="Download/install Argos models")
    p.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep per translation call")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    desc_in = data_dir / "Description_PROC.csv"
    reviews_in = data_dir / "Reviews_PROC.csv"
    desc_out = data_dir / "Description_PROC_en.csv"
    reviews_out = data_dir / "Reviews_PROC_en.csv"

    if not desc_in.exists() or not reviews_in.exists():
        raise SystemExit(f"Missing input CSV(s) in {data_dir}")

    translator = Translator(cache_path=Path(".cache/translation_cache.json"))
    translator.load()

    if args.install_models:
        translator.ensure_models(verbose=True)

    # Translation is currently offline/Argos-only; if models aren’t installed for a language, text is left as-is.
    # Columns to translate:
    desc_text_cols = [
        "area_description",
        "property_description",
        "check_out_policy",
        "pet_policy",
        "children_and_extra_bed_policy",
        "check_in_instructions",
        "know_before_you_go",
        # Amenity lists are mostly English tokens; leave them unless you know they contain non-English.
    ]

    reviews_text_cols = ["review_title", "review_text"]

    normalize_csv(desc_in, desc_out, desc_text_cols, translator)
    normalize_csv(reviews_in, reviews_out, reviews_text_cols, translator)


if __name__ == "__main__":
    main()

