# %% [markdown]
# # 01 Corpus Cleaning For Autodesk RAG
#
# This notebook cleans raw Autodesk HTML files from `raw_corpus/` and writes
# RAG-ready Markdown documents to `cleaned_corpus/`.
#
# The cleaning strategy is intentionally two-stage:
#
# 1. Deterministic HTML cleanup with BeautifulSoup removes executable content,
#    hidden elements, layout regions, page chrome, cookie banners, and other
#    likely boilerplate.
# 2. Main-content extraction with Trafilatura extracts document content as
#    Markdown while preserving useful structures such as headings, lists,
#    tables, links, and code-like blocks where possible.
#
# For RAG, the goal is not simply to strip tags. Headings support
# heading-aware chunking, tables often contain dense technical facts, code
# blocks and short statements can be critical, and links can support source
# citation. Repeated navigation and global page chrome add embedding noise and
# reduce retrieval precision.
#
# GPU acceleration is not useful for this HTML cleaning stage because the work
# is mostly parsing, string processing, and boilerplate detection. GPU may be
# useful later for embeddings, reranking, or local LLM inference.

# %%
from __future__ import annotations

import concurrent.futures as futures
import importlib.util
import math
import os
import re
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

# %% [markdown]
# ## 1. Setup And Configuration
#
# Edit these variables before running if you want different behavior. The
# defaults read from `raw_corpus/`, write Markdown to `cleaned_corpus/`,
# overwrite prior cleaned outputs, save diagnostics, and use all available CPU
# cores via a thread pool.

# %%
RAW_CORPUS_DIR = Path("raw_corpus")
CLEANED_CORPUS_DIR = Path("cleaned_corpus")
CLEANED_CORPUS_INFO_DIR = Path("cleaned_corpus_info")
OUTPUT_FORMAT = "markdown"
MIN_CONTENT_CHARS = 300
N_WORKERS = None  # Use all available CPU cores by default.

FILE_EXTENSIONS = (".html", ".htm")
OVERWRITE_EXISTING = True
SAVE_EXTRACTION_DIAGNOSTICS = True
MIN_ACCEPTABLE_CLEANED_CHARS = MIN_CONTENT_CHARS
REPEATED_LINE_THRESHOLD = 0.10
REMOVE_REPEATED_LINES = False
SAMPLE_N = 5
PREFER_TABLE_PRESERVING_FALLBACK = True
TABLE_FALLBACK_MAX_CHAR_MULTIPLIER = 12
TABLE_FALLBACK_ABSOLUTE_MAX_CHARS = 150_000
RICH_FALLBACK_MIN_TRAFILATURA_CHARS = 1_000
RICH_FALLBACK_MIN_MULTIPLIER = 2.5
RICH_FALLBACK_ABSOLUTE_MAX_CHARS = 80_000
METADATA_MAX_HEADINGS = 30
METADATA_MAX_HEADING_CHARS = 120
LANGUAGE_DETECTION_MIN_CHARS = 200
TFIDF_MAX_KEYWORDS = 12
TFIDF_MAX_FEATURES = 8_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_EXCLUDED_TERMS = {"autodesk", "www", "com", "https", "http"}
PURGE_MIN_MARKDOWN_BYTES = 600
PURGE_NON_ENGLISH_DOCUMENTS = True
RETAIN_UNKNOWN_LANGUAGE_DOCUMENTS = True

BOILERPLATE_KEYWORDS = [
    "breadcrumb",
    "breadcrumbs",
    "nav",
    "navigation",
    "navbar",
    "footer",
    "header",
    "sidebar",
    "cookie",
    "consent",
    "banner",
    "promo",
    "promotion",
    "subscribe",
    "related",
    "recommended",
    "social",
    "share",
    "language-selector",
    "global-navigation",
    "globalnav",
    "masthead",
    "flyout",
    "megamenu",
    "menu",
    "skip-link",
    "skipcontent",
    "search",
    "signin",
    "sign-in",
    "login",
    "account",
    "profile",
    "modal",
    "overlay",
    "popover",
    "tooltip",
    "survey",
    "ratings",
    "rate-this",
    "page-tools",
    "print",
    "newsletter",
    "teaser",
    "carousel",
    "adsk-nav",
    "adsk-footer",
    "adsk-header",
    "onetrust",
    "trustarc",
    "ot-sdk",
]

BOILERPLATE_COMPOUND_PATTERNS = [
    "language-selector",
    "global-navigation",
    "skip-link",
    "rate-this",
    "page-tools",
    "adsk-nav",
    "adsk-footer",
    "adsk-header",
    "ot-sdk",
]

NEVER_DECOMPOSE_BY_ATTR_TAGS = {"html", "body", "main", "article"}

# %% [markdown]
# ## 2. Environment And Dependency Check
#
# The notebook uses the current virtual Python environment. It does not create
# or modify environments. If a package is missing, install it in the active
# environment before rerunning.

# %%
REQUIRED_PACKAGES = {
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "trafilatura": "trafilatura",
    "lingua-language-detector": "lingua",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
    "pandas": "pandas",
}


def check_dependencies() -> None:
    missing = [
        pkg
        for pkg, module in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print("Missing required packages:", ", ".join(missing))
        print("Install suggestion:")
        print("pip install beautifulsoup4 lxml trafilatura lingua-language-detector scikit-learn tqdm pandas")
        raise ImportError(f"Missing required packages: {missing}")
    print("All required packages are available.")


# Run the dependency check before importing third-party packages so missing
# packages produce a clear install suggestion.
check_dependencies()

import pandas as pd
import trafilatura
from bs4 import BeautifulSoup, Comment
from lingua import LanguageDetectorBuilder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm


# %% [markdown]
# ## 3. File Discovery
#
# The manifest records each source path, planned output path, file size,
# processing status, character counts, extraction method, warnings, and errors.
# It is saved to `cleaned_corpus_info/cleaning_manifest.csv` so diagnostics do
# not become part of the RAG corpus.

# %%
def planned_output_path(source_path: Path) -> Path:
    relative_path = source_path.relative_to(RAW_CORPUS_DIR)
    suffix = ".md" if OUTPUT_FORMAT == "markdown" else ".txt"
    return CLEANED_CORPUS_DIR / relative_path.with_suffix(suffix)


def discover_source_files() -> list[Path]:
    return sorted(
        path
        for path in RAW_CORPUS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in FILE_EXTENSIONS
    )


def build_initial_manifest(source_files: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_file_path": str(path),
                "relative_path": str(path.relative_to(RAW_CORPUS_DIR)),
                "file_size_bytes": path.stat().st_size,
                "planned_output_path": str(planned_output_path(path)),
                "processing_status": "pending",
                "char_count_before_cleaning": None,
                "char_count_after_cleaning": None,
                "cleaned_html_char_count": None,
                "extraction_method_used": None,
                "title": None,
                "warnings": None,
                "errors": None,
            }
            for path in source_files
        ]
    )


# %% [markdown]
# ## 4. Cleaning Function
#
# This deterministic pass removes common non-content regions before
# Trafilatura sees the page. It intentionally preserves semantic content such
# as headings, paragraphs, lists, tables, links, and code/pre blocks.

# %%
EXECUTABLE_OR_VISUAL_ONLY_TAGS = [
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "template",
    "picture",
    "source",
]
COMMON_LAYOUT_TAGS = ["nav", "footer", "header", "aside"]
BOILERPLATE_ROLES = {"navigation", "banner", "contentinfo", "complementary", "search"}
HIDDEN_STYLE_PATTERNS = [
    "display:none",
    "display: none",
    "visibility:hidden",
    "visibility: hidden",
    "opacity:0",
    "opacity: 0",
]


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_title_from_soup(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in [
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "twitter:title"}),
        ("meta", {"name": "title"}),
    ]:
        tag = soup.find(*selector)
        if tag and tag.get("content"):
            candidates.append(tag.get("content"))
    if soup.title and soup.title.get_text(strip=True):
        candidates.append(soup.title.get_text(" ", strip=True))
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        candidates.append(h1.get_text(" ", strip=True))
    for candidate in candidates:
        cleaned = compact_whitespace(candidate)
        if cleaned:
            return cleaned
    return "Untitled Autodesk Page"


def element_attr_blob(el) -> str:
    values = []
    for attr_name in [
        "class",
        "id",
        "role",
        "aria-label",
        "data-testid",
        "data-component",
        "data-module",
    ]:
        value = el.get(attr_name)
        if not value:
            continue
        if isinstance(value, list):
            values.extend(str(item).lower() for item in value)
        else:
            values.append(str(value).lower())
    return " ".join(values)


def has_boilerplate_attr(el) -> bool:
    """Return True for boilerplate-looking class/id attrs without broad substring matches."""
    if el.name in NEVER_DECOMPOSE_BY_ATTR_TAGS:
        return False

    attrs = element_attr_blob(el)
    if not attrs:
        return False

    if any(pattern in attrs for pattern in BOILERPLATE_COMPOUND_PATTERNS):
        return True

    tokens = {token for token in re.split(r"[^a-z0-9]+", attrs) if token}
    return any(keyword in tokens for keyword in BOILERPLATE_KEYWORDS)


def is_hidden_element(el) -> bool:
    if el is None or getattr(el, "attrs", None) is None:
        return False
    if el.has_attr("hidden"):
        return True
    if str(el.get("aria-hidden", "")).lower() == "true":
        return True
    style = str(el.get("style", "")).replace(" ", "").lower()
    return any(pattern.replace(" ", "") in style for pattern in HIDDEN_STYLE_PATTERNS)


def clean_autodesk_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    for tag in soup(EXECUTABLE_OR_VISUAL_ONLY_TAGS):
        tag.decompose()

    for tag in soup.find_all(COMMON_LAYOUT_TAGS):
        tag.decompose()

    for el in list(soup.find_all(True)):
        if getattr(el, "name", None) is None or getattr(el, "attrs", None) is None:
            continue
        if is_hidden_element(el):
            el.decompose()
            continue
        role = str(el.get("role", "")).lower()
        if role in BOILERPLATE_ROLES:
            el.decompose()
            continue
        if has_boilerplate_attr(el):
            el.decompose()

    for el in reversed(list(soup.find_all(True))):
        if getattr(el, "name", None) is None or getattr(el, "attrs", None) is None:
            continue
        if el.name in {"td", "th", "tr", "table", "code", "pre", "a", "img"}:
            continue
        if not el.get_text(" ", strip=True) and not el.find(["table", "pre", "code"]):
            el.decompose()

    return str(soup)


# %% [markdown]
# ## 5. Trafilatura Extraction Function And Fallback
#
# Trafilatura performs the main-content extraction. If it returns nothing or
# very short output, a BeautifulSoup fallback preserves headings, paragraphs,
# lists, tables, pre/code blocks, and useful links in Markdown-like text.
# The fallback is also preferred for table-heavy documentation pages when
# Trafilatura produces a compact answer but silently drops useful tables.

# %%
def text_with_markdown_links(el) -> str:
    clone = BeautifulSoup(str(el), "lxml")
    for media in clone.find_all(["img", "svg"]):
        label = compact_whitespace(
            media.get("alt")
            or media.get("aria-label")
            or media.get("title")
            or media.get_text(" ", strip=True)
        )
        if label.lower() in {"checkmark", "check", "checked", "tick"}:
            label = "Yes"
        elif label.lower() in {"x", "close", "cross", "not included", "minus"}:
            label = "No"
        if label:
            media.replace_with(label)
        else:
            media.decompose()
    for link in clone.find_all("a"):
        label = compact_whitespace(link.get_text(" ", strip=True))
        href = compact_whitespace(link.get("href", ""))
        if label and href and not href.startswith("#"):
            link.replace_with(f"[{label}]({href})")
        elif label:
            link.replace_with(label)
        else:
            link.decompose()
    return compact_whitespace(clone.get_text(" ", strip=True))


def table_to_markdown(table) -> str:
    rows = []
    row_tags = table.find_all("tr")
    has_header = bool(row_tags and row_tags[0].find_all("th"))
    for tr in row_tags:
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([text_with_markdown_links(cell).replace("|", "\\|") for cell in cells])
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    if has_header:
        header = rows[0]
        body_rows = rows[1:]
    else:
        if width == 4:
            header = ["Feature", "Standard", "Premium", "Enterprise"]
        elif width == 3:
            header = ["Feature", "Standard", "Premium"]
        elif width == 2:
            header = ["Field", "Value"]
        else:
            header = [f"Column {idx}" for idx in range(1, width + 1)]
        body_rows = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def normalize_markdown(markdown_text: str) -> str:
    text = (markdown_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def markdown_table_count(markdown_text: str) -> int:
    lines = [line.strip() for line in (markdown_text or "").splitlines()]
    count = 0
    for idx, line in enumerate(lines[:-1]):
        if not line.startswith("|") or not line.endswith("|"):
            continue
        next_line = lines[idx + 1]
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", next_line):
            count += 1
    return count


def fallback_markdown_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[str] = []

    def emit(block: str) -> None:
        block = block.strip()
        if block:
            blocks.append(block)

    def walk(node) -> None:
        if getattr(node, "name", None) is None:
            return
        name = node.name.lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            emit(f"{'#' * int(name[1])} {text_with_markdown_links(node)}")
            return
        if name == "p":
            emit(text_with_markdown_links(node))
            return
        if name == "pre":
            text = node.get_text("\n", strip=True)
            if text:
                emit(f"```\n{text}\n```")
            return
        if name == "code" and node.parent and node.parent.name != "pre":
            text = node.get_text(" ", strip=True)
            if text:
                emit(f"`{text}`")
            return
        if name == "table":
            emit(table_to_markdown(node))
            return
        if name in {"ul", "ol"}:
            for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
                bullet = f"{idx}." if name == "ol" else "-"
                text = text_with_markdown_links(li)
                if text:
                    emit(f"{bullet} {text}")
            return
        if name == "li":
            text = text_with_markdown_links(node)
            if text:
                emit(f"- {text}")
            return
        if name == "br":
            return
        for child in node.children:
            walk(child)

    walk(root)
    return normalize_markdown("\n\n".join(blocks))


def extract_markdown_from_html(html_text: str) -> tuple[str, str, str]:
    warnings = []
    output_format = "markdown" if OUTPUT_FORMAT == "markdown" else "txt"
    html_table_count = len(BeautifulSoup(html_text, "lxml").find_all("table"))
    try:
        extracted = trafilatura.extract(
            html_text,
            include_tables=True,
            include_links=True,
            include_comments=False,
            deduplicate=True,
            output_format=output_format,
        )
    except TypeError:
        extracted = trafilatura.extract(
            html_text,
            include_tables=True,
            include_links=True,
            include_comments=False,
            output_format=output_format,
        )
    except Exception as exc:
        extracted = None
        warnings.append(f"trafilatura_error: {type(exc).__name__}: {exc}")

    extracted = normalize_markdown(extracted or "")
    extracted_table_count = markdown_table_count(extracted)
    fallback = ""

    if (
        PREFER_TABLE_PRESERVING_FALLBACK
        and html_table_count > 0
        and extracted_table_count < html_table_count
    ):
        fallback = fallback_markdown_from_html(html_text)
        fallback_table_count = markdown_table_count(fallback)
        max_allowed_chars = min(
            TABLE_FALLBACK_ABSOLUTE_MAX_CHARS,
            max(MIN_CONTENT_CHARS, len(extracted)) * TABLE_FALLBACK_MAX_CHAR_MULTIPLIER,
        )
        if (
            fallback_table_count > extracted_table_count
            and len(fallback) >= MIN_CONTENT_CHARS
            and len(fallback) <= max_allowed_chars
        ):
            warnings.append(
                f"used_table_preserving_fallback: html_tables={html_table_count}, "
                f"trafilatura_tables={extracted_table_count}, fallback_tables={fallback_table_count}"
            )
            return fallback, "beautifulsoup_table_fallback", "; ".join(warnings)

    if len(extracted) >= MIN_CONTENT_CHARS and len(extracted) < RICH_FALLBACK_MIN_TRAFILATURA_CHARS:
        fallback = fallback or fallback_markdown_from_html(html_text)
        if (
            len(fallback) >= MIN_CONTENT_CHARS
            and len(fallback) >= len(extracted) * RICH_FALLBACK_MIN_MULTIPLIER
            and len(fallback) <= RICH_FALLBACK_ABSOLUTE_MAX_CHARS
        ):
            warnings.append(
                f"used_richer_beautifulsoup_fallback: "
                f"trafilatura_chars={len(extracted)}, fallback_chars={len(fallback)}"
            )
            return fallback, "beautifulsoup_rich_fallback", "; ".join(warnings)

    if len(extracted) >= MIN_CONTENT_CHARS:
        return extracted, "trafilatura", "; ".join(warnings)

    fallback = fallback or fallback_markdown_from_html(html_text)
    if len(fallback) > len(extracted):
        warnings.append("used_beautifulsoup_fallback")
        return fallback, "beautifulsoup_fallback", "; ".join(warnings)

    warnings.append("very_short_trafilatura_output")
    return extracted, "trafilatura_short", "; ".join(warnings)


# %% [markdown]
# ## 6. Metadata Preservation
#
# Each cleaned Markdown file starts with a YAML-style metadata block. These
# fields support downstream RAG indexing and source citation.

# %%
def yaml_quote(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)


def parse_metadata_value(value: str) -> Any:
    text = value.strip()
    if text.lower() == "null":
        return None
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text.strip("'")


def parse_metadata_block(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = parse_metadata_value(value)
    return metadata, text[match.end() :]


def ensure_title_heading(content: str, title: str) -> str:
    content = normalize_markdown(content)
    title = compact_whitespace(title)
    if not title:
        return content
    first_nonempty = next((line.strip() for line in content.splitlines() if line.strip()), "")
    comparable_first = re.sub(r"^#+\s*", "", first_nonempty).strip().lower()
    if comparable_first == title.lower():
        return content
    return normalize_markdown(f"# {title}\n\n{content}")


def trim_metadata_text(value: str, max_chars: int = METADATA_MAX_HEADING_CHARS) -> str:
    text = compact_whitespace(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_heading_metadata(content: str) -> tuple[list[str], list[str]]:
    headings: list[str] = []
    subheadings: list[str] = []
    in_code_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if not match:
            continue
        level = len(match.group(1))
        heading_text = trim_metadata_text(match.group(2))
        if not heading_text:
            continue
        headings.append(f"h{level}: {heading_text}")
        if level > 1:
            subheadings.append(heading_text)
        if len(headings) >= METADATA_MAX_HEADINGS:
            break
    return headings, subheadings


def join_metadata_list(values: list[str], max_items: int = METADATA_MAX_HEADINGS) -> str:
    selected = values[:max_items]
    suffix = " | ..." if len(values) > max_items else ""
    return " | ".join(selected) + suffix


@lru_cache(maxsize=1)
def get_language_detector():
    return LanguageDetectorBuilder.from_all_languages().build()


def detect_document_language(content: str) -> dict[str, Any]:
    text = compact_whitespace(re.sub(r"[#*_`>\[\]()|]+", " ", content))
    if len(text) < LANGUAGE_DETECTION_MIN_CHARS:
        return {
            "document_language": "",
            "document_language_name": "",
            "document_language_confidence": None,
        }

    detector = get_language_detector()
    language = detector.detect_language_of(text)
    if language is None:
        return {
            "document_language": "",
            "document_language_name": "",
            "document_language_confidence": None,
        }

    confidence = None
    for value in detector.compute_language_confidence_values(text):
        if value.language == language:
            confidence = round(float(value.value), 4)
            break

    return {
        "document_language": language.iso_code_639_1.name.lower(),
        "document_language_name": language.name.replace("_", " ").title(),
        "document_language_confidence": confidence,
    }


def add_metadata_block(content: str, metadata: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in metadata.items():
        if value is None:
            lines.append(f"{key}: null")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {yaml_quote(value)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n" + normalize_markdown(content) + "\n"


def normalize_tfidf_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^#{1,6}\s+", " ", text, flags=re.M)
    return compact_whitespace(text)


def select_tfidf_keywords(feature_names: list[str], row, max_keywords: int = TFIDF_MAX_KEYWORDS) -> list[str]:
    if row.nnz == 0:
        return []
    keyword_scores = zip(row.indices, row.data)
    ranked = sorted(keyword_scores, key=lambda item: item[1], reverse=True)
    keywords: list[str] = []
    seen: set[str] = set()
    for feature_index, _score in ranked:
        term = feature_names[feature_index].strip().lower()
        if not term or term in seen:
            continue
        tokens = term.split()
        if any(token in TFIDF_EXCLUDED_TERMS for token in tokens):
            continue
        if len(term) < 3 or not re.search(r"[a-zA-Z]", term):
            continue
        keywords.append(term)
        seen.add(term)
        if len(keywords) >= max_keywords:
            break
    return keywords


def enrich_cleaned_files_with_tfidf(manifest: pd.DataFrame) -> pd.DataFrame:
    success_manifest = manifest[manifest["processing_status"].eq("success")].copy()
    records: list[dict[str, Any]] = []
    documents: list[str] = []

    for row in success_manifest.itertuples(index=False):
        output_path = Path(row.planned_output_path)
        if not output_path.exists():
            continue
        text = output_path.read_text(encoding="utf-8", errors="replace")
        metadata, body = parse_metadata_block(text)
        normalized = normalize_tfidf_text(body)
        if not normalized:
            continue
        records.append({"path": output_path, "metadata": metadata, "body": body, "relative_path": row.relative_path})
        documents.append(normalized)

    if not documents:
        manifest["tfidf_keywords"] = ""
        return manifest

    min_df = TFIDF_MIN_DF if len(documents) >= TFIDF_MIN_DF else 1
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=min_df,
        max_features=TFIDF_MAX_FEATURES,
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9+.-]{1,}\b",
    )
    matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out().tolist()
    keywords_by_relative_path: dict[str, str] = {}

    for index, record in enumerate(records):
        keywords = select_tfidf_keywords(feature_names, matrix.getrow(index))
        keyword_text = join_metadata_list(keywords, TFIDF_MAX_KEYWORDS)
        record["metadata"]["tfidf_keyword_count"] = len(keywords)
        record["metadata"]["tfidf_keywords"] = keyword_text
        output_text = add_metadata_block(record["body"], record["metadata"])
        record["path"].write_text(output_text, encoding="utf-8")
        keywords_by_relative_path[record["relative_path"]] = keyword_text

    manifest["tfidf_keywords"] = manifest["relative_path"].map(keywords_by_relative_path).fillna("")
    return manifest


def purge_low_value_cleaned_files(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    for column in ["purge_reason", "cleaned_file_size_bytes", "document_language"]:
        if column not in manifest.columns:
            manifest[column] = ""

    purge_records: list[dict[str, Any]] = []
    success_mask = manifest["processing_status"].eq("success")

    for index, row in manifest.loc[success_mask].iterrows():
        output_path = Path(row["planned_output_path"])
        if not output_path.exists():
            continue

        file_size = output_path.stat().st_size
        text = output_path.read_text(encoding="utf-8", errors="replace")
        metadata, _body = parse_metadata_block(text)
        language = str(metadata.get("document_language") or "").strip().lower()
        reasons: list[str] = []

        if file_size < PURGE_MIN_MARKDOWN_BYTES:
            reasons.append(f"markdown_file_size_below_{PURGE_MIN_MARKDOWN_BYTES}_bytes")
        if PURGE_NON_ENGLISH_DOCUMENTS:
            if language and language != "en":
                reasons.append(f"non_english_language_{language}")
            elif not language and not RETAIN_UNKNOWN_LANGUAGE_DOCUMENTS:
                reasons.append("unknown_document_language")

        manifest.at[index, "cleaned_file_size_bytes"] = file_size
        manifest.at[index, "document_language"] = language

        if not reasons:
            continue

        reason_text = "; ".join(reasons)
        output_path.unlink()
        existing_warnings = str(manifest.at[index, "warnings"] or "").strip()
        manifest.at[index, "processing_status"] = "purged"
        manifest.at[index, "purge_reason"] = reason_text
        manifest.at[index, "warnings"] = "; ".join([value for value in [existing_warnings, reason_text] if value])
        purge_records.append(
            {
                "relative_path": row["relative_path"],
                "planned_output_path": row["planned_output_path"],
                "cleaned_file_size_bytes": file_size,
                "document_language": language,
                "purge_reason": reason_text,
            }
        )

    CLEANED_CORPUS_INFO_DIR.mkdir(parents=True, exist_ok=True)
    purge_report_path = CLEANED_CORPUS_INFO_DIR / "purged_cleaned_documents.csv"
    purge_df = pd.DataFrame(
        purge_records,
        columns=[
            "relative_path",
            "planned_output_path",
            "cleaned_file_size_bytes",
            "document_language",
            "purge_reason",
        ],
    )
    purge_df.to_csv(purge_report_path, index=False)
    print(f"Purged {len(purge_df):,} low-value cleaned Markdown file(s).")
    print(f"Saved purge report to {purge_report_path}")
    return manifest


# %% [markdown]
# ## 7. Per-File Processing And Error Handling
#
# Every file is isolated behind exception handling. One broken HTML file should
# produce a `failed` manifest row, not stop the full run.

# %%
def process_one_file(source_path: Path) -> dict[str, Any]:
    output_path = planned_output_path(source_path)
    relative_path = source_path.relative_to(RAW_CORPUS_DIR)
    result = {
        "source_file_path": str(source_path),
        "relative_path": str(relative_path),
        "file_size_bytes": source_path.stat().st_size if source_path.exists() else None,
        "planned_output_path": str(output_path),
        "processing_status": "pending",
        "char_count_before_cleaning": None,
        "char_count_after_cleaning": None,
        "cleaned_html_char_count": None,
        "extraction_method_used": None,
        "title": None,
        "warnings": "",
        "errors": "",
    }

    try:
        if output_path.exists() and not OVERWRITE_EXISTING:
            existing = output_path.read_text(encoding="utf-8", errors="replace")
            result.update(
                {
                    "processing_status": "skipped",
                    "char_count_after_cleaning": len(existing),
                    "warnings": "output_exists_and_overwrite_disabled",
                }
            )
            return result

        raw_html = source_path.read_text(encoding="utf-8", errors="replace")
        result["char_count_before_cleaning"] = len(raw_html)

        title_soup = BeautifulSoup(raw_html, "lxml")
        title = extract_title_from_soup(title_soup)
        result["title"] = title

        cleaned_html = clean_autodesk_html(raw_html)
        result["cleaned_html_char_count"] = len(cleaned_html)

        extracted, method, warnings = extract_markdown_from_html(cleaned_html)
        extracted = ensure_title_heading(extracted, title)
        content_len = len(extracted)
        result["char_count_after_cleaning"] = content_len
        result["extraction_method_used"] = method
        headings, subheadings = extract_heading_metadata(extracted)
        language_metadata = detect_document_language(extracted)

        extra_warnings = []
        if warnings:
            extra_warnings.append(warnings)
        if content_len < MIN_ACCEPTABLE_CLEANED_CHARS:
            extra_warnings.append("very_short_cleaned_content")

        metadata = {
            "source_file": source_path.as_posix(),
            "relative_source_path": relative_path.as_posix(),
            "title": title,
            "cleaned_format": OUTPUT_FORMAT,
            "extraction_method": method,
            **language_metadata,
            "heading_count": len(headings),
            "subheading_count": len(subheadings),
            "headings": join_metadata_list(headings),
            "subheadings": join_metadata_list(subheadings),
            "raw_char_count": result["char_count_before_cleaning"],
            "cleaned_char_count": content_len,
        }
        output_text = add_metadata_block(extracted, metadata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")

        result["processing_status"] = "success"
        result["warnings"] = "; ".join([w for w in extra_warnings if w])
        return result

    except Exception as exc:
        result["processing_status"] = "failed"
        result["errors"] = f"{type(exc).__name__}: {exc}"
        result["warnings"] = traceback.format_exc(limit=3)
        return result


# %% [markdown]
# ## 8. Multicore Processing
#
# The notebook uses `ThreadPoolExecutor` by default. It is safer than
# notebook-based process pools on Windows and still gives useful concurrency for
# mixed parsing, extraction, and file I/O. Set `N_WORKERS` to a smaller integer
# if you want to leave more CPU for other work.

# %%
def process_corpus(source_files: list[Path]) -> pd.DataFrame:
    worker_count = N_WORKERS or os.cpu_count() or 1
    print(f"Processing {len(source_files):,} files with {worker_count} worker(s).")
    results: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {executor.submit(process_one_file, path): path for path in source_files}
        for future in tqdm(
            futures.as_completed(future_to_path),
            total=len(future_to_path),
            desc="Cleaning HTML",
        ):
            results.append(future.result())

    manifest = pd.DataFrame(results).sort_values("relative_path").reset_index(drop=True)
    print("Adding corpus-level TF-IDF keywords to cleaned document metadata.")
    manifest = enrich_cleaned_files_with_tfidf(manifest)
    print("Purging very small and non-English cleaned Markdown files.")
    manifest = purge_low_value_cleaned_files(manifest)
    CLEANED_CORPUS_INFO_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CLEANED_CORPUS_INFO_DIR / "cleaning_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")
    return manifest


# %% [markdown]
# ## 9. Diagnostics And Quality Control
#
# The summary helps identify whether cleaning was too aggressive or too weak.
# Large reductions are expected because raw Autodesk pages include navigation,
# scripts, styles, page chrome, and repeated global content.

# %%
def safe_sum(series: pd.Series) -> int:
    return int(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def dataframe_to_markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "No rows."
    return df[columns].head(max_rows).to_markdown(index=False)


def build_diagnostics(manifest: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    success_mask = manifest["processing_status"].eq("success")
    skipped_mask = manifest["processing_status"].eq("skipped")
    failed_mask = manifest["processing_status"].eq("failed")
    purged_mask = manifest["processing_status"].eq("purged")
    cleaned_counts = pd.to_numeric(manifest["char_count_after_cleaning"], errors="coerce").fillna(0)
    short_mask = (success_mask | purged_mask) & (cleaned_counts < MIN_ACCEPTABLE_CLEANED_CHARS)
    initially_cleaned_count = int((success_mask | purged_mask).sum())
    retained_cleaned_count = int(success_mask.sum())
    purged_count = int(purged_mask.sum())
    file_count_reduction_pct = (
        purged_count / initially_cleaned_count * 100 if initially_cleaned_count else 0
    )

    total_raw_chars = safe_sum(manifest["char_count_before_cleaning"])
    total_cleaned_chars = safe_sum(manifest.loc[success_mask, "char_count_after_cleaning"])
    reduction_pct = (1 - (total_cleaned_chars / total_raw_chars)) * 100 if total_raw_chars else 0

    summary = {
        "raw_html_files_found": len(manifest),
        "successfully_cleaned": int(success_mask.sum()),
        "initially_cleaned_before_purge": initially_cleaned_count,
        "retained_cleaned_after_purge": retained_cleaned_count,
        "purged_after_cleaning": purged_count,
        "cleaned_file_count_reduction_percentage": round(file_count_reduction_pct, 2),
        "skipped": int(skipped_mask.sum()),
        "failed": int(failed_mask.sum()),
        "very_short_extracted_content": int(short_mask.sum()),
        "total_raw_characters": total_raw_chars,
        "total_cleaned_characters": total_cleaned_chars,
        "approximate_reduction_percentage": round(reduction_pct, 2),
    }

    analysis_df = manifest.copy()
    analysis_df["char_count_before_cleaning"] = pd.to_numeric(
        analysis_df["char_count_before_cleaning"], errors="coerce"
    ).fillna(0).astype(int)
    analysis_df["char_count_after_cleaning"] = pd.to_numeric(
        analysis_df["char_count_after_cleaning"], errors="coerce"
    ).fillna(0).astype(int)
    analysis_df["char_reduction"] = (
        analysis_df["char_count_before_cleaning"] - analysis_df["char_count_after_cleaning"]
    )
    analysis_df["reduction_pct"] = analysis_df.apply(
        lambda row: round(
            row["char_reduction"] / row["char_count_before_cleaning"] * 100,
            2,
        )
        if row["char_count_before_cleaning"]
        else 0,
        axis=1,
    )

    success_df = analysis_df[analysis_df["processing_status"].eq("success")]
    top_largest_reduction = success_df.sort_values("char_reduction", ascending=False).head(20)
    top_shortest_outputs = success_df.sort_values("char_count_after_cleaning", ascending=True).head(20)
    return summary, analysis_df, top_largest_reduction, top_shortest_outputs


def save_summary_reports(
    manifest: pd.DataFrame,
    summary: dict[str, Any],
    analysis_df: pd.DataFrame,
    top_largest_reduction: pd.DataFrame,
    top_shortest_outputs: pd.DataFrame,
) -> dict[str, Any]:
    success_mask = manifest["processing_status"].eq("success")
    status_counts = manifest["processing_status"].value_counts().to_dict()
    method_counts = manifest["extraction_method_used"].fillna("none").value_counts().to_dict()

    CLEANED_CORPUS_INFO_DIR.mkdir(parents=True, exist_ok=True)
    summary_report_path = CLEANED_CORPUS_INFO_DIR / "cleaning_summary.md"
    before_after_stats_path = CLEANED_CORPUS_INFO_DIR / "before_after_processing_stats.md"

    summary_report = f"""# Corpus Cleaning Summary

Generated: {datetime.now().isoformat(timespec='seconds')}

## Overall Results

- Raw HTML files found: {summary['raw_html_files_found']:,}
- Initially cleaned before purge: {summary['initially_cleaned_before_purge']:,}
- Retained cleaned Markdown files after purge: {summary['retained_cleaned_after_purge']:,}
- Purged after cleaning: {summary['purged_after_cleaning']:,}
- Cleaned file count reduction from purge: {summary['cleaned_file_count_reduction_percentage']}%
- Skipped: {summary['skipped']:,}
- Failed: {summary['failed']:,}
- Very short cleaned documents: {summary['very_short_extracted_content']:,}
- Total raw characters: {summary['total_raw_characters']:,}
- Total cleaned characters: {summary['total_cleaned_characters']:,}
- Approximate character reduction: {summary['approximate_reduction_percentage']}%

## Status Counts

{pd.Series(status_counts, name='count').to_markdown()}

## Extraction Method Counts

{pd.Series(method_counts, name='count').to_markdown()}

## Purge Policy

After cleaning and metadata enrichment, the pipeline deletes cleaned Markdown files smaller than {PURGE_MIN_MARKDOWN_BYTES:,} bytes. It also deletes cleaned Markdown files whose `document_language` header is a known non-English language. Documents with unknown language are {'retained' if RETAIN_UNKNOWN_LANGUAGE_DOCUMENTS else 'purged'}.

## Largest Character Reductions

{dataframe_to_markdown_table(top_largest_reduction, ['relative_path', 'char_count_before_cleaning', 'char_count_after_cleaning', 'char_reduction', 'reduction_pct', 'extraction_method_used'])}

## Shortest Cleaned Outputs

{dataframe_to_markdown_table(top_shortest_outputs, ['relative_path', 'title', 'char_count_after_cleaning', 'extraction_method_used', 'warnings'])}
"""
    summary_report_path.write_text(summary_report, encoding="utf-8")

    file_size_raw = int(manifest["file_size_bytes"].fillna(0).sum())
    cleaned_files = [Path(path) for path in manifest.loc[success_mask, "planned_output_path"]]
    cleaned_file_size = sum(path.stat().st_size for path in cleaned_files if path.exists())
    file_size_reduction_pct = (1 - cleaned_file_size / file_size_raw) * 100 if file_size_raw else 0

    stats_report = f"""# Before / After Corpus Processing Stats

Generated: {datetime.now().isoformat(timespec='seconds')}

## Size Overview

- Raw HTML file count: {len(manifest):,}
- Initially cleaned Markdown file count before purge: {summary['initially_cleaned_before_purge']:,}
- Retained cleaned Markdown file count after purge: {summary['retained_cleaned_after_purge']:,}
- Purged cleaned Markdown file count: {summary['purged_after_cleaning']:,}
- Cleaned file count reduction from purge: {summary['cleaned_file_count_reduction_percentage']:.2f}%
- Raw corpus file size: {file_size_raw:,} bytes ({file_size_raw / (1024**2):.2f} MB)
- Cleaned corpus Markdown size: {cleaned_file_size:,} bytes ({cleaned_file_size / (1024**2):.2f} MB)
- Approximate file size reduction: {file_size_reduction_pct:.2f}%
- Raw character count: {summary['total_raw_characters']:,}
- Cleaned character count: {summary['total_cleaned_characters']:,}
- Approximate character reduction: {summary['approximate_reduction_percentage']:.2f}%

## Interpretation For RAG

The cleaned corpus should be much smaller than the raw HTML corpus because scripts, styles, navigation, menus, cookie notices, page chrome, and repeated Autodesk layout elements are removed. This is desirable for RAG because boilerplate can dominate embeddings and cause retrieval to return pages for shared navigation text rather than useful technical content.

The cleaning process preserves headings, lists, links, tables, and code-like blocks where possible. These structures are useful for heading-aware chunking, source citation, and technical answer grounding.

After enrichment, the pipeline purges very small cleaned Markdown files and known non-English documents so downstream retrieval indexes focus on substantive English Autodesk content. Deleted files are listed in `purged_cleaned_documents.csv`.

## Extraction Method Counts

{pd.Series(method_counts, name='count').to_markdown()}

## Largest Reductions

{dataframe_to_markdown_table(top_largest_reduction, ['relative_path', 'char_count_before_cleaning', 'char_count_after_cleaning', 'char_reduction', 'reduction_pct', 'extraction_method_used'])}

## Shortest Cleaned Documents

{dataframe_to_markdown_table(top_shortest_outputs, ['relative_path', 'title', 'char_count_after_cleaning', 'extraction_method_used', 'warnings'])}
"""
    before_after_stats_path.write_text(stats_report, encoding="utf-8")
    print(f"Saved summary report to {summary_report_path}")
    print(f"Saved before/after stats report to {before_after_stats_path}")

    return {
        "cleaned_files": cleaned_files,
        "file_size_raw": file_size_raw,
        "cleaned_file_size": cleaned_file_size,
        "file_size_reduction_pct": file_size_reduction_pct,
    }


# %% [markdown]
# ## 10. Boilerplate Frequency Review
#
# This optional review counts repeated non-empty lines across cleaned Markdown
# files. Lines appearing in many files can indicate residual boilerplate. The
# notebook saves candidates for manual inspection but does not delete them by
# default.

# %%
def content_lines_without_metadata(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                lines = lines[idx + 1 :]
                break
    cleaned = []
    for line in lines:
        line = compact_whitespace(line)
        if len(line) >= 20:
            cleaned.append(line)
    return cleaned


def save_repeated_line_candidates(cleaned_files: list[Path]) -> pd.DataFrame:
    line_doc_counts = Counter()
    line_examples = defaultdict(list)
    for path in tqdm(cleaned_files, desc="Counting repeated lines"):
        if not path.exists():
            continue
        unique_lines = set(content_lines_without_metadata(path))
        for line in unique_lines:
            line_doc_counts[line] += 1
            if len(line_examples[line]) < 3:
                line_examples[line].append(str(path.relative_to(CLEANED_CORPUS_DIR)))

    threshold_count = max(2, math.ceil(max(1, len(cleaned_files)) * REPEATED_LINE_THRESHOLD))
    repeated_rows = [
        {
            "line": line,
            "document_count": count,
            "document_percentage": round(count / max(1, len(cleaned_files)) * 100, 2),
            "example_files": "; ".join(line_examples[line]),
        }
        for line, count in line_doc_counts.items()
        if count >= threshold_count
    ]
    repeated_df = (
        pd.DataFrame(repeated_rows).sort_values("document_count", ascending=False)
        if repeated_rows
        else pd.DataFrame(columns=["line", "document_count", "document_percentage", "example_files"])
    )
    CLEANED_CORPUS_INFO_DIR.mkdir(parents=True, exist_ok=True)
    repeated_path = CLEANED_CORPUS_INFO_DIR / "repeated_line_candidates.csv"
    repeated_df.to_csv(repeated_path, index=False)
    print(f"Saved {len(repeated_df):,} repeated-line candidates to {repeated_path}")

    if REMOVE_REPEATED_LINES and not repeated_df.empty:
        repeated_lines = set(repeated_df["line"])
        for path in tqdm(cleaned_files, desc="Removing repeated lines"):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            filtered = [line for line in lines if compact_whitespace(line) not in repeated_lines]
            path.write_text("\n".join(filtered).rstrip() + "\n", encoding="utf-8")
        print("Repeated lines removed because REMOVE_REPEATED_LINES=True")
    else:
        print("Repeated lines were not removed. Inspect the CSV before enabling removal.")
    return repeated_df


# %% [markdown]
# ## 11. Sample Inspection
#
# Review these examples to judge whether cleaning is too aggressive or too weak.
# Good outputs should retain the page title and technical content while
# excluding menus, banners, cookie text, and repeated navigation.

# %%
def print_sample_inspection(analysis_df: pd.DataFrame) -> None:
    sample_df = analysis_df[analysis_df["processing_status"].eq("success")].copy()
    if len(sample_df) > SAMPLE_N:
        sample_df = sample_df.sample(SAMPLE_N, random_state=42)

    for _, row in sample_df.iterrows():
        output_path = Path(row["planned_output_path"])
        text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
        preview = text[:1500]
        print("=" * 100)
        print(f"Raw HTML path: {row['source_file_path']}")
        print(f"Extracted title: {row['title']}")
        print(f"Raw character count: {row['char_count_before_cleaning']:,}")
        print(f"Cleaned character count: {row['char_count_after_cleaning']:,}")
        print(f"Reduction percentage: {row['reduction_pct']}%")
        print("-" * 100)
        print(preview)
        print()


# %% [markdown]
# ## 12. Run Cleaning Pipeline
#
# This cell executes the full local, reproducible cleaning workflow. It does not
# modify or delete files in `raw_corpus/`.

# %%
def main() -> None:
    CLEANED_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_CORPUS_INFO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Raw corpus: {RAW_CORPUS_DIR.resolve()}")
    print(f"Cleaned corpus: {CLEANED_CORPUS_DIR.resolve()}")
    print(f"Cleaning diagnostics: {CLEANED_CORPUS_INFO_DIR.resolve()}")

    source_files = discover_source_files()
    initial_manifest = build_initial_manifest(source_files)
    print(f"Discovered {len(source_files):,} HTML files under {RAW_CORPUS_DIR}/")
    if initial_manifest.empty:
        print("No source files found. Nothing to clean.")
        return

    manifest = process_corpus(source_files)
    summary, analysis_df, top_largest_reduction, top_shortest_outputs = build_diagnostics(manifest)
    print(pd.DataFrame([summary]).to_string(index=False))

    print("\nTop files with largest reduction:")
    print(
        top_largest_reduction[
            [
                "relative_path",
                "char_count_before_cleaning",
                "char_count_after_cleaning",
                "char_reduction",
                "reduction_pct",
                "extraction_method_used",
            ]
        ].to_string(index=False)
    )

    print("\nTop files with shortest cleaned output:")
    print(
        top_shortest_outputs[
            [
                "relative_path",
                "title",
                "char_count_after_cleaning",
                "extraction_method_used",
                "warnings",
            ]
        ].to_string(index=False)
    )

    report_info = save_summary_reports(
        manifest,
        summary,
        analysis_df,
        top_largest_reduction,
        top_shortest_outputs,
    )
    repeated_df = save_repeated_line_candidates(report_info["cleaned_files"])
    if not repeated_df.empty:
        print("\nTop repeated-line candidates:")
        print(repeated_df.head(20).to_string(index=False))

    print("\nSample before/after inspection:")
    print_sample_inspection(analysis_df)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## 13. Next Steps
#
# Recommended follow-up checks:
#
# - Inspect `cleaned_corpus_info/repeated_line_candidates.csv` for residual
#   boilerplate.
# - Inspect very short cleaned outputs in the manifest.
# - Review the before/after examples above before committing to downstream
#   chunking.
# - Use the cleaned Markdown files for heading-aware chunking, hybrid retrieval
#   indexing, and later RAGAS evaluation.
