"""Streamlit UI for the alcohol label verification prototype.

A reviewer-focused compliance dashboard with two modes:
  - Single label: upload one product's label image(s) + the application values; see the
    per-field verdict with drill-down evidence.
  - Batch: upload many labels at once, optionally with an application-data file (CSV/JSON,
    matched to products by filename stem) so each product is verified against its submitted
    values; products without application data are screened against the fixed rules only.
    Results persist in the session.

The deterministic verdict (verification.py) is central; the model's observations are shown
as evidence the reviewer can inspect, never as the judgment itself.
"""
import csv
import html
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# Make API keys available before clients/functions are created (extraction's client is lazy,
# but config reads the env at import time -- this block MUST stay above those imports).
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # fall back to keys already set in the environment

from config import BATCH_MAX_WORKERS
from extraction import extract_fields, failure_kind
from verification import verify, verify_label_only, PASS, REVIEW, FAIL

_log = logging.getLogger(__name__)

st.set_page_config(page_title="TTB Label Verification", layout="wide")

# --- styling: restrained enterprise palette, resolved per the user's ACTUAL theme -----------
# Streamlit's CSS theme variables are not reliably defined for custom st.markdown HTML, so
# relying on var(--text-color) left dark-theme users with dark-on-dark text (reported bug).
# Instead the session's resolved theme is read in Python (st.context.theme, Streamlit >=1.46)
# and concrete colors are baked into the stylesheet via the __TEXT__/__MUTED__ tokens.

def _is_dark_theme():
    try:
        return getattr(getattr(st.context, "theme", None), "type", None) == "dark"
    except Exception:
        return False   # bare mode / very old Streamlit: assume the light default


_TEXT = "#f1f5f9" if _is_dark_theme() else "#0f172a"    # primary text
_MUTED = "#a0aec0" if _is_dark_theme() else "#64748b"   # secondary text
# panels/borders use translucent slate, which reads correctly on BOTH backgrounds
_CSS = """
<style>
/* leave room for Streamlit's fixed top toolbar so the page header is never cut off */
.block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1280px; }

/* header */
.lv-header { border-bottom: 1px solid rgba(148,163,184,.35); padding-bottom: 0.9rem;
             margin-bottom: 0.4rem; }
.lv-header h1 { font-size: 2.1rem; font-weight: 700; color: __TEXT__;
                margin: 0 0 0.25rem 0; letter-spacing: -0.01em; }
.lv-header p { color: __MUTED__; font-size: 1rem; margin: 0; }

/* section titles */
/* font sizes float at or near the 1rem default: half the user base is 50+ (audit finding:
   the previous 0.72-0.9rem range put all the critical content below default size) */
.lv-section { color: __MUTED__; font-size: 0.9rem; font-weight: 650;
              letter-spacing: .05em; text-transform: uppercase; margin-bottom: 0.35rem; }

/* status badges (text + color: never color-only; fixed light fills read on any background) */
.lv-badge { display: inline-block; padding: 2px 10px; border-radius: 4px; font-size: 0.85rem;
            font-weight: 650; letter-spacing: .03em; border: 1px solid; white-space: nowrap; }
.lv-badge.pass   { color: #166534; background: #f0fdf4; border-color: #bbf7d0; }
.lv-badge.review { color: #92400e; background: #fffbeb; border-color: #fde68a; }
.lv-badge.fail   { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
.lv-badge.error  { color: #475569; background: #f8fafc; border-color: #cbd5e1; }

/* overall verdict strip */
.lv-verdict { border: 1px solid rgba(148,163,184,.35); border-left: 4px solid #94a3b8;
              border-radius: 6px; background: rgba(148,163,184,.12);
              padding: 12px 16px; margin: 0.4rem 0 0.9rem 0; }
.lv-verdict.pass   { border-left-color: #16a34a; }
.lv-verdict.review { border-left-color: #d97706; }
.lv-verdict.fail   { border-left-color: #dc2626; }
.lv-verdict .lv-v-title { font-size: 1.05rem; font-weight: 650; color: __TEXT__; }
.lv-verdict .lv-v-sub { color: __MUTED__; font-size: 0.95rem; margin-top: 2px; }

/* field-results table */
table.lv-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; margin: 0.2rem 0 0.4rem 0; }
table.lv-table th { text-align: left; color: __MUTED__; font-weight: 650; font-size: 0.85rem;
                    text-transform: uppercase; letter-spacing: .05em;
                    border-bottom: 1px solid rgba(148,163,184,.35); padding: 6px 10px;
                    background: rgba(148,163,184,.12); }
table.lv-table td { border-bottom: 1px solid rgba(148,163,184,.2); padding: 7px 10px;
                    vertical-align: top; color: __TEXT__; }
table.lv-table td.lv-dim { color: __MUTED__; }
table.lv-table tr:last-child td { border-bottom: none; }

/* key-value evidence rows */
.lv-kv { font-size: 0.95rem; color: __TEXT__; margin: 2px 0; }
.lv-kv .k { color: __MUTED__; display: inline-block; min-width: 220px; }

.lv-note { color: __TEXT__; background: rgba(148,163,184,.12);
           border: 1px solid rgba(148,163,184,.35); border-radius: 6px;
           padding: 8px 12px; font-size: 0.95rem; margin: 0.3rem 0; }
</style>
"""
st.markdown(_CSS.replace("__TEXT__", _TEXT).replace("__MUTED__", _MUTED),
            unsafe_allow_html=True)

_LABEL = {PASS: "Pass", REVIEW: "Needs review", FAIL: "Fail"}
_BADGE = {PASS: "pass", REVIEW: "review", FAIL: "fail"}
_FRIENDLY_ERROR = (
    "We couldn't read this label. This is usually a photo issue — try a clearer, "
    "straight-on image with the whole label in frame and no glare. "
    "(If it keeps happening, the label-reading service may be unavailable.)"
)
# Failure-specific guidance, keyed by extraction.failure_kind (audit finding: every failure
# class used to render as "a photo issue", so a missing API key or a rate-limited batch told
# the user to retake their photo). _FRIENDLY_ERROR stays the fallback for unknown failures.
_ERROR_MESSAGES = {
    "auth": "The label-reading service isn't set up: its access key is missing or invalid. "
            "This is a setup problem, not a photo problem — ask whoever runs this app to "
            "check the OpenAI API key.",
    "quota": "The label-reading service's account is out of credits (or billing isn't set "
             "up). This is a setup problem, not a photo problem — ask whoever runs this app "
             "to check the OpenAI account's billing.",
    "rate_limit": "The label-reading service is briefly over capacity (too many labels at "
                  "once). Your photo is fine — wait a minute and try again.",
    "timeout": "The label-reading service took too long to answer. Check your internet "
               "connection and try again.",
    "connection": "Couldn't reach the label-reading service. Check your internet connection "
                  "and try again.",
    "bad_response": "The label-reading service returned an unusable answer. This is usually "
                    "temporary — try again.",
}
# short forms of the same causes for the batch results table
_ERROR_SHORT = {
    "auth": "service not set up", "quota": "service out of credits",
    "rate_limit": "service busy — try again",
    "timeout": "service timeout — try again", "connection": "no connection to service",
    "bad_response": "bad service reply — try again",
}
# evidence-only extraction fields: captured for the reviewer, never judged by the verifier
_EVIDENCE_FIELDS = ("fanciful_name", "statement_of_composition", "sulfite_declaration")
# application fields the verifier compares against (the batch application file's columns)
_APP_FIELDS = ("brand_name", "class_type", "alcohol_content", "net_contents",
               "name_and_address", "country_of_origin")
def _extract(images, media_type="image/png"):
    """Extract label fields from one product's image(s)."""
    return extract_fields(images, media_type)


def _stem(filename):
    """Product stem of a filename: extension dropped and a TRAILING side marker stripped --
    ``_Front``, ``-other``, `` back``, ``_Label`` (optionally followed by a copy number).
    Anchored at the end, so words inside a product name (e.g. ``back_forty_ipa``) are never
    eaten. Shared by upload grouping and application-data matching."""
    stem = os.path.splitext(filename)[0]
    return re.sub(r"[ _\-]+(front|other|back|label)([ _\-]*\d+|\s*\(\d+\))?$", "",
                  stem, flags=re.IGNORECASE) or stem


def _group_uploads(files, group_pairs):
    """Turn uploaded files into a list of (label, [files]) products. With grouping on, files
    that share a name stem (see _stem) are read together as one product, so a front+back
    pair is screened as one label instead of false-failing the front for the warning that
    lives on the back. With grouping off, each file is its own product. Upload order is
    preserved."""
    if not group_pairs:
        # each file is its own product, but the LABEL is still the stem so application-data
        # matching works with grouping off (review finding: raw filenames with extensions
        # could never match an application row's product stem -> silent rules-only downgrade)
        return [(_stem(f.name), [f]) for f in files]
    groups, order = {}, []
    for f in files:
        key = _stem(f.name)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(f)
    return [(key, groups[key]) for key in order]


def _norm_header(key):
    """Normalize an application-file column name: 'Brand Name' / 'brand-name' -> 'brand_name'."""
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _parse_applications(file):
    """Parse an uploaded application-data file into {product_key_lower: {field: value}}.

    CSV: a ``product`` column plus any of the verifier fields (``_APP_FIELDS``; header
    spelling is normalized, so 'Brand Name' / 'brand-name' both work).
    JSON: either a mapping ``{"product_stem": {fields...}}`` or a list of objects each with a
    ``"product"`` key. Returns (mapping, error_message_or_None, warnings) — duplicate product
    keys are last-row-wins but REPORTED (review finding: silently verifying a label against
    the wrong application's values is a data-integrity hazard). A bad file never blocks the
    batch -- it just falls back to rules-only screening."""
    try:
        text = file.getvalue().decode("utf-8-sig")
        mapping, dups = {}, []
        if file.name.lower().endswith(".json"):
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("products"), list):
                data = data["products"]
            if isinstance(data, dict):
                # the mapping key IS the product and must beat any inner product-ish field:
                # normalize the inner keys FIRST, then assert the key last — an inner
                # 'Product' would otherwise normalize onto 'product' later in iteration
                # order and override k (review finding)
                rows = [{**{_norm_header(kk): vv for kk, vv in v.items()}, "product": k}
                        for k, v in data.items() if isinstance(v, dict)]
            elif isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict)]
            else:
                return {}, "JSON must be a mapping of product -> fields, or a list of objects", []
        else:
            rows = list(csv.DictReader(io.StringIO(text)))
        # normalize header/key spelling: the previous exact-lowercase match silently produced
        # all-empty values for a 'Brand Name'-headed file while still reporting "matched"
        # (audit finding). DictReader puts overflow cells under a None key — drop those.
        rows = [{_norm_header(k): v for k, v in row.items() if k is not None} for row in rows]
        for row in rows:
            prod = str(row.get("product") or "").strip()
            if not prod:
                continue
            if prod.lower() in mapping:
                dups.append(prod)
            mapping[prod.lower()] = {f: str(row.get(f) or "").strip() for f in _APP_FIELDS}
        if not mapping:
            return {}, "no rows with a 'product' value were found", []
        warnings = []
        if not any(f in row for row in rows for f in _APP_FIELDS):
            warnings.append("no recognized application-field columns were found (expected any "
                            "of: " + ", ".join(_APP_FIELDS) + ") — check the header row; "
                            "products will effectively be screened against the fixed rules only")
        if dups:
            warnings.append("duplicate product row(s): " + ", ".join(sorted(set(dups)))
                            + " — the last row wins; check the file")
        return mapping, None, warnings
    except UnicodeDecodeError:
        return {}, "the file isn't plain text — save it as a regular CSV or JSON file", []
    except json.JSONDecodeError:
        return {}, "the JSON file could not be read — it isn't valid JSON", []
    except Exception as exc:
        return {}, str(exc)[:160], []


def _pick_application_row(mapping, stem):
    """Choose the application row for SINGLE mode: the row matching the uploaded image's
    product stem; else, if the file holds exactly one row, that row. Ambiguity returns None
    rather than silently prefilling the wrong product's values."""
    if stem and stem.lower() in mapping:
        return mapping[stem.lower()], f"matched product '{stem}'"
    if len(mapping) == 1:
        prod = next(iter(mapping))
        return mapping[prod], f"single row ('{prod}') used"
    return None, (f"no row matches product '{stem}'" if stem else "no image uploaded to match") + \
                 " — fields left for manual entry"


# session-state keys for the single-mode application form (prefillable)
_FORM_KEYS = {"brand_name": "app_brand", "class_type": "app_class",
              "alcohol_content": "app_abv", "net_contents": "app_net",
              "name_and_address": "app_addr", "country_of_origin": "app_country"}


def _set_form_values(values, source):
    """Write application values into the form's session-state keys. Must run BEFORE the
    text_input widgets are instantiated in the rerun (file prefill) or from an on_click
    callback (the copy button) — both run-paths satisfy that."""
    for field, key in _FORM_KEYS.items():
        st.session_state[key] = values.get(field) or ""
    st.session_state["app_source"] = source


def _copy_extracted_to_form():
    """Explicit convenience: copy the LAST read's extracted values into the application form.
    This makes the next comparison non-independent (model vs itself), so the source is tracked
    and the verification result carries a clear not-independent notice."""
    extracted = (st.session_state.get("single") or {}).get("extracted") or {}
    values = {field: (extracted.get(field) or {}).get("value") or "" for field in _APP_FIELDS}
    _set_form_values(values, "copied")
    # snapshot the copied values: at verify time the form is compared against this, so a user
    # who EDITS the values escapes the "not independent" flag (review finding: 'copied' had
    # no path back to manual)
    st.session_state["_copied_snapshot"] = values
    # the button sits below the result but fills the form far above it — without feedback a
    # user has no cue that anything happened (audit finding)
    st.toast("Copied into the application form above — review or edit the values, "
             "then click Verify label again.")


# File uploaders cannot be emptied programmatically; the supported pattern is a nonce in the
# widget key — incrementing it makes Streamlit render a fresh, empty uploader. The clear
# callbacks below run before widgets are instantiated, so resetting form keys here is safe.
# Clearing removes WORK PRODUCT (files, values, results) but not preferences (the batch
# group-by-filename checkbox keeps its setting).

def _clear_single():
    """Start a new single-label review: drop uploads, form values, tracking, and the result."""
    st.session_state["single_nonce"] = st.session_state.get("single_nonce", 0) + 1
    st.session_state.pop("single", None)
    st.session_state.pop("_single_app_fid", None)
    st.session_state.pop("_single_app_msg", None)
    st.session_state.pop("_copied_snapshot", None)
    st.session_state.pop("app_source", None)
    for key in _FORM_KEYS.values():
        st.session_state[key] = ""


def _clear_batch():
    """Start a new batch: drop uploads, the application file, and all results (also frees the
    session memory the per-product photos occupy)."""
    st.session_state["batch_nonce"] = st.session_state.get("batch_nonce", 0) + 1
    st.session_state.pop("batch", None)


# --- rendering helpers --------------------------------------------------------

def _esc(value):
    return html.escape(str(value)) if value not in (None, "") else "&mdash;"


def _badge(status):
    label = _LABEL.get(status, str(status).capitalize())
    cls = _BADGE.get(status, "error")
    return f'<span class="lv-badge {cls}">{html.escape(label)}</span>'


def _verdict_strip(result, elapsed=None):
    """The overall verdict: prominent, quiet, with per-status counts."""
    overall = result["overall"]
    counts = {PASS: 0, REVIEW: 0, FAIL: 0}
    for f in result["fields"]:
        counts[f.status] = counts.get(f.status, 0) + 1
    bits = [f"{counts[FAIL]} fail", f"{counts[REVIEW]} needs review", f"{counts[PASS]} pass"]
    bev = result.get("beverage_type")
    sub = " &middot; ".join(bits)
    if bev and bev != "unknown":
        sub += f" &middot; detected beverage type: {html.escape(bev)}"
    if elapsed is not None:
        sub += f" &middot; verified in {elapsed:.1f}s"
    st.markdown(
        f'<div class="lv-verdict {_BADGE.get(overall, "error")}">'
        f'<div class="lv-v-title">Overall: {_LABEL.get(overall, overall)} &nbsp;{_badge(overall)}</div>'
        f'<div class="lv-v-sub">{sub}</div></div>',
        unsafe_allow_html=True)


def _fields_table(fields):
    """Field-level results as a scannable table."""
    rows = []
    for f in fields:
        name = f.field.replace("_", " ").capitalize()
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{_badge(f.status)}</td>"
            f"<td>{_esc(f.reason)}</td>"
            f"<td class='lv-dim'>{_esc(f.extracted)}</td>"
            f"<td class='lv-dim'>{_esc(f.expected)}</td>"
            "</tr>")
    st.markdown(
        "<table class='lv-table'>"
        "<thead><tr><th>Field</th><th>Status</th><th>Detail</th>"
        "<th>Label value</th><th>Application value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True)


def _kv(key, value):
    st.markdown(f"<div class='lv-kv'><span class='k'>{html.escape(key)}</span> {_esc(value)}</div>",
                unsafe_allow_html=True)


def _yn(value):
    return {True: "yes", False: "no"}.get(value, "not determinable")


def _warning_evidence(extracted):
    """The model's government-warning observations (evidence, not judgment)."""
    gw = extracted.get("government_warning") or {}
    _kv("Warning found on label", _yn(gw.get("present")))
    _kv("Header in capital letters", _yn(gw.get("header_all_caps")))
    _kv("Header printed in bold",
        f"{_yn(gw.get('header_bold'))}  ({gw.get('header_bold_confidence') or '—'} confidence)")
    _kv("Body text printed in bold",
        f"{_yn(gw.get('body_bold'))}  ({gw.get('body_bold_confidence') or '—'} confidence)")
    _kv("Basis for the bold reading", gw.get("header_bold_basis"))
    if gw.get("image_quality_notes"):
        _kv("Warning-read note", gw.get("image_quality_notes"))
    text = gw.get("text")
    if text:
        st.markdown("<div class='lv-kv'><span class='k'>Transcribed warning text</span></div>",
                    unsafe_allow_html=True)
        st.code(text, language=None, wrap_lines=True)


def _evidence_panel(extracted, result):
    """Evidence-only extraction fields + additional statements: shown for the reviewer,
    never auto-checked."""
    shown = False
    for name in _EVIDENCE_FIELDS:
        obj = extracted.get(name) or {}
        if obj.get("present") or obj.get("value"):
            _kv(name.replace("_", " ").capitalize(),
                f"{obj.get('value') or '(present, unreadable)'}"
                f"  ({obj.get('confidence') or '—'} confidence)")
            shown = True
    stmts = result.get("additional_statements") or []
    for s in stmts:
        kind = f" [{s.get('kind')}]" if s.get("kind") else ""
        _kv(f"Other statement{kind}", s.get("value"))
        shown = True
    note = result.get("image_quality_notes")
    if note:
        _kv("Image quality note", note)
        shown = True
    if not shown:
        st.caption("No evidence-only fields or additional statements were found on this label.")


def _render_product(result, extracted, images=None, elapsed=None):
    """One product's verification result: verdict strip, field table, evidence expanders."""
    _verdict_strip(result, elapsed)

    if images:
        img_col, detail_col = st.columns([1, 3])
        for b in images:
            img_col.image(b, width="stretch")
        box = detail_col
    else:
        box = st.container()

    with box:
        _fields_table(result["fields"])

        # surface the model's photo note at the verdict level (audit finding: it was only
        # visible inside a collapsed expander, while the reframed field reasons said "submit
        # a clearer label image" without saying what was wrong with the photo)
        note = result.get("image_quality_notes")
        if note:
            st.markdown(f"<div class='lv-note'><strong>Photo note:</strong> {html.escape(note)}"
                        "</div>", unsafe_allow_html=True)

        # a missing warning is usually a back label that wasn't uploaded; branch on the
        # machine-readable cause, not the display reason (which may be reworded freely)
        warn = next((f for f in result["fields"] if f.field == "government_warning"), None)
        if warn is not None and warn.status == FAIL and getattr(warn, "cause", None) == "absence":
            st.markdown("<div class='lv-note'>No government warning was found in the image(s). "
                        "It is usually on the back/other label — include that image too if "
                        "you have it.</div>", unsafe_allow_html=True)

        with st.expander("Government warning — what was seen on the label"):
            _warning_evidence(extracted or {})
        with st.expander("Other label details (for reference — not auto-checked)"):
            _evidence_panel(extracted or {}, result)
        with st.expander("Full technical readout (JSON)"):
            st.json(extracted or {})


def _thumb(image_bytes, max_px=560):
    """Downscaled display copy for session storage. Results persist across reruns, so storing
    full-resolution bytes doubled per-session memory with no cap (review finding); the
    uploader buffers keep the originals, and model calls always use those originals — this
    copy is only ever displayed."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes))
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception:
        return image_bytes   # display fallback — never block a result on a thumbnail


# --- header --------------------------------------------------------------------

st.markdown(
    '<div class="lv-header"><h1>TTB Label Verification</h1>'
    '<p>Checks an alcohol beverage label against the federal labeling rules (27 CFR) and '
    'the values on the application. A screening aid — not a final legal determination: '
    'anything that cannot be confirmed from the photo goes to a human reviewer, never an '
    'automatic pass.</p></div>',
    unsafe_allow_html=True)

# Tabs, not a mode radio: tab bodies render on EVERY rerun, so switching views no longer
# unmounts the other view's widgets — which is what silently wiped the uploads and all six
# typed form values on a Single -> Batch -> Single round-trip (Streamlit drops keyed widget
# state when a widget doesn't render; audit finding). Tabs are also a more familiar control
# than a label-collapsed radio in the corner.
tab_single, tab_batch = st.tabs(["Single label", "Batch screening"])


# --- Single-label flow -----------------------------------------------------------
with tab_single:
    nonce = st.session_state.get("single_nonce", 0)   # bumped by Clear to reset the uploaders
    # Safety net: if the form is entirely blank (e.g. the user manually cleared every field),
    # reset the prefill/copy tracking too — otherwise a stale 'copied' flag could mislabel
    # freshly typed values and a re-uploaded application file would never re-prefill (review
    # finding). Tabs keep the widgets mounted, so mode switches no longer blank the form;
    # this now only fires for a genuinely emptied form.
    if not any(st.session_state.get(k) for k in _FORM_KEYS.values()):
        st.session_state.pop("_single_app_fid", None)
        st.session_state.pop("_single_app_msg", None)
        st.session_state.pop("_copied_snapshot", None)
        st.session_state.pop("app_source", None)
    # the second condition hides the banner on the very rerun a verification completes —
    # without it, the result is stored mid-run AFTER this line already rendered, so the
    # how-to banner sat above the first result for one run (review finding)
    if not st.session_state.get("single") and not st.session_state.get("single_verify"):
        st.info("**How it works** — 1. Upload the label photo(s), front and back.  "
                "2. Type the application values, or leave them all blank to check just the "
                "basic rules.  3. Click **Verify label**.")
    up_col, app_col = st.columns(2)
    with up_col, st.container(border=True):
        st.markdown('<div class="lv-section">Label upload</div>', unsafe_allow_html=True)
        image_files = st.file_uploader(
            "Label image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
            key=f"single_files_{nonce}",
            help="Upload the front and back/'other' labels together — the government "
                 "warning, net contents, and name/address are usually on the back. "
                 "iPhone HEIC photos aren't supported: export or save them as JPEG first.")
    with app_col, st.container(border=True):
        st.markdown('<div class="lv-section">Application data</div>', unsafe_allow_html=True)
        app_file = st.file_uploader(
            "Prefill from application file (optional, CSV or JSON)", type=["csv", "json"],
            key=f"single_app_file_{nonce}",
            help="Same format as the batch application file. The row whose 'product' value "
                 "matches the image's file name (ignoring endings like _front/_back) "
                 "prefills the fields below; review and edit before verifying.")
        if app_file is not None:
            # Gate on file id + current image stem: a new file OR a new/changed image
            # re-attempts the match (review finding: gating on the file id alone meant
            # uploading the application file BEFORE the image left the form permanently
            # unfilled). The messages persist in session state so the outcome stays visible
            # across reruns instead of vanishing after one.
            stem = _stem(image_files[0].name) if image_files else None
            fid = getattr(app_file, "file_id", None) or f"{app_file.name}:{app_file.size}"
            gate = f"{fid}|{stem or ''}"
            if st.session_state.get("_single_app_fid") != gate:
                st.session_state["_single_app_fid"] = gate
                mapping, err, warns = _parse_applications(app_file)
                msgs = []
                if err:
                    msgs.append(("warn", f"Could not use the application file ({err})."))
                else:
                    msgs.extend(("warn", f"Application file: {w}") for w in warns)
                    row, msg = _pick_application_row(mapping, stem)
                    if row:
                        _set_form_values(row, "file")
                    msgs.append(("info", f"Application file: {msg}."))
                st.session_state["_single_app_msg"] = msgs
        elif st.session_state.pop("_single_app_fid", None):
            st.session_state.pop("_single_app_msg", None)   # file removed -> drop its messages
        for kind, text in st.session_state.get("_single_app_msg", []):
            (st.warning if kind == "warn" else st.caption)(text)
        brand = st.text_input("Brand name", key="app_brand")
        class_type = st.text_input(
            "Class / type", key="app_class",
            help="The specific identity, e.g. 'Kentucky Straight Bourbon Whiskey' or 'India Pale Ale'.")
        abv = st.text_input("Alcohol content", key="app_abv", placeholder="e.g. 45% Alc./Vol.")
        net = st.text_input("Net contents", key="app_net", placeholder="e.g. 750 mL")
        name_addr = st.text_input(
            "Name & address", key="app_addr",
            placeholder="e.g. Bottled by Old Tom Distillery, Bardstown, KY")
        country = st.text_input(
            "Country of origin", key="app_country",
            placeholder="imports only — leave blank if domestic")
        st.caption("Leave every field blank to screen against the fixed rules only "
                   "(government warning + mandatory-field presence).")

    act_col, _gap, clear_col = st.columns([2, 6, 1])
    verify_clicked = act_col.button("Verify label", type="primary", key="single_verify",
                                    disabled=not image_files, width="stretch")
    # two-step confirm (audit finding): one stray click on a bare Clear button threw away
    # the uploads, six typed fields, and the result, with no undo
    with clear_col.popover("Clear", width="stretch"):
        st.caption("Removes the uploaded files, application values, and the result below.")
        st.button("Yes, start over", key="single_clear", on_click=_clear_single,
                  width="stretch")
    if verify_clicked:
        application = {
            "brand_name": brand, "class_type": class_type, "alcohol_content": abv,
            "net_contents": net, "name_and_address": name_addr, "country_of_origin": country,
        }
        has_app = any((v or "").strip() for v in application.values())
        # 'copied' only sticks if the form still equals the copied snapshot — a user who
        # edited the values has made them (at least partly) independent again (review
        # finding: the flag previously had no path back to 'manual')
        source = st.session_state.get("app_source", "manual")
        if source == "copied":
            snap = st.session_state.get("_copied_snapshot") or {}
            if any((application.get(f) or "").strip() != (snap.get(f) or "").strip()
                   for f in _APP_FIELDS):
                source = "manual"
                st.session_state["app_source"] = "manual"
        images = [(f.getvalue(), f.type or "image/png") for f in image_files]
        with st.spinner("Reading the label..."):
            try:
                start = time.perf_counter()
                extracted = _extract(images)
                # blank form -> honest rules-only screening, NEVER auto-filled values: the
                # application must stay an independent witness (model-vs-itself comparisons
                # would trivially pass — see _copy_extracted_to_form for the explicit,
                # clearly-labeled convenience path).
                result = verify(extracted, application) if has_app else verify_label_only(extracted)
                elapsed = time.perf_counter() - start
                item = {
                    "label": _stem(image_files[0].name),
                    "files": [f.name for f in image_files],
                    "result": result, "extracted": extracted, "elapsed": elapsed,
                    "images": [_thumb(b) for b, _ in images],   # display copies only
                    "matched": has_app, "application": dict(application),
                    "screening": "application" if has_app else "rules_only",
                    "app_source": source if has_app else None,
                }
                st.session_state["single"] = item
            except Exception as exc:
                # classified message + a log line (audit finding: every failure class showed
                # the photo-advice text, and nothing was ever logged for whoever debugs it)
                _log.exception("single-label verification failed")
                st.session_state.pop("single", None)
                st.error(_ERROR_MESSAGES.get(failure_kind(exc), _FRIENDLY_ERROR))

    single = st.session_state.get("single")
    if single:
        st.markdown('<div class="lv-section">Verification result</div>', unsafe_allow_html=True)
        # staleness cue (review finding): the stored result keeps rendering after the inputs
        # change — in a compliance tool a reviewer must never read an old verdict against new
        # files/values, so compare the stored inputs to the current ones
        current_files = sorted(f.name for f in image_files) if image_files else []
        current_app = {f: (st.session_state.get(k) or "").strip() for f, k in _FORM_KEYS.items()}
        stored_app = {f: ((single.get("application") or {}).get(f) or "").strip()
                      for f in _APP_FIELDS}
        if current_files != sorted(single["files"]) or current_app != stored_app:
            st.markdown("<div class='lv-note'><strong>Inputs changed:</strong> the result below "
                        "is from a previous read and may not match the files or application "
                        "values currently entered above — click Verify label to refresh, or "
                        "Clear to start over.</div>", unsafe_allow_html=True)
        if single.get("screening") == "rules_only":
            st.markdown("<div class='lv-note'>No application data was provided — this label was "
                        "screened against the fixed rules only (government warning + "
                        "mandatory-field presence). Fill in the application fields to also "
                        "verify label-vs-application consistency.</div>", unsafe_allow_html=True)
        elif single.get("app_source") == "copied":
            st.markdown("<div class='lv-note'><strong>Not an independent comparison:</strong> "
                        "the application values were copied from a label read, so field matches "
                        "compare the model against itself. Confirm the values against the actual "
                        "application before relying on this result.</div>", unsafe_allow_html=True)
        _render_product(single["result"], single["extracted"],
                        images=single["images"], elapsed=single["elapsed"])
        st.button("Copy extracted values into the application form",
                  on_click=_copy_extracted_to_form,
                  help="Convenience only: fills the form from this read so you can edit and "
                       "re-verify. A result produced from unedited copied values is flagged as "
                       "not independent — the proper inputs are the applicant's submitted values.")


# --- Batch flow -------------------------------------------------------------------
with tab_batch:
    # gated on the Screen click too, for the same first-result reason as the single banner
    if not st.session_state.get("batch") and not st.session_state.get("batch_screen"):
        st.info("**How it works** — 1. Upload all the label photos.  2. Add the application "
                "data file if you have one.  3. Click **Screen products**.")
    with st.container(border=True):
        st.markdown('<div class="lv-section">Batch upload</div>', unsafe_allow_html=True)
        st.caption(
            "Upload label images, and optionally an application-data file so each product is "
            "verified against its submitted values. Products without application data are "
            "screened against the fixed rules only (government warning + mandatory-field "
            "presence).")
        nonce = st.session_state.get("batch_nonce", 0)   # bumped by Clear to reset the uploaders
        files = st.file_uploader(
            "Label images", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
            key=f"batch_files_{nonce}",
            help="iPhone HEIC photos aren't supported: export or save them as JPEG first.")
        # the pairing rule lives in visible text, not only a hover tooltip (audit finding:
        # camera filenames like IMG_1234.jpg silently become one product per file)
        st.caption("Name each product's photos with the same beginning plus _front / _back — "
                   "e.g. **oldtom_front.jpg** and **oldtom_back.jpg** are read together as "
                   "one product. Files named like IMG_1234.jpg are each treated as a "
                   "separate product.")
        group_pairs = st.checkbox(
            "Group front/back images of one product by filename", value=True,
            help="Read files that share a name stem (e.g. brandX_front.jpg + brandX_back.jpg) as "
                 "one product. Otherwise each image is screened on its own — and a front-only "
                 "image will fail for the government warning, which is usually on the back.")

        app_file = st.file_uploader(
            "Application data (optional, CSV or JSON)", type=["csv", "json"],
            key=f"batch_app_file_{nonce}",
            help="One row/entry per product. The 'product' value must match the shared "
                 "beginning of that product's file names (e.g. files brandX_front.jpg + "
                 "brandX_back.jpg -> product 'brandX'). Matching ignores capitalization.")
        # the expected columns live in visible text (audit finding: they were documented
        # nowhere in the app, so a mis-headed file silently matched with empty values)
        st.caption("Columns: **product** (required) plus any of "
                   + ", ".join(_APP_FIELDS) + ".")
        applications, app_err = ({}, None)
        if app_file is not None:
            applications, app_err, app_warns = _parse_applications(app_file)
            if app_err:
                st.warning(f"Could not use the application file ({app_err}) — "
                           f"all products will be screened against the fixed rules only.")
            else:
                for w in app_warns:
                    st.warning(f"Application file: {w}")
                st.caption(f"Application data loaded for {len(applications)} product(s).")

        def _app_row_for(label):
            """The product's application row, or None when there is no row OR the row is
            entirely blank — a blank row must screen rules-only, exactly like single mode's
            blank form (audit finding: dict truthiness sent all-blank matched rows through
            verify(), producing a wall of 'no application value provided' reviews)."""
            row = applications.get(label.lower())
            return row if row and any(v.strip() for v in row.values()) else None

        products = _group_uploads(files, group_pairs) if files else []
        if products:
            st.caption(f"{len(products)} product(s) from {len(files)} file(s):")
            preview = [{"Product": label,
                        "Files": ", ".join(g.name for g in group),
                        "Application data": "matched" if _app_row_for(label)
                                            else ("row found but empty"
                                                  if label.lower() in applications else "—")}
                       for label, group in products]
            st.dataframe(preview, width="stretch", hide_index=True)
            # aggregate cue for typo'd 'product' values (audit finding: a row matching no
            # uploaded product was visible only as one easy-to-miss dash among 20 rows)
            stems = {label.lower() for label, _ in products}
            unused = sorted(p for p in applications if p not in stems)
            if unused:
                st.warning(f"{len(unused)} application row(s) match no uploaded product: "
                           + ", ".join(unused[:8]) + ("…" if len(unused) > 8 else "")
                           + " — check the 'product' values if this is unexpected.")

    act_col, _gap, clear_col = st.columns([2, 6, 1])
    # rendered disabled, not hidden, while there are no files yet — single mode does the
    # same, and a first-time user should see what the next step will be (audit finding)
    screen_clicked = act_col.button(
        f"Screen {len(products)} product(s)" if products else "Screen products",
        type="primary", disabled=not products, width="stretch", key="batch_screen")
    with clear_col.popover("Clear", width="stretch"):
        st.caption("Removes the uploaded files, the application data file, and all "
                   "results below.")
        st.button("Yes, start over", key="batch_clear", on_click=_clear_batch,
                  width="stretch")
    if screen_clicked:
        progress = st.progress(0.0, text="Starting…")
        start = time.perf_counter()
        total = len(products)
        items = [None] * total

        def _process(idx, label, group_files, application):
            try:
                t0 = time.perf_counter()
                images = [(g.getvalue(), g.type or "image/png") for g in group_files]
                extracted = _extract(images)
                result = (verify(extracted, application) if application
                          else verify_label_only(extracted))
                secs = round(time.perf_counter() - t0, 1)
                thumbs = [_thumb(b) for b, _ in images]   # display copies only (memory cap)
                return idx, label, result, extracted, thumbs, secs, None
            except Exception as exc:  # one bad product must not sink the batch
                _log.exception("batch product %r failed", label)
                return idx, label, None, None, [], None, (str(exc), failure_kind(exc))

        done = 0
        with ThreadPoolExecutor(max_workers=BATCH_MAX_WORKERS) as pool:
            # key results by index, not label — distinct products can share a stem/name
            # blank/missing application rows screen rules-only (see _app_row_for)
            futures = [pool.submit(_process, i, label, gf, _app_row_for(label))
                       for i, (label, gf) in enumerate(products)]
            for fut in as_completed(futures):
                idx, name, res, extracted, image_bytes, secs, err = fut.result()
                done += 1
                progress.progress(done / total, text=f"Processed {done}/{total} — {name}")
                items[idx] = {"label": name,
                              "files": [g.name for g in products[idx][1]],
                              "result": res, "extracted": extracted, "images": image_bytes,
                              "seconds": secs,
                              "matched": _app_row_for(name) is not None,
                              "error": err[0] if err else None,    # raw, for logs only
                              "error_kind": err[1] if err else None}
        progress.empty()
        elapsed_total = round(time.perf_counter() - start, 1)
        st.session_state["batch"] = {
            "items": items, "elapsed": elapsed_total,
            # input signature for the staleness cue below
            "inputs_sig": {"files": sorted(f.name for f in files), "group": group_pairs,
                           "apps": sorted(applications)},
        }

    batch = st.session_state.get("batch")
    if batch:
        items = batch["items"]
        counts = {"fail": 0, "needs_review": 0, "pass": 0, "error": 0}
        for item in items:
            key = "error" if item.get("error") else item["result"]["overall"]
            counts[key] = counts.get(key, 0) + 1

        st.markdown('<div class="lv-section">Batch results</div>', unsafe_allow_html=True)
        # staleness cue (review finding): results persist while the inputs above can change
        sig_now = {"files": sorted(f.name for f in files) if files else [],
                   "group": group_pairs, "apps": sorted(applications)}
        if batch.get("inputs_sig") is not None and batch["inputs_sig"] != sig_now:
            st.markdown("<div class='lv-note'><strong>Inputs changed:</strong> the results "
                        "below are from a previous screening and may not match the files, "
                        "grouping, or application data currently set above — click Screen to "
                        "refresh, or Clear to start over.</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="lv-verdict {"fail" if counts["fail"] or counts["error"] else ("review" if counts["needs_review"] else "pass")}">'
            f'<div class="lv-v-title">Screened {len(items)} product(s)</div>'
            f'<div class="lv-v-sub">{counts["fail"]} fail &middot; {counts["needs_review"]} needs review &middot; '
            f'{counts["pass"]} pass &middot; {counts["error"]} error &middot; '
            f'{batch["elapsed"]}s total</div></div>',
            unsafe_allow_html=True)

        order = {"fail": 0, "error": 1, "needs_review": 2, "pass": 3}
        ranked = sorted(range(len(items)),
                        key=lambda i: order.get(
                            "error" if items[i].get("error") else items[i]["result"]["overall"], 9))
        rows = []
        for i in ranked:
            item = items[i]
            if item.get("error"):
                rows.append({"Product": item["label"], "Files": ", ".join(item["files"]),
                             "Result": "Error", "Gov. warning": "—",
                             "Application data": "matched" if item.get("matched") else "—",
                             "Flagged fields": _ERROR_SHORT.get(item.get("error_kind"),
                                                                "could not read image")})
            else:
                res = item["result"]
                flags = [f.field.replace("_", " ") for f in res["fields"] if f.status != PASS]
                warn = next((f.status for f in res["fields"]
                             if f.field == "government_warning"), None)
                rows.append({"Product": item["label"], "Files": ", ".join(item["files"]),
                             "Result": _LABEL.get(res["overall"], res["overall"]),
                             "Gov. warning": _LABEL.get(warn, "—"),
                             "Application data": "matched" if item.get("matched") else "—",
                             "Flagged fields": ", ".join(flags) if flags else "—"})
        st.dataframe(rows, width="stretch", hide_index=True)

        # The detail section is a fragment with pagination: expander BODIES execute on every
        # rerun regardless of expanded state (review finding), so rendering all N field
        # tables + JSON dumps + images scaled badly with batch size. Pagination bounds the
        # per-run work; the fragment keeps page flips from rerunning the whole app.
        @st.fragment
        def _batch_detail():
            st.markdown('<div class="lv-section">Per-product detail</div>',
                        unsafe_allow_html=True)
            page_size = 10
            n_pages = (len(ranked) + page_size - 1) // page_size
            page = 1
            if n_pages > 1:
                page = st.selectbox("Detail page", list(range(1, n_pages + 1)),
                                    format_func=lambda p: f"Page {p} of {n_pages} "
                                                          f"({page_size} products per page)",
                                    key="batch_detail_page")
            for i in ranked[(page - 1) * page_size: page * page_size]:
                item = items[i]
                status = "Error" if item.get("error") else _LABEL.get(item["result"]["overall"])
                with st.expander(f"{item['label']}  —  {status}  ({', '.join(item['files'])})"):
                    if item.get("error"):
                        st.error(_ERROR_MESSAGES.get(item.get("error_kind"), _FRIENDLY_ERROR))
                    else:
                        _render_product(item["result"], item.get("extracted"),
                                        images=item.get("images"),
                                        elapsed=item.get("seconds"))

        _batch_detail()
