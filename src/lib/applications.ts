/**
 * Application-data file parsing — a 1:1 port of app.py's `_parse_applications`
 * / `_app_row_for` / `_pick_application_row`. The file pairs each product (by
 * its filename stem) with the applicant-submitted values; parsing failures
 * never block a batch — products just screen rules-only.
 */

/** The application columns the verifier compares against (matches app.py's
 *  `_APP_FIELDS`; the batch file does not carry the two union-only fields). */
export const APP_FIELDS = [
  "brand_name",
  "class_type",
  "alcohol_content",
  "net_contents",
  "name_and_address",
  "country_of_origin",
] as const;

export type AppRow = Record<(typeof APP_FIELDS)[number], string>;

export interface ParsedApplications {
  /** product stem (lowercased) -> row of trimmed values */
  mapping: Record<string, AppRow>;
  error: string | null;
  warnings: string[];
}

/** Normalize a column name: 'Brand Name' / 'brand-name' -> 'brand_name'. */
export function normHeader(key: string): string {
  return String(key).trim().toLowerCase().replace(/[\s\-]+/g, "_");
}

/** Python-truthiness string coercion, mirroring app.py's `str(x or "")`:
 *  0, false, null, "", and empty arrays/objects all collapse to "" (a JSON
 *  application value of 0/false must not become the string "0"/"false" here
 *  when the Python app would treat it as blank). */
function pyStr(v: unknown): string {
  if (!v) return "";
  if (Array.isArray(v) && v.length === 0) return "";
  if (typeof v === "object" && !Array.isArray(v) && Object.keys(v as object).length === 0) return "";
  return String(v);
}

/** Own-property read — a product stem like "constructor" or "toString" must
 *  not resolve through Object.prototype the way a bare `mapping[key]` would. */
export function ownRow(
  mapping: Record<string, AppRow>,
  key: string,
): AppRow | undefined {
  return Object.hasOwn(mapping, key) ? mapping[key] : undefined;
}

/** Minimal RFC-4180 CSV parser (quoted fields, "" escapes, CRLF/LF). No
 *  dependency: the format here is a simple header + rows file. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      cell += ch;
      i += 1;
      continue;
    }
    if (ch === '"' && cell === "") {
      // a quote only starts quoted mode at the BEGINNING of a cell — Python's
      // csv keeps a mid-field quote (inch marks, nicknames) literally
      inQuotes = true;
      i += 1;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
      i += 1;
    } else if (ch === "\n" || ch === "\r") {
      row.push(cell);
      cell = "";
      rows.push(row);
      row = [];
      i += ch === "\r" && text[i + 1] === "\n" ? 2 : 1;
    } else {
      cell += ch;
      i += 1;
    }
  }
  if (cell !== "" || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  // mirror csv.DictReader: a truly blank line yields no row (our scanner emits
  // it as a single empty cell), but a line of empty cells like "," is kept —
  // it is dropped later by the blank-product rule, same as Python's csv.
  return rows.filter((r) => !(r.length === 1 && r[0] === ""));
}

function csvRows(text: string): Record<string, string>[] {
  const grid = parseCsv(text);
  if (grid.length === 0) return [];
  const header = grid[0];
  return grid.slice(1).map((cells) => {
    const row: Record<string, string> = {};
    header.forEach((h, idx) => {
      row[h] = cells[idx] ?? ""; // extra cells beyond the header are dropped
    });
    return row;
  });
}

function jsonRows(text: string): Record<string, unknown>[] | string {
  let data: unknown = JSON.parse(text);
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const products = (data as Record<string, unknown>).products;
    if (Array.isArray(products)) data = products;
  }
  if (data && typeof data === "object" && !Array.isArray(data)) {
    // the mapping key IS the product and must beat any inner product-ish field:
    // normalize the inner keys FIRST, then assert the key last (mirrors app.py)
    return Object.entries(data as Record<string, unknown>)
      .filter(([, v]) => v && typeof v === "object" && !Array.isArray(v))
      .map(([k, v]) => ({
        ...Object.fromEntries(
          Object.entries(v as Record<string, unknown>).map(([kk, vv]) => [normHeader(kk), vv]),
        ),
        product: k,
      }));
  }
  if (Array.isArray(data)) {
    return data.filter(
      (r): r is Record<string, unknown> => !!r && typeof r === "object" && !Array.isArray(r),
    );
  }
  return "JSON must be a mapping of product -> fields, or a list of objects";
}

/** Parse an application-data file (CSV or JSON) into {product -> row}.
 *  Duplicate product keys are last-row-wins but REPORTED; a bad file never
 *  blocks the batch — it falls back to rules-only screening. */
export function parseApplications(rawText: string, filename: string): ParsedApplications {
  try {
    const text = rawText.replace(/^\uFEFF/, ""); // tolerate a UTF-8 BOM
    // File.text() never throws on binary input \u2014 invalid bytes become U+FFFD \u2014
    // so detect the .xlsx-renamed-to-.csv mistake here (app.py's
    // UnicodeDecodeError branch) instead of failing with a misleading
    // "no rows" error.
    if (text.includes("\uFFFD") || text.includes("\u0000")) {
      return { mapping: {}, error: "the file isn't plain text \u2014 save it as a regular CSV or JSON file", warnings: [] };
    }
    let rows: Record<string, unknown>[];
    if (filename.toLowerCase().endsWith(".json")) {
      const parsed = jsonRows(text);
      if (typeof parsed === "string") return { mapping: {}, error: parsed, warnings: [] };
      rows = parsed;
    } else {
      rows = csvRows(text);
    }
    // normalize header/key spelling so 'Brand Name' / 'brand-name' both work
    rows = rows.map((row) =>
      Object.fromEntries(Object.entries(row).map(([k, v]) => [normHeader(k), v])),
    );

    const mapping: Record<string, AppRow> = {};
    const dups: string[] = [];
    for (const row of rows) {
      const prod = pyStr(row.product).trim();
      if (!prod) continue;
      const key = prod.toLowerCase();
      if (Object.hasOwn(mapping, key)) dups.push(prod);
      mapping[key] = Object.fromEntries(
        APP_FIELDS.map((f) => [f, pyStr(row[f]).trim()]),
      ) as AppRow;
    }
    if (Object.keys(mapping).length === 0) {
      return { mapping: {}, error: "no rows with a 'product' value were found", warnings: [] };
    }
    const warnings: string[] = [];
    if (!rows.some((row) => APP_FIELDS.some((f) => f in row))) {
      warnings.push(
        "no recognized application-field columns were found (expected any of: " +
          APP_FIELDS.join(", ") +
          ") — check the header row; products will effectively be screened against the " +
          "fixed rules only",
      );
    }
    if (dups.length > 0) {
      warnings.push(
        "duplicate product row(s): " +
          [...new Set(dups)].sort().join(", ") +
          " — the last row wins; check the file",
      );
    }
    return { mapping, error: null, warnings };
  } catch (exc) {
    if (exc instanceof SyntaxError) {
      return { mapping: {}, error: "the JSON file could not be read — it isn't valid JSON", warnings: [] };
    }
    return { mapping: {}, error: String(exc).slice(0, 160), warnings: [] };
  }
}

/** The product's application row, or null when there is no row OR the row is
 *  entirely blank — a blank row must screen rules-only, exactly like a blank
 *  form in single mode. */
export function appRowFor(
  mapping: Record<string, AppRow>,
  label: string,
): AppRow | null {
  const row = ownRow(mapping, label.toLowerCase());
  if (row && Object.values(row).some((v) => v.trim())) return row;
  return null;
}

/** Choose the application row for SINGLE mode: the row matching the uploaded
 *  image's product stem; else, if the file holds exactly one row, that row.
 *  Ambiguity returns null rather than silently prefilling the wrong product. */
export function pickApplicationRow(
  mapping: Record<string, AppRow>,
  imageStem: string | null,
): { row: AppRow | null; message: string } {
  const matched = imageStem ? ownRow(mapping, imageStem.toLowerCase()) : undefined;
  if (matched) {
    return { row: matched, message: `matched product '${imageStem}'` };
  }
  const keys = Object.keys(mapping);
  if (keys.length === 1) {
    return { row: mapping[keys[0]], message: `single row ('${keys[0]}') used` };
  }
  const why = imageStem ? `no row matches product '${imageStem}'` : "no image uploaded to match";
  return { row: null, message: `${why} — fields left for manual entry` };
}
