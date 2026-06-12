/**
 * Application-data file parsing — the row-matching semantics are a 1:1 port of
 * the retired Streamlit prototype's `_parse_applications` / `_app_row_for` /
 * `_pick_application_row` (dev-archive branch). The file pairs each product
 * (by its filename stem) with the applicant-submitted values; parsing failures
 * never block a batch — products just screen rules-only.
 *
 * Input format is Excel only (`.xlsx`/`.xls`/`.xlsm`, read via SheetJS): the
 * first sheet with a `product` column is the data sheet, so the downloadable
 * template can carry an instructions sheet alongside it. The earlier CSV/JSON
 * formats were retired in favor of one self-documenting template.
 */

/** The application columns the verifier compares against (the batch file does
 *  not carry the two union-only fields). */
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

/** Extensions the application-file inputs accept (Excel workbooks only). */
export const EXCEL_EXTENSIONS = [".xlsx", ".xls", ".xlsm"] as const;
export const EXCEL_ACCEPT = EXCEL_EXTENSIONS.join(",");

/** A workbook is a small table of typed values; anything bigger is a mistake
 *  (and a multi-MB parse would freeze the tab before erroring). */
const MAX_APP_FILE_BYTES = 10 * 1024 * 1024;

export const TEMPLATE_FILENAME = "ttb-application-template.xlsx";

/** Normalize a column name: 'Brand Name' / 'brand-name' -> 'brand_name'. */
export function normHeader(key: string): string {
  return String(key).trim().toLowerCase().replace(/[\s\-]+/g, "_");
}

/** Python-truthiness string coercion, mirroring the prototype's `str(x or "")`:
 *  0, false, null, "", and empty arrays/objects all collapse to "" (a value of
 *  0/false must not become the string "0"/"false" here when the Python app
 *  would treat it as blank). */
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

/** SheetJS ships as CJS; under webpack the namespace carries the exports while
 *  under plain Node (the test runner) they may only be on `.default`. */
async function loadXlsx(): Promise<typeof import("xlsx")> {
  const mod: unknown = await import("xlsx");
  const ns = mod as { default?: unknown; read?: unknown };
  return (ns.read ? ns : ns.default) as typeof import("xlsx");
}

/** Header row + cell grid -> one object per data row (extra cells beyond the
 *  header are dropped; on a duplicate header the rightmost column wins). */
function gridRows(grid: unknown[][]): Record<string, unknown>[] {
  const header = (grid[0] ?? []).map((h) => String(h));
  return grid.slice(1).map((cells) => {
    const row: Record<string, unknown> = {};
    header.forEach((h, idx) => {
      row[h] = cells[idx] ?? "";
    });
    return row;
  });
}

/** Shared tail of every parse path: normalize header spelling, key rows by
 *  lowercased product, report duplicates and missing field columns. */
function rowsToMapping(rawRows: Record<string, unknown>[]): ParsedApplications {
  // normalize header/key spelling so 'Brand Name' / 'brand-name' both work
  const rows = rawRows.map((row) =>
    Object.fromEntries(Object.entries(row).map(([k, v]) => [normHeader(k), v])),
  );

  // Object.create(null): a product literally named "__proto__" (reachable
  // from a file like __proto___front.jpg) would otherwise hit the prototype
  // setter — silently dropping the row and polluting the object. A null-proto
  // map stores it as a plain own key and makes every `in`/bracket read safe.
  const mapping: Record<string, AppRow> = Object.create(null);
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
}

/** Parse an application-data Excel workbook into {product -> row}.
 *  Reads the first sheet whose header row has a 'product' column (so the
 *  template's instructions sheet is skipped). Duplicate product keys are
 *  last-row-wins but REPORTED; a bad file never blocks the batch — it falls
 *  back to rules-only screening. */
export async function parseApplicationsFile(file: File): Promise<ParsedApplications> {
  const name = file.name.toLowerCase();
  if (!EXCEL_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return {
      mapping: {},
      error:
        "only Excel files are accepted (.xlsx) — open your data in Excel and save it " +
        "as .xlsx, or start from the downloadable template",
      warnings: [],
    };
  }
  if (file.size > MAX_APP_FILE_BYTES) {
    return {
      mapping: {},
      error: "the file is over 10 MB — the application list should be a small spreadsheet",
      warnings: [],
    };
  }
  // the chunk load is NOT the user's fault — a failed dynamic import (offline,
  // deploy skew) must never be reported as a corrupt file
  let XLSX: Awaited<ReturnType<typeof loadXlsx>>;
  try {
    XLSX = await loadXlsx();
  } catch {
    return {
      mapping: {},
      error:
        "the spreadsheet reader could not be loaded — check your connection, reload " +
        "the page, and try again",
      warnings: [],
    };
  }
  let grids: { sheet: string; grid: unknown[][] }[];
  try {
    const workbook = XLSX.read(new Uint8Array(await file.arrayBuffer()), { type: "array" });
    grids = workbook.SheetNames.map((sheet) => ({
      sheet,
      // raw:false reads the FORMATTED cell text (what the user sees in Excel) —
      // a percent-formatted ABV cell must arrive as "45%", not "0.45"
      grid: XLSX.utils.sheet_to_json<unknown[]>(workbook.Sheets[sheet], {
        header: 1,
        raw: false,
        defval: "",
        blankrows: false,
      }),
    }));
  } catch {
    return {
      mapping: {},
      error: "the file could not be read as an Excel workbook — re-save it as .xlsx in Excel",
      warnings: [],
    };
  }
  const withProduct = grids.filter(({ grid }) =>
    (grid[0] ?? []).some((h) => normHeader(String(h)) === "product"),
  );
  if (withProduct.length === 0) {
    return {
      mapping: {},
      error:
        "no sheet has a 'product' column — see the downloadable template for the " +
        "expected layout",
      warnings: [],
    };
  }
  // Prefer the first product-bearing sheet that actually yields rows — a
  // header-only decoy (a cleared/duplicated template sheet) must not shadow
  // the real data sheet and turn into a misleading "no rows" error.
  let chosen = withProduct[0].sheet;
  let parsed = rowsToMapping(gridRows(withProduct[0].grid));
  for (const cand of withProduct.slice(1)) {
    if (parsed.error === null) break;
    const next = rowsToMapping(gridRows(cand.grid));
    if (next.error === null) {
      chosen = cand.sheet;
      parsed = next;
    }
  }
  if (parsed.error === null && withProduct.length > 1) {
    parsed.warnings.push(
      `more than one sheet has a 'product' column — only '${chosen}' was read`,
    );
  }
  return parsed;
}

/** The downloadable template: an example data sheet plus an instructions sheet
 *  (parseApplicationsFile keys on the 'product' column, so the extra sheet is
 *  ignored on the way back in). Returns the .xlsx file bytes. */
export async function buildApplicationsTemplate(): Promise<ArrayBuffer> {
  const XLSX = await loadXlsx();
  const dataSheet = XLSX.utils.aoa_to_sheet([
    ["product", ...APP_FIELDS],
    [
      "oldtom",
      "OLD TOM RESERVE",
      "Kentucky Straight Bourbon Whiskey",
      "45% Alc./Vol. (90 Proof)",
      "750 mL",
      "Bottled by Old Tom Distilling Co., Louisville, KY",
      "",
    ],
    [
      "riverbend",
      "RIVERBEND CELLARS",
      "Red Wine",
      "13% Alc. by Vol.",
      "750 mL",
      "Imported by Riverbend Imports LLC, New York, NY",
      "Product of France",
    ],
  ]);
  dataSheet["!cols"] = [12, 22, 32, 24, 12, 46, 18].map((wch) => ({ wch }));
  const helpSheet = XLSX.utils.aoa_to_sheet([
    ["How the verifier uses this file"],
    [""],
    ["• One row per product on the 'Applications' sheet. Only 'product' is required;"],
    ["  leave any other cell blank to skip checking that field."],
    [""],
    ["• The 'product' value must match the product's image file names: take the file"],
    ["  name, drop the extension, and drop a trailing _front / _back / _label / _other"],
    ["  marker. Capitalization doesn't matter."],
    [""],
    ["• Front + back photos: oldtom_front.jpg and oldtom_back.jpg are read together"],
    ["  as one product — its row is product = oldtom."],
    [""],
    ["• A single combined (stitched front-and-back) image works the same way: a file"],
    ["  named oldtom.jpg or oldtom_label.jpg also matches product = oldtom."],
    [""],
    ["• Single-label mode: the row matching the front image is prefilled into the"],
    ["  form; if the file has exactly one row, that row is used automatically."],
    [""],
    ["• Extra columns are ignored. The first sheet with a 'product' column is read,"],
    ["  so this instructions sheet is skipped."],
  ]);
  helpSheet["!cols"] = [{ wch: 86 }];
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, dataSheet, "Applications");
  XLSX.utils.book_append_sheet(workbook, helpSheet, "How to fill this in");
  return XLSX.write(workbook, { type: "array", bookType: "xlsx" }) as ArrayBuffer;
}

/** Browser-only: build the template and hand it to the user as a download. */
export async function downloadApplicationsTemplate(): Promise<void> {
  const bytes = await buildApplicationsTemplate();
  const url = URL.createObjectURL(
    new Blob([bytes], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }),
  );
  try {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = TEMPLATE_FILENAME;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
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
