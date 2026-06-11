import type { AdditionalStatement, Extraction } from "@/lib/types";

/** "yes" / "no" / "not determinable". */
function yn(value: boolean | null | undefined): string {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "not determinable";
}

function Kv({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-1 text-sm sm:flex-row sm:gap-3">
      <span className="shrink-0 text-slate-500 sm:w-56">{k}</span>
      <span className="text-slate-800">{children}</span>
    </div>
  );
}

function Expander({ summary, children }: { summary: string; children: React.ReactNode }) {
  return (
    <details className="group rounded-xl border border-slate-200 bg-white">
      <summary className="cursor-pointer select-none rounded-xl px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50 group-open:border-b group-open:border-slate-100">
        {summary}
      </summary>
      <div className="px-4 py-3">{children}</div>
    </details>
  );
}

/** The model's government-warning OBSERVATIONS (evidence, not judgment). */
export function WarningEvidence({ extracted }: { extracted: Extraction }) {
  const gw = extracted.government_warning;
  return (
    <Expander summary="Government warning — what was seen on the label">
      <Kv k="Warning found on label">{yn(gw.present)}</Kv>
      <Kv k="Header in capital letters">{yn(gw.header_all_caps)}</Kv>
      <Kv k="Header printed in bold">
        {yn(gw.header_bold)} ({gw.header_bold_confidence || "—"} confidence)
      </Kv>
      <Kv k="Body text printed in bold">
        {yn(gw.body_bold)} ({gw.body_bold_confidence || "—"} confidence)
      </Kv>
      <Kv k="Basis for the bold reading">{gw.header_bold_basis || "—"}</Kv>
      {gw.text && (
        <div className="mt-2">
          <p className="mb-1 text-sm text-slate-500">Transcribed warning text</p>
          <pre className="whitespace-pre-wrap break-words rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-800">
            {gw.text}
          </pre>
        </div>
      )}
    </Expander>
  );
}

/** Evidence-only extraction fields + additional statements: shown for the
 *  reviewer, never auto-checked. */
const EVIDENCE_FIELDS = [
  "fanciful_name",
  "statement_of_composition",
  "sulfite_declaration",
] as const;

export function OtherLabelDetails({
  extracted,
  additionalStatements,
  imageQualityNotes,
}: {
  extracted: Extraction;
  additionalStatements: AdditionalStatement[];
  imageQualityNotes: string | null;
}) {
  const rows: { k: string; v: string }[] = [];
  for (const name of EVIDENCE_FIELDS) {
    const obj = extracted[name];
    if (obj.present || obj.value) {
      rows.push({
        k: name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()),
        v: `${obj.value || "(present, unreadable)"} (${obj.confidence || "—"} confidence)`,
      });
    }
  }
  for (const s of additionalStatements) {
    rows.push({ k: s.kind ? `Other statement [${s.kind}]` : "Other statement", v: s.value });
  }
  if (imageQualityNotes) {
    rows.push({ k: "Image quality note", v: imageQualityNotes });
  }
  return (
    <Expander summary="Other label details (for reference — not auto-checked)">
      {rows.length > 0 ? (
        rows.map((r, i) => (
          <Kv key={i} k={r.k}>
            {r.v}
          </Kv>
        ))
      ) : (
        <p className="text-sm text-slate-500">
          No evidence-only fields or additional statements were found on this label.
        </p>
      )}
    </Expander>
  );
}

/** Full technical readout of the raw extraction. */
export function JsonReadout({ extracted }: { extracted: Extraction }) {
  return (
    <Expander summary="Full technical readout (JSON)">
      <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-800">
        {JSON.stringify(extracted, null, 2)}
      </pre>
    </Expander>
  );
}
