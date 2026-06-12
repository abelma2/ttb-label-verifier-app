/**
 * 1:1 mirror of the Pydantic models in api/_models.py — change them together.
 * The response shape is, in turn, the engine's verify()/verify_label_only()
 * contract with FieldResult dataclasses serialized verbatim.
 */

export type Status = "pass" | "needs_review" | "fail";
export type Confidence = "high" | "medium" | "low";
export type VerifyMode = "application_match" | "rules_only";

/** Applicant-submitted values; all optional. Blank/absent => rules-only screening. */
export interface ApplicationData {
  brand_name?: string;
  fanciful_name?: string;
  class_type?: string;
  statement_of_composition?: string;
  alcohol_content?: string;
  net_contents?: string;
  name_and_address?: string;
  country_of_origin?: string;
}

export const APPLICATION_KEYS = [
  "brand_name",
  "fanciful_name",
  "class_type",
  "statement_of_composition",
  "alcohol_content",
  "net_contents",
  "name_and_address",
  "country_of_origin",
] as const satisfies readonly (keyof ApplicationData)[];

/** Machine-readable verdict cause. "absence"/"wording"/"caps"/"bold" are
 *  emitted by the government-warning check; "low_confidence" may appear on ANY
 *  field whose passing read was downgraded to needs_review. */
export type VerdictCause = "absence" | "wording" | "caps" | "bold" | "low_confidence";

/** Serialized verification.FieldResult. */
export interface FieldVerdict {
  field: string;
  status: Status;
  reason: string;
  extracted: string;
  expected: string;
  cause: VerdictCause | null;
}

export interface AdditionalStatement {
  value: string;
  kind: string | null;
  confidence: Confidence;
}

/** One extracted scalar field, per the engine's coerced schema. */
export interface ExtractedField {
  present: boolean;
  value: string | null;
  confidence: Confidence;
}

export interface AlcoholContentField extends ExtractedField {
  abv_percent: number | null;
  proof: number | null;
}

/** The model's warning OBSERVATIONS (evidence, not judgment). When the warning-only
 *  supplement reader ran (WARNING_SUPPLEMENT_MODEL), present/text/caps/bold carry ITS
 *  read and the main model's original read sits in the main_* fields;
 *  warning_observer says which reader is in effect ("supplement" | "main-fallback",
 *  absent when the supplement is disabled). */
export interface GovernmentWarningField {
  present: boolean;
  text: string | null;
  header_all_caps: boolean | null;
  header_bold: boolean | null;
  header_bold_confidence: Confidence;
  header_bold_basis: string | null;
  body_bold: boolean | null;
  body_bold_confidence: Confidence;
  confidence: Confidence;
  warning_observer?: string | null;
  main_present?: boolean | null;
  main_text?: string | null;
  main_header_all_caps?: boolean | null;
  main_header_bold?: boolean | null;
  main_header_bold_confidence?: Confidence | null;
  main_header_bold_basis?: string | null;
  main_body_bold?: boolean | null;
  main_body_bold_confidence?: Confidence | null;
}

/** The engine's coerced extraction (api/_models.py Extraction). */
export interface Extraction {
  beverage_type: string;
  brand_name: ExtractedField;
  fanciful_name: ExtractedField;
  class_type: ExtractedField;
  statement_of_composition: ExtractedField;
  net_contents: ExtractedField;
  name_and_address: ExtractedField;
  country_of_origin: ExtractedField;
  appellation: ExtractedField;
  vintage: ExtractedField;
  sulfite_declaration: ExtractedField;
  alcohol_content: AlcoholContentField;
  government_warning: GovernmentWarningField;
  additional_statements: AdditionalStatement[];
  image_quality_notes: string | null;
}

export interface VerifyResponse {
  mode: VerifyMode;
  overall: Status;
  beverage_type: string;
  fields: FieldVerdict[];
  additional_statements: AdditionalStatement[];
  image_quality_notes: string | null;
  extracted: Extraction;
}

export interface ErrorResponse {
  error: { kind: string; message: string };
}

/** Human-readable labels for engine field names. */
export const FIELD_LABELS: Record<string, string> = {
  brand_name: "Brand name",
  class_type: "Class / type designation",
  alcohol_content: "Alcohol content",
  net_contents: "Net contents",
  name_and_address: "Name & address",
  country_of_origin: "Country of origin",
  government_warning: "Government warning",
  appellation: "Appellation of origin",
};

export function fieldLabel(field: string): string {
  return FIELD_LABELS[field] ?? field.replace(/_/g, " ");
}
