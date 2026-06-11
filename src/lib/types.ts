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

/** Serialized verification.FieldResult. */
export interface FieldVerdict {
  field: string;
  status: Status;
  reason: string;
  extracted: string;
  expected: string;
  /** Machine-readable verdict cause (government warning only):
   *  "absence" | "wording" | "caps" | "bold" | "low_confidence". */
  cause: string | null;
}

export interface AdditionalStatement {
  value: string;
  kind: string | null;
  confidence: Confidence;
}

export interface VerifyResponse {
  mode: VerifyMode;
  overall: Status;
  beverage_type: string;
  fields: FieldVerdict[];
  additional_statements: AdditionalStatement[];
  image_quality_notes: string | null;
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
