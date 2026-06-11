"""Pydantic request/response models for the verification API.

These mirror two existing contracts and add nothing to them:
  - ApplicationData mirrors the application fields verification.verify() reads
    (the same keys the Streamlit form and the batch application files use).
  - VerifyResponse mirrors verify()/verify_label_only()'s return shape, with
    FieldResult dataclasses serialized verbatim.

The TypeScript types in src/lib/types.ts are a 1:1 mirror of the response
models; change them together.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

Status = Literal["pass", "needs_review", "fail"]
Confidence = Literal["high", "medium", "low"]


class ApplicationData(BaseModel):
    """The applicant-submitted values a label is matched against.

    All fields optional: a missing/blank form means rules-only screening.
    extra="forbid" so a drifted frontend payload fails loudly at the boundary
    instead of being silently ignored.
    """
    model_config = ConfigDict(extra="forbid")

    brand_name: Optional[str] = None
    fanciful_name: Optional[str] = None
    class_type: Optional[str] = None
    statement_of_composition: Optional[str] = None
    alcohol_content: Optional[str] = None
    net_contents: Optional[str] = None
    name_and_address: Optional[str] = None
    country_of_origin: Optional[str] = None

    def cleaned(self) -> dict:
        """Trimmed, non-empty values only — a blank form must stay an empty dict
        so the API preserves the engine's rules-only screening path (the
        application is an independent witness; it is never auto-populated)."""
        out = {}
        for key, value in self.model_dump().items():
            if isinstance(value, str) and value.strip():
                out[key] = value.strip()
        return out


class AdditionalStatement(BaseModel):
    """One transcribed conditional-disclosure statement (no pass/fail logic —
    surfaced to the reviewer, mirroring the engine's deliberate scope)."""
    value: str
    kind: Optional[str] = None
    confidence: Confidence = "low"

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, v):
        """Under the engine's json_object fallback (models without Structured
        Outputs) the model may emit a non-string kind, which extraction._coerce
        passes through raw — coerce instead of crashing response serialization."""
        if v is None or isinstance(v, str):
            return v
        return str(v)


class ExtractedField(BaseModel):
    """One extracted scalar field: {present, value, confidence}. extraction._coerce
    guarantees this shape; extras are ignored (not forbidden) so a future engine
    field doesn't break response serialization."""
    present: bool
    value: Optional[str] = None
    confidence: Confidence = "low"


class AlcoholContentField(ExtractedField):
    abv_percent: Optional[float] = None
    proof: Optional[float] = None


class GovernmentWarningField(BaseModel):
    """The model's warning OBSERVATIONS (evidence, not judgment) — surfaced so the
    reviewer can see what the bold/caps verdict was based on."""
    present: bool
    text: Optional[str] = None
    header_all_caps: Optional[bool] = None
    header_bold: Optional[bool] = None
    header_bold_confidence: Confidence = "low"
    header_bold_basis: Optional[str] = None
    body_bold: Optional[bool] = None
    body_bold_confidence: Confidence = "low"
    confidence: Confidence = "low"


class Extraction(BaseModel):
    """The engine's coerced extraction schema (see extraction._EXTRACTION_SCHEMA).
    Returned alongside the verdicts so the UI can show the model's raw read —
    evidence-only fields, warning observations, and the full JSON readout —
    exactly as the Streamlit prototype does."""
    beverage_type: str
    brand_name: ExtractedField
    fanciful_name: ExtractedField
    class_type: ExtractedField
    statement_of_composition: ExtractedField
    net_contents: ExtractedField
    name_and_address: ExtractedField
    country_of_origin: ExtractedField
    appellation: ExtractedField
    vintage: ExtractedField
    sulfite_declaration: ExtractedField
    alcohol_content: AlcoholContentField
    government_warning: GovernmentWarningField
    additional_statements: list[AdditionalStatement]
    image_quality_notes: Optional[str] = None


class FieldVerdict(BaseModel):
    """Serialized verification.FieldResult.

    `cause` values "absence"/"wording"/"caps"/"bold" come from the
    government-warning check; "low_confidence" may appear on ANY field whose
    passing read was downgraded to needs_review (verification._escalate)."""
    field: str
    status: Status
    reason: str
    extracted: str
    expected: str
    cause: Optional[str] = None


class VerifyResponse(BaseModel):
    mode: Literal["application_match", "rules_only"]
    overall: Status
    beverage_type: str
    fields: list[FieldVerdict]
    additional_statements: list[AdditionalStatement]
    image_quality_notes: Optional[str] = None
    extracted: Extraction


class ErrorBody(BaseModel):
    """Machine-readable error kind + human-readable message. `kind` extends
    extraction.failure_kind()'s vocabulary with the API-side validation kinds."""
    kind: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
