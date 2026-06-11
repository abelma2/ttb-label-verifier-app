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

from pydantic import BaseModel, ConfigDict

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


class FieldVerdict(BaseModel):
    """Serialized verification.FieldResult."""
    field: str
    status: Status
    reason: str
    extracted: str
    expected: str
    cause: Optional[str] = None


class AdditionalStatement(BaseModel):
    """One transcribed conditional-disclosure statement (no pass/fail logic —
    surfaced to the reviewer, mirroring the engine's deliberate scope)."""
    value: str
    kind: Optional[str] = None
    confidence: Confidence = "low"


class VerifyResponse(BaseModel):
    mode: Literal["application_match", "rules_only"]
    overall: Status
    beverage_type: str
    fields: list[FieldVerdict]
    additional_statements: list[AdditionalStatement]
    image_quality_notes: Optional[str] = None


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
