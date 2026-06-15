"""Personality injection types and contributor (Pods 3.3a + 3.3b).

Provides the frozen value types that describe a personality profile and the
``PersonalityContributor`` that slots rendered profile text into the
``personality`` context slot.

The profile is rendered ONCE at construction time (the profile is frozen and
``render_profile`` is a pure function), so the contributor is cheap to call and
consistent across every step in a drive (S6/S9 guarantee).

Dependency direction (CANON D3): runtime -> context -> injection -> recall.
This module does NOT import ``cogworx.runtime``.

CANON sections cited:
  - S1  -- model work off the write path; this contributor never calls the model.
  - S4  -- model-agnostic; no provider reference anywhere.
  - S8  -- graceful degradation; empty profile -> status="empty", not an exception.
  - S9  -- structure over prompting; rendered text derives only from declared fields.
  - S11 -- cost bounded structurally; the assembler owns drop-whole atomicity for this
          slot -- the contributor does NOT truncate (see PersonalityContributor.contribute).

Contract changelog:
  - 2026-06-13 (Pod 3.3a): initial surface -- PersonalityAttribute, PersonalityProfile,
    render_profile.
  - 2026-06-13 (Pod 3.3b): PersonalityContributor.
  - 2026-06-13 (Pod 3.3 red-team F1/F2/F3): extended newline guard to all string fields.
    _NEWLINE_CHARS now covers newline, CR, VT, FF, U+2028 LINE SEP, U+2029 PARA SEP.
    _strip_optional and _strip_tuple_items now reject embedded newlines (previously only
    PersonalityAttribute was guarded -- injection through name/tone/preamble/style/traits
    was possible).
  - 2026-06-13 (Pod 3.4a): extracted _NEWLINE_CHARS + _has_newline to cogworx.context._text.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from cogworx.context._text import _has_newline
from cogworx.context.types import (
    ContextRequest,
    SlotAllocation,
    SlotChunk,
    SlotContent,
)

__all__ = [
    "PersonalityAttribute",
    "PersonalityContributor",
    "PersonalityProfile",
    "render_profile",
]

# ---------------------------------------------------------------------------
# PersonalityAttribute
# ---------------------------------------------------------------------------


class PersonalityAttribute(BaseModel):
    """A single key/value metadata pair on a personality profile.

    Both ``key`` and ``value`` are stripped of leading/trailing whitespace,
    must be non-empty after stripping, and must not contain embedded newlines.

    Args:
        key:   Attribute name (e.g. ``"language"``).
        value: Attribute content (e.g. ``"formal English"``).
    """

    model_config = ConfigDict(frozen=True)

    key: str
    value: str

    @field_validator("key", "value", mode="before")
    @classmethod
    def _strip_and_check(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("must be non-empty after stripping whitespace")
        if _has_newline(stripped):
            raise ValueError("must not contain embedded newlines or line-separator characters")
        return stripped


# ---------------------------------------------------------------------------
# PersonalityProfile
# ---------------------------------------------------------------------------


class PersonalityProfile(BaseModel):
    """Declarative description of an agent's personality.

    All string fields are optional; the profile is considered empty when every
    field is ``None`` or an empty tuple (see :meth:`is_empty`).  Fields that
    accept a string are stripped and validated non-empty on assignment.

    Args:
        name:       The agent's name (e.g. ``"Aria"``).
        role:       The agent's role (e.g. ``"a helpful assistant"``).
        tone:       Tone descriptor (e.g. ``"concise and direct"``).
        style:      Ordered style directives (e.g. ``("Use bullet points.",)``).
        traits:     Behavioural traits (e.g. ``("Always cite sources.",)``).
        attributes: Arbitrary key/value metadata pairs.
        preamble:   Verbatim opening text prepended after all other sections.
    """

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    role: str | None = None
    tone: str | None = None
    style: tuple[str, ...] = ()
    traits: tuple[str, ...] = ()
    attributes: tuple[PersonalityAttribute, ...] = ()
    preamble: str | None = None

    @field_validator("name", "role", "tone", "preamble", mode="before")
    @classmethod
    def _strip_optional(cls, v: object) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("must be a string or None")
        stripped = v.strip()
        if not stripped:
            raise ValueError("must be non-empty after stripping whitespace")
        if _has_newline(stripped):
            raise ValueError("must not contain embedded newlines or line-separator characters")
        return stripped

    @field_validator("style", "traits", mode="before")
    @classmethod
    def _strip_tuple_items(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, (list, tuple)):
            result: list[str] = []
            for item in v:
                if not isinstance(item, str):
                    raise ValueError("each item must be a string")
                stripped = item.strip()
                if not stripped:
                    raise ValueError("each item must be non-empty after stripping whitespace")
                if _has_newline(stripped):
                    raise ValueError(
                        "each item must not contain embedded newlines or line-separator characters"
                    )
                result.append(stripped)
            return tuple(result)
        raise ValueError("must be a list or tuple")

    def is_empty(self) -> bool:
        """Return True iff every field is None or an empty tuple."""
        return (
            self.name is None
            and self.role is None
            and self.tone is None
            and not self.style
            and not self.traits
            and not self.attributes
            and self.preamble is None
        )


# ---------------------------------------------------------------------------
# render_profile
# ---------------------------------------------------------------------------


def render_profile(profile: PersonalityProfile) -> str:
    """Render a ``PersonalityProfile`` to a canonical prompt-ready string.

    Sections are joined by ``"\\n\\n"``; empty sections are skipped entirely.
    Returns ``""`` for an empty profile.

    Section order:

    1. Identity line -- name, role, or both.
    2. Tone.
    3. Style block (bullet list).
    4. Behaviors block (bullet list).
    5. Additional block (key: value pairs).
    6. Preamble (verbatim).

    Args:
        profile: The frozen profile to render.

    Returns:
        A multi-section string with no trailing newline, or ``""`` when empty.
    """
    if profile.is_empty():
        return ""

    sections: list[str] = []

    # 1. Identity
    if profile.name is not None and profile.role is not None:
        sections.append(f"You are {profile.name}, {profile.role}.")
    elif profile.name is not None:
        sections.append(f"You are {profile.name}.")
    elif profile.role is not None:
        sections.append(f"You are {profile.role}.")

    # 2. Tone
    if profile.tone is not None:
        sections.append(f"Tone: {profile.tone}")

    # 3. Style
    if profile.style:
        sections.append("Style:\n" + "\n".join(f"- {item}" for item in profile.style))

    # 4. Behaviors
    if profile.traits:
        sections.append("Behaviors:\n" + "\n".join(f"- {item}" for item in profile.traits))

    # 5. Additional
    if profile.attributes:
        sections.append(
            "Additional:\n" + "\n".join(f"- {a.key}: {a.value}" for a in profile.attributes)
        )

    # 6. Preamble
    if profile.preamble is not None:
        sections.append(profile.preamble)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# PersonalityContributor
# ---------------------------------------------------------------------------


class PersonalityContributor:
    """``ContextContributor``-compatible contributor for the ``personality`` slot.

    The profile is rendered ONCE at construction time (the profile is frozen, the
    renderer is a pure function) -- this is the S6/S9 guarantee that the personality
    text is identical across every step in a drive.

    The contributor does NOT truncate against ``allocation.max_tokens``.  The assembler
    owns drop-whole atomicity for the ``personality`` slot.  Truncating here would create
    two arbiters of the same decision (the ``RegistryToolContributor`` single-source lesson
    from Pod 3.2c -- S9).

    Args:
        profile:     The frozen personality profile to inject.
        key:         Stable chunk key (default ``"personality:profile"``).
        source_slot: Slot name for provenance (default ``"personality"``).
    """

    def __init__(
        self,
        profile: PersonalityProfile,
        *,
        key: str = "personality:profile",
        source_slot: str = "personality",
    ) -> None:
        # Render ONCE at construction (profile is frozen, renderer is pure -> S6/S9 guarantee).
        self._rendered: str = render_profile(profile)
        self._key = key
        self._source_slot = source_slot

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Return the pre-rendered personality text as a single chunk.

        Returns ``SlotContent(status="empty")`` when the profile was empty at
        construction time.  Never truncates against ``allocation.max_tokens`` --
        the assembler owns drop-whole atomicity for this slot (S9 single-source).
        Never calls the model (S1).
        """
        if not self._rendered:
            return SlotContent(status="empty")
        return SlotContent(
            chunks=(
                SlotChunk(
                    text=self._rendered,
                    key=self._key,
                    source_slot=self._source_slot,
                ),
            )
        )
