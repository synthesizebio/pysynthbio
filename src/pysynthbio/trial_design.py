"""Build, inspect and edit VERDICT trial designs in Python.

A :class:`Trial` is an editable ``verdict.clinical-design/v1`` design: the same
clinical-name contract the VERDICT form submits. Nothing here contacts the API
until you call :meth:`Trial.review` or :meth:`Trial.run`; the server remains the
authority on name resolution, analysis availability and validation.

Quick tour::

    from pysynthbio import Trial, VerdictClient, trial_vocabulary

    trial_vocabulary()                 # every allowed value, offline
    Trial.describe()                   # every editable field, as a table

    trial = Trial("My phase 3 trial")
    trial.population.set_disease("hepatocellular carcinoma")
    trial.details.edit(phase="phase 3", allocation="randomized", blinding="double")
    trial.arms.add("Durvalumab", type="experimental", treatments=["durvalumab"])
    trial.arms.add("Sorafenib", type="active comparator", treatments=["sorafenib"])
    trial.endpoints.add("Overall survival", family="overall_survival",
                        effect_measure="hazard ratio", hypothesis="superiority")
    trial.comparisons.add("Durvalumab", "Sorafenib")
    trial                               # renders the design in a notebook

    client = VerdictClient()
    trial.review(client)                # what the server resolved and will run
    result = trial.run(client, idempotency_key="hcc-phase3-v1")

    by_sex = trial.vary(sex=["female", "male"])
    by_sex.run(client, idempotency_prefix="hcc-by-sex").table
"""

from __future__ import annotations

import copy
import hashlib
import html
import itertools
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = "verdict.clinical-design/v1"

# The pinned contract enums (verdict webapp app/lib/trial-catalog.ts). The server
# validates authoritatively; these allow friendly offline checks and discovery.
VOCABULARY: dict[str, tuple[str, ...]] = {
    "phase": ("EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"),
    "sex": ("all", "female", "male"),
    "arm_type": (
        "experimental",
        "active_comparator",
        "placebo_comparator",
        "no_intervention",
    ),
    "treatment_kind": (
        "drug",
        "biologic",
        "procedure",
        "radiation",
        "surgery",
        "device",
        "transplant",
        "behavioral",
        "dietary_supplement",
        "other",
    ),
    "treatment_role": (
        "study",
        "background",
        "placebo",
        "supportive",
        "premedication",
        "control",
        "other",
    ),
    "timing": (
        "concurrent",
        "lead_in",
        "sequential",
        "before",
        "after",
        "maintenance",
        "unspecified",
        "other",
    ),
    "endpoint_category": ("overall_survival", "progression_free_survival", "other"),
    "endpoint_role": ("primary", "secondary", "unknown"),
    "effect_measure": (
        "mean_difference",
        "hazard_ratio",
        "geometric_mean_ratio",
        "percentage_difference",
        "risk_difference",
        "odds_ratio",
        "risk_ratio",
        "median_difference",
        "ratio",
        "difference",
        "other",
    ),
    "hypothesis": ("superiority", "noninferiority", "equivalence", "unspecified"),
    "allocation": ("RANDOMIZED", "NON_RANDOMIZED", "NA"),
    "blinding": ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"),
    "purpose": (
        "TREATMENT",
        "PREVENTION",
        "SUPPORTIVE_CARE",
        "DIAGNOSTIC",
        "SCREENING",
        "HEALTH_SERVICES_RESEARCH",
        "BASIC_SCIENCE",
        "DEVICE_FEASIBILITY",
        "OTHER",
    ),
    "sponsor_type": (
        "INDUSTRY",
        "NIH",
        "FED",
        "OTHER_GOV",
        "NETWORK",
        "INDIV",
        "UNKNOWN",
        "OTHER",
    ),
    "seriousness": ("serious", "other"),
    "analysis": ("trial_characteristics", "efficacy", "toxicity", "biology"),
}

# Requested by default: every analysis a custom design can feed. Historical Odds
# needs an original registry submission, so it is opt-in.
DEFAULT_ANALYSES = ("efficacy", "toxicity", "biology")

_ALIASES = {
    "phase": {
        "1": "PHASE1",
        "2": "PHASE2",
        "3": "PHASE3",
        "4": "PHASE4",
        "EARLY_1": "EARLY_PHASE1",
        "0": "EARLY_PHASE1",
    },
    "blinding": {"OPEN": "NONE", "OPEN_LABEL": "NONE"},
    "arm_type": {
        "placebo": "placebo_comparator",
        "comparator": "active_comparator",
        "active": "active_comparator",
        "none": "no_intervention",
        "no_treatment": "no_intervention",
    },
}


class TrialDesignError(ValueError):
    """An edit or design the contract does not accept, with the allowed values."""


def _key(value: str) -> str:
    return re.sub(r"[\s\-/]+", "_", value.strip()).upper()


def choose(field: str, value: Any, *, nullable: bool = True) -> str | None:
    """Normalize a vocabulary value forgivingly ("phase 3", "double", "placebo").

    Raises TrialDesignError listing the allowed values when nothing matches.
    """
    allowed = VOCABULARY[field]
    if value is None:
        if nullable:
            return None
        raise TrialDesignError(f"{field} is required; choose one of {list(allowed)}")
    if not isinstance(value, str):
        raise TrialDesignError(
            f"{field} must be a string; choose one of {list(allowed)}"
        )
    aliases = {_key(k): _key(v) for k, v in _ALIASES.get(field, {}).items()}
    wanted = _key(value)
    candidates = {wanted, wanted.replace("_", "")}
    for candidate in list(candidates):
        bare = (
            candidate.removeprefix("PHASE_").removeprefix("PHASE")
            if field == "phase"
            else candidate
        )
        for key in (candidate, bare):
            if key in aliases:
                candidates.add(aliases[key])
    for option in allowed:
        if _key(option) in candidates or _key(option).replace("_", "") in candidates:
            return option
    raise TrialDesignError(
        f"{value!r} is not a valid {field}; choose one of {list(allowed)}"
    )


def _phases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p for p in re.split(r"[/,+&]| and ", value) if p.strip()]
        if len(parts) > 1:
            # "Phase 1/2" names its first phase once; later parts are bare numbers.
            return [choose("phase", p, nullable=False) for p in parts]  # type: ignore[misc]
        value = [value]
    return [choose("phase", p, nullable=False) for p in value]  # type: ignore[misc]


def _age(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise TrialDesignError(
            f"{field} must be a non-negative number of years or None"
        )
    return value


def _positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrialDesignError(f"{field} must be a positive whole number or None")
    return value


def _text(value: Any, field: str, *, required: bool = False) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise TrialDesignError(f"{field} must be a nonblank string")
        return None
    if not isinstance(value, str):
        raise TrialDesignError(f"{field} must be a string")
    return value.strip()


def _start_date(value: Any) -> str | None:
    value = _text(value, "start_date")
    if value is not None and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", value):
        raise TrialDesignError("start_date must be YYYY, YYYY-MM or YYYY-MM-DD")
    return value


def _form_number(value: Any) -> int | float | None:
    """A number from form text ("18", "2.5"); blank means not stated."""
    if value in (None, ""):
        return None
    return float(value) if "." in str(value) else int(value)


def _section(title: str, body: str) -> str:
    return f"<h4 style='margin:14px 0 4px'>{title}</h4>{body}"


def _escape(value: Any) -> str:
    return html.escape("—" if value is None or value == "" else str(value), quote=True)


_DISPLAY = {
    "NONE": "Open label",
    "NA": "Not applicable",
    "NON_RANDOMIZED": "Non-randomized",
}


def _label(value: str | None) -> str:
    if not value:
        return "—"
    if value in _DISPLAY:
        return _DISPLAY[value]
    text = value.replace("_", " ")
    return text if any(c.islower() for c in value) else text.title()


def _table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    cell = (
        "padding:4px 10px;border-bottom:1px solid #e5e5e5;"
        "text-align:left;vertical-align:top"
    )
    head = "".join(f"<th style='{cell}'>{_escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td style='{cell}'>{_escape(v)}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return (
        "<table style='border-collapse:collapse'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


# ---- Vocabulary and field discovery ------------------------------------------------


class TrialVocabulary(dict):
    """Allowed values for every enumerated trial field (a dict of tuples)."""

    def describe(self, field: str) -> tuple[str, ...]:
        """The allowed values of one field, e.g. ``describe("blinding")``."""
        if field not in self:
            raise KeyError(f"Unknown field {field!r}; known: {sorted(self)}")
        return self[field]

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {', '.join(v)}" for k, v in self.items())

    def _repr_html_(self) -> str:
        return _table(
            ["Field", "Allowed values"], [(k, ", ".join(v)) for k, v in self.items()]
        )


def trial_vocabulary() -> TrialVocabulary:
    """Every enumerated value a trial design accepts. Offline; no API call.

    Endpoint families come from the release catalog: ``client.trial_catalog()``.
    """
    return TrialVocabulary(VOCABULARY)


# Field path, type, allowed values (vocabulary key or None) and meaning. This is the
# map an agent or person needs to construct and edit a valid design.
_FIELDS: tuple[tuple[str, str, str | None, str], ...] = (
    ("details.name", "text", None, "Trial name (required)"),
    ("details.phases", "list", "phase", "Trial phase(s); 'phase 1/2' is accepted"),
    (
        "details.start_date",
        "YYYY[-MM[-DD]]",
        None,
        "Planned or actual start; limits evidence",
    ),
    ("details.allocation", "choice", "allocation", "Randomization"),
    ("details.blinding", "choice", "blinding", "Masking"),
    ("details.purpose", "choice", "purpose", "Primary purpose"),
    ("details.sponsor_type", "choice", "sponsor_type", "Lead sponsor class"),
    ("details.arm_count", "int", None, "Registered arms, if more than modeled"),
    ("details.primary_endpoint_count", "int", None, "Primary endpoints in the trial"),
    ("population.disease", "text", None, "Condition name; MONDO resolves on review"),
    (
        "population.mondo_id",
        "MONDO:#######",
        None,
        "Optional explicit disease identity",
    ),
    ("population.sex", "choice", "sex", "Sex eligibility"),
    ("population.min_age", "years", None, "Minimum age (None = not stated)"),
    ("population.max_age", "years", None, "Maximum age (None = not stated)"),
    ("population.healthy_volunteers", "bool", None, "Accepts healthy volunteers"),
    ("population.notes", "text", None, "Eligibility as written"),
    ("arms[].label", "text", None, "Arm name (unique)"),
    ("arms[].type", "choice", "arm_type", "Arm type"),
    (
        "arms[].treatments[].name",
        "text",
        None,
        "Treatment name; resolves to ChEMBL on review",
    ),
    ("arms[].treatments[].role", "choice", "treatment_role", "Role in the regimen"),
    ("arms[].treatments[].timing", "choice", "timing", "When it is given"),
    ("arms[].treatments[].kind", "choice", "treatment_kind", "Treatment type"),
    ("arms[].treatments[].chembl", "CHEMBL#", None, "Explicit ChEMBL identity"),
    ("arms[].treatments[].smiles", "SMILES", None, "Explicit structure"),
    (
        "arms[].treatments[].name_only",
        "bool",
        None,
        "No molecular identity (e.g. radiation)",
    ),
    ("arms[].treatments[].dose", "text", None, "Dose as stated (reference only)"),
    ("endpoints[].title", "text", None, "Registered endpoint title"),
    ("endpoints[].role", "choice", "endpoint_role", "Primary (default) or secondary"),
    (
        "endpoints[].family",
        "catalog key",
        None,
        "Release family; see client.trial_catalog()",
    ),
    ("endpoints[].category", "choice", "endpoint_category", "Scientific category"),
    (
        "endpoints[].effect_measure",
        "choice",
        "effect_measure",
        "Planned effect measure",
    ),
    ("endpoints[].hypothesis", "choice", "hypothesis", "Planned hypothesis"),
    ("endpoints[].time_frame", "text", None, "Time frame notes"),
    (
        "comparisons[]",
        "arm → arm",
        None,
        "Experimental arm vs comparator, on endpoints",
    ),
    (
        "safety_queries[]",
        "term + seriousness",
        "seriousness",
        "Only for non-catalog releases",
    ),
    (
        "analyses",
        "list",
        "analysis",
        "Requested analyses (default efficacy, toxicity, biology)",
    ),
)


# ---- Building blocks ----------------------------------------------------------------


class _Record:
    """A dict-backed record with attribute access and validated edits."""

    _fields: tuple[str, ...] = ()

    def __init__(self, trial: "Trial | None" = None):
        self._trial = trial

    def to_dict(self) -> dict[str, Any]:
        return {f: copy.deepcopy(getattr(self, f)) for f in self._fields}

    def edit(self, **changes: Any):
        """Change fields by keyword; unknown keys raise. Returns self for chaining."""
        unknown = set(changes) - set(self._fields)
        if unknown:
            raise TrialDesignError(
                f"Unknown {type(self).__name__.lower()} field(s) {sorted(unknown)}; "
                f"editable: {list(self._fields)}"
            )
        staged = copy.copy(self)
        object.__setattr__(staged, "_source", self)
        for name, value in changes.items():
            setattr(staged, name, value)  # validates
        for name in changes:
            object.__setattr__(self, name, getattr(staged, name))
        return self

    def show(self) -> None:
        print(self)

    def __repr__(self) -> str:
        values = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._fields)
        return f"{type(self).__name__}({values})"

    def _repr_html_(self) -> str:
        return _table(["Field", "Value"], [(f, getattr(self, f)) for f in self._fields])


class Treatment(_Record):
    """One concurrent component of an arm's regimen."""

    _fields = (
        "name",
        "role",
        "timing",
        "kind",
        "chembl",
        "smiles",
        "name_only",
        "dose",
        "dose_stated",
    )

    def __init__(
        self,
        name: str,
        *,
        role=None,
        timing=None,
        kind=None,
        chembl=None,
        smiles=None,
        name_only=False,
        dose=None,
        dose_stated=None,
    ):
        super().__init__()
        self.name = name
        self.role, self.timing, self.kind = role, timing, kind
        self.chembl, self.smiles, self.name_only = chembl, smiles, name_only
        self.dose, self.dose_stated = dose, dose_stated

    def __setattr__(self, name, value):
        if name == "name":
            value = _text(value, "treatment name", required=True)
        elif name == "role":
            value = choose("treatment_role", value)
        elif name == "timing":
            value = choose("timing", value)
        elif name == "kind":
            value = choose("treatment_kind", value)
        elif name == "chembl" and value is not None:
            value = _text(value, "chembl", required=True).upper()
            if not re.fullmatch(r"CHEMBL\d+", value):
                raise TrialDesignError("chembl must look like CHEMBL1336")
        elif name == "smiles" and value is not None:
            value = _text(value, "smiles", required=True)
            if re.search(r"\s", value):
                raise TrialDesignError("smiles must not contain spaces")
        elif name == "dose":
            value = _text(value, "dose")
        object.__setattr__(self, name, value)
        identities = [
            getattr(self, "chembl", None),
            getattr(self, "smiles", None),
            getattr(self, "name_only", False) or None,
        ]
        if sum(v is not None for v in identities) > 1:
            raise TrialDesignError("Set at most one of chembl, smiles or name_only")

    @property
    def identity(self) -> str:
        """``chembl``/``smiles``/``name_only`` if explicit, else ``resolve by name``."""
        if self.chembl:
            return f"ChEMBL {self.chembl}"
        if self.smiles:
            return "SMILES"
        if self.name_only:
            return "name only"
        return "resolve by name"

    def _contract(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        for key, field in (("kind", "kind"), ("role", "role"), ("timing", "timing")):
            if getattr(self, field) is not None:
                out[key] = getattr(self, field)
        if self.chembl:
            out["identifier"] = {"kind": "chembl", "value": self.chembl}
        elif self.smiles:
            out["identifier"] = {"kind": "smiles", "value": self.smiles}
        elif self.name_only:
            out["identifier"] = {"kind": "name_only"}
        if self.dose:
            out["doseNotes"] = self.dose
        if self.dose_stated is not None or self.dose:
            out["doseStated"] = (
                bool(self.dose) if self.dose_stated is None else self.dose_stated
            )
        return out

    @classmethod
    def _from_contract(cls, data: dict[str, Any]) -> "Treatment":
        identifier = data.get("identifier") or {}
        return cls(
            data["name"],
            role=data.get("role"),
            timing=data.get("timing"),
            kind=data.get("kind"),
            chembl=(
                identifier.get("value") if identifier.get("kind") == "chembl" else None
            ),
            smiles=(
                identifier.get("value") if identifier.get("kind") == "smiles" else None
            ),
            name_only=identifier.get("kind") == "name_only",
            dose=data.get("doseNotes"),
            dose_stated=data.get("doseStated"),
        )


class _Collection:
    """Shared list behaviour: lookup by position or name, iteration, tables."""

    _noun = "item"
    _name_attr = "label"

    def __init__(self, trial: "Trial"):
        self._trial = trial
        self._items: list[Any] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(list(self._items))

    def __getitem__(self, key: int | str):
        return self.get(key)

    def __contains__(self, key: Any) -> bool:
        try:
            self.get(key)
            return True
        except (KeyError, IndexError):
            return False

    def list(self) -> list[Any]:
        """The items, in order."""
        return list(self._items)

    def names(self) -> list[str]:
        return [getattr(item, self._name_attr) for item in self._items]

    def get(self, key: Any):
        """Look up by position, by name (case-insensitive) or by the item itself."""
        if any(key is item for item in self._items):
            return key
        if isinstance(key, bool):
            raise KeyError(key)
        if isinstance(key, int):
            try:
                return self._items[key]
            except IndexError:
                raise IndexError(
                    f"No {self._noun} at position {key}; there are {len(self)}"
                ) from None
        if isinstance(key, str):
            wanted = key.strip().lower()
            for item in self._items:
                if getattr(item, self._name_attr).lower() == wanted:
                    return item
        raise KeyError(f"No {self._noun} {key!r}; have {self.names()}")

    def index(self, key: Any) -> int:
        item = self.get(key)
        return next(i for i, other in enumerate(self._items) if other is item)

    def move(self, key: Any, position: int):
        """Move an item to a new position. Returns the trial for chaining."""
        item = self.get(key)
        self._items.remove(item)
        self._items.insert(position, item)
        return self._trial

    def show(self) -> None:
        print(self)

    def to_frame(self):
        """The items as a pandas DataFrame, one row each."""
        import pandas as pd

        return pd.DataFrame(self._rows(), columns=self._headers())

    def __repr__(self) -> str:
        if not self._items:
            return f"No {self._noun}s"
        return "\n".join(
            f"{i}. " + " · ".join(str(v) for v in row if v not in (None, ""))
            for i, row in enumerate(self._rows())
        )

    def _repr_html_(self) -> str:
        if not self._items:
            return f"<p>No {self._noun}s</p>"
        return _table(self._headers(), self._rows())

    def _headers(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _rows(self) -> list[list[Any]]:  # pragma: no cover - overridden
        raise NotImplementedError


class TreatmentList(_Collection):
    """An arm's treatments: ``add``, ``remove``, ``edit``, ``list``, ``get``."""

    _noun = "treatment"
    _name_attr = "name"

    def add(self, name: "str | Treatment", **fields: Any) -> Treatment:
        """Add a treatment by name (resolved on review) or a :class:`Treatment`."""
        treatment = name if isinstance(name, Treatment) else Treatment(name, **fields)
        if len(self._items) >= 20:
            raise TrialDesignError("An arm can have at most 20 treatments")
        self._items.append(treatment)
        return treatment

    def remove(self, key: Any) -> Treatment:
        item = self.get(key)
        self._items.remove(item)
        return item

    def edit(self, key: Any, **changes: Any) -> Treatment:
        return self.get(key).edit(**changes)

    def clear(self) -> None:
        self._items.clear()

    def _headers(self):
        return ["Treatment", "Role", "Timing", "Dose", "Identity"]

    def _rows(self):
        return [
            [t.name, _label(t.role), _label(t.timing), t.dose, t.identity]
            for t in self._items
        ]


class Arm(_Record):
    """A trial arm: label, type and treatments."""

    _fields = ("label", "type")

    def __init__(self, trial: "Trial", label: str, type: str | None = None):
        object.__setattr__(self, "_trial", trial)
        object.__setattr__(self, "treatments", TreatmentList(trial))
        self.label = label
        self.type = type

    def __setattr__(self, name, value):
        if name == "label":
            value = _text(value, "arm label", required=True)
            arms = self._trial.arms if hasattr(self._trial, "arms") else None
            original = self.__dict__.get("_source", self)
            if arms is not None and any(
                other is not original and other.label.lower() == value.lower()
                for other in arms
            ):
                raise TrialDesignError(f"An arm named {value!r} already exists")
        elif name == "type":
            value = choose("arm_type", value)
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "type": self.type,
            "treatments": [t.to_dict() for t in self.treatments],
        }

    def __repr__(self) -> str:
        regimen = " + ".join(t.name for t in self.treatments) or "no treatments"
        return f"Arm({self.label!r}, {_label(self.type)}: {regimen})"

    def _repr_html_(self) -> str:
        return (
            f"<p><strong>{_escape(self.label)}</strong> · "
            f"{_escape(_label(self.type))}</p>" + self.treatments._repr_html_()
        )


class ArmList(_Collection):
    """The trial's arms: add, remove, edit, rename, list, get and move."""

    _noun = "arm"

    def add(
        self, label: str, type: str | None = None, *, treatments: Iterable[Any] = ()
    ) -> Arm:
        """Add an arm; ``treatments`` takes names, field dicts or Treatments."""
        if len(self._items) >= 12:
            raise TrialDesignError("A trial can have at most 12 arms")
        arm = Arm(self._trial, label, type)
        for item in treatments:
            if isinstance(item, dict):
                arm.treatments.add(**item)
            else:
                arm.treatments.add(item)
        self._items.append(arm)
        return arm

    def remove(self, key: Any) -> Arm:
        """Remove an arm and every comparison that uses it."""
        arm = self.get(key)
        self._trial.comparisons._drop_arm(arm)
        self._items.remove(arm)
        return arm

    def edit(self, key: Any, **changes: Any) -> Arm:
        return self.get(key).edit(**changes)

    def rename(self, key: Any, label: str) -> Arm:
        return self.edit(key, label=label)

    def clear(self) -> None:
        for arm in list(self._items):
            self.remove(arm)

    def _headers(self):
        return ["Arm", "Type", "Treatments"]

    def _rows(self):
        return [
            [
                a.label,
                _label(a.type),
                " + ".join(
                    t.name + (f" ({t.dose})" if t.dose else "") for t in a.treatments
                )
                or "—",
            ]
            for a in self._items
        ]


class Endpoint(_Record):
    """A trial endpoint and its planned analysis."""

    _fields = (
        "title",
        "role",
        "family",
        "category",
        "effect_measure",
        "hypothesis",
        "time_frame",
        "notes",
    )

    def __init__(
        self,
        trial: "Trial",
        title: str,
        *,
        role=None,
        family=None,
        category=None,
        effect_measure=None,
        hypothesis=None,
        time_frame=None,
        notes=None,
    ):
        object.__setattr__(self, "_trial", trial)
        self.title, self.role = title, role
        self.family, self.category = family, category
        self.effect_measure, self.hypothesis = effect_measure, hypothesis
        self.time_frame, self.notes = time_frame, notes

    def __setattr__(self, name, value):
        if name == "title":
            value = _text(value, "endpoint title", required=True)
        elif name == "family":
            value = _text(value, "family")
            value = value.lower().replace(" ", "_") if value else None
        elif name == "category":
            value = choose("endpoint_category", value)
        elif name == "role":
            value = choose("endpoint_role", value)
        elif name == "effect_measure":
            value = choose("effect_measure", value)
        elif name == "hypothesis":
            value = choose("hypothesis", value)
        elif name in ("time_frame", "notes"):
            value = _text(value, name)
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        plan = " · ".join(
            _label(v) for v in (self.effect_measure, self.hypothesis) if v
        )
        return f"Endpoint({self.title!r}{', ' + plan if plan else ''})"


class EndpointList(_Collection):
    """The trial's endpoints: add, remove, edit, list, get and move."""

    _noun = "endpoint"
    _name_attr = "title"

    def add(self, title: str, **fields: Any) -> Endpoint:
        """Add an endpoint.

        Fields: family, category, effect_measure, hypothesis, time_frame, notes.

        New endpoints join every existing comparison; edit comparisons to change that.
        """
        if len(self._items) >= 20:
            raise TrialDesignError("A trial can have at most 20 endpoints")
        if title.strip().lower() in (n.lower() for n in self.names()):
            raise TrialDesignError(f"An endpoint titled {title!r} already exists")
        endpoint = Endpoint(self._trial, title, **fields)
        self._items.append(endpoint)
        for comparison in self._trial.comparisons:
            comparison.endpoints.append(endpoint)
        return endpoint

    def remove(self, key: Any) -> Endpoint:
        """Remove an endpoint, and comparisons left with no endpoint."""
        endpoint = self.get(key)
        self._items.remove(endpoint)
        self._trial.comparisons._drop_endpoint(endpoint)
        return endpoint

    def edit(self, key: Any, **changes: Any) -> Endpoint:
        return self.get(key).edit(**changes)

    def clear(self) -> None:
        for endpoint in list(self._items):
            self.remove(endpoint)

    def _headers(self):
        return [
            "Endpoint",
            "Role",
            "Family",
            "Effect measure",
            "Hypothesis",
            "Time frame",
        ]

    def _rows(self):
        return [
            [
                e.title,
                _label(e.role or "primary"),
                _label(e.family),
                _label(e.effect_measure),
                _label(e.hypothesis),
                e.time_frame,
            ]
            for e in self._items
        ]


class Comparison:
    """An experimental arm compared against a comparator on some endpoints."""

    def __init__(self, experimental: Arm, comparator: Arm, endpoints: list[Endpoint]):
        self.experimental, self.comparator, self.endpoints = (
            experimental,
            comparator,
            endpoints,
        )

    @property
    def label(self) -> str:
        return f"{self.experimental.label} vs {self.comparator.label}"

    def __repr__(self) -> str:
        return (
            f"Comparison({self.label} on {', '.join(e.title for e in self.endpoints)})"
        )


class ComparisonList(_Collection):
    """Comparisons: add, remove, edit, swap, list, get and clear.

    Look one up by position, by ``"A vs B"``, or with ``get("A", "B")``.
    """

    _noun = "comparison"

    def get(self, key: Any, comparator: Any = None):  # type: ignore[override]
        if comparator is not None:
            experimental = self._trial.arms.get(key)
            comparator = self._trial.arms.get(comparator)
            for item in self._items:
                if item.experimental is experimental and item.comparator is comparator:
                    return item
            raise KeyError(f"No comparison {experimental.label} vs {comparator.label}")
        if isinstance(key, str) and " vs " in key.lower():
            left, right = re.split(r"\s+vs\.?\s+", key, maxsplit=1, flags=re.I)
            return self.get(left, right)
        return super().get(key)

    def add(
        self, experimental: Any, comparator: Any, endpoints: Iterable[Any] | None = None
    ) -> Comparison:
        """Compare two arms on some endpoints (default: every endpoint).

        Adding an existing pair in the other direction reverses it instead.
        """
        exp, comp = self._trial.arms.get(experimental), self._trial.arms.get(comparator)
        if exp is comp:
            raise TrialDesignError("A comparison needs two different arms")
        chosen = self._endpoints(endpoints)
        for item in self._items:
            if {item.experimental, item.comparator} == {exp, comp}:
                item.experimental, item.comparator, item.endpoints = exp, comp, chosen
                return item
        if len(self._items) >= 24:
            raise TrialDesignError("A trial can have at most 24 comparisons")
        comparison = Comparison(exp, comp, chosen)
        self._items.append(comparison)
        return comparison

    def remove(self, key: Any, comparator: Any = None) -> Comparison:
        item = self.get(key, comparator)
        self._items.remove(item)
        return item

    def edit(
        self,
        key: Any,
        comparator: Any = None,
        *,
        endpoints=None,
        add_endpoints=(),
        remove_endpoints=(),
    ) -> Comparison:
        """Replace, add or remove a comparison's endpoints."""
        item = self.get(key, comparator)
        chosen = (
            self._endpoints(endpoints)
            if endpoints is not None
            else list(item.endpoints)
        )
        for endpoint in self._endpoints(add_endpoints, default_all=False):
            if endpoint not in chosen:
                chosen.append(endpoint)
        removing = self._endpoints(remove_endpoints, default_all=False)
        chosen = [e for e in chosen if e not in removing]
        if not chosen:
            raise TrialDesignError(
                "A comparison needs at least one endpoint; remove it instead"
            )
        item.endpoints = chosen
        return item

    def swap(self, key: Any, comparator: Any = None) -> Comparison:
        """Reverse which arm is experimental."""
        item = self.get(key, comparator)
        item.experimental, item.comparator = item.comparator, item.experimental
        return item

    def clear(self) -> None:
        self._items.clear()

    def all_pairs(
        self, comparator: Any, endpoints: Iterable[Any] | None = None
    ) -> list[Comparison]:
        """Compare every other arm against one comparator (e.g. a shared control)."""
        control = self._trial.arms.get(comparator)
        return [
            self.add(arm, control, endpoints)
            for arm in self._trial.arms
            if arm is not control
        ]

    def _endpoints(
        self, keys: Iterable[Any] | None, default_all: bool = True
    ) -> list[Endpoint]:
        if keys is None:
            if default_all:
                if not len(self._trial.endpoints):
                    raise TrialDesignError("Add an endpoint before comparing arms")
                return self._trial.endpoints.list()
            return []
        if isinstance(keys, (str, int, Endpoint)):
            keys = [keys]
        return [self._trial.endpoints.get(k) for k in keys]

    def _drop_arm(self, arm: Arm) -> None:
        self._items = [
            c for c in self._items if arm not in (c.experimental, c.comparator)
        ]

    def _drop_endpoint(self, endpoint: Endpoint) -> None:
        for item in self._items:
            item.endpoints = [e for e in item.endpoints if e is not endpoint]
        self._items = [c for c in self._items if c.endpoints]

    def _headers(self):
        return ["Experimental", "Comparator", "Endpoints"]

    def _rows(self):
        return [
            [
                c.experimental.label,
                c.comparator.label,
                ", ".join(e.title for e in c.endpoints),
            ]
            for c in self._items
        ]


class SafetyQueryList(_Collection):
    """Adverse-event queries; only for releases without the full Safety catalog."""

    _noun = "safety query"
    _name_attr = "term"

    class Query:
        def __init__(self, term: str, seriousness: str):
            self.term = _text(term, "term", required=True)
            self.seriousness = choose("seriousness", seriousness, nullable=False)

        def __repr__(self):
            return f"Query({self.term!r}, {self.seriousness})"

    def add(self, term: str, seriousness: str = "serious"):
        query = self.Query(term, seriousness)
        self._items.append(query)
        return query

    def remove(self, key: Any):
        item = self.get(key)
        self._items.remove(item)
        return item

    def clear(self) -> None:
        self._items.clear()

    def _headers(self):
        return ["Term", "Seriousness"]

    def _rows(self):
        return [[q.term, q.seriousness] for q in self._items]


class TrialDetails(_Record):
    """Trial-level details: name, phase(s), start date and design metadata."""

    _fields = (
        "name",
        "phases",
        "start_date",
        "allocation",
        "blinding",
        "purpose",
        "sponsor_type",
        "arm_count",
        "primary_endpoint_count",
    )

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        for field in self._fields[1:]:
            setattr(self, field, [] if field == "phases" else None)

    def edit(self, **changes: Any):
        if "phase" in changes:  # the singular is the common case
            changes["phases"] = changes.pop("phase")
        return super().edit(**changes)

    def __setattr__(self, name, value):
        if name == "name":
            value = _text(value, "name", required=True)
        elif name == "phases":
            value = _phases(value)
        elif name == "start_date":
            value = _start_date(value)
        elif name in ("allocation", "blinding", "purpose", "sponsor_type"):
            value = choose(name, value)
        elif name in ("arm_count", "primary_endpoint_count"):
            value = _positive_int(value, name)
        object.__setattr__(self, name, value)

    @property
    def phase(self) -> str:
        return (
            " / ".join(
                p.replace("EARLY_PHASE", "Early phase ").replace("PHASE", "Phase ")
                for p in self.phases
            )
            or "—"
        )


class Population(_Record):
    """Who the trial enrols: disease, sex, ages, healthy volunteers and notes."""

    _fields = ("conditions", "sex", "min_age", "max_age", "healthy_volunteers", "notes")

    def __init__(self):
        super().__init__()
        self.conditions = []
        self.sex = self.min_age = self.max_age = self.healthy_volunteers = (
            self.notes
        ) = None

    def __setattr__(self, name, value):
        if name == "conditions":
            value = [self._condition(c) for c in value]
            if len(value) > 20:
                raise TrialDesignError("At most 20 conditions")
        elif name == "sex":
            value = choose("sex", value)
        elif name in ("min_age", "max_age"):
            value = _age(value, name)
            other = getattr(self, "max_age" if name == "min_age" else "min_age", None)
            low, high = (value, other) if name == "min_age" else (other, value)
            if low is not None and high is not None and low > high:
                raise TrialDesignError("min_age must not exceed max_age")
        elif name == "healthy_volunteers":
            if value is not None and not isinstance(value, bool):
                raise TrialDesignError("healthy_volunteers must be True, False or None")
        elif name == "notes":
            value = _text(value, "notes")
        object.__setattr__(self, name, value)

    @staticmethod
    def _condition(value: Any) -> dict[str, str]:
        if isinstance(value, str):
            value = {"name": value}
        name = _text(value.get("name"), "disease", required=True)
        out = {"name": name}
        mondo = value.get("mondoId") or value.get("mondo_id")
        if mondo:
            mondo = mondo.strip().upper()
            if not re.fullmatch(r"MONDO:\d{7}", mondo):
                raise TrialDesignError("mondo_id must look like MONDO:0007256")
            out["mondoId"] = mondo
        return out

    @property
    def disease(self) -> str | None:
        return self.conditions[0]["name"] if self.conditions else None

    def set_disease(self, name: str, mondo_id: str | None = None):
        """Set the trial's single disease (replacing others). Returns self."""
        self.conditions = [
            {"name": name, **({"mondoId": mondo_id} if mondo_id else {})}
        ]
        return self

    @property
    def summary(self) -> str:
        parts = [self.disease]
        if self.sex and self.sex != "all":
            parts.append(f"{self.sex} only")
        if self.min_age is not None or self.max_age is not None:
            low = "?" if self.min_age is None else f"{self.min_age:g}"
            high = "+" if self.max_age is None else f"–{self.max_age:g}"
            parts.append(f"ages {low}{high}")
        if self.healthy_volunteers:
            parts.append("healthy volunteers")
        return " · ".join(p for p in parts if p) or "—"


# ---- The trial ----------------------------------------------------------------------

_DETAIL_KEYS = set(TrialDetails._fields) | {"phase"}
_POPULATION_KEYS = set(Population._fields) | {"disease", "mondo_id"}


class Trial:
    """An editable VERDICT trial design (``verdict.clinical-design/v1``).

    Sections: :attr:`details`, :attr:`population`, :attr:`arms`, :attr:`endpoints`,
    :attr:`comparisons`, :attr:`safety_queries` and :attr:`analyses`. Display it in a
    notebook, ``print`` it, or use ``to_dict()``/``save()``. Every edit is validated
    against the contract vocabulary offline; :meth:`review` asks the server.
    """

    def __init__(self, name: str = "Untitled trial"):
        self.details = TrialDetails(name)
        self.population = Population()
        self.arms = ArmList(self)
        self.endpoints = EndpointList(self)
        self.comparisons = ComparisonList(self)
        self.safety_queries = SafetyQueryList(self)
        self._analyses = list(DEFAULT_ANALYSES)

    # -- discovery -------------------------------------------------------------------

    @staticmethod
    def describe():
        """Every editable field, its type, allowed values and meaning (DataFrame)."""
        import pandas as pd

        return pd.DataFrame(
            [
                (path, kind, ", ".join(VOCABULARY[vocab]) if vocab else "", meaning)
                for path, kind, vocab, meaning in _FIELDS
            ],
            columns=["field", "type", "allowed values", "meaning"],
        )

    # -- convenience -----------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.details.name

    @name.setter
    def name(self, value: str) -> None:
        self.details.name = value

    @property
    def analyses(self) -> list[str]:
        return list(self._analyses)

    @analyses.setter
    def analyses(self, values: Iterable[str]) -> None:
        chosen = [choose("analysis", v, nullable=False) for v in values]
        if not chosen:
            raise TrialDesignError("Request at least one analysis")
        self._analyses = list(dict.fromkeys(chosen))  # type: ignore[arg-type]

    def edit(self, **changes: Any) -> "Trial":
        """Change details and population fields in one call. Returns self.

        Example: ``trial.edit(phase="phase 2", sex="male", min_age=18)``.
        """
        details = {k: v for k, v in changes.items() if k in _DETAIL_KEYS}
        population = {k: v for k, v in changes.items() if k in _POPULATION_KEYS}
        unknown = set(changes) - set(details) - set(population) - {"analyses"}
        if unknown:
            raise TrialDesignError(
                f"Unknown field(s) {sorted(unknown)}; "
                "see Trial.describe() for editable fields"
            )
        staged = self.copy()
        staged._apply(details, population, changes.get("analyses"))
        self._apply(details, population, changes.get("analyses"))
        return self

    def _apply(self, details, population, analyses) -> None:
        if details:
            self.details.edit(**details)
        disease = population.pop("disease", None)
        mondo = population.pop("mondo_id", None)
        if disease is not None or mondo is not None:
            self.population.set_disease(disease or self.population.disease or "", mondo)
        if population:
            self.population.edit(**population)
        if analyses is not None:
            self.analyses = analyses

    def copy(self, name: str | None = None) -> "Trial":
        """An independent copy (optionally renamed)."""
        twin = Trial.from_dict(self.to_dict())
        if name:
            twin.details.name = name
        return twin

    def variant(self, name: str | None = None, **changes: Any) -> "Trial":
        """A copy with details/population changes: ``trial.variant(sex="male")``."""
        twin = self.copy(name)
        twin.edit(**changes)
        return twin

    def vary(self, **sweeps: Iterable[Any]) -> "TrialSet":
        """Every combination of the given values, as a :class:`TrialSet`.

        ``trial.vary(sex=["female", "male"], min_age=[18, 65])`` makes four trials.
        Keys are details/population fields (see :meth:`describe`).
        """
        if not sweeps:
            raise TrialDesignError("Pass at least one field to vary")
        names = list(sweeps)
        variants = TrialSet()
        for values in itertools.product(*(list(sweeps[n]) for n in names)):
            label = ", ".join(f"{n}={v}" for n, v in zip(names, values))
            changes = dict(zip(names, values))
            variants.add(label, self.variant(f"{self.name} ({label})", **changes))
        return variants

    # -- serialization ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The ``verdict.clinical-design/v1`` JSON body the API accepts."""
        d, p = self.details, self.population
        endpoints = self.endpoints.list()
        arms = self.arms.list()
        design: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "name": d.name,
            "population": {
                "conditions": copy.deepcopy(p.conditions),
                "phases": list(d.phases),
                "sex": p.sex,
                "healthyVolunteers": p.healthy_volunteers,
                "minAge": p.min_age,
                "maxAge": p.max_age,
                **({"notes": p.notes} if p.notes else {}),
            },
            "arms": [
                {
                    "label": a.label,
                    "type": a.type,
                    "components": [t._contract() for t in a.treatments],
                }
                for a in arms
            ],
            "endpoints": [
                {
                    k: v
                    for k, v in {
                        "title": e.title,
                        "role": e.role,
                        "family": e.family,
                        "category": e.category,
                        "effectMeasure": e.effect_measure,
                        "hypothesis": e.hypothesis,
                        "timeFrame": e.time_frame,
                        "notes": e.notes,
                    }.items()
                    if v is not None or k in ("effectMeasure", "hypothesis")
                }
                for e in endpoints
            ],
            "metadata": {
                "allocation": d.allocation,
                "blinding": d.blinding,
                "purpose": d.purpose,
                "sponsorType": d.sponsor_type,
                # Registered arms cannot be fewer than the arms being modeled.
                "armCount": (
                    None if d.arm_count is None else max(d.arm_count, len(arms))
                ),
                "primaryEndpointCount": d.primary_endpoint_count,
                **({"startDate": d.start_date} if d.start_date else {}),
            },
            "requestedAnalyses": list(self._analyses),
        }
        # Two arms and no comparison: the server compares the first against the second.
        if len(self.comparisons) or len(arms) != 2:
            design["comparisons"] = [
                {
                    "experimentalArm": arms.index(c.experimental),
                    "comparatorArm": arms.index(c.comparator),
                    "endpoints": [endpoints.index(e) for e in c.endpoints],
                }
                for c in self.comparisons
            ]
        if len(self.safety_queries):
            design["safetyQueries"] = [
                {"term": q.term, "seriousness": q.seriousness}
                for q in self.safety_queries
            ]
        return design

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the design as JSON; never calls the API."""
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trial":
        """Read a design saved with :meth:`save` (or any clinical-design JSON)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, design: dict[str, Any]) -> "Trial":
        """Build a Trial from a ``verdict.clinical-design/v1`` dict."""
        if design.get("schemaVersion", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise TrialDesignError(f"Expected schemaVersion {SCHEMA_VERSION!r}")
        trial = cls(design.get("name") or "Untitled trial")
        population = design.get("population") or {}
        metadata = design.get("metadata") or {}
        trial.details.edit(
            phases=population.get("phases") or [],
            start_date=metadata.get("startDate"),
            allocation=metadata.get("allocation"),
            blinding=metadata.get("blinding"),
            purpose=metadata.get("purpose"),
            sponsor_type=metadata.get("sponsorType"),
            arm_count=metadata.get("armCount"),
            primary_endpoint_count=metadata.get("primaryEndpointCount"),
        )
        trial.population.edit(
            conditions=population.get("conditions") or [],
            sex=population.get("sex"),
            min_age=population.get("minAge"),
            max_age=population.get("maxAge"),
            healthy_volunteers=population.get("healthyVolunteers"),
            notes=population.get("notes"),
        )
        for arm in design.get("arms") or []:
            added = trial.arms.add(arm["label"], arm.get("type"))
            for component in arm.get("components") or []:
                added.treatments.add(Treatment._from_contract(component))
        for endpoint in design.get("endpoints") or []:
            trial.endpoints._items.append(
                Endpoint(
                    trial,
                    endpoint.get("title")
                    or _label(endpoint.get("category"))
                    or "Endpoint",
                    role=endpoint.get("role"),
                    family=endpoint.get("family"),
                    category=endpoint.get("category"),
                    effect_measure=endpoint.get("effectMeasure"),
                    hypothesis=endpoint.get("hypothesis"),
                    time_frame=endpoint.get("timeFrame"),
                    notes=endpoint.get("notes"),
                )
            )
        arms, endpoints = trial.arms.list(), trial.endpoints.list()
        for comparison in design.get("comparisons") or []:
            trial.comparisons._items.append(
                Comparison(
                    arms[comparison["experimentalArm"]],
                    arms[comparison["comparatorArm"]],
                    [endpoints[i] for i in comparison["endpoints"]],
                )
            )
        for query in design.get("safetyQueries") or []:
            trial.safety_queries.add(query["term"], query["seriousness"])
        if design.get("requestedAnalyses"):
            trial.analyses = design["requestedAnalyses"]
        return trial

    @classmethod
    def from_import(cls, proposal: dict[str, Any]) -> "Trial":
        """Build a Trial from ``client.import_trial(...)`` output.

        Review the result: imported values are suggestions, and the import's
        evidence stays in ``proposal["metadata"]``.
        """
        draft = proposal.get("draft", proposal)
        custom = draft.get("custom") or {}
        design = custom if "arms" in custom else draft
        trial = cls(design.get("name") or "Imported trial")
        outcome = design.get("outcomeDesign") or {}
        number = _form_number
        endpoints = design.get("endpoints") or []
        counts = {e.get("analysis", {}).get("primaryEndpoints") for e in endpoints} - {
            None,
            "",
        }
        trial.details.edit(
            phases=design.get("phases") or [],
            start_date=design.get("startDate") or None,
            allocation=outcome.get("allocation") or None,
            blinding=outcome.get("masking") or None,
            purpose=outcome.get("primaryPurpose") or None,
            sponsor_type=outcome.get("sponsorClass") or None,
            arm_count=number(outcome.get("numberOfArms")),
            primary_endpoint_count=number(counts.pop()) if len(counts) == 1 else None,
        )
        trial.population.edit(
            conditions=[c for c in design.get("conditions") or [] if c.get("name")],
            sex=design.get("sex") or None,
            min_age=number(design.get("minAge")),
            max_age=number(design.get("maxAge")),
            healthy_volunteers={"yes": True, "no": False}.get(
                design.get("healthyVolunteers")
            ),
            notes=design.get("population") or None,
        )
        arm_ids: dict[str, Arm] = {}
        for arm in design.get("arms") or []:
            kind = (arm.get("registeredType") or "").lower()
            kind = {"sham_comparator": "placebo_comparator"}.get(kind, kind)
            added = trial.arms.add(
                arm.get("label") or f"Arm {len(trial.arms) + 1}",
                kind if kind in VOCABULARY["arm_type"] else None,
            )
            arm_ids[arm.get("id")] = added
            for t in arm.get("treatments") or []:
                if not t.get("name"):
                    continue
                identity = t.get("identity")
                added.treatments.add(
                    t["name"],
                    role=t.get("role") if t.get("role") != "active" else "study",
                    timing=t.get("timing") or None,
                    kind=t.get("kind") or None,
                    chembl=t.get("chemblId") or None if identity == "chembl" else None,
                    smiles=t.get("smiles") or None if identity == "smiles" else None,
                    name_only=identity == "name_only",
                    dose=t.get("dose") or None,
                    dose_stated=t.get("doseStated"),
                )
        endpoint_ids: dict[str, Endpoint] = {}
        for e in endpoints:
            analysis = e.get("analysis") or {}
            title = (
                analysis.get("title")
                or e.get("name")
                or f"Endpoint {len(trial.endpoints) + 1}"
            )
            if title.lower() in (n.lower() for n in trial.endpoints.names()):
                title = f"{title} ({len(trial.endpoints) + 1})"
            endpoint = Endpoint(
                trial,
                title,
                role=analysis.get("endpointRole") or None,
                family=e.get("family") or None,
                category=(
                    e.get("kind")
                    if e.get("kind") in VOCABULARY["endpoint_category"]
                    else None
                ),
                effect_measure=analysis.get("effectType") or None,
                hypothesis=analysis.get("hypothesis") or None,
                time_frame=e.get("timeFrame") or None,
                notes=e.get("definition") or None,
            )
            trial.endpoints._items.append(endpoint)
            endpoint_ids[e.get("id")] = endpoint
        for c in design.get("comparisons") or []:
            exp, comp = arm_ids.get(c.get("experimental")), arm_ids.get(
                c.get("comparator")
            )
            chosen = [
                endpoint_ids[i] for i in c.get("endpoints") or [] if i in endpoint_ids
            ]
            if exp and comp and exp is not comp and chosen:
                trial.comparisons._items.append(Comparison(exp, comp, chosen))
        for q in draft.get("queries") or []:
            if q.get("term"):
                trial.safety_queries.add(q["term"], q.get("seriousness") or "serious")
        return trial

    # -- server --------------------------------------------------------------------

    def review(self, client, *, idempotency_key: str | None = None) -> "TrialReview":
        """Ask the server to resolve names and check every analysis. Runs nothing."""
        return TrialReview(
            client.prepare_trial_design(self.to_dict(), idempotency_key), self
        )

    def run(
        self,
        client,
        *,
        idempotency_key: str,
        skip_unavailable: bool = True,
        wait: bool = True,
        timeout: float = 3600,
        poll_interval: float = 5,
    ):
        """Review, submit and (by default) wait for a :class:`~pysynthbio.TrialResult`.

        Analyses the deployed models cannot score are skipped (and named in the
        review) instead of refusing the run, like the VERDICT form. Reuse
        ``idempotency_key`` to recover the same run; use a new one for a new run.
        With ``wait=False`` the submission receipt (``id``, ``status``) is returned.
        """
        design = self.to_dict()
        review = TrialReview(client.prepare_trial_design(design, idempotency_key), self)
        if skip_unavailable:
            skipped = set(review.unavailable)
            # Releases without the full Safety catalog need a query to run Toxicity.
            if "safetyQueries" not in design and any(
                str(i.get("field", "")).startswith("query.") for i in review.issues
            ):
                skipped.add("toxicity")
            remaining = [a for a in design["requestedAnalyses"] if a not in skipped]
            if skipped and remaining:
                design["requestedAnalyses"] = remaining
                review = TrialReview(
                    client.prepare_trial_design(design, idempotency_key), self
                )
        if not review.ok:
            raise TrialReviewError(review)
        receipt = client.run_trial_design(
            design, review.fingerprint, idempotency_key=idempotency_key
        )
        if not wait:
            return receipt
        return client.wait_for_trial(
            receipt["id"], timeout=timeout, poll_interval=poll_interval
        )

    # -- display ---------------------------------------------------------------------

    def __repr__(self) -> str:
        d = self.details
        lines = [
            f"Trial: {d.name}",
            f"  {d.phase} · {self.population.summary}",
            (
                "  Design: "
                + " · ".join(
                    _label(v) for v in (d.allocation, d.blinding, d.purpose) if v
                )
                if any((d.allocation, d.blinding, d.purpose))
                else "  Design: not stated"
            ),
            "Arms:",
            *(f"  {r}" for r in repr(self.arms).splitlines()),
            "Endpoints:",
            *(f"  {r}" for r in repr(self.endpoints).splitlines()),
            "Comparisons:",
            *(f"  {r}" for r in repr(self.comparisons).splitlines()),
            "Analyses: " + ", ".join(self._analyses),
        ]
        return "\n".join(lines)

    def show(self) -> None:
        print(self)

    def _repr_html_(self) -> str:
        d, p = self.details, self.population
        overview = _table(
            ["", ""],
            [
                ("Disease", p.disease),
                ("Phase", d.phase),
                ("Population", p.summary),
                ("Allocation", _label(d.allocation)),
                ("Blinding", _label(d.blinding)),
                ("Purpose", _label(d.purpose)),
                ("Sponsor", _label(d.sponsor_type)),
                ("Start", d.start_date),
                ("Analyses", ", ".join(self._analyses)),
            ],
        )
        section = _section
        return (
            f"<div><h3 style='margin:0 0 6px'>{_escape(d.name)}</h3>{overview}"
            + section("Arms", self.arms._repr_html_())
            + section("Endpoints", self.endpoints._repr_html_())
            + section("Comparisons", self.comparisons._repr_html_())
            + (
                section("Safety queries", self.safety_queries._repr_html_())
                if len(self.safety_queries)
                else ""
            )
            + "</div>"
        )


# ---- Review ------------------------------------------------------------------------


class TrialReviewError(RuntimeError):
    """The server found issues; nothing was submitted. ``.review`` has the details."""

    def __init__(self, review: "TrialReview"):
        self.review = review
        problems = (
            "; ".join(i.get("message", "") for i in review.issues)
            or "no runnable analysis"
        )
        super().__init__(f"Trial design not runnable: {problems}")


class TrialReview:
    """What the server resolved for a design and whether each analysis can run."""

    def __init__(self, response: dict[str, Any], trial: Trial | None = None):
        self.raw = response
        self.trial = trial
        self.issues: list[dict[str, Any]] = response.get("issues") or []
        self.entities: list[dict[str, Any]] = response.get("entities") or []
        self.assumptions: list[str] = response.get("assumptions") or []
        self.availability: list[dict[str, Any]] = response.get("availability") or []
        self.fingerprint: str | None = response.get("reviewFingerprint")
        self.normalized = response.get("normalized")

    @property
    def ok(self) -> bool:
        """True when the design is valid and at least one analysis can run."""
        return bool(self.fingerprint) and not self.issues and bool(self.available)

    @property
    def available(self) -> list[str]:
        return [
            a["analysis"]
            for a in self.availability
            if a.get("outcome") != "unavailable"
        ]

    @property
    def unavailable(self) -> list[str]:
        return [
            a["analysis"]
            for a in self.availability
            if a.get("outcome") == "unavailable"
        ]

    def to_frame(self):
        """Analysis availability as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                (a["analysis"], a.get("outcome"), " ".join(a.get("reasons") or []))
                for a in self.availability
            ],
            columns=["analysis", "outcome", "reasons"],
        )

    def __repr__(self) -> str:
        lines = [f"Review: {'ready to run' if self.ok else 'needs attention'}"]
        lines += [
            f"  issue: {i.get('field', '')} {i.get('message')}".rstrip()
            for i in self.issues
        ]
        lines += [
            f"  {a['analysis']}: {a.get('outcome')}"
            + (f" — {' '.join(a['reasons'])}" if a.get("reasons") else "")
            for a in self.availability
        ]
        unresolved = [e for e in self.entities if e.get("status") != "resolved"]
        lines += [
            f"  unresolved: {e.get('name')} ({e.get('status')})" for e in unresolved
        ]
        if self.assumptions:
            lines.append(f"  {len(self.assumptions)} model notes: see .assumptions")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        parts = [f"<h4>Review: {'ready to run' if self.ok else 'needs attention'}</h4>"]
        if self.issues:
            parts.append(
                _table(
                    ["Field", "Issue"],
                    [(i.get("field"), i.get("message")) for i in self.issues],
                )
            )
        parts.append(
            _table(
                ["Analysis", "Outcome", "Reasons"],
                [
                    (a["analysis"], a.get("outcome"), " ".join(a.get("reasons") or []))
                    for a in self.availability
                ],
            )
        )
        if self.entities:
            parts.append(
                _table(
                    ["Name", "Status", "Resolved to"],
                    [
                        (
                            e.get("name"),
                            e.get("status"),
                            e.get("identity") or e.get("chemblId") or e.get("mondoId"),
                        )
                        for e in self.entities
                    ],
                )
            )
        if self.assumptions:
            parts.append(
                f"<details><summary>{len(self.assumptions)} model notes</summary><ul>"
                + "".join(f"<li>{_escape(a)}</li>" for a in self.assumptions)
                + "</ul></details>"
            )
        return "".join(parts)


# ---- Sets of variants ----------------------------------------------------------------


def _variant_key(prefix: str, label: str) -> str:
    """A valid, stable idempotency key for one variant (letters, digits, ``_.:-``)."""
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", label).strip("-")
    key = f"{prefix}:{slug}"
    if len(key) <= 180:
        return key
    return f"{prefix[:120]}:{hashlib.sha256(label.encode()).hexdigest()[:40]}"


class TrialSet:
    """Named trial variants to review and run together (see :meth:`Trial.vary`)."""

    def __init__(self, trials: dict[str, Trial] | None = None):
        self._trials: dict[str, Trial] = dict(trials or {})

    def add(self, label: str, trial: Trial) -> "TrialSet":
        if label in self._trials:
            raise TrialDesignError(f"A variant named {label!r} already exists")
        self._trials[label] = trial
        return self

    def __getitem__(self, label: str) -> Trial:
        return self._trials[label]

    def __iter__(self):
        return iter(self._trials.items())

    def __len__(self) -> int:
        return len(self._trials)

    def labels(self) -> list[str]:
        return list(self._trials)

    def review(self, client):
        """Review every variant (no runs); a DataFrame of readiness per variant."""
        import pandas as pd

        rows = []
        for label, trial in self._trials.items():
            review = trial.review(client)
            rows.append(
                (
                    label,
                    review.ok,
                    ", ".join(review.available),
                    ", ".join(review.unavailable),
                    "; ".join(i.get("message", "") for i in review.issues),
                )
            )
        return pd.DataFrame(
            rows, columns=["variant", "ready", "will run", "skipped", "issues"]
        )

    def run(
        self,
        client,
        *,
        idempotency_prefix: str,
        timeout: float = 3600,
        poll_interval: float = 5,
    ):
        """Submit every variant, then wait for all. Returns a :class:`TrialSetResult`.

        Each variant's key is ``"{idempotency_prefix}:{label}"``; rerun with the same
        prefix to recover the same runs.
        """
        receipts = {
            label: trial.run(
                client,
                idempotency_key=_variant_key(idempotency_prefix, label),
                wait=False,
            )
            for label, trial in self._trials.items()
        }
        results = {
            label: client.wait_for_trial(
                receipt["id"], timeout=timeout, poll_interval=poll_interval
            )
            for label, receipt in receipts.items()
        }
        return TrialSetResult(results)

    def __repr__(self) -> str:
        return "\n".join(
            f"{label}: {trial.population.summary}"
            for label, trial in self._trials.items()
        )

    def _repr_html_(self) -> str:
        return _table(
            ["Variant", "Population", "Phase"],
            [
                (label, t.population.summary, t.details.phase)
                for label, t in self._trials.items()
            ],
        )


class TrialSetResult:
    """Results of a :class:`TrialSet`, side by side."""

    def __init__(self, results: dict[str, Any]):
        self.results = results

    def __getitem__(self, label: str):
        return self.results[label]

    @property
    def table(self):
        """One row per variant: status and headline estimates (DataFrame)."""
        import pandas as pd

        rows = []
        for label, result in self.results.items():
            rows.append({"variant": label, **result.headlines()})
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return self.table.to_string(index=False)

    def _repr_html_(self) -> str:
        return self.table.to_html(index=False, na_rep="—")
