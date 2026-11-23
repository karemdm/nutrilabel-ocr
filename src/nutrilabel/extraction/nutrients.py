"""Nutrient catalogue and multilingual label lexicon.

Patterns run on *normalised* text (lower-case, accents stripped, ``ß`` -> ``ss``),
see :func:`normalize_text`. Order matters: more specific nutrients are matched
first and consume their span, so "saturated fat" is never also read as "fat",
and "of which sugars" is never read as "carbohydrates".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

BASES = ("100g", "serving")
IGNORED = "_ignored"


@dataclass(frozen=True)
class Nutrient:
    """One nutrient: its output name, value kind and the regex that finds its label."""

    name: str
    kind: str  # "energy_kj" | "energy_kcal" | "energy" | "mass"
    pattern: re.Pattern[str]


def _rx(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{a})" for a in alternatives))


# Ordered from most to least specific (see module docstring).
NUTRIENTS: tuple[Nutrient, ...] = (
    # Rows we do not report but must recognise, so their values are not given to a
    # neighbouring nutrient (e.g. "monounsaturated", "trans fat", "starch", "calcium").
    Nutrient(
        IGNORED,
        "mass",
        _rx(
            r"\w*(?:in|un)satur\w*(?:\s+fat\w*)?",
            r"\bmono\w*",
            r"\bpoly\w*",
            r"(?:\bgordura\w*\s+|\bgrasas?\s+)?\btrans\b(?:\s+fat\w*)?",
            r"\bomega\w*",
            r"\bstarch\w*",
            r"\bamidon\w*",
            r"\balmidon\w*",
            r"\bamido\b",
            r"\bstarke\b",
            r"\bzetmeel\b",
            r"\bcholest\w*",
            r"\bcolesterol\w*",
            r"\bcalci?um\b",
            r"\bcalcio\b",
            r"\biron\b",
            r"\bfer\b",
            r"\bpotassium\b",
            r"\bvitamin\w*",
            r"\blactose\b",
            r"\balcohol\b",
            r"\bpolyol\w*",
            r"\berythritol\b",
            r"\badded sugars?\b",
            r"\bincludes?\b.*added",
        ),
    ),
    Nutrient(
        "saturated_fat",
        "mass",
        _rx(
            # saturated, saturates, satures, saturadas, saturi, saturados - with the
            # surrounding words so that "acides gras" is not read as "fat".
            r"(?:(?:acid[eiao]s?|fatty acids?)\s+)?(?:gras\w*\s+|gordura\w*\s+)?satur\w*"
            r"(?:\s+(?:fat\w*|fatty acids|grasas?|gras\b))?",
            r"gesatt\w*(?:\s+fettsaur\w*)?",  # gesattigte Fettsauren (de)
            r"verzadig\w*(?:\s+vetzur\w*)?",  # verzadigde vetzuren (nl)
            r"nasycen\w*",  # nasycene (cs/pl)
            r"\bzasic\w*",  # zasicene (hr)
            r"\bnasic\w*",  # nasicene (sl)
            r"telitett\w*",  # (hu)
            r"\bmattede\b",  # (da)
            r"mattat\w*",  # mattatut (fi)
            r"\bsatd\b",
        ),
    ),
    Nutrient(
        "sugars",
        "mass",
        _rx(
            r"\bsugars?\b",
            r"\bsucres?\b",
            r"azucar\w*",
            r"zuccher\w*",
            r"\bzucker\b",
            r"suiker\w*",
            r"acucar\w*",
            r"\bcukr\w*",
            r"\bcukier\w*",
            r"\bcukor\w*",  # (hu)
            r"\bsecer\w*",  # secera (hr/sr)
            r"sladkor\w*",  # (sl)
            r"\bsukker\w*",  # (da/no)
            r"\bsokeri\w*",  # (fi)
            r"\bzahar\w*",  # zaharuri (ro)
            r"\bsocker(arter)?\b",
        ),
    ),
    Nutrient(
        "fiber",
        "mass",
        _rx(
            r"\bfib(?:re|er|ra|res|ers|ras)\b",
            r"ballaststoff\w*",
            r"\bvezels?\b",
            r"vlaknin\w*",
            r"blonnik\w*",
            r"\bvlakn\w*",  # vlakna, vlaknine (hr/sl)
            r"\brost\b",  # (hu)
            r"\bkuitu\w*",  # (fi)
            r"kostfiber",
        ),
    ),
    Nutrient(
        "carbohydrates",
        "mass",
        _rx(
            r"carbo\w*",  # carbohydrate(s), carboidrati, carboidratos, carb
            r"\bcarbs?\b",
            r"glucid\w*",
            r"kohlenhydr\w*",
            r"koolhydr\w*",
            r"hidrat\w*",  # hidratos de carbono (es/pt), ugljikohidrati (hr), ogljikovi hidrati (sl)
            r"sacharid\w*",
            r"uhlohydr\w*",
            r"weglowodan\w*",
            r"szenhidrat\w*",  # (hu)
            r"kolhydrat\w*",  # (sv)
            r"kulhydrat\w*",  # (da)
            r"hiilihydraat\w*",  # (fi)
        ),
    ),
    Nutrient(
        "fat",
        "mass",
        _rx(
            r"\bfats?\b",
            r"\bfat/",
            r"mat\w*\.? ?grass\w*",  # matieres grasses, mat. grasses, mat.grasses
            r"\bgrasses\b",  # "matieres" and "grasses" split over two lines
            r"\bgraisses?\b",
            r"\blipid\w*",
            r"\bgrasas?\b",
            r"\bgrassi\b",
            r"\bfett\b",
            r"\bvetten?\b",
            r"\bgordura\w*",
            r"\btuky?\b",
            r"\btluszcz\w*",
            r"\bmasti\b",  # (hr/sr)
            r"\bmascob\w*",  # (sl)
            r"\bzsir\w*",  # (hu)
            r"\bfedt\b",  # (da)
            r"\brasva\w*",  # (fi)
            r"\bgrasimi\b",  # (ro)
            r"\bmatg\b",
        ),
    ),
    Nutrient(
        "proteins",
        "mass",
        _rx(
            r"\bprote?i[nr]\w*",  # protein(s), proteines, proteinas, proteine, OCR "proteires"
            r"eiwei\w*",
            r"eiwit\w*",
            r"bilkovin\w*",
            r"bialk\w*",
            r"\bb[je]?l?jel?ancevin\w*",  # bjelancevine / belancevine (hr/sr)
            r"beljakovin\w*",  # (sl)
            r"\bfeherje\w*",  # (hu)
        ),
    ),
    Nutrient(
        "sodium",
        "mass",
        _rx(r"\bsodi\w*", r"\bnatrium\b"),
    ),
    Nutrient(
        "salt",
        "mass",
        _rx(
            r"\bsalt\b",
            r"\bsel\b",
            r"\bsal\b",
            r"\bsale\b",
            r"\bsalz\b",
            r"\bzout\b",
            r"\bsul\b",
            r"\bsol\b",  # (hr/sl/pl)
            r"\bsuola\b",  # (fi)
            r"\bsare\b",  # (ro)
        ),
    ),
    Nutrient(
        "energy",
        "energy",
        _rx(
            r"\benerg\w*",  # energy, energie, energia, energija, energetico, energetique
            r"\bener[gr]\w*ij?a\b",  # common OCR slips: "Erergija"
            r"brennwert\w*",
            r"\benergiaertek\w*",
            r"\bvalor energ\w*",
            r"\bcalories?\b",
            r"\bcalorias?\b",
        ),
    ),
)

NUTRIENT_NAMES: tuple[str, ...] = tuple(
    n.name for n in NUTRIENTS if n.name not in ("energy", IGNORED)
) + (
    "energy_kj",
    "energy_kcal",
)

#: Every field key the extractor can produce, e.g. ``fat_100g``, ``energy_kcal_serving``.
SUPPORTED_KEYS: tuple[str, ...] = tuple(sorted(f"{n}_{b}" for n in NUTRIENT_NAMES for b in BASES))

# Header cues that tell which column holds "per 100 g" vs "per serving" values.
HEADER_100G = re.compile(r"100\s*(?:g|gr|grs|gramm?e?s?|ml|m1)\b|\b100\s*$|^100(?:g|ml)")
HEADER_SERVING = re.compile(
    r"serving|portion|porcion|porzione|porce|portie|porcao|porcja|\bper\s+\w+\s*\(|"
    r"\bpar\s+part|\bpro\s+portion|\bbar\b|\bbiscuit|\bslice|\bpiece|\bpack\b|\bcup\b"
)
# "% of reference intake / daily value" columns, which are not reported.
HEADER_PCT = re.compile(
    r"%|\b(?:ar|ri|gda|dv|vd|vq|vnr|nrv|ajr|cdr|ir|rda|vrn|ddr|rm|ri\*+|ar\*+)\b|"
    r"\breference intake|\bapports? de reference|\breferenzmenge|\bassunzioni di riferimento|"
    r"\bingestas? de referencia|\bdaily value|\bvaleur quotidienne"
)
# Markers of US/Canada-style "Nutrition Facts" panels (values are per serving).
USDA_MARKERS = re.compile(
    r"nutrition facts|amount per serving|daily value|valeur quotidienne|% dv|servings? per"
)
# Lines that are not part of the nutrition table and should not create anchors.
NOISE_LINE = re.compile(
    r"reference intake|apport de reference|apports de reference|referenzmenge|"
    r"ingredient|ingredien|zutaten|ingredienti|ingrediente|contains|contient|"
    r"percent daily values are based|ri\s*\*|\*\s*ri\b"
)


def normalize_text(text: str) -> str:
    """Lower-case, strip accents and unify a few look-alike characters."""
    text = text.replace("ß", "ss").replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def nutrient_kind(name: str) -> str:
    """Return the kind (``mass``, ``energy_kj`` or ``energy_kcal``) of a nutrient name; ``KeyError`` if unknown."""
    for n in NUTRIENTS:
        if n.name == name:
            return n.kind
    if name in ("energy_kj", "energy_kcal"):
        return name
    raise KeyError(name)
