"""Procedural authoring of the corpus's game events (tattoo-shop-simulator domain -- this project's
own game, `Claude/CLAUDE.md`: "You're Getting A Wizard: A Tattoo Shop Simulator").

Ten event *kinds* (categories of shop event a customer reaction responds to), each carrying a pool
of twenty concrete, hand-authored `detail` phrases. Per kind, the first fifteen details are reserved
for the training split and the last five for the held-out split -- disjoint at the lexical level, not
merely by combination index, so a held-out event is never a paraphrase of a training event (plan Risk
#6: "Curie authors held-out events with novel surface phrasing, not paraphrases of training events").
Event ids are deterministic (`<kind>-<split>-<index>`) so the corpus is reproducible from this module
alone, with no randomness anywhere in event construction.

`required_info` is two facts per event, each a short clause naming what a faithful reaction must
convey; `intended_attitude` is a short descriptor shared across all three race adapters (voice is a
race-level property of the *adapter*, never of the event -- canonical design Sec.2/Sec.4). Every
detail string below is written to be self-contained: it reads as a complete idea when substituted
into a required_info template, with no upstream context needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .events import CorpusEntry, GameEvent

N_HELD_OUT_PER_KIND = 5
N_TRAIN_PER_KIND = 15


@dataclass(frozen=True)
class EventKind:
    slug: str
    required_info_templates: tuple[str, str]
    intended_attitude: str
    details: tuple[str, ...]  # exactly N_TRAIN_PER_KIND + N_HELD_OUT_PER_KIND entries

    def __post_init__(self) -> None:
        expected = N_TRAIN_PER_KIND + N_HELD_OUT_PER_KIND
        if len(self.details) != expected:
            raise ValueError(f"EventKind {self.slug}: expected {expected} details, got {len(self.details)}")
        if len(set(self.details)) != len(self.details):
            raise ValueError(f"EventKind {self.slug}: duplicate detail strings")


_KINDS: tuple[EventKind, ...] = (
    EventKind(
        slug="botched_request",
        required_info_templates=(
            "the shop acknowledges that {detail}",
            "the shop offers a free touch-up appointment",
        ),
        intended_attitude="apologetic and reassuring",
        details=(
            # -- train (15) --
            "the linework on the forearm piece came out shakier than the reference design",
            "the shading on the back panel healed patchy in two spots",
            "the lettering was spaced too tightly and reads cramped",
            "the color saturation faded unevenly after the first week",
            "the outline drifted off the stencil placement by a few millimeters",
            "the symmetry on the matching pair of pieces is slightly off",
            "the fine detail work blew out and lost definition",
            "the black fill has visible patchiness near the elbow crease",
            "the gradient transition looks banded instead of smooth",
            "a small section of the design was accidentally mirrored",
            "the portrait's proportions came out subtly distorted",
            "the script font used was not the one the customer approved",
            "the placement ended up a full inch lower than discussed",
            "the needle depth left the linework looking slightly blown out",
            "the stencil transfer smudged before the artist could trace it",
            # -- held-out (5), disjoint vocabulary --
            "the geometric mandala's spokes are visibly uneven on one side",
            "the watercolor-style splashes bled together into a muddy blur",
            "the negative-space lettering filled in instead of staying open",
            "the customer's chosen quote was misspelled by one letter",
            "the wrap-around design does not line up when the arm bends",
        ),
    ),
    EventKind(
        slug="brag_result",
        required_info_templates=(
            "the shop compliments the finished {detail}",
            "the shop invites the customer to share photos on the shop's wall of fame",
        ),
        intended_attitude="warm and proud",
        details=(
            "full sleeve of overlapping koi and wave work",
            "chest piece built around a family crest",
            "backpiece phoenix that took three sessions to finish",
            "forearm script tribute to the customer's late grandmother",
            "thigh piece of a detailed clockwork owl",
            "ribcage floral bouquet in fine linework",
            "shoulder cap of geometric wolf silhouette",
            "calf piece of a lighthouse in a storm",
            "hand-and-fingers set of tiny botanical stamps",
            "collarbone piece of a minimalist mountain range",
            "half-sleeve blackwork sleeve of forest scenery",
            "ankle piece of a coiled serpent",
            "spine piece running the length of the back",
            "bicep piece of a roaring bear portrait",
            "neckline piece of delicate lace patterning",
            # -- held-out (5) --
            "knee-to-hip leg piece of a cathedral facade",
            "underarm piece of a swarm of dragonflies",
            "sternum piece shaped like a slow hourglass with sand mid-fall",
            "wraparound wrist cuff of interlocking chain links",
            "temple-to-jaw piece of a crescent moon and stars",
        ),
    ),
    EventKind(
        slug="scope_creep",
        required_info_templates=(
            "the shop notes that {detail} was not part of the original booking",
            "the shop offers to schedule a follow-up session for the extra work",
        ),
        intended_attitude="firm but accommodating",
        details=(
            "adding a second full sleeve to match the first, mid-session",
            "extending the backpiece down to cover the entire lower back",
            "swapping the agreed black-and-grey style for full color partway through",
            "asking for three extra names to be added to the memorial piece",
            "requesting the design be doubled in size once the outline was done",
            "wanting an entirely new second tattoo squeezed into the same slot",
            "asking to add a matching piece on the other arm right after",
            "requesting the shading be redone in a completely different technique",
            "wanting the piece extended to wrap around to the back",
            "asking for a friend's walk-in piece to be worked in immediately",
            "requesting an additional four hours of work beyond the booked two",
            "wanting the original small piece turned into a full sleeve on the spot",
            "asking to add a portrait into a design that had none planned",
            "requesting the appointment be extended to cover a second body part",
            "wanting the whole design re-drawn from scratch after the outline started",
            # -- held-out (5) --
            "asking for a matching set of five tiny tattoos instead of the one booked",
            "requesting the piece get mirrored onto the opposite leg right after this session",
            "wanting a completely unrelated cover-up squeezed in before the booked piece begins",
            "asking to fold a second client's design into the same booked slot",
            "requesting the linework redone in a style never discussed at booking",
        ),
    ),
    EventKind(
        slug="underage_or_bad_id",
        required_info_templates=(
            "the shop states that {detail} must be resolved before any work begins",
            "the shop asks for a second form of identification",
        ),
        intended_attitude="stern and procedural",
        details=(
            "the birthdate on the ID does not match the customer's stated age",
            "the ID's photo does not clearly resemble the customer",
            "the ID appears to have a laminate edge that has been tampered with",
            "the ID was issued in a format the shop cannot verify on-site",
            "the customer could not produce any ID when asked at check-in",
            "the ID presented is expired by several years",
            "the ID's listed address does not match the customer's paperwork",
            "the customer's guardian consent form is missing a required signature",
            "the ID font and spacing do not match the state's known template",
            "the customer appears visibly younger than the ID's stated birth year",
            "the ID was reported as a photocopy rather than an original",
            "the barcode on the ID fails the shop's scanner check",
            "the customer's name on the ID does not match the booking name",
            "the ID's hologram seal is missing under the shop's UV light",
            "the customer admits the ID belongs to an older sibling",
            # -- held-out (5) --
            "the passport presented has a visibly re-glued photo page",
            "the customer's student ID has no birthdate printed on it at all",
            "the ID's raised text does not respond to the shop's verification light",
            "the customer's temporary paper ID was issued only yesterday",
            "the ID's signature strip has clearly been rewritten over",
        ),
    ),
    EventKind(
        slug="supply_shortage",
        required_info_templates=(
            "the shop explains that {detail} is currently out of stock",
            "the shop offers to reschedule once the supply order arrives",
        ),
        intended_attitude="apologetic and practical",
        details=(
            "the specific white ink needed for the highlight work",
            "the fine-gauge needles required for the detailed linework",
            "the UV-reactive ink the customer specifically requested",
            "the exact shade of teal the reference design calls for",
            "the numbing cream the customer asked to use beforehand",
            "the extra-large stencil paper needed for the backpiece",
            "the specific grey wash set used for the portrait style",
            "the disposable grips sized for the artist's preferred machine",
            "the gold-tone metallic ink for the accent details",
            "the hypoallergenic ink line for the customer's sensitive skin",
            "the extra-fine liner cartridges the piece requires",
            "the skin-prep wipes the shop normally uses before outlining",
            "the specific green shade needed for the botanical piece",
            "the wide-format bandage wrap for the large finished piece",
            "the touch-up ink batch matched to the customer's healed piece",
            # -- held-out (5) --
            "the glow-in-the-dark ink the customer wanted for the accent lines",
            "the extra-wide shading cartridges the backpiece design calls for",
            "the specific matte-black ink used for the blackout sleeve style",
            "the sterile cover sheets sized for the reclining chair setup",
            "the custom stencil transfer gel the artist prefers for large pieces",
        ),
    ),
    EventKind(
        slug="walk_in_rush",
        required_info_templates=(
            "the shop acknowledges that {detail}",
            "the shop offers the next available appointment slot",
        ),
        intended_attitude="sympathetic but time-pressured",
        details=(
            "the customer's flight leaves in three hours and they want it done now",
            "the customer is on a lunch break with only forty-five minutes free",
            "the customer just won a bet and wants proof inked immediately",
            "the customer's ride is waiting outside with the meter running",
            "the customer has a work event tonight and wants it healed by then",
            "the customer is leaving town tomorrow for a long deployment",
            "the customer's friend is getting married this weekend and it's a group tattoo",
            "the customer wants it done before a video call in twenty minutes",
            "the customer's babysitter is only booked for another hour",
            "the customer is celebrating a milestone that happens today only",
            "the customer's train departs from the station across town shortly",
            "the customer has a hospital visit later and wants this done first",
            "the customer's shift at work starts in under an hour",
            "the customer is meeting someone for a surprise reveal tonight",
            "the customer's rental car needs to be returned within the hour",
            # -- held-out (5) --
            "the customer's ferry to the island leaves at the top of the hour",
            "the customer's court appearance comes up later today and this matters to them now",
            "the customer's band goes on stage in under two hours",
            "the customer's parole check-in comes later and they wanted this done first",
            "the customer's graduation ceremony starts in ninety minutes",
        ),
    ),
    EventKind(
        slug="haggle_price",
        required_info_templates=(
            "the shop responds to the customer's point that {detail}",
            "the shop states the shop's price is fixed for the requested work",
        ),
        intended_attitude="polite but firm",
        details=(
            "a shop across town quoted a lower price for a similar piece",
            "the customer is a regular and expects a loyalty discount",
            "the customer is paying in cash and expects a cash discount",
            "the customer's friend got a better rate last month for less work",
            "the design looks simple to the customer so it should cost less",
            "the customer is bringing in three friends and wants a group rate",
            "the customer found a cheaper flash-sheet option online",
            "the customer says the quoted time seems shorter than the price implies",
            "the customer wants to skip the deposit to save money upfront",
            "the customer offers to trade a favor instead of paying full price",
            "the customer says the ink alone can't cost what's being charged",
            "the customer wants a discount for referring two new clients today",
            "the customer says a student discount should apply here",
            "the customer wants to pay less since they're supplying their own reference art",
            "the customer says the piece is small so the minimum charge feels unfair",
            # -- held-out (5) --
            "the customer mentions it's their birthday and expects a lower total because of it",
            "the customer wants a bundle rate for two unrelated small tattoos today",
            "the customer says the shop's own website listed a lower starting price",
            "the customer offers to leave a five-star review in exchange for a discount",
            "the customer says they'll book a much bigger piece later for a discount now",
        ),
    ),
    EventKind(
        slug="aftercare_worry",
        required_info_templates=(
            "the shop addresses the customer's concern about {detail}",
            "the shop recommends the customer come back in for a look if it does not improve",
        ),
        intended_attitude="calm and reassuring",
        details=(
            "redness that has not gone down after five days",
            "a small area that feels warmer than the skin around it",
            "flaking that looks heavier than what the aftercare sheet described",
            "a faint ring of irritation around the edges of the piece",
            "ink that seems to be fading unusually fast in the first week",
            "a raised, slightly puffy texture over part of the linework",
            "itching that has not settled down after several days",
            "a small cluster of tiny bumps near the shading",
            "the piece looking duller than expected once the peeling finished",
            "a patch that stayed shiny and tight-feeling longer than the rest",
            "mild swelling that appeared after a long day in the sun",
            "a slightly different color developing in one small section",
            "tenderness that returned after the customer went back to the gym",
            "a scab that came off earlier than the aftercare sheet expected",
            "a faint line of dryness tracking along the tattoo's edge",
            # -- held-out (5) --
            "a cool, clammy patch of skin the customer noticed this morning",
            "a faint discoloration spreading just past the tattoo's border",
            "unusual tightness in the skin whenever the customer stretches",
            "a small blister that appeared overnight near the wrist",
            "the piece looking oddly glossy days after it should have matted down",
        ),
    ),
    EventKind(
        slug="referral_praise",
        required_info_templates=(
            "the shop thanks the customer for {detail}",
            "the shop mentions a referral discount for the customer's next visit",
        ),
        intended_attitude="grateful and warm",
        details=(
            "sending their coworker in for a matching piece",
            "posting about the shop on their social media page",
            "recommending the shop to their entire book club",
            "bringing their sibling in for a first tattoo",
            "telling their gym group about the shop's custom work",
            "leaving a glowing review after their last visit",
            "referring a coworker who now wants a full sleeve",
            "bringing three friends in together for a group session",
            "recommending the shop during a local radio interview",
            "telling their tattoo-collector friends about the shop's linework",
            "bringing their partner in after loving their own piece",
            "sharing healed photos that brought in two new walk-ins",
            "recommending the shop to their entire wedding party",
            "telling a stranger at a coffee shop about their piece",
            "bringing their neighbor in for a cover-up consultation",
            # -- held-out (5) --
            "vouching for the shop to a visiting out-of-town friend",
            "recommending the shop in a local community newsletter",
            "bringing their entire hiking club in for matching pieces",
            "telling a tattoo convention booth about the shop's work",
            "referring their landlord, who booked a large backpiece",
        ),
    ),
    EventKind(
        slug="custom_design_pitch",
        required_info_templates=(
            "the shop responds to the customer's idea of {detail}",
            "the shop asks to schedule a design consultation before booking the session",
        ),
        intended_attitude="engaged and curious",
        details=(
            "a full-back mural depicting their family's migration story",
            "a sleeve blending three unrelated mythologies into one scene",
            "a piece that changes meaning depending on the angle it's viewed from",
            "a design built entirely from their grandmother's handwriting",
            "a piece incorporating their own hand-drawn childhood sketches",
            "a design that maps constellations onto their exact birth date and location",
            "a piece built around a recurring dream they've had for years",
            "a design combining their pet's silhouette with a family motto",
            "a piece that tells their recovery story in three linked scenes",
            "a design based on a song's lyrics rendered as visual metaphor",
            "a piece incorporating their late father's fingerprint pattern",
            "a design that reproduces a page from their favorite book verbatim",
            "a piece built from a collage of their travel photographs",
            "a design representing every city they've ever lived in",
            "a piece that incorporates their own handwriting alongside an illustration",
            # -- held-out (5) --
            "a design built from a heart-monitor readout during a meaningful hospital stay",
            "a piece mapping their family tree as a branching tree illustration",
            "a design translating their favorite recipe into symbolic imagery",
            "a piece built around the exact coordinates of their childhood home",
            "a design that reinterprets their old varsity jersey number as art",
        ),
    ),
)


def kind_by_slug(slug: str) -> EventKind:
    for k in _KINDS:
        if k.slug == slug:
            return k
    raise KeyError(slug)


def all_kinds() -> tuple[EventKind, ...]:
    return _KINDS


def build_entries() -> list[CorpusEntry]:
    """Materialize the full corpus: 10 kinds x 15 train + 10 kinds x 5 held-out = 150 train events,
    50 held-out events. The held-out count (50) is the plan's own binding number
    (`RuntimeLoRA_VoiceDemo_Plan.md` Sec.5 item 1); the train count (150) is an engineering-judgment
    proposal within this build's discretion, sized 3x held-out and revisable, same standing as the
    plan's own N=50 (Sec.5's closing line: "All values here ... are proposals within this plan's
    discretion, revisable by Charpy or by measurement")."""

    entries: list[CorpusEntry] = []
    for kind in _KINDS:
        for i, detail in enumerate(kind.details[:N_TRAIN_PER_KIND]):
            entries.append(_make_entry(kind, detail, split="train", index=i))
        for i, detail in enumerate(kind.details[N_TRAIN_PER_KIND:]):
            entries.append(_make_entry(kind, detail, split="held_out", index=i))
    return entries


def _make_entry(kind: EventKind, detail: str, split: str, index: int) -> CorpusEntry:
    event_id = f"{kind.slug}-{split}-{index:02d}"
    event = GameEvent(id=event_id, kind=kind.slug, payload={"detail": detail})
    required_info = tuple(t.format(detail=detail) for t in kind.required_info_templates)
    return CorpusEntry(
        event=event,
        required_info=required_info,
        intended_attitude=kind.intended_attitude,
        split=split,
        validation_reactions=(),
    )
