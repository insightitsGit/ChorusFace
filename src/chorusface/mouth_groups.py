"""Named mouth cell groups — lips, teeth, cavity — with retargetable membership.

Vision
------
Mouth motion is not one blob. It is **named groups of cells** that the word
clock can drive differently:

===============  ==============================================
Group            Role
===============  ==============================================
``upper_lip``    Outer upper lip flesh — opens up / presses down
``lower_lip``    Outer lower lip flesh — opens down / presses up
``lip_corners``  Commissures — EE widens, OU rounds
``teeth``        Visible dental band (retargetable; often cavity rim)
``cavity``       Deep oral interior — soft follow, never identity
===============  ==============================================

Why groups are first-class
--------------------------
Teeth and lips will need different timing and different cells after the next
capture pass. Membership is a **table you can rewrite** without changing the
viseme clock: ``retarget_group("teeth", cells=...)`` or reload from a part
atlas. Recipes per viseme say how hard each group moves (open / width / round /
close), so AH can reveal teeth while PP kills the teeth group.

Detection today is geometric on ``mouth_unlocked``. When a part atlas is
present, prefer atlas labels (upper/lower lip, cavity) and split cavity into
``teeth`` (near the lip line) vs deep ``cavity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Iterable, Mapping, Sequence

from chorusface.parts import (
    PART_LOWER_LIP,
    PART_MOUTH_CAVITY,
    PART_UPPER_LIP,
)
from chorusface.speech import canonical_viseme

if TYPE_CHECKING:
    from chorusface.mouth_cell_plan import DetectedCell
else:
    # Runtime import kept local to avoid cycles with mouth_cell_plan.
    DetectedCell = object  # type: ignore[misc,assignment]

#: Stable group ids used in recipes and /cells snapshots.
MOUTH_GROUP_NAMES: Final[tuple[str, ...]] = (
    "upper_lip",
    "lower_lip",
    "lip_corners",
    "teeth",
    "cavity",
)

#: |side| above this on a lip cell → also tagged as a corner articulator.
CORNER_SIDE: Final = 0.55
#: Cavity cells with |lip| above this (near parting line) default into teeth.
TEETH_LIP_BAND: Final = 0.22


@dataclass(frozen=True, slots=True)
class GroupMotion:
    """How hard one group should follow the active viseme flow."""

    open_scale: float = 1.0
    width_scale: float = 1.0
    round_scale: float = 1.0
    #: Extra inward press when the mouth closes (PP / CLOSED).
    close_scale: float = 0.0
    #: When False the group is skipped for that viseme (e.g. hide teeth on PP).
    active: bool = True

    def scaled_flow(
        self, open_n: float, width_n: float, round_n: float
    ) -> tuple[float, float, float, float]:
        if not self.active:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(open_n) * float(self.open_scale),
            float(width_n) * float(self.width_scale),
            float(round_n) * float(self.round_scale),
            float(self.close_scale),
        )


def _gm(
    open_scale: float = 1.0,
    width_scale: float = 1.0,
    round_scale: float = 1.0,
    close_scale: float = 0.0,
    *,
    active: bool = True,
) -> GroupMotion:
    return GroupMotion(
        open_scale=open_scale,
        width_scale=width_scale,
        round_scale=round_scale,
        close_scale=close_scale,
        active=active,
    )


#: Default word → group recipe. Override per character with ``set_recipe``.
DEFAULT_VISEME_GROUP_RECIPES: Final[dict[str, dict[str, GroupMotion]]] = {
    "REST": {
        "upper_lip": _gm(0.0, 0.0, 0.0),
        "lower_lip": _gm(0.0, 0.0, 0.0),
        "lip_corners": _gm(0.0, 0.0, 0.0),
        "teeth": _gm(active=False),
        "cavity": _gm(active=False),
    },
    "CLOSED": {
        "upper_lip": _gm(0.0, 0.0, 0.1, close_scale=1.0),
        "lower_lip": _gm(0.0, 0.0, 0.1, close_scale=1.0),
        "lip_corners": _gm(0.0, 0.0, 0.2, close_scale=0.6),
        "teeth": _gm(active=False),
        "cavity": _gm(active=False),
    },
    "PP": {
        "upper_lip": _gm(0.0, 0.0, 0.15, close_scale=1.2),
        "lower_lip": _gm(0.0, 0.0, 0.15, close_scale=1.2),
        "lip_corners": _gm(0.0, 0.0, 0.25, close_scale=0.8),
        "teeth": _gm(active=False),
        "cavity": _gm(active=False),
    },
    "MM": {
        "upper_lip": _gm(0.0, 0.1, 0.1, close_scale=0.9),
        "lower_lip": _gm(0.0, 0.1, 0.1, close_scale=0.9),
        "lip_corners": _gm(0.0, 0.15, 0.1, close_scale=0.5),
        "teeth": _gm(active=False),
        "cavity": _gm(active=False),
    },
    "AH": {
        "upper_lip": _gm(1.0, 0.1, 0.0),
        "lower_lip": _gm(1.25, 0.1, 0.0),
        "lip_corners": _gm(0.35, 0.2, 0.0),
        "teeth": _gm(0.55, 0.05, 0.0),
        "cavity": _gm(0.25, 0.0, 0.0),
    },
    "AA": {
        "upper_lip": _gm(0.95, 0.15, 0.0),
        "lower_lip": _gm(1.1, 0.15, 0.0),
        "lip_corners": _gm(0.4, 0.25, 0.0),
        "teeth": _gm(0.5, 0.08, 0.0),
        "cavity": _gm(0.22, 0.0, 0.0),
    },
    "OH": {
        "upper_lip": _gm(0.7, 0.05, 0.45),
        "lower_lip": _gm(0.85, 0.05, 0.45),
        "lip_corners": _gm(0.3, 0.05, 0.9),
        "teeth": _gm(0.35, 0.0, 0.2),
        "cavity": _gm(0.2, 0.0, 0.15),
    },
    "OU": {
        "upper_lip": _gm(0.5, 0.0, 0.7),
        "lower_lip": _gm(0.55, 0.0, 0.7),
        "lip_corners": _gm(0.25, 0.0, 1.1),
        "teeth": _gm(0.25, 0.0, 0.35),
        "cavity": _gm(0.15, 0.0, 0.25),
    },
    "EE": {
        "upper_lip": _gm(0.35, 0.7, 0.0),
        "lower_lip": _gm(0.3, 0.7, 0.0),
        "lip_corners": _gm(0.2, 1.2, 0.0),
        "teeth": _gm(0.4, 0.35, 0.0),
        "cavity": _gm(0.1, 0.1, 0.0),
    },
    "EH": {
        "upper_lip": _gm(0.55, 0.35, 0.0),
        "lower_lip": _gm(0.6, 0.35, 0.0),
        "lip_corners": _gm(0.3, 0.55, 0.0),
        "teeth": _gm(0.35, 0.2, 0.0),
        "cavity": _gm(0.12, 0.05, 0.0),
    },
    "IH": {
        "upper_lip": _gm(0.4, 0.45, 0.0),
        "lower_lip": _gm(0.35, 0.45, 0.0),
        "lip_corners": _gm(0.25, 0.7, 0.0),
        "teeth": _gm(0.3, 0.25, 0.0),
        "cavity": _gm(0.1, 0.08, 0.0),
    },
    "FF": {
        "upper_lip": _gm(0.15, 0.15, 0.05, close_scale=0.4),
        "lower_lip": _gm(0.25, 0.1, 0.05),
        "lip_corners": _gm(0.1, 0.2, 0.05),
        "teeth": _gm(0.2, 0.05, 0.0),
        "cavity": _gm(active=False),
    },
    "SS": {
        "upper_lip": _gm(0.2, 0.3, 0.0),
        "lower_lip": _gm(0.15, 0.3, 0.0),
        "lip_corners": _gm(0.15, 0.45, 0.0),
        "teeth": _gm(0.45, 0.2, 0.0),
        "cavity": _gm(0.08, 0.05, 0.0),
    },
    "TH": {
        "upper_lip": _gm(0.25, 0.1, 0.0),
        "lower_lip": _gm(0.35, 0.1, 0.0),
        "lip_corners": _gm(0.15, 0.15, 0.0),
        "teeth": _gm(0.55, 0.05, 0.0),
        "cavity": _gm(0.1, 0.0, 0.0),
    },
}


def _default_recipe(phoneme: str) -> dict[str, GroupMotion]:
    key = canonical_viseme(phoneme)
    if key in DEFAULT_VISEME_GROUP_RECIPES:
        return dict(DEFAULT_VISEME_GROUP_RECIPES[key])
    # Unknown: gentle open on lips only.
    return {
        "upper_lip": _gm(0.4, 0.15, 0.0),
        "lower_lip": _gm(0.45, 0.15, 0.0),
        "lip_corners": _gm(0.25, 0.25, 0.0),
        "teeth": _gm(0.2, 0.05, 0.0),
        "cavity": _gm(0.08, 0.0, 0.0),
    }


@dataclass(slots=True)
class GroupedCell:
    """Detected cell plus its primary mouth group (and optional corner tag)."""

    cell: DetectedCell
    group: str
    is_corner: bool = False


@dataclass(slots=True)
class MouthGroupMap:
    """Membership tables for every mouth group — rewrite to retarget teeth/lips."""

    cells: list[GroupedCell] = field(default_factory=list)
    #: group → list of indices into ``cells``.
    members: dict[str, list[int]] = field(default_factory=dict)
    source: str = "geometry"

    def counts(self) -> dict[str, int]:
        return {name: len(self.members.get(name, ())) for name in MOUTH_GROUP_NAMES}

    def cells_for(self, group: str) -> list[DetectedCell]:
        return [self.cells[i].cell for i in self.members.get(group, ())]

    def retarget(
        self,
        group: str,
        coordinates: Sequence[tuple[int, int]],
        *,
        as_corner: bool = False,
    ) -> None:
        """Replace membership of ``group`` with an explicit cell list.

        Other groups keep cells that were not claimed. Claimed cells are moved
        into ``group`` so teeth vs lips can be reauthored without a reseed.
        """
        if group not in MOUTH_GROUP_NAMES:
            raise KeyError(f"unknown mouth group {group!r}")
        claim = {(int(x), int(y)) for x, y in coordinates}
        # Drop claimed coords from every group.
        for name in MOUTH_GROUP_NAMES:
            kept: list[int] = []
            for idx in self.members.get(name, []):
                cell = self.cells[idx].cell
                if (cell.x, cell.y) in claim:
                    continue
                kept.append(idx)
            self.members[name] = kept
        # Ensure each claimed cell exists in ``cells``.
        index_by_xy = {
            (g.cell.x, g.cell.y): i for i, g in enumerate(self.cells)
        }
        new_indices: list[int] = []
        for x, y in claim:
            key = (int(x), int(y))
            if key in index_by_xy:
                idx = index_by_xy[key]
                self.cells[idx].group = group
                self.cells[idx].is_corner = bool(as_corner) or self.cells[idx].is_corner
                new_indices.append(idx)
            else:
                # Minimal detection stub for a retargeted cell outside the map.
                from chorusface.mouth_cell_plan import DetectedCell as _DetectedCell

                stub = _DetectedCell(
                    x=key[0], y=key[1], side=0.0, lip=0.0, radial=0.5
                )
                self.cells.append(
                    GroupedCell(cell=stub, group=group, is_corner=bool(as_corner))
                )
                new_indices.append(len(self.cells) - 1)
        self.members[group] = new_indices
        self.source = f"retarget:{group}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "groups": self.counts(),
            "vision": {
                "upper_lip": "outer upper lip flesh",
                "lower_lip": "outer lower lip flesh",
                "lip_corners": "commissures — width / round",
                "teeth": "dental band (retargetable)",
                "cavity": "deep oral interior",
            },
        }


def assign_groups_geometric(detected: Sequence[DetectedCell]) -> MouthGroupMap:
    """Split detected mouth cells into lips / corners / teeth / cavity."""
    from chorusface.mouth_cell_plan import DetectedCell as _DetectedCell

    grouped: list[GroupedCell] = []
    members: dict[str, list[int]] = {name: [] for name in MOUTH_GROUP_NAMES}
    for cell in detected:
        assert isinstance(cell, _DetectedCell)
        is_corner = abs(cell.side) >= CORNER_SIDE and abs(cell.lip) >= 0.08
        # Interior (low |lip|) → cavity / teeth; rim → lips.
        if abs(cell.lip) < TEETH_LIP_BAND:
            if abs(cell.lip) < 0.08 and cell.radial < 0.40:
                group = "cavity"
            else:
                group = "teeth"
        elif cell.lip >= 0.0:
            group = "upper_lip"
        else:
            group = "lower_lip"
        idx = len(grouped)
        grouped.append(GroupedCell(cell=cell, group=group, is_corner=is_corner))
        members[group].append(idx)
        if is_corner and group in {"upper_lip", "lower_lip"}:
            # Corners are also listed under lip_corners for EE/OU recipes.
            members["lip_corners"].append(idx)
    return MouthGroupMap(cells=grouped, members=members, source="geometry")


def assign_groups_from_part_ids(
    detected: Sequence[DetectedCell],
    part_id_at: Mapping[tuple[int, int], int],
) -> MouthGroupMap:
    """Prefer atlas part labels; split cavity into teeth vs deep cavity."""
    from chorusface.mouth_cell_plan import DetectedCell as _DetectedCell

    grouped: list[GroupedCell] = []
    members: dict[str, list[int]] = {name: [] for name in MOUTH_GROUP_NAMES}
    for cell in detected:
        assert isinstance(cell, _DetectedCell)
        code = int(part_id_at.get((cell.x, cell.y), 0))
        is_corner = abs(cell.side) >= CORNER_SIDE
        if code == PART_UPPER_LIP:
            group = "upper_lip"
        elif code == PART_LOWER_LIP:
            group = "lower_lip"
        elif code == PART_MOUTH_CAVITY:
            group = "teeth" if abs(cell.lip) >= TEETH_LIP_BAND else "cavity"
        else:
            # Fall back to geometry for unlocked soft cells without a part label.
            if abs(cell.lip) < TEETH_LIP_BAND:
                group = "cavity" if abs(cell.lip) < 0.08 else "teeth"
            elif cell.lip >= 0.0:
                group = "upper_lip"
            else:
                group = "lower_lip"
        idx = len(grouped)
        grouped.append(GroupedCell(cell=cell, group=group, is_corner=is_corner))
        members[group].append(idx)
        if is_corner and group in {"upper_lip", "lower_lip"}:
            members["lip_corners"].append(idx)
    return MouthGroupMap(cells=grouped, members=members, source="part_atlas")


@dataclass(slots=True)
class MouthGroupPlan:
    """Group-aware motion plan: viseme recipes × retargetable cell sets."""

    groups: MouthGroupMap
    recipes: dict[str, dict[str, GroupMotion]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.recipes:
            self.recipes = {
                key: dict(value) for key, value in DEFAULT_VISEME_GROUP_RECIPES.items()
            }

    def set_recipe(
        self, phoneme: str, group: str, motion: GroupMotion
    ) -> None:
        key = canonical_viseme(phoneme)
        table = self.recipes.setdefault(key, _default_recipe(key))
        if group not in MOUTH_GROUP_NAMES:
            raise KeyError(f"unknown mouth group {group!r}")
        table[group] = motion

    def recipe_for(self, phoneme: str) -> dict[str, GroupMotion]:
        key = canonical_viseme(phoneme)
        return self.recipes.get(key) or _default_recipe(key)

    def retarget_group(
        self,
        group: str,
        coordinates: Sequence[tuple[int, int]],
        *,
        as_corner: bool = False,
    ) -> None:
        self.groups.retarget(group, coordinates, as_corner=as_corner)

    def iter_active_cells(
        self, phoneme: str
    ) -> Iterable[tuple[GroupedCell, GroupMotion, str]]:
        """Yield (grouped cell, motion, group_name) for active groups."""
        recipe = self.recipe_for(phoneme)
        # Prefer dedicated corner recipe when a cell is listed under lip_corners.
        corner_motion = recipe.get("lip_corners")
        seen: set[int] = set()
        if corner_motion is not None and corner_motion.active:
            for idx in self.groups.members.get("lip_corners", ()):
                seen.add(idx)
                yield self.groups.cells[idx], corner_motion, "lip_corners"
        for group_name in MOUTH_GROUP_NAMES:
            if group_name == "lip_corners":
                continue
            motion = recipe.get(group_name) or _gm(active=False)
            if not motion.active:
                continue
            for idx in self.groups.members.get(group_name, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                yield self.groups.cells[idx], motion, group_name

    def snapshot(self) -> dict[str, Any]:
        ah = {
            name: {
                "open_scale": m.open_scale,
                "width_scale": m.width_scale,
                "round_scale": m.round_scale,
                "close_scale": m.close_scale,
                "active": m.active,
            }
            for name, m in self.recipe_for("AH").items()
        }
        return {
            "groups": self.groups.snapshot(),
            "recipe_example_AH": ah,
            "group_names": list(MOUTH_GROUP_NAMES),
        }


def build_mouth_group_plan(
    detected: Sequence[Any],
    *,
    part_id_at: Mapping[tuple[int, int], int] | None = None,
) -> MouthGroupPlan:
    if part_id_at:
        groups = assign_groups_from_part_ids(detected, part_id_at)
    else:
        groups = assign_groups_geometric(detected)
    return MouthGroupPlan(groups=groups)


__all__ = [
    "CORNER_SIDE",
    "DEFAULT_VISEME_GROUP_RECIPES",
    "GroupMotion",
    "GroupedCell",
    "MOUTH_GROUP_NAMES",
    "MouthGroupMap",
    "MouthGroupPlan",
    "TEETH_LIP_BAND",
    "assign_groups_from_part_ids",
    "assign_groups_geometric",
    "build_mouth_group_plan",
]
