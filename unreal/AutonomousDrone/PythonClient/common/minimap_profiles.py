"""Persistent minimap profiles and automatic AirSim scene matching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


_DYNAMIC_NAME_PARTS = (
    "simpleflight",
    "vehiclepawn",
    "dronepawn",
    "externalcamera",
    "cameradirector",
    "enemydrone",
    "ainormalpeople",
    "humantarget",
    "flockcharacter",
    "boidcharacter",
    "birdtarget",
    "crow",
    # Engine/runtime actors are nearly identical in every BlenderGIS level and
    # therefore must not be used as map fingerprints.
    "gameplaydebugger",
    "brush",
    "simhud",
    "playerstart",
    "playercontroller",
    "skylight",
    "airsimgamemode",
    "pipcamera",
    "volumetriccloud",
    "playercameramanager",
    "exponentialheightfog",
    "playerstate",
    "scenecapture2d",
    "skyatmosphere",
    "worldsettings",
    "weatheractor",
    "defaultphysicsvolume",
    "particleeventmanager",
    "menuactor",
    "navdata",
    "gamenetworkmanager",
    "droneruntime",
    "gamestate",
    "gamesession",
    "chaosdebugdrawactor",
    "simmodeworld",
    "directionallight",
    "staticmeshactor",
)
_PIE_PREFIX = re.compile(r"uedpie_\d+_", re.IGNORECASE)
_INSTANCE_SUFFIX = re.compile(r"(?:_c)?_\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class MinimapProfile:
    """One image-to-NED-coordinate mapping for an Unreal level."""

    profile_id: str
    display_name: str
    image: str
    center_x_m: float
    center_y_m: float
    half_extent_m: float
    scene_signature: str = ""
    scene_anchors: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict) -> "MinimapProfile":
        center = value.get("center_xy_m", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("center_xy_m은 [X, Y] 두 값이어야 합니다.")
        half_extent = float(value.get("half_extent_m", 0.0))
        if half_extent <= 0.0:
            raise ValueError("half_extent_m은 0보다 커야 합니다.")
        return cls(
            profile_id=str(value["profile_id"]),
            display_name=str(value.get("display_name", value["profile_id"])),
            image=str(value["image"]),
            center_x_m=float(center[0]),
            center_y_m=float(center[1]),
            half_extent_m=half_extent,
            scene_signature=str(value.get("scene_signature", "")),
            scene_anchors=tuple(str(item) for item in value.get("scene_anchors", [])),
        )

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "image": self.image,
            "center_xy_m": [self.center_x_m, self.center_y_m],
            "half_extent_m": self.half_extent_m,
            "scene_signature": self.scene_signature,
            "scene_anchors": list(self.scene_anchors),
        }


def default_profile() -> MinimapProfile:
    return MinimapProfile(
        profile_id="airbase",
        display_name="AirBase (기본 맵)",
        image="assets/Minimap_AirBase.PNG",
        center_x_m=53.31,
        center_y_m=159.39,
        half_extent_m=700.0,
    )


def load_minimap_profiles(path: Path) -> list[MinimapProfile]:
    """Load a registry, falling back to the built-in AirBase mapping."""
    path = Path(path)
    if not path.exists():
        return [default_profile()]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("profiles", []) if isinstance(payload, dict) else payload
        profiles = [MinimapProfile.from_dict(row) for row in rows]
    except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError):
        return [default_profile()]
    return profiles or [default_profile()]


def save_minimap_profiles(path: Path, profiles: Iterable[MinimapProfile]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "profiles": [profile.to_dict() for profile in profiles],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def upsert_minimap_profile(
    profiles: Iterable[MinimapProfile],
    profile: MinimapProfile,
) -> list[MinimapProfile]:
    result = list(profiles)
    for index, current in enumerate(result):
        if current.profile_id.casefold() == profile.profile_id.casefold():
            result[index] = profile
            return result
    result.append(profile)
    return result


def profile_id_from_name(display_name: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_-]+", "_", display_name).strip("_-").lower()
    if candidate:
        return candidate[:48]
    digest = hashlib.sha1(display_name.encode("utf-8")).hexdigest()[:8]
    return f"map_{digest}"


def unique_profile_id(
    profiles: Iterable[MinimapProfile],
    display_name: str,
) -> str:
    """Return an unused ID so a new level can never overwrite another map."""
    existing_ids = {profile.profile_id.casefold() for profile in profiles}
    base = profile_id_from_name(display_name)
    if base.casefold() not in existing_ids:
        return base
    suffix = 2
    while f"{base}_{suffix}".casefold() in existing_ids:
        suffix += 1
    return f"{base}_{suffix}"


def resolve_profile_image(root: Path, profile: MinimapProfile) -> Path:
    image = Path(profile.image)
    return image if image.is_absolute() else Path(root) / image


def _normalized_static_names(scene_object_names: Iterable[str]) -> list[str]:
    names: set[str] = set()
    for raw_name in scene_object_names:
        name = _PIE_PREFIX.sub("", str(raw_name)).strip().casefold()
        if not name or any(part in name for part in _DYNAMIC_NAME_PARTS):
            continue
        name = _INSTANCE_SUFFIX.sub("", name)
        if name:
            names.add(name)
    return sorted(names)


def build_scene_identity(scene_object_names: Iterable[str]) -> dict[str, object]:
    """Create a stable signature and a small fallback anchor set."""
    names = _normalized_static_names(scene_object_names)
    joined = "\n".join(names).encode("utf-8")
    signature = hashlib.sha256(joined).hexdigest() if names else ""
    # Hash ranking avoids depending on alphabetical prefixes shared by maps.
    ranked = sorted(
        names,
        key=lambda item: hashlib.sha1(item.encode("utf-8")).hexdigest(),
    )
    return {
        "signature": signature,
        "anchors": ranked[:64],
        "object_count": len(names),
    }


def with_scene_identity(
    profile: MinimapProfile,
    identity: dict[str, object],
) -> MinimapProfile:
    return replace(
        profile,
        scene_signature=str(identity.get("signature", "")),
        scene_anchors=tuple(str(item) for item in identity.get("anchors", [])),
    )


def match_minimap_profile(
    scene_object_names: Iterable[str],
    profiles: Iterable[MinimapProfile],
) -> tuple[MinimapProfile | None, str, float]:
    """Match by optional MAPID marker, exact signature, then anchor overlap."""
    raw_names = [str(item) for item in scene_object_names]
    profile_list = list(profiles)
    by_id = {profile.profile_id.casefold(): profile for profile in profile_list}
    for raw_name in raw_names:
        lowered = raw_name.casefold()
        for profile_id, profile in by_id.items():
            marker = re.compile(
                rf"mapid[_-]{re.escape(profile_id)}(?:_c)?(?:_\d+)?(?:$|[^a-z0-9_-])"
            )
            if marker.search(lowered):
                return profile, "MAPID 마커", 1.0

    identity = build_scene_identity(raw_names)
    signature = str(identity["signature"])
    object_count = int(identity.get("object_count", 0))
    # Similar BlenderGIS levels often expose only the same generic actor class
    # names. Automatic signature matching is allowed only when enough
    # map-specific names remain; otherwise MAPID or manual selection is safer.
    if signature and object_count >= 12:
        for profile in profile_list:
            if profile.scene_signature and profile.scene_signature == signature:
                return profile, "장면 서명", 1.0

    # Compare stored anchors against every current static name. Newly added
    # actors must not displace older anchors merely because their hash sorts
    # earlier in the current sample.
    current = set(_normalized_static_names(raw_names))
    best: MinimapProfile | None = None
    best_score = 0.0
    for profile in profile_list:
        expected = set(profile.scene_anchors)
        if len(expected) < 12 or object_count < 12:
            continue
        score = len(current & expected) / len(expected)
        if score > best_score:
            best = profile
            best_score = score
    if best is not None and best_score >= 0.72:
        return best, "장면 객체 비교", best_score
    return None, "미등록 장면", best_score
