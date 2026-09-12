"""Print the Python-visible API and relevant properties of BP_GoodSky."""

from __future__ import annotations

import unreal


def main() -> None:
    blueprint_class = unreal.EditorAssetLibrary.load_blueprint_class(
        "/Game/GoodSky/Blueprint/BP_GoodSky"
    )
    if blueprint_class is None:
        raise RuntimeError("BP_GoodSky class was not found")
    default = unreal.get_default_object(blueprint_class)
    for name in sorted(dir(default)):
        lowered = name.lower()
        if any(
            token in lowered
            for token in ("sky", "time", "sun", "light", "refresh", "update")
        ):
            unreal.log(f"GoodSkyAPI: {name}")


if __name__ == "__main__":
    main()
