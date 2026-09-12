"""Print Good SKY material parameters relevant to horizon/background color."""

from __future__ import annotations

import unreal


MATERIALS = (
    "/Game/GoodSky/Resource/Materials/M_GoodSky_Base",
    "/Game/GoodSky/Resource/Materials/M_GoodSky_Moon",
    "/Game/GoodSky/Resource/Materials/M_GoodSky_Sun_Stars",
    "/Game/GoodSky/Resource/Materials/M_GoodSky_Sun_Stars_Moon",
)


def main() -> None:
    library = unreal.MaterialEditingLibrary
    for path in MATERIALS:
        material = unreal.load_asset(path)
        if material is None:
            unreal.log_warning(f"GoodSkyMaterial: missing {path}")
            continue
        unreal.log(f"GoodSkyMaterial: MATERIAL {path}")
        for name in library.get_scalar_parameter_names(material):
            lowered = str(name).lower()
            if any(word in lowered for word in ("horizon", "background", "dome", "zenith", "opacity", "alpha", "sun", "solar", "radius", "size", "scale")):
                value = library.get_material_default_scalar_parameter_value(material, name)
                unreal.log(f"GoodSkyMaterial: scalar {name}={value}")
        for name in library.get_vector_parameter_names(material):
            lowered = str(name).lower()
            if any(word in lowered for word in ("horizon", "background", "dome", "zenith", "overlay", "color", "sun", "solar")):
                value = library.get_material_default_vector_parameter_value(material, name)
                unreal.log(f"GoodSkyMaterial: vector {name}={value}")


if __name__ == "__main__":
    main()
