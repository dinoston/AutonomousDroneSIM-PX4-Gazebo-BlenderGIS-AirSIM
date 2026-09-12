"""Increase AirSim rain visibility for bright daytime scenes.

Run this script with UnrealEditor-Cmd. It edits the AirSim rain material
instance used by P_Weather_RainFX and saves the asset in the current project.
"""

from __future__ import annotations

import unreal


RAIN_MATERIAL = "/AirSim/Weather/WeatherFX/Materials/M_RainDrop_Inst"
TARGET_EMISSIVE = 2.5


def main() -> None:
    material = unreal.load_asset(RAIN_MATERIAL)
    if material is None:
        raise RuntimeError(f"Rain material was not found: {RAIN_MATERIAL}")

    library = unreal.MaterialEditingLibrary
    scalar_names = list(library.get_scalar_parameter_names(material))
    unreal.log(f"DroneRain: scalar parameters: {[str(name) for name in scalar_names]}")

    emissive_name = next(
        (name for name in scalar_names if str(name).lower() == "emissive"), None
    )
    if emissive_name is None:
        raise RuntimeError("M_RainDrop_Inst has no Emissive scalar parameter")

    previous = library.get_material_instance_scalar_parameter_value(
        material, emissive_name
    )
    changed = library.set_material_instance_scalar_parameter_value(
        material, emissive_name, TARGET_EMISSIVE
    )

    # AirSim's weather assets were authored in UE4.18. In UE5.5 the editing
    # library can report False for these migrated instances, so update the
    # serialized override struct directly as a compatibility fallback.
    if not changed:
        overrides = list(material.get_editor_property("scalar_parameter_values"))
        override_found = False
        for override in overrides:
            parameter_info = override.get_editor_property("parameter_info")
            parameter_name = parameter_info.get_editor_property("name")
            if str(parameter_name).lower() == "emissive":
                override.set_editor_property("parameter_value", TARGET_EMISSIVE)
                override_found = True
                break
        if not override_found:
            raise RuntimeError("Emissive override was not found in M_RainDrop_Inst")
        material.set_editor_property("scalar_parameter_values", overrides)

    material.modify()

    if not unreal.EditorAssetLibrary.save_loaded_asset(material, False):
        raise RuntimeError("Failed to save M_RainDrop_Inst")

    unreal.log(
        f"DroneRain: daytime visibility applied: Emissive "
        f"{previous:.3f} -> {TARGET_EMISSIVE:.3f}"
    )


if __name__ == "__main__":
    main()
