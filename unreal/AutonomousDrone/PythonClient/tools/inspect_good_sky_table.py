"""Print sun-related fields from the Good SKY preset data table."""

from __future__ import annotations

import json
import unreal


TABLE_PATH = "/Game/GoodSky/Resource/Data/DataTable/DataTable_GoodSky"


def main() -> None:
    table = unreal.load_asset(TABLE_PATH)
    if table is None:
        raise RuntimeError(f"Missing table: {TABLE_PATH}")
    raw = unreal.DataTableFunctionLibrary.export_data_table_to_json_string(table)
    rows = json.loads(raw)
    for row in rows:
        row_name = str(row.get("Name", row.get("RowName", row.get("---", ""))))
        unreal.log(f"GoodSkyTableName: {row_name}")
        if not any(token in row_name.lower() for token in ("sunset", "noon", "sunrise")):
            continue
        unreal.log(f"GoodSkyTable: ROW {row_name}")
        for key, value in row.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("sun", "radius", "size", "scale", "glow")):
                unreal.log(f"GoodSkyTable: {key}={value}")


if __name__ == "__main__":
    main()
