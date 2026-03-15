"""
Energy & Water Usage Comparison Study

Compares the energy and water footprint of everyday activities, digital
services, and physical goods. Items that only have one metric are
auto-converted using a configurable conversion factor.

Run with:
    python -m studies.energy_and_water.main
"""

from studies.energy_and_water.data import ITEMS, CONVERSIONS

# ---------------------------------------------------------------------------
# Preset comparisons
# ---------------------------------------------------------------------------
# Each preset has:
#   title  – display name
#   metric – "energy" or "water"
#   items  – list of {name, multiplier} dicts
#             multiplier scales the item's per-unit value for comparison.
#             e.g. multiplier=300 on a "per query" item → 300 queries.

PRESETS: list[dict] = [
    {
        "title": "Digital water: one action at a time",
        "metric": "water",
        "baseline": "GPT-5 query",
        # All at ×1 — shows the mL-scale cost of a single digital action
        "items": [
            {"name": "Google search",      "multiplier": 1},
            {"name": "GPT-5 query",        "multiplier": 1},
            {"name": "Email (text only)",  "multiplier": 1},
            {"name": "TikTok",             "multiplier": 1},
        ],
    },
    {
        "title": "30 min online vs. your morning routine",
        "metric": "water",
        "baseline": "GPT-5 query",
        # Scale digital items to ~30 min of use; compare against common L-scale
        # household water uses. GPT-5 ×300 ≈ TikTok ×30 ≈ 1 min of shower.
        "items": [
            {"name": "Toilet flush",   "multiplier": 1},
            {"name": "Shower",         "multiplier": 1},       # 1 minute
            {"name": "GPT-5 query",    "multiplier": 100},     # ~10 min of chatting
            {"name": "TikTok",         "multiplier": 30},      # 30 min of scrolling
            {"name": "Almond milk",    "multiplier": 1},       # 1 cup with breakfast
            {"name": "Bath (full tub)","multiplier": 1},
        ],
    },
    {
        "title": "What you eat vs. what you query",
        "metric": "water",
        "baseline": "GPT-5 query",
        # Food water footprints dwarf digital — but how many queries equals a burger?
        "items": [
            {"name": "Almond milk",              "multiplier": 1},       # 87.9 L/cup
            {"name": "GPT-5 query",              "multiplier": 1_000},   # 32 L
            {"name": "Avocado",                  "multiplier": 1},       # 227 L
            {"name": "Beef burger (quarter-pound)", "multiplier": 1},    # 1,741 L
            {"name": "Cotton t-shirt",           "multiplier": 1},       # 2,700 L
        ],
    },
    {
        "title": "Same energy as 1 Bitcoin transaction",
        "metric": "energy",
        "baseline": "GPT-5 query",
        # 1 Bitcoin ≈ 3 million GPT-5 queries ≈ 3,500 miles EV ≈ 1,000 miles gas
        "items": [
            {"name": "GPT-5 query",      "multiplier": 1_000},
            {"name": "Electric car",     "multiplier": 3_500},    # miles
            {"name": "Gasoline car",     "multiplier": 1_000},    # miles
            {"name": "Domestic flight",  "multiplier": 1},
            {"name": "Bitcoin transaction", "multiplier": 1},
        ],
    },
    {
        "title": "100 queries vs. your home appliances",
        "metric": "energy",
        "baseline": "GPT-5 query",
        # 100 GPT-5 queries sit just below a washer load
        "items": [
            {"name": "Google search",    "multiplier": 1_000},
            {"name": "GPT-5 query",      "multiplier": 100},
            {"name": "Clothes washer",   "multiplier": 1},
            {"name": "Dishwasher cycle", "multiplier": 1},
            {"name": "Clothes dryer",    "multiplier": 1},
        ],
    },
    {
        "title": "Production scale: AI training vs. real-world megaprojects",
        "metric": "energy",
        "baseline": "GPT-4 training run",
        # Training a frontier model rivals manufacturing a jumbo jet;
        # one hyperscale data center annually dwarfs it;
        # the Bitcoin network dwarfs all of Google's data centers.
        "items": [
            {"name": "Big-budget film (production)",  "multiplier": 1},   # ~8.7 GWh
            {"name": "Boeing 787 (manufacture)",      "multiplier": 1},   # ~25 GWh
            {"name": "GPT-4 training run",            "multiplier": 1},   # ~50 GWh
            {"name": "EAF steel mini-mill",           "multiplier": 1},   # ~713 GWh/yr
            {"name": "Hyperscale data center",        "multiplier": 1},   # ~876 GWh/yr
            {"name": "Primary aluminum smelter",      "multiplier": 1},   # ~11 TWh/yr
            {"name": "Google data centers",           "multiplier": 1},   # ~32.7 TWh/yr
            {"name": "Bitcoin network",               "multiplier": 1},   # ~175 TWh/yr
        ],
    },
]

_ITEM_BY_NAME: dict[str, dict] = {item["name"]: item for item in ITEMS}


def _resolve_preset_items(preset: dict) -> list[tuple[dict, float]]:
    """Return [(item_dict, multiplier), ...] for a preset."""
    resolved = []
    for entry in preset["items"]:
        name = entry["name"]
        multiplier = entry.get("multiplier", 1)
        item = _ITEM_BY_NAME.get(name)
        if item is None:
            print(f"  WARNING: preset item '{name}' not found in catalogue — skipped.")
        else:
            resolved.append((item, multiplier))
    return resolved


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def format_energy(wh: float) -> str:
    """Format a watt-hour value with appropriate unit prefix."""
    if wh < 0.001:
        return f"{wh * 1_000_000:.2f} µWh"
    if wh < 1:
        return f"{wh * 1_000:.2f} mWh"
    if wh < 1_000:
        return f"{wh:.4g} Wh"
    if wh < 1_000_000:
        return f"{wh / 1_000:.4g} kWh"
    if wh < 1_000_000_000:
        return f"{wh / 1_000_000:.4g} MWh"
    if wh < 1_000_000_000_000:
        return f"{wh / 1_000_000_000:,.2f} GWh"
    return f"{wh / 1_000_000_000_000:,.2f} TWh"


def format_water(ml: float) -> str:
    """Format a milliliter value with appropriate unit prefix."""
    if ml < 1_000:
        return f"{ml:.2f} mL"
    if ml < 1_000_000:
        return f"{ml / 1_000:.2f} L"
    return f"{ml / 1_000_000:.2f} m³"


def format_amount(amount_number: float, amount_unit: str, multiplier: float) -> str:
    n = amount_number * multiplier
    n_int = int(n)
    return f"{n_int:,} {amount_unit}" if n == n_int else f"{n:g} {amount_unit}"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
# Priority order for source fields: energy_wh > co2_g > water_ml

def _co2_to_wh(co2_g: float) -> float:
    return co2_g * CONVERSIONS["co2_to_energy"]["factor"]


def get_energy(item: dict) -> tuple[float | None, str]:
    """Return (energy_wh, conversion_path). Path is '' if value is direct."""
    if item.get("energy_wh") is not None:
        return item["energy_wh"], ""
    if item.get("co2_g") is not None:
        return _co2_to_wh(item["co2_g"]), "CO₂ → energy"
    if item.get("water_ml") is not None:
        return item["water_ml"] * CONVERSIONS["water_to_energy"]["factor"], "water → energy"
    return None, ""


def get_water(item: dict) -> tuple[float | None, str]:
    """Return (water_ml, conversion_path). Path is '' if value is direct."""
    if item.get("water_ml") is not None:
        return item["water_ml"], ""
    if item.get("energy_wh") is not None:
        return item["energy_wh"] * CONVERSIONS["energy_to_water"]["factor"], "energy → water"
    if item.get("co2_g") is not None:
        wh = _co2_to_wh(item["co2_g"])
        return wh * CONVERSIONS["energy_to_water"]["factor"], "CO₂ → energy → water"
    return None, ""


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_PATH_TO_CONV_KEYS: dict[str, list[str]] = {
    "energy → water":       ["energy_to_water"],
    "water → energy":       ["water_to_energy"],
    "CO₂ → energy":         ["co2_to_energy"],
    "CO₂ → energy → water": ["co2_to_energy", "energy_to_water"],
}


def _raw_source_value(item: dict, metric: str) -> str:
    """Return a human-readable string of the raw stored value used as the source."""
    unit = item["amount_unit"]
    if metric == "energy":
        if item.get("energy_wh") is not None:
            return f"{format_energy(item['energy_wh'])}/{unit}"
        if item.get("co2_g") is not None:
            return f"{item['co2_g']:.3g} gCO₂/{unit}"
        if item.get("water_ml") is not None:
            return f"{format_water(item['water_ml'])}/{unit}"
    else:
        if item.get("water_ml") is not None:
            return f"{format_water(item['water_ml'])}/{unit}"
        if item.get("energy_wh") is not None:
            return f"{format_energy(item['energy_wh'])}/{unit}"
        if item.get("co2_g") is not None:
            return f"{item['co2_g']:.3g} gCO₂/{unit}"
    return "?"


def display_comparison(item_pairs: list[tuple[dict, float]], metric: str, title: str = "", baseline: str | None = None):
    """Print a sorted comparison table for the chosen metric."""
    rows = []
    for item, multiplier in item_pairs:
        if metric == "energy":
            raw_val, path = get_energy(item)
        else:
            raw_val, path = get_water(item)

        val = raw_val * multiplier if raw_val is not None else None
        fmt = (format_energy(val) if metric == "energy" else format_water(val)) if val is not None else "N/A"
        amount_str = format_amount(item["amount_number"], item["amount_unit"], multiplier)

        rows.append({
            "name": item["name"],
            "amount": amount_str,
            "val": val,
            "fmt": fmt,
            "path": path,
            "item": item,
        })

    rows.sort(key=lambda r: r["val"] if r["val"] is not None else float("inf"))

    col_name   = max(len(r["name"])   for r in rows) + 2
    col_amount = max(len(r["amount"]) for r in rows) + 2
    col_val    = max(len(r["fmt"])    for r in rows) + 2

    print(f"\n{'=' * 80}")
    if title:
        print(f"  {title}")
    print(f"  METRIC: {metric.upper()}")
    print(f"{'=' * 80}")
    print()
    header = f"  {'Item':<{col_name}} {'Amount':<{col_amount}} {metric.capitalize():<{col_val}} Note"
    print(header)
    print(f"  {'-' * (col_name + col_amount + col_val + 20)}")
    for r in rows:
        note = f"({r['path']})" if r["path"] else ""
        print(f"  {r['name']:<{col_name}} {r['amount']:<{col_amount}} {r['fmt']:<{col_val}} {note}")

    if len(rows) >= 2:
        baseline_row = next((r for r in rows if r["name"] == baseline), None) if baseline else None
        if baseline_row is None:
            baseline_row = rows[0]
        if baseline_row["val"] and baseline_row["val"] > 0:
            print(f"\n  Relative to '{baseline_row['name']}' ({baseline_row['amount']}):")
            for r in rows:
                if r is baseline_row or not r["val"]:
                    continue
                ratio = r["val"] / baseline_row["val"]
                print(f"    {r['name']} ({r['amount']}): {ratio:,.1f}×")

    print(f"\n  Sources:")
    for r in rows:
        item = r["item"]
        raw_val_str = _raw_source_value(item, metric)
        # Use metric-specific source if available (e.g. items with both energy and water)
        source_str = item.get(f"{metric}_source", item["source"])
        source_lines = [s.strip() for s in source_str.split(" — ")]
        print(f"    {r['name']} ({r['amount']}): {raw_val_str}")
        for line in source_lines:
            print(f"      {line}")
        for conv_key in _PATH_TO_CONV_KEYS.get(r["path"], []):
            conv = CONVERSIONS[conv_key]
            print(f"      → converted via {conv['description']}: {conv['source']}")
    print()


# ---------------------------------------------------------------------------
# Selection menus
# ---------------------------------------------------------------------------

def select_metric() -> str:
    print("\n=== Select Metric ===")
    print("  [1] Energy (Wh)")
    print("  [2] Water (mL / L)")
    choice = input("\nChoice: ").strip()
    if choice == "2":
        return "water"
    if choice != "1":
        print("Invalid choice, defaulting to energy.")
    return "energy"


def select_items() -> list[tuple[dict, float]]:
    """Prompt user to select items; all use multiplier=1."""
    print("\n=== Select Items to Compare ===")
    for i, item in enumerate(ITEMS, 1):
        has_energy = "E" if item.get("energy_wh") is not None else " "
        has_co2    = "C" if item.get("co2_g")     is not None else " "
        has_water  = "W" if item.get("water_ml")  is not None else " "
        flags = f"[{has_energy}{has_co2}{has_water}]"
        amt = format_amount(item["amount_number"], item["amount_unit"], 1)
        print(f"  [{i:>2}] {flags} {item['name']} ({amt})")

    print("\n  (E = direct energy, C = CO₂ source, W = direct water)")
    print("  Enter comma-separated numbers, or 'all' for everything.")
    picks = input("\nSelection: ").strip()

    if picks.lower() == "all":
        return [(item, 1) for item in ITEMS]

    selected = []
    for p in picks.split(","):
        try:
            idx = int(p.strip()) - 1
            if 0 <= idx < len(ITEMS):
                selected.append((ITEMS[idx], 1))
        except ValueError:
            pass
    return selected


def print_catalogue():
    """Print all items with their raw source values (no conversions applied)."""
    print(f"\n{'=' * 80}")
    print("  FULL CATALOGUE  (raw source values, per 1 unit)")
    print(f"{'=' * 80}")
    col = max(len(item["name"]) for item in ITEMS) + 2
    print(f"\n  {'Item':<{col}} {'Per':<30} {'Energy':>15} {'CO₂':>12} {'Water':>18}")
    print(f"  {'-' * (col + 30 + 15 + 12 + 18 + 4)}")
    for item in ITEMS:
        e = format_energy(item["energy_wh"]) if item.get("energy_wh") is not None else "-"
        c = f"{item['co2_g']:.2f} gCO₂"    if item.get("co2_g")     is not None else "-"
        w = format_water(item["water_ml"])   if item.get("water_ml")  is not None else "-"
        amt = format_amount(item["amount_number"], item["amount_unit"], 1)
        print(f"  {item['name']:<{col}} {amt:<30} {e:>15} {c:>12} {w:>18}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_presets():
    print("\n=== Preset Comparisons ===")
    for i, preset in enumerate(PRESETS, 1):
        print(f"  [{i}] [{preset['metric'][0].upper()}] {preset['title']}")
    print("\n  Enter number(s) to run (comma-separated), or 'all'.")
    picks = input("\nSelection: ").strip()

    if picks.lower() == "all":
        selected = PRESETS
    else:
        selected = []
        for p in picks.split(","):
            try:
                idx = int(p.strip()) - 1
                if 0 <= idx < len(PRESETS):
                    selected.append(PRESETS[idx])
            except ValueError:
                pass

    for preset in selected:
        pairs = _resolve_preset_items(preset)
        if not pairs:
            print(f"  Skipping '{preset['title']}' — no valid items.")
            continue
        display_comparison(pairs, preset["metric"], title=preset["title"], baseline=preset.get("baseline"))


def main():
    print("\n" + "=" * 60)
    print("  STUDY: Energy & Water Usage Comparison")
    print("=" * 60)
    print(f"\n  {len(ITEMS)} items  |  {len(PRESETS)} presets loaded.")

    print("\n=== Main Menu ===")
    print("  [1] Run preset comparisons")
    print("  [2] Custom comparison")
    print("  [3] View full catalogue")
    choice = input("\nChoice: ").strip()

    if choice == "3":
        print_catalogue()
        return

    if choice == "2":
        metric = select_metric()
        pairs = select_items()
        if not pairs:
            print("No items selected. Exiting.")
            return
        display_comparison(pairs, metric)
        return

    run_presets()


if __name__ == "__main__":
    main()
