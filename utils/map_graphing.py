"""Map visualization utilities using geopandas.

Provides choropleth maps for:
- US states (via public GeoJSON)
- World countries (via bundled or public shapefile)
"""

import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt

US_STATES_GEOJSON = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)

WORLD_SHP_PATH = Path("/Users/brogle/workspace/brogle/states/shp/world.shp")

# Caches
_states_gdf_cache = None
_world_gdf_cache = None


def _load_states() -> gpd.GeoDataFrame:
    global _states_gdf_cache
    if _states_gdf_cache is None:
        _states_gdf_cache = gpd.read_file(US_STATES_GEOJSON)
    return _states_gdf_cache.copy()


def _load_world() -> gpd.GeoDataFrame:
    global _world_gdf_cache
    if _world_gdf_cache is None:
        _world_gdf_cache = gpd.read_file(WORLD_SHP_PATH)
    return _world_gdf_cache.copy()


def get_all_state_names() -> list[str]:
    """Return all US state names from the GeoJSON."""
    return list(_load_states()["name"])


def get_all_country_names() -> list[str]:
    """Return all country names from the world shapefile."""
    return list(_load_world()["NAME"])


# ---------------------------------------------------------------------------
# US State Choropleth
# ---------------------------------------------------------------------------

def us_state_choropleth(
    state_values: dict[str, float],
    title: str,
    save_path: str | Path,
    value_range: tuple[float, float] | None = None,
    cmap: matplotlib.colors.Colormap | str = "coolwarm",
    label_low: str = "",
    label_high: str = "",
    missing_color: str = "#DDDDDD",
):
    """Draw and save a US state choropleth map.

    Args:
        state_values: {state_name: numeric_value} (e.g. {"Texas": 0.3, ...})
        title: Chart title
        save_path: Output file path
        value_range: (vmin, vmax) for the color scale; auto-detected if None
        cmap: Matplotlib colormap name or object
        label_low: Label for the low end of the colorbar
        label_high: Label for the high end of the colorbar
        missing_color: Color for states with no data
    """
    states = _load_states()

    df = pd.DataFrame([
        {"name": name, "value": val}
        for name, val in state_values.items()
    ])
    states = states.merge(df, on="name", how="left")

    if value_range is None:
        vals = [v for v in state_values.values() if v is not None]
        value_range = (min(vals), max(vals)) if vals else (-1, 1)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    states.plot(
        column="value",
        ax=ax,
        legend=True,
        legend_kwds={
            "shrink": 0.6,
            "orientation": "horizontal",
            "aspect": 40,
            "pad": 0.02,
        },
        cmap=cmap,
        edgecolor="black",
        linewidth=0.5,
        vmin=value_range[0],
        vmax=value_range[1],
        missing_kwds={"color": missing_color, "label": "No data"},
    )

    # Add colorbar endpoint labels
    if label_low or label_high:
        cbar = ax.get_figure().axes[-1]
        if label_low:
            cbar.text(0, -1.5, label_low, transform=cbar.transAxes,
                      fontsize=11, ha="left", va="top")
        if label_high:
            cbar.text(1, -1.5, label_high, transform=cbar.transAxes,
                      fontsize=11, ha="right", va="top")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.axis("off")

    # Crop to continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(23, 50)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# World Choropleth
# ---------------------------------------------------------------------------

def world_choropleth(
    country_values: dict[str, float],
    title: str,
    save_path: str | Path,
    value_range: tuple[float, float] | None = None,
    cmap: matplotlib.colors.Colormap | str = "coolwarm",
    label_low: str = "",
    label_high: str = "",
    missing_color: str = "#DDDDDD",
    region: str | None = None,
):
    """Draw and save a world choropleth map.

    Args:
        country_values: {country_name: numeric_value} using names from the
            world shapefile (call get_all_country_names() to see them).
        title: Chart title
        save_path: Output file path
        value_range: (vmin, vmax) for the color scale; auto-detected if None
        cmap: Matplotlib colormap name or object
        label_low: Label for the low end of the colorbar
        label_high: Label for the high end of the colorbar
        missing_color: Color for countries with no data
        region: Optional region crop. One of "europe", "asia", "north_america",
            "south_america", "africa", or None for full world.
    """
    world = _load_world()

    df = pd.DataFrame([
        {"NAME": name, "value": val}
        for name, val in country_values.items()
    ])
    world = world.merge(df, on="NAME", how="left")

    if value_range is None:
        vals = [v for v in country_values.values() if v is not None]
        value_range = (min(vals), max(vals)) if vals else (-1, 1)

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    world.plot(
        column="value",
        ax=ax,
        legend=True,
        legend_kwds={
            "shrink": 0.6,
            "orientation": "horizontal",
            "aspect": 40,
            "pad": 0.02,
        },
        cmap=cmap,
        edgecolor="black",
        linewidth=0.3,
        vmin=value_range[0],
        vmax=value_range[1],
        missing_kwds={"color": missing_color, "label": "No data"},
    )

    if label_low or label_high:
        cbar = ax.get_figure().axes[-1]
        if label_low:
            cbar.text(0, -1.5, label_low, transform=cbar.transAxes,
                      fontsize=11, ha="left", va="top")
        if label_high:
            cbar.text(1, -1.5, label_high, transform=cbar.transAxes,
                      fontsize=11, ha="right", va="top")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
    ax.axis("off")

    # Region crops
    bounds = {
        "europe": (-25, 45, 35, 72),
        "asia": (25, 180, -10, 75),
        "north_america": (-170, -50, 10, 75),
        "south_america": (-85, -30, -60, 15),
        "africa": (-20, 55, -37, 38),
    }
    if region and region in bounds:
        xmin, xmax, ymin, ymax = bounds[region]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
