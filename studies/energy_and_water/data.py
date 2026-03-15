"""
Data for the Energy & Water usage comparison study.

All energy values are stored in Wh (watt-hours).
All water values are stored in mL (milliliters).

Items that only have one metric will be auto-converted using CONVERSIONS
when the other metric is requested.

Each item stores its value PER ONE amount_unit (amount_number=1).
Presets apply a multiplier to scale comparisons into the same order of magnitude.

Derivation constants used throughout:
  CO2 → energy : 1 kWh / 386 g CO2 x 1000 Wh/kWh = 2.590 Wh / g CO2
  CO2 → water  : 1000 mL/L x 4.65 L/kWh / 386 g/kWh = 12.05 mL / g CO2
  energy → water: 4.65 L / kWh  = 4.65 mL / Wh
"""

# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------
# Each item must have: name, amount_number, amount_unit, source
# Each item should have: energy_wh (float | None), water_ml (float | None)
# At least one of energy_wh, water_ml, or co2_g must be non-None.
# Values are stored per ONE amount_unit (amount_number is always 1).

ITEMS: list[dict] = [

    # =========================================================================
    # AI
    # =========================================================================

    {
        "name": "GPT-5 query",
        "amount_number": 1,
        "amount_unit": "query",
        # Jegham et al. (2025) 'How Hungry is AI?' — GPT-5 query water 25–39 mL,
        # midpoint 32 mL; GPT-4o inference ~0.42 Wh.
        "energy_wh": 0.42,
        "water_ml": 32.0,
        "source": "Jegham et al. 'How Hungry is AI?' (2025); Live Microsoft Power BI dashboard, GPT-5 low: 32 mL/medium query",
    },
    {
        "name": "GPT-4 training run",
        "amount_number": 1,
        "amount_unit": "training run",
        # MIT Technology Review (2025): GPT-4 training ~50 GWh.
        "energy_wh": 50_000_000_000.0,  # 50 GWh
        "water_ml": None,
        "source": "MIT Technology Review (2025): GPT-4 training ~50 GWh; https://www.technologyreview.com/2025/05/20/1116327/ai-energy-usage-climate-footprint-big-tech/",
    },

    # =========================================================================
    # SEARCH & EMAIL
    # =========================================================================

    {
        "name": "Google search",
        "amount_number": 1,
        "amount_unit": "search",
        "energy_wh": 0.3,           # 0.0003 kWh per Google's official figure
        "water_ml": None,
        "source": "Google official statement via Chipkin/RW Digital: 0.0003 kWh per average search query; https://store.chipkin.com/articles/did-you-know-it-takes-00003-kwh-per-google-search-and-more",
    },
    {
        "name": "Email (text only)",
        "amount_number": 1,
        "amount_unit": "email",
        # Berners-Lee / Carbon Literacy Project: ~4 g CO2/email.
        # 4 g ÷ 386 g/kWh × 1000 Wh/kWh ≈ 10.36 Wh
        "energy_wh": 10.36,
        "water_ml": None,
        "source": "Mike Berners-Lee, 'How Bad Are Bananas?' (2020) via Carbon Literacy Project: ~4 g CO2 per text email; converted at 386 g CO2/kWh; https://carbonliteracy.com/the-carbon-cost-of-an-email/",
    },
    {
        "name": "Email (with attachment)",
        "amount_number": 1,
        "amount_unit": "email",
        # BBC Science Focus / Carbon Literacy Project: ~50 g CO2 (image attachment).
        # 50 g ÷ 386 g/kWh × 1000 ≈ 129.5 Wh
        "energy_wh": 129.5,
        "water_ml": None,
        "source": "BBC Science Focus / Carbon Literacy Project: ~50 g CO2 for email with image attachment; converted at 386 g CO2/kWh; https://carbonliteracy.com/the-carbon-cost-of-an-email/",
    },

    # =========================================================================
    # SOCIAL MEDIA & STREAMING
    # =========================================================================

    {
        "name": "TikTok",
        "amount_number": 1,
        "amount_unit": "minute",
        # Greenspector (2021): 2.63 g CO2/min.
        # Energy: 2.63 g × 2.590 Wh/g CO2 = 5.49 Wh
        # Water: direct figure from joburgetc.com
        "energy_wh": 5.49,
        "water_ml": 270.0,
        "energy_source": "Greenspector (2021) 2.63 g CO2/min × 2.59 Wh/g CO2 = 5.49 Wh; https://greenspector.com/en/social-media-2021/",
        "water_source": "270 mL/min; https://www.joburgetc.com/news/artificial-intelligence/ai-vs-social-media-water-energy-use/",
        "source": "Energy: Greenspector (2021) 2.63 g CO2/min × 2.59 Wh/g CO2 = 5.49 Wh; https://greenspector.com/en/social-media-2021/ — Water: 270 mL/min; https://www.joburgetc.com/news/artificial-intelligence/ai-vs-social-media-water-energy-use/",
    },
    {
        "name": "Instagram",
        "amount_number": 1,
        "amount_unit": "minute",
        # TikTok ≈ Instagram in emissions per trtworld
        "energy_wh": None,
        "water_ml": None,
        "co2_g": 303 / 60,          # 5.05 g CO2/min (303 g/hr avg of 270, 336 g/hr)
        "source": "https://www.trtworld.com/article/18245163 (TikTok ≈ Instagram in emissions); ResearchGate/358794471",
    },
    {
        "name": "YouTube (HD)",
        "amount_number": 1,
        "amount_unit": "hour",
        # IEA (George Kamiya, 2020): 0.036 kWh/hr for network + data-centre
        # infrastructure (device excluded).
        "energy_wh": 36.0,
        "water_ml": None,
        "source": "IEA commentary (George Kamiya, 2020), 'The carbon footprint of streaming video: fact-checking the headlines'; https://www.iea.org/commentaries/the-carbon-footprint-of-streaming-video-fact-checking-the-headlines",
    },
    {
        "name": "Netflix (HD)",
        "amount_number": 1,
        "amount_unit": "hour",
        "energy_wh": 36.0,
        "water_ml": None,
        "source": "IEA commentary (George Kamiya, 2020), 'The carbon footprint of streaming video: fact-checking the headlines'; https://www.iea.org/commentaries/the-carbon-footprint-of-streaming-video-fact-checking-the-headlines",
    },

    # =========================================================================
    # CRYPTO
    # =========================================================================

    {
        "name": "Bitcoin transaction",
        "amount_number": 1,
        "amount_unit": "transaction",
        # Digiconomist (2025): ~1,249 kWh/transaction (annualised network ÷ tx count).
        "energy_wh": 1_249_000.0,
        "water_ml": None,
        "source": "Digiconomist Bitcoin Energy Consumption Index (2025): ~1,249 kWh per transaction; https://digiconomist.net/bitcoin-energy-consumption",
    },

    # =========================================================================
    # FOOD & DRINK
    # =========================================================================

    {
        "name": "Almond milk",
        "amount_number": 1,
        "amount_unit": "cup",
        # 371 L/L almond milk ÷ 4.22 cups per liter = 87.9 L/cup
        "energy_wh": None,
        "water_ml": 87_900.0,       # 87.9 L
        "source": "https://sentientmedia.org/is-almond-milk-bad-for-the-environment",
    },
    {
        "name": "Avocado",
        "amount_number": 1,
        "amount_unit": "avocado",
        # Water Footprint Network global average ~227 L/fruit (~60 US gallons).
        "energy_wh": None,
        "water_ml": 227_000.0,      # 227 L (~60 gallons)
        "source": "Water Footprint Network / UNEP: ~227 L (60 gal) per avocado; https://www.unep.org/news-and-stories/story/whats-your-burger-more-you-think",
    },
    {
        "name": "Beef burger (quarter-pound)",
        "amount_number": 1,
        "amount_unit": "burger",
        # USGS Water Science School: ~460 gallons per quarter-pound burger.
        # 460 gal × 3,785.41 mL/gal = 1,741,289 mL
        "energy_wh": None,
        "water_ml": 1_741_289.0,    # 460 gallons (USGS)
        "source": "USGS Water Science School: ~460 gallons of water per quarter-pound burger; https://water.usgs.gov/edu/activity-watercontent.php",
    },
    {
        "name": "Beef",
        "amount_number": 1,
        "amount_unit": "lb",
        # Water Footprint Network / Denver Water: ~1,800 gallons (6,813 L) per lb.
        "energy_wh": None,
        "water_ml": 6_813_000.0,    # 1,800 gallons
        "source": "Water Footprint Network / Denver Water: ~1,800 gallons (6,813 L) per pound of beef; https://www.denverwater.org/tap/whats-beef-water",
    },

    # =========================================================================
    # FASHION
    # =========================================================================

    {
        "name": "Pair of jeans",
        "amount_number": 1,
        "amount_unit": "pair",
        "energy_wh": None,
        "water_ml": 10_000_000.0,   # 10,000 L
        "source": "https://d3.harvard.edu/platform-rctom/submission/levi-strauss-taking-the-water-out-of-jeans/",
    },
    {
        "name": "Cotton t-shirt",
        "amount_number": 1,
        "amount_unit": "shirt",
        # WWF / Water Footprint Network: ~2,700 L per cotton t-shirt.
        "energy_wh": None,
        "water_ml": 2_700_000.0,    # 2,700 L
        "source": "WWF / Water Footprint Network via TriplePundit: ~2,700 L per cotton t-shirt; https://www.triplepundit.com/story/2013/it-takes-2700-liters-water-make-t-shirt/54321",
    },

    # =========================================================================
    # HOUSEHOLD
    # =========================================================================

    {
        "name": "Shower",
        "amount_number": 1,
        "amount_unit": "minute",
        # EPA standard showerhead: 2.5 gal/min = 9,463.5 mL/min
        "energy_wh": None,
        "water_ml": 9_463.5,        # 2.5 gallons / 9.46 L per minute
        "source": "US EPA / EPAct 1992: standard showerhead max 2.5 gal/min; https://www.epa.gov/watersense/showerheads",
    },
    {
        "name": "Bath (full tub)",
        "amount_number": 1,
        "amount_unit": "bath",
        # Standard US bathtub ~36 gallons = 136,275 mL.
        "energy_wh": None,
        "water_ml": 136_275.0,      # 36 gallons
        "source": "EPA WaterSense: average US standard bathtub ~36 gallons (136 L); https://www.epa.gov/watersense/statistics-and-facts",
    },
    {
        "name": "Toilet flush",
        "amount_number": 1,
        "amount_unit": "flush",
        # Federal standard since EPAct 1994: 1.6 gal/flush = 6,057 mL.
        "energy_wh": None,
        "water_ml": 6_057.0,        # 1.6 gallons
        "source": "US EPA / EPAct 1994: federal maximum 1.6 gallons per flush; https://www.epa.gov/watersense/toilets",
    },
    {
        "name": "Dishwasher cycle",
        "amount_number": 1,
        "amount_unit": "cycle",
        # Energy: ~1.2 kWh/cycle (Direct Energy, standard model).
        # Water: ENERGY STAR ≤ 3.5 gal/cycle = 13,249 mL.
        "energy_wh": 1_200.0,
        "water_ml": 13_249.0,       # 3.5 gallons (ENERGY STAR limit)
        "source": "Energy: Direct Energy: ~1.2 kWh per standard cycle — Water: EPA ENERGY STAR Version 6: ≤3.5 gal/cycle; https://www.energystar.gov/products/dishwashers/key_product_criteria",
    },
    {
        "name": "Clothes washer",
        "amount_number": 1,
        "amount_unit": "load",
        # ENERGY STAR HE front-loader: ~500 Wh/cycle; ≤ 15 gal/cycle = 56,775 mL.
        "energy_wh": 500.0,
        "water_ml": 56_775.0,       # 15 gallons
        "source": "DOE/EnergySage: ENERGY STAR HE washer ~500 Wh/cycle; water ≤15 gal/cycle; https://www.energystar.gov/products/clothes_washers",
    },
    {
        "name": "Clothes dryer",
        "amount_number": 1,
        "amount_unit": "cycle",
        # Average electric dryer: 2,250–3,000 Wh/cycle; 2,450 Wh mid-range.
        "energy_wh": 2_450.0,
        "water_ml": None,
        "source": "EnergySage / Arcadia: average electric dryer ~2,250–3,000 Wh per 45-min cycle; 2,450 Wh mid-range; https://www.energysage.com/electricity/house-watts/how-many-watts-does-a-dryer-use/",
    },
    {
        "name": "Lawn sprinkler",
        "amount_number": 1,
        "amount_unit": "hour",
        # Typical rotary/impact sprinkler ~1 gal/min = 60 gal/hr = 227,125 mL.
        "energy_wh": None,
        "water_ml": 227_125.0,      # 60 gallons/hr
        "source": "EPA WaterSense: typical lawn sprinkler ~1 gal/min = 60 gal/hr; https://www.epa.gov/watersense/outdoor-water-use-us",
    },
    {
        "name": "Swimming pool (fill)",
        "amount_number": 1,
        "amount_unit": "fill",
        # Pool & Hot Tub Alliance: average US in-ground pool ~15,000 gal.
        "energy_wh": None,
        "water_ml": 56_781_000.0,   # 15,000 gallons
        "source": "Pool & Hot Tub Alliance (APSP): average US in-ground residential pool ~15,000 gallons; https://www.poolspahot.com",
    },

    # =========================================================================
    # TRANSPORT
    # =========================================================================

    {
        "name": "Domestic flight",
        "amount_number": 1,
        "amount_unit": "flight (NYC→LA)",
        # US airlines 2018 fleet average: 4.06 L/100 p-km (EIA/BTS).
        # Jet fuel: 10 kWh/L. 4.06 × 39.4 × 10,000 Wh/L ≈ 1,600,000 Wh.
        "energy_wh": 1_600_000.0,
        "water_ml": None,
        "source": "US EIA/BTS domestic fleet avg 4.06 L/100 p-km (2018); jet fuel 10 kWh/L (MacKay); NYC–LA ~3,940 km; https://www.eia.gov/todayinenergy/detail.php?id=31512",
    },
    {
        "name": "Transatlantic flight",
        "amount_number": 1,
        "amount_unit": "flight (NYC→London)",
        # IATA 2018 global avg 88 g CO2/RPK → ~3.5 L/100 p-km.
        "energy_wh": 1_940_000.0,
        "water_ml": None,
        "source": "IATA 2018 global avg 88 g CO2/RPK → ~3.5 L/100 p-km; jet fuel 10 kWh/L; NYC–LHR ~5,540 km; https://en.wikipedia.org/wiki/Fuel_economy_in_aircraft",
    },
    {
        "name": "Gasoline car",
        "amount_number": 1,
        "amount_unit": "mile",
        # EPA 2024 avg new light-duty vehicle: ~28 mpg combined.
        # 1 ÷ 28 gal × 33,700 Wh/gal = 1,203.6 Wh/mile
        "energy_wh": 1_202.0,
        "water_ml": None,
        "source": "EPA 2024 avg new vehicle ~28 mpg combined; 1 US gal gasoline = 33.7 kWh (EPA/DOE); https://www.fueleconomy.gov",
    },
    {
        "name": "Electric car",
        "amount_number": 1,
        "amount_unit": "mile",
        # DOE AFDC / EPA (2024): 25–40 kWh/100 miles; 35 kWh → 350 Wh/mile.
        "energy_wh": 350.0,
        "water_ml": None,
        "source": "US DOE AFDC / EPA (2024): light-duty EVs 25–40 kWh per 100 miles; 35 kWh central estimate → 350 Wh/mile; https://afdc.energy.gov/fuels/electricity-benefits",
    },

    # =========================================================================
    # LARGE-SCALE PRODUCTION
    # =========================================================================

    {
        "name": "Boeing 787 (manufacture)",
        "amount_number": 1,
        "amount_unit": "aircraft",
        # Materials: CF 50,000 kg × 286 MJ/kg + Al 24,000 kg × 46.8 MJ/kg ≈ 4,284 MWh.
        # Total incl. assembly, engines, avionics: ~20,000–30,000 MWh; 25,000 MWh midpoint.
        "energy_wh": 25_000_000_000.0,  # 25,000 MWh
        "water_ml": None,
        "source": "designlife-cycle.com: CF 50k kg×286 MJ/kg + Al 24k kg×46.8 MJ/kg → ~4,284 MWh materials; total w/ assembly ~20,000–30,000 MWh; 25,000 MWh midpoint; http://www.designlife-cycle.com/boeing-787",
    },
    # =========================================================================
    # INFRASTRUCTURE / ANNUAL SCALE
    # =========================================================================

    {
        "name": "Hyperscale data center",
        "amount_number": 1,
        "amount_unit": "year (100 MW facility)",
        # 100 MW × 8,760 hr/yr = 876,000 MWh = 876 GWh/yr.
        "energy_wh": 876_000_000_000.0,  # 876 GWh
        "water_ml": None,
        "source": "C&C Technology Group / Statista / IEA: hyperscale data centers typically draw ≥100 MW continuously → 100 MW × 8,760 hr = 876 GWh/yr; https://cc-techgroup.com/how-much-power-does-a-hyperscale-data-center-use/",
    },
    {
        "name": "EAF steel mini-mill",
        "amount_number": 1,
        "amount_unit": "year (1.5 Mt/yr output)",
        # EAF electricity intensity: ~475 kWh/t (industry average).
        # 1,500,000 t × 475 kWh/t = 712,500 MWh = 712.5 GWh/yr.
        "energy_wh": 712_500_000_000.0,  # 712.5 GWh
        "water_ml": None,
        "source": "EAF electricity intensity ~475 kWh/t, industry average (Wikipedia / HeatTreatConsortium); http://heattreatconsortium.com/metals-advisor/electric-arc-furnace/ — 1.5 Mtpa × 475 kWh/t = 712.5 GWh/yr electricity",
    },
    {
        "name": "Primary aluminum smelter",
        "amount_number": 1,
        "amount_unit": "year (750 kt/yr smelter)",
        # IAI 2021: ~14,114 kWh/t of primary aluminum.
        # 750,000 t × 14,114 kWh/t ≈ 10.6 TWh; Aluminum Association cites ~11 TWh/yr.
        "energy_wh": 11_000_000_000_000.0,  # 11 TWh
        "water_ml": None,
        "source": "Aluminum Association: a single new aluminum smelter uses ~11 TWh/yr (equivalent to Boston or Nashville annual electricity demand); https://www.aluminum.org/policy-agenda/energy — IAI 2021 global avg smelting intensity ~14,114 kWh/t",
    },
    {
        "name": "Google data centers",
        "amount_number": 1,
        "amount_unit": "year (all facilities, 2024)",
        # Google 2025 Environmental Report: 32,727.8 GWh in 2024.
        "energy_wh": 32_727_800_000_000.0,  # 32,727.8 GWh ≈ 32.7 TWh
        "water_ml": None,
        "source": "Google 2025 Environmental Report (cited by BestBrokers.com, Dec 2025): Google global electricity consumption 32,727.8 GWh in 2024, more than double 2020 levels; https://www.bestbrokers.com/2025/12/19/bitcoin-energy-cost/",
    },
    {
        "name": "Bitcoin network",
        "amount_number": 1,
        "amount_unit": "year (global network, 2025)",
        # Digiconomist / CBECI 2025: ~175 TWh/yr (range 143–195 TWh).
        # Water: ~2,772 GL/yr (Digiconomist 2025).
        "energy_wh": 175_000_000_000_000.0,  # 175 TWh
        "water_ml": 2_772_000_000_000_000.0,  # 2,772 GL
        "source": "Digiconomist Bitcoin Energy Consumption Index (2025): ~175.87 TWh/yr annualized; water 2,772 GL/yr; https://digiconomist.net/bitcoin-energy-consumption — Cambridge Bitcoin Electricity Consumption Index (CBECI)",
    },

    {
        "name": "Big-budget film (production)",
        "amount_number": 1,
        "amount_unit": "film",
        # Sustainable Production Alliance (2021) via Variety: avg ~3,370 metric tons CO2.
        # 3,370 t × 1,000,000 g/t × 2.590 Wh/g CO2 ≈ 8,731,300,000 Wh
        "energy_wh": 8_731_000_000.0,   # ~8,731 MWh
        "water_ml": None,
        "source": "Sustainable Production Alliance (2021) via Variety: avg big-budget film ~3,370 metric tons CO2; at 2.590 Wh/g CO2 → ~8,731 MWh; https://variety.com/2021/film/news/sustainable-production-alliance-carbon-emissions-report-1234942580/",
    },
]

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
# Conversion factors used when an item only has one metric and the other
# is requested. Each entry has a factor, description, and source.

CONVERSIONS: dict[str, dict] = {
    "energy_to_water": {
        # 1 kWh → 4.65 L water = 4.65 mL / Wh (US average WUE, data-centre weighted)
        "factor": 4.65,
        "description": "mL of water per Wh of energy (1 kWh = 4.65 L, US avg WUE)",
        "source": "Jegham et al. 'How Hungry is AI?' (2025); https://arxiv.org/html/2505.09598v1",
    },
    "water_to_energy": {
        # Derived: 1 / 4.65 Wh per mL
        "factor": 1 / 4.65,
        "description": "Wh of energy per mL of water (derived from energy_to_water)",
        "source": "derived",
    },
    "co2_to_energy": {
        # US grid average: 386 g CO2/kWh → 1000/386 = 2.590 Wh per g CO2
        "factor": 1000 / 386,
        "description": "Wh of energy per gram of CO2 (US grid average 386 g CO2/kWh)",
        "source": "MIT (2024): US grid average 386 g CO2/kWh; https://news.mit.edu/2024/cutting-carbon-emissions-us-power-grid-0311",
    },
    "co2_to_water": {
        # 1000 mL/L × 4.65 L/kWh × 1 kWh/386 g CO2 = 12.05 mL/g CO2
        "factor": 1000 * 4.65 / 386,
        "description": "mL of water per gram of CO2 (= energy_to_water × co2_to_energy chain)",
        "source": "derived from energy_to_water and co2_to_energy factors above",
    },
}
