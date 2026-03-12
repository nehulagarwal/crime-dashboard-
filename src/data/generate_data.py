import pandas as pd
import json

# ── Load data ─────────────────────────────────────────────────────────
print("Loading CSVs...")
train    = pd.read_csv('train_data.csv')
test     = pd.read_csv('test_data.csv')
all_data = pd.concat([train, test], ignore_index=True)

print(f"Train rows : {len(train)}")
print(f"Test rows  : {len(test)}")
print(f"Total rows : {len(all_data)}")
print(f"Columns    : {len(all_data.columns)}")
print()

GROUPS = ['SC', 'ST', 'Women', 'Children']

# ── 1. overview.json ──────────────────────────────────────────────────
print("Generating overview.json...")

top_states = (
    all_data.groupby('state_name')['total_crimes']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

overview = {
    "dataset": {
        "total_records" : 21067,
        "train_records" : 17379,
        "test_records"  : 3688,
        "states"        : 36,
        "districts"     : 700,
        "features"      : 188,
        "years"         : [2017, 2018, 2019, 2020, 2021, 2022]
    },
    "paper": {
        "title"    : "Fairness-Constrained Multi-Task Learning for Crime Prediction",
        "subtitle" : "Addressing Bias Against Vulnerable Populations",
        "venue"    : "SRM Institute of Science and Technology, Delhi NCR Campus",
        "team"     : ["Rishabh Singh", "Nehul Agarwal", "Farah Masoodi", "Archit Raghav"],
        "guide"    : "Prof. Geetanjali Tyagi"
    },
    "groups": [
        {
            "name"       : "SC",
            "records"    : 5323,
            "categories" : 39,
            "color"      : "#64B5F6",
            "desc"       : "Scheduled Castes — caste-based discrimination and social exclusion"
        },
        {
            "name"       : "ST",
            "records"    : 5101,
            "categories" : 39,
            "color"      : "#81C784",
            "desc"       : "Scheduled Tribes — geographic isolation, limited access to justice"
        },
        {
            "name"       : "Women",
            "records"    : 5322,
            "categories" : 43,
            "color"      : "#FF7043",
            "desc"       : "Gender-based crimes — domestic violence, sexual assault, harassment"
        },
        {
            "name"       : "Children",
            "records"    : 5321,
            "categories" : 52,
            "color"      : "#FFB74D",
            "desc"       : "Minors — trafficking, abuse, exploitation under POCSO Act"
        }
    ],
    "top_states": [
        {"state": k, "avg": round(float(v), 1)}
        for k, v in top_states.items()
    ]
}

with open('overview.json', 'w') as f:
    json.dump(overview, f, indent=2)
print("✓ overview.json saved")

# ── 2. trends.json ────────────────────────────────────────────────────
print("Generating trends.json...")

# average crimes per group per year
yearly = []
for yr in sorted(all_data['year'].unique()):
    row = {"year": int(yr)}
    for g in GROUPS:
        mask    = (all_data['year'] == yr) & (all_data['protected_group'] == g)
        avg     = all_data.loc[mask, 'total_crimes'].mean()
        row[g]  = round(float(avg), 1)
    yearly.append(row)

# top 10 states by average crimes
top10 = (
    all_data.groupby('state_name')['total_crimes']
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

# yearly trend for each of those top 10 states
state_trends = {}
for state in top10:
    sd = all_data[all_data['state_name'] == state]
    state_trends[state] = [
        {
            "year" : int(y),
            "avg"  : round(float(sd[sd['year'] == y]['total_crimes'].mean()), 1)
        }
        for y in sorted(sd['year'].unique())
    ]

trends = {
    "yearly"       : yearly,
    "top_states"   : top10,
    "state_trends" : state_trends,
    "all_states"   : sorted(all_data['state_name'].unique().tolist())
}

with open('trends.json', 'w') as f:
    json.dump(trends, f, indent=2)
print("✓ trends.json saved")

# ── 3. cities.json ────────────────────────────────────────────────────
print("Generating cities.json...")

cities_map = {}

for group, fname in [
    ("Women",    "Women_Crimes_2021_2023.csv"),
    ("SC",       "SC_Crimes_2021_2023.csv"),
    ("ST",       "ST_Crimes_2021_2023.csv"),
    ("Children", "Children_Crimes_2021_2023.csv"),
]:
    df = pd.read_csv(fname)

    for _, row in df.iterrows():
        city = str(row.get('City', '')).strip()
        if not city:
            continue
        if city not in cities_map:
            cities_map[city] = {"city": city, "groups": {}}

        cities_map[city]["groups"][group] = {
            "2021" : int(row.get('2021', 0) or 0),
            "2022" : int(row.get('2022', 0) or 0),
            "2023" : int(row.get('2023', 0) or 0),
        }

cities = {
    "cities": list(cities_map.values())
}

with open('cities.json', 'w') as f:
    json.dump(cities, f, indent=2)
print("✓ cities.json saved")

# ── Done ──────────────────────────────────────────────────────────────
print()
print("✅ All 3 JSON files generated!")
print("   overview.json  — dataset stats, paper info, group breakdown")
print("   trends.json    — yearly crime averages by group and state")
print("   cities.json    — 34 metro cities data 2021-2023")