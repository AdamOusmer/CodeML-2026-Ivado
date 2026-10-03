REGIONS = (
    "Montreal", "Capitale-Nationale", "Bas-Saint-Laurent", "Cote-Nord",
    "Gaspesie-Iles-de-la-Madeleine",
)
REMOTE_REGIONS = ("Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine")


def is_remote(df):
    return df["region_administrative"].isin(REMOTE_REGIONS).to_numpy().astype(int)
