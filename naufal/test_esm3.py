from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

model = ESM3.from_pretrained("esm3-open").to("cuda")
protein = ESMProtein(sequence="MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPR")

model(protein)

print("Keys:", protein.representations.keys())
