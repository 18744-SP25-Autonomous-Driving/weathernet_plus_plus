import pandas as pd

# Load the CSV file
df = pd.read_csv('labels_train.csv')

# Check for invalid rows
fog = df[~df['fog'].isin([0, 1])]
glare = df[~df['glare'].isin([0, 1])]
road = df[~df['road'].isin([0, 1, 2])]
traffic = df[~df['traffic'].isin([0, 1, 2])]
weather = df[~df['weather'].isin([0, 1, 2, 3, 4, 5])]
scene = df[~df['scene'].isin([0, 1, 2, 3])]
timeofday = df[~df['timeofday'].isin([0, 1, 2, 3])]


print(f"Invalid labels: {len(fog), len(glare), len(road), len(traffic), len(weather), len(scene), len(timeofday)}")
if (len(fog)): print(f"Invalid fog labels:\n{fog}")
if (len(glare)): print(f"Invalid glare labels:\n{glare}")
if (len(road)): print(f"Invalid road labels:\n{road}")
if (len(traffic)): print(f"Invalid traffic labels:\n{traffic}")
if (len(weather)): print(f"Invalid weather labels:\n{weather}")
if (len(scene)): print(f"Invalid scene labels:\n{scene}")
if (len(timeofday)): print(f"Invalid timeofday labels:\n{timeofday}")