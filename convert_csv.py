import os
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from translate.storage.tmx import tmxfile

# http://docs.translatehouse.org/projects/translate-toolkit/en/latest/api/storage.html

INPUT_DIR = "corpus/"
OUTPUT_PATH = "csv_corpus/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

x = []

# For each language file
for file_name in tqdm(os.listdir(INPUT_DIR)):

    if ".tmx" not in file_name:
        continue

    # Documents counter
    cpt = 0

    # Lenght of the sentences
    source_lens = []
    target_lens = []

    print("file_name : ", file_name)

    # Get the languages
    LANG_PAIR = file_name.split(".")[0]
    L1, L2 = LANG_PAIR.split("-")

    data = []

    df = pd.DataFrame(data={
        'id': [],
        'lang': [],
        'source_text': [],
        'target_text': []
    })

    # Read the file
    with open(INPUT_DIR + file_name, 'rb') as fin:

        # For each sentence
        for node in tmxfile(fin, L1, L2).unit_iter():

            # Add the sentence pair
            data.append({
                'id': node.getid(),
                'lang': LANG_PAIR,
                'source_text': node.source,
                'target_text': node.target
            })

            cpt += 1
            source_lens.append(len(node.source.split(" ")))
            target_lens.append(len(node.target.split(" ")))

    x.append([
        L2,
        cpt,
        int(sum(source_lens) / len(source_lens)),
        int(sum(target_lens) / len(target_lens))
    ])

    # Add to the data frame
    df = df.append(data)

    # Convert to CSV
    df.to_csv(OUTPUT_PATH + LANG_PAIR + ".csv", index=False)

log_file = open("stats.md","w")
log_file.write(str(
    tabulate(x, headers=['Lang', '# Docs', 'Avg. # Source Tokens', 'Avg. # Target Tokens'], tablefmt='orgtbl')
).replace("+","|"))
log_file.close()
