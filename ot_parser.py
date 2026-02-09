# FILE: text_parsers.py
# AUTHOR: Caleb M. Lees
# LAST MODIFIED: 9 February 2026

import re
import xml.etree.ElementTree as ET
import json
from pathlib import Path


def get_hebrew_verse_dictionary(file_path: str):
    """
    Converts a hebrew book XML file into a dictionary of lematized values.
    Lemmatized values are Strong's Concordance values without prefixes/suffixes or morphological variant markers
    (i.e., just raw numerical values without modifiers). The basis for this lemmatization strategy is that
    while grammatical features are ignored, broader semantic structural features are preserved. This is theoretically 
    ideal for capturing global narrative structures like "Redemption" or "The Fall."
    """

    ns_url = "http://www.bibletechnologies.net/2003/OSIS/namespace"
    ns = {'ns': ns_url}

    # Parse the full tree
    tree = ET.parse(file_path)
    root = tree.getroot()
    lemma_dict = {}

    # Extract verses and words
    for verse in root.findall('.//ns:verse', ns):
        verse_id = verse.get('osisID')
        
        # split the osisID to get the chapter number
        parts = verse_id.split('.')
        chapter_num = parts[1]
        if chapter_num not in lemma_dict:
            lemma_dict[chapter_num] = {}

        lemma_words = []
        for w in verse.findall('ns:w', ns):
            raw_lemma = w.get('lemma')
            if raw_lemma:
                clean_lemma = re.findall(r'\d+', raw_lemma)
                lemma_words.extend(clean_lemma)

        if lemma_words:
            lemma_dict[chapter_num][verse_id] = " ".join(lemma_words)

    return lemma_dict

def batch_xml_to_json(source_dir: str, output_dir: str):
    """
    Converts all Tanak books (XMLs) in a given directory to JSONs.
    """
    src = Path(source_dir)
    out = Path(output_dir)

    out.mkdir(parents=True, exist_ok=True)

    for xml_file in src.glob("*.xml"):
        print(f"Processing: {xml_file.name}...")
        data_dict = get_hebrew_verse_dictionary(xml_file)
        json_file_path = out / xml_file.with_suffix(".json").name

        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=4)

        print("\nBatch conversion complete!")

ot_xml_dir = './texts/manuscripts/ot'
ot_json_dir = './texts/lemmatized_manuscripts/ot'

batch_xml_to_json(ot_xml_dir, ot_json_dir)


