# FILE: book_visualizer.py
# AUTHOR: Caleb Lees
# LAST MODIFIED: 10 February 2026

from embedding_utilities import get_bert_verse_vector
from matplotlib import pyplot as plt
from pathlib import Path
import json

def scroll_scatter(scroll_name:str):
    scroll_dir = Path('./texts/lemmatized_manuscripts/ot')

    # Search for the scroll JSON file
    scroll_stubs = []
    scroll_path = None
    for scroll_json in scroll_dir.glob("*.json"):
        this_scroll_stub = scroll_json.name.split('.')[0]
        if this_scroll_stub == scroll_name:
            scroll_path = scroll_dir / scroll_json.name
    
    # If scroll not found, print error
    if scroll_path == None:
        print(f"Error: scroll '{scroll_name}' not recognized. Acceptable scroll names are:")
        print(scroll_stubs)
        return

    with open(scroll_path, 'r') as f:
        scroll_dict = json.load(f)
        scroll_verses = []
        for chapter in scroll_dict.keys():
            for verse in scroll_dict[chapter].keys():
                scroll_verses.append(scroll_dict[chapter][verse])
    
    print('done')


scroll_scatter('Gen')

