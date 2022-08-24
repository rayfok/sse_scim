import glob
import json
import os

START = 1
END = 250

def main():
    OUTPUT_DIR = "output"
    papers = {}
    for i in range(START, END+1):
        fpath = os.path.join(OUTPUT_DIR, f"2022.naacl-main.{i}.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                paper_data = json.load(f)

            paper_id = os.path.splitext(os.path.basename(fpath))[0]
            papers[paper_id] = paper_data
        else:
            print(f"=> 2022.naacl-main.{i}.json missing")
    with open(os.path.join(OUTPUT_DIR, "facets.json"), "w") as out:
        json.dump(papers, out)


if __name__ == "__main__":
    main()
