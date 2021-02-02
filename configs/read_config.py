import json
import argparse




class ConfigReader():
    def __init__(self, data):
        self.dataset_name = data["dataset_name"]
        self.coco_images = data["coco_images"]
        self.annotation_json = data["annotation_json"]
        self.query_path = data["query_path"]



if __name__ == '__main__' :
    args = argparse.ArgumentParser()

    args.add_argument('cfg_path', help='config file with paths', type=str)

    params = args.parse_args()
    print(params)

    with open("local_DocExplore.json") as json_data_file:
        data = json.load(json_data_file)
    
    cfg = ConfigReader(data)

    params.query_path = data['query_path']

    print(params)
