import json
import os
import argparse as ap

def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-j", "--json_path", type=str, required=True,
                        help="Path to JSON file.")
    parser.add_argument('-i','--img_path', type=str, required=True,
                        help="Path to images folder")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with open(args.json_path, 'r') as f:
      data = json.load(f)
    failed=0
    correct=0
    for item in data:
        if not os.path.isfile(os.path.join(args.img_path,item['name'])):
            print('No image for label')
            failed+=1
        else:
            correct+=1

    print('Correct image-label mappings: %d' % correct)
    print('Failed image-label mappings: %d' % failed)


