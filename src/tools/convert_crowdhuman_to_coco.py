import os
import numpy as np
import json
import cv2

DATA_PATH = '../../data/crowdhuman/'
OUT_PATH = f'{DATA_PATH}annotations/'
SPLITS = ['val', 'train']
DEBUG = False

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    return [json.loads(line.strip('\n')) for line in lines]

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
      os.mkdir(OUT_PATH)
    video_cnt = 0
    for split in SPLITS:
        data_path = DATA_PATH + split
        out_path = OUT_PATH + '{}.json'.format(split)
        ann_path = DATA_PATH + '/annotation_{}.odgt'.format(split)
        anns_data = load_func(ann_path)
        ann_cnt = 0
        out = {'images': [], 'annotations': [], 
           'categories': [{'id': 1, 'name': 'person'}]}
        for image_cnt, ann_data in enumerate(anns_data, start=1):
            image_info = {'file_name': '{}.jpg'.format(ann_data['ID']),
                          'id': image_cnt}
            out['images'].append(image_info)
            if split != 'test':
              anns = ann_data['gtboxes']
              for i in range(len(anns)):
                ann_cnt += 1
                ann = {'id': ann_cnt,
                       'category_id': 1,
                       'image_id': image_cnt,
                       'bbox_vis': anns[i]['vbox'],
                       'bbox': anns[i]['fbox'],
                       'iscrowd': 1 if 'extra' in anns[i] and \
                                       'ignore' in anns[i]['extra'] and \
                                       anns[i]['extra']['ignore'] == 1 else 0}
                out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(
          split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
        
        

