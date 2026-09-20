'''
Copied from: https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

Modified by: Maxlinn

1. adapt calculation of CHAIR-i and CHAIR-s for Python3, supports for both json and jsonl file input.
2. integrate synonyms.txt to make the script standalone.
3. remove machine-translation based metrics BLEU-n, CIDEr, ROGUE
4. add new metric Recall, which represents the node words(i.e. lemmas of objects) coverage overall.
5. add pickle cache mechanism to make it fast for repetitive evaluations.
'''


import os
import sys
import nltk
import json
import argparse
import tqdm
import pickle
from collections import defaultdict
from synonyms import synonyms_txt_original

import inflect
# from pattern.en import singularize

p = inflect.engine()

def singularize(word):
    return p.singular_noun(word) or word
# we ignore the detection of desk/table, since they are annotated incorrectly in many cases.
# desk, workstation, bureau
# table, diningtable, tableware, dining table, desk
# plate, dish, platter
# bowl, basin, dish, bowl, container

def combine_coco_captions(annotation_path):

    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' %(annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps 

def combine_coco_instances(annotation_path):

    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' %(annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' %(annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances 

class CHAIRObject365(object):

    def __init__(self):

        self.imid_to_objects = defaultdict(list) # later become a dict of sets


        #read in synonyms
        synonyms = synonyms_txt_original.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = [] #mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        self.synonym_dict = {}
        for synonym in synonyms[1:]:
            self.synonym_dict[synonym[0]] = synonym
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        #common 'double words' in MSCOCO that should be treated as a single word
        # coco_double_words = ['other shoes', 'pickup truck', 'street lights', 'storage box', 'leather shoes', 'potted plant', 'wine glass', 'traffic light', 'street light', 'traffic signal', 'stop light', 'bow tie', 'trash bin can', 'wild bird', 'high heels', 'motor bike', 'motor cycle', 'cell phone', 'mobile phone', 'traffic cone', 'stuffed toy', 'laptop computer', 'power outlet', 'air conditioner', 'hockey stick', 'traffic sign', 'other fish', 'machine vehicle', 'green vegetables', 'baseball glove', 'air plane', 'suit case', 'tea pot', 'head phone', 'sports car', 'stop sign', ' freezer', 'stove top oven', 'gas stove', 'baseball bat', 'surveillnace camera', 'skating shoes', 'skiing shoes', 'other balls', 'computer box', 'toilet paper', 'cleaning products', 'cutting board', 'coffee table', 'side table', 'fire hydrant', 'fire extinguisher', 'fire truck', 'golf club', 'paint brush', 'extension cord', 'tennis racket', 'american football', 'coffee machine', 'green beans', 'washing machine', 'ice cream', 'hotir ballon', 'french fries', 'french fry', 'hot dog', 'golf ball', 'parking meter', 'fishing rod', 'green onion', 'speed limit sign', 'induction cooker', 'kiwi fruit', 'poker card', 'tape measur', 'crosswalk sign', 'meat ball', 'rice cooker', 'electric drill', 'hair dryer', 'hair drier', 'egg tart', 'game board', 'spring rolls', 'pencil case', 'scallop mollusk', 'table teniis paddle', 'cosmetics brush', 'eyeliner pencil', 'cosmetics mirror', 'table tennis']
        coco_double_words = [
                            "other shoes",
                            "street lights",
                            "potted plant",
                            "leather shoes",
                            "wine glass",
                            "trash bin can",
                            "wild bird",
                            "high heels",
                            "cell phone",
                            "machinery vehicle",
                            "green vegetables",
                            "baseball glove",
                            "skating and skiing shoes",
                            "bow tie",
                            "other balls",
                            "chopping board",
                            "coffee table",
                            "side table",
                            "american football",
                            "washing machine",
                            "drying machine",
                            "french fries",
                            "hot dog",
                            "speed limit sign",
                            "green beans",
                            "hotair ballon",
                            "crosswalk sign",
                            "tape measur",
                            "table teniis paddle",
                            "cosmetics brush",
                            "eyeliner pencil",
                            "formula 1",
                            "meat ball",
                            "red cabbage",
                            "spring rolls",
                            "table tennis"
                        ]
        coco_double_words = [word.lower() for word in coco_double_words]
        
        #Hard code some rules for special cases in MSCOCO
        #qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = mammals_non_human = [
                'dog', 'puppy', 'cat', 'kitten', 'horse', 'stallion', 'mare', 'colt', 'pony',
                'racehorse', 'equine', 'foal', 'palomino', 'mustang', 'clydesdale', 'bronc', 'bronco',
                'cow', 'cattle', 'oxen', 'ox', 'calf', 'holstein', 'heifer', 'buffalo', 'bull', 'zebu', 'bison',
                'sheep', 'lamb', 'ovine', 'ram', 'goat', 'ewe',
                'elephant', 'pachyderm',
                'pig', 'hog', 'swine',
                'rabbit', 'hare', 'bunny',
                'deer', 'cervid',
                'bear', 'ursine', 'panda',
                'lion',
                'monkey',
                'donkey', 'ass', 'mule',
                'camel', 'dromedary',
                'yak',
                'antelope', 'ungulate',
                'seal', 'pinniped',
                'dolphin', 'cetacean'
            ]
        #qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['bus','minibus','train','locomotive','tramway','taxi','cab','taxicab','ferry','ferryboat']
                                
        #double_word_dict will map double words to the word they should be treated as in our analysis
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
        with open("./data/object365/object365_ground_truth_val.json", "r") as f:
            f = json.load(f)
            for meta_data in f:
                self.imid_to_objects[meta_data['image_id']] = set([d.lower() for d in meta_data['objects']])


        
    def caption_to_words(self, caption):
    
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''
    
        #standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]
        #p = inflect.engine()
        #words = [p.singular_noun(w) for w in words]
        #replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words
    
        #toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']
    
        #get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        #return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words


    def compute_hallucinations(self, imid, cap, args = None):        
        imid_to_objects = self.imid_to_objects
     
        #get all words in the caption, as well as corresponding node word
        words, node_words, idxs, raw_words = self.caption_to_words(cap) 

        gt_objects_ = imid_to_objects[imid]

        gt_objects  = []
        ## 
        for gt in gt_objects_:
            gt_objects.append(gt.lower())
            if gt in self.synonym_dict:
                gt_objects = gt_objects + self.synonym_dict[gt.lower()]
                syno = self.inverse_synonym_dict[gt.lower()]
                if syno in self.synonym_dict and syno != gt_objects:
                    gt_objects = gt_objects + self.synonym_dict[syno]
            if gt in self.inverse_synonym_dict:
                gt_objects = gt_objects + [self.inverse_synonym_dict[gt.lower()]]
                syno = self.inverse_synonym_dict[gt.lower()]
                gt_objects = gt_objects + self.synonym_dict[syno]
        
        cap_dict = {
            'mscoco_hallucinated_words': [],
            'mscoco_gt_words': list(gt_objects),
            'mscoco_generated_words': list(node_words),
            'hallucination_idxs': [], 
            'hallucinated_words': 0,
            'recall_words': [],
            'recall_idxs': [],
        }
        # f = open(f'./log/hallucination_words_{args.lvlm}.txt', '+a')
        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                cap_dict['hallucinated_words'] += 1
                cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                cap_dict['hallucination_idxs'].append(idx)

            else:
                cap_dict['recall_words'].append((word, node_word))
                cap_dict['recall_idxs'].append(idx)
        # f.write(args.current_id + '\n')
        # for word in list(set(gt_objects_)):
        #     f.write(word + ',')
        # f.write('\n')
        # for word in list(set(cap_dict['mscoco_hallucinated_words'])):
        #     f.write(word[0] + ',')
        # f.write('\n')
        # f.flush()
        return cap_dict

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):

        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        #Read in captions        
        self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def compute_chair(self, cap_file, image_id_key, caption_key):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)
        
        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids
 
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        
        # :add:
        num_recall_gt_objects = 0.
        num_gt_objects = 0.

        output = {'sentences': []} 
        
        for i in tqdm.trange(len(caps)):
            cap :str = caps[i]
            imid :int = eval_imids[i]
    
            #get all words in the caption, as well as corresponding node word
            words, node_words, idxs, raw_words = self.caption_to_words(cap) 
 
            gt_objects = imid_to_objects[imid]
            cap_dict = {'image_id': imid, 
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'hallucination_idxs': [], 
                        'raw_words': raw_words,
                        'words': words 
                        }

            # :add:
            cap_dict['metrics'] = {'CHAIRs': 0,
                                   'CHAIRi': 0,
                                   'Recall': 0}

            #count hallucinated words
            coco_word_count += len(node_words) 
            hallucinated = False
            
            # add
            recall_gt_objects = set()
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)

            #count hallucinated caps
            num_caps += 1
            if hallucinated:
               num_hallucinated_caps += 1
            
            # add
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            cap_dict['metrics']['Recall'] = 0.
            
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))
            
            # add
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)

            output['sentences'].append(cap_dict)

        chair_s = (num_hallucinated_caps/num_caps)
        chair_i = (hallucinated_word_count/coco_word_count)

        recall = num_recall_gt_objects / num_gt_objects

        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i,
                                     'Recall': recall}

        return output 

def load_generated_captions(cap_file, image_id_key:str, caption_key:str):
    #Read in captions        
    # it should be list of dict
    ext = os.path.splitext(cap_file)[-1]
    with open(cap_file, "r") as f:
        caps = [json.loads(line) for line in f]
    # if ext == '.json':
    #     caps = json.load(open(cap_file))
    # elif ext == '.jsonl':
    #     caps = [json.loads(s) for s in open(cap_file)]
    # else:
    #     raise ValueError(f'Unspported extension {ext} for cap_file: {cap_file}')
    
    # list of int
    imids = [obj[image_id_key] for obj in caps]
    
    # list of str
    caps = [obj[caption_key] for obj in caps]
       
    return caps, imids

def save_hallucinated_words(cap_file, cap_dict): 
    with open(cap_file, 'w') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    
    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(10)
        v_str = f'{v * 100:.2f}'
        print(k_str, v_str, sep=': ')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cap_file", type=str, default='',
                        help="path towards json or jsonl saving image ids and their captions in list of dict.")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="in each dict of cap_file, which key stores image id of coco.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="in each dict of cap_file, which key stores caption of the image.")
    
    parser.add_argument("--cache", type=str, default="chair.pkl",
                        help="pre inited CHAIR evaluator object, for fast loading.")
    parser.add_argument("--coco_path", type=str, default='coco_annotations',
                        help="only use for regenerating CHAIR evaluator object, will be ignored if uses cached evaluator.")
    
    parser.add_argument("--save_path", type=str, default="",
                        help="saving CHAIR evaluate and results to json, useful for debugging the caption model.")
    
    args = parser.parse_args()
    
    if args.cache and os.path.exists(args.cache):
        evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"loaded evaluator from cache: {args.cache}")
    else:
        print(f"cache not setted or not exist yet, building from scratch...")
        evaluator = CHAIRObject365()
        pickle.dump(evaluator, open(args.cache, 'wb'))
        print(f"cached evaluator to: {args.cache}")

    cap_dict = evaluator.compute_chair(args.cap_file, args.image_id_key, args.caption_key) 
    
    print_metrics(cap_dict)
    
#     if args.save_path:
#         save_hallucinated_words(args.save_path, cap_dict)