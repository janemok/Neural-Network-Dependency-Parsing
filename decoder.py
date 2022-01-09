from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            #pass
            # TODO: Write the body of this loop for part 4 
            features = self.extractor.get_input_representation(words, pos, state).reshape(1,-1)
            possible_actions = self.model(features)[0]
            asc_sorted = np.argsort(possible_actions)[::-1]
            highest_permitted = False
            
            i = 0
            while highest_permitted == False:
                trans,dep = self.output_labels[asc_sorted[i]]
                if trans == 'right_arc' and state.stack:
                    state.right_arc(dep)
                    highest_permitted = True
                elif trans == 'shift' and len(state.stack) == 0:
                    state.shift()
                    highest_permitted = True
                elif trans == 'shift' and len(state.buffer) > 1:
                    state.shift()
                    highest_permitted = True
                elif trans == 'left_arc' and state.stack and state.stack[-1] != 0:
                    state.left_arc(dep)
                    highest_permitted = True 
                else:
                    i+=1

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
