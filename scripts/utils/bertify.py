import sys
import torch
from utils.logger import logger
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

N_BERT_LAYERS = 12

class bertify:
    def __init__(self, device):
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
        # Load pre-trained model (weights)
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.eval()
        bert_model.to(device)


    def tokenize(sentence, tokenizer):
        # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
        logging.basicConfig(level=logging.INFO)

        # Tokenized input
        if isinstance(sentence, str):
            sentence = sentence.split(' ')
        sentence = ' '.join(['[CLS]'] + sentence + ['[SEP]'])
        #print(sentence)
        tokenized_sentence = tokenizer.tokenize(sentence)

        #print('tokenized: ', tokenized_sentence)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        return tokens_tensor, tokenized_sentence


    def encode(tokens_tensor, model):
        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor)
            #print(encoded_layers[0].shape)

        return encoded_layers


    def correct_bert_tokenization(bert_encoding, bert_sentence):
        print(bert_sentence)

        all_layers = []
        for layer in range(N_BERT_LAYERS):
            current_layer = []

            prev_token = bert_encoding[layer][0,0,:] # This is [CLS]!
            sequence_len = bert_encoding[layer].shape[1]
            print('cbt: seq_len:', sequence_len)

            accum = 1
            for token_id in range(1,sequence_len):
                if  bert_sentence[token_id][:2] != '##':
                    current_layer.append(prev_token.view(1,1,ENC_DIM) / accum) # Average pooling
                    accum = 1
                    prev_token = bert_encoding[layer][0,token_id,:]
                else:
                    prev_token += bert_encoding[layer][0,token_id,:] # Average pooling
                    accum += 1
            # Add the final token too:
            current_layer.append(prev_token.view(1,1,ENC_DIM) / accum) # Average pooling

            current_layer_tensor = torch.cat(current_layer, dim=1)
            # Get rid of the [CLS] and [SEP]
            current_layer_tensor = current_layer_tensor[:,1:-1,:]

            all_layers.append(current_layer_tensor)
            print('cbt: all_layers[0]', all_layers[0].shape)

        return all_layers