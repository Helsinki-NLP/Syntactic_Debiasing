import sys
import torch
from utils.logger import logger
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


class bertify:
    def __init__(self, device):
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.model.to(self.device)

        self.N_LAYERS = 12
        self.ENC_DIM = 768

    def tokenize(self, sentence):
        # Tokenized input
        if isinstance(sentence, str):
            sentence = sentence.split(' ')
        sentence = ' '.join(['[CLS]'] + sentence + ['[SEP]'])
        #print(sentence)
        tokenized_sentence = self.tokenizer.tokenize(sentence)

        #print('tokenized: ', tokenized_sentence)

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        return tokens_tensor, tokenized_sentence


    def encode(self, tokens_tensor):
        tokens_tensor = tokens_tensor.to(self.device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor)
            #print(encoded_layers[0].shape)

        return encoded_layers


    def correct_bert_tokenization(self, bert_encoding, bert_sentence):
        print(bert_sentence)

        all_layers = []
        for layer in range(self.N_LAYERS):
            current_layer = []

            prev_token = bert_encoding[layer][0,0,:] # This is [CLS]!
            sequence_len = bert_encoding[layer].shape[1]
            print('cbt: seq_len:', sequence_len)

            accum = 1
            for token_id in range(1,sequence_len):
                if  bert_sentence[token_id][:2] != '##':
                    current_layer.append(prev_token.view(1,1,self.ENC_DIM) / accum) # Average pooling
                    accum = 1
                    prev_token = bert_encoding[layer][0,token_id,:]
                else:
                    prev_token += bert_encoding[layer][0,token_id,:] # Average pooling
                    accum += 1
            # Add the final token too:
            current_layer.append(prev_token.view(1,1,self.ENC_DIM) / accum) # Average pooling

            current_layer_tensor = torch.cat(current_layer, dim=1)
            # Get rid of the [CLS] and [SEP]
            current_layer_tensor = current_layer_tensor[:,1:-1,:]

            all_layers.append(current_layer_tensor)
            print('cbt: all_layers[0]', all_layers[0].shape)

        return all_layers