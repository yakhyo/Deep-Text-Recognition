import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Configuration:
    def __init__(self):
        """ Training configurations """
        self.exp_name = ''
        self.train_data = 'data/trainset'
        self.valid_data = 'data/testset'
        self.manualSeed = 1111
        self.workers = 4
        self.batch_size = 192
        self.num_iter = 50000
        self.valInterval = 2000
        self.saved_model = 'weights/TPS-ResNet-BiLSTM-Attn-FT.pth'  # 'best_accuracy.pth'
        self.FT = False
        self.lr = 1
        self.beta1 = 0.9
        self.rho = 0.95
        self.eps = 1e-8
        self.grad_clip = 5
        self.select_data = '/'
        self.batch_ratio = '1'
        self.total_data_usage_ratio = '1.0'
        self.num_class = None
        self.adam = False
        """ Data processing """
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.sensitive = False
        self.PAD = False
        self.data_filtering_off = False
        """ Model Architecture """
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256

        """ Test Configurations """
        self.eval_data = 'data/testset'  # evaluation data folder
        self.benchmark_all_eval = False

        """ Demo """
        self.image_folder = 'images'


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            # text = [self.dict[char] if char in self.dict else self.dict['[UNK]'] for char in text]
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
