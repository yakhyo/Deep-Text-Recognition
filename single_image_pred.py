import string
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms

from utils.utils import AttnLabelConverter, Configuration
from net.model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt, img_path):
    """ model configuration """
    converter = AttnLabelConverter(opt.character)

    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    img = Image.open(f'{img_path}').convert('L')  # for single image

    img = img.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    tranform = transforms.ToTensor()
    img = tranform(img)
    img.sub_(0.5).div_(0.5)
    img = img.unsqueeze(0)
    image_path_list = 'images/img.png'

    # predict
    model.eval()
    with torch.no_grad():
        image = img.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * 1).to(device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

        preds = model(image, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        dashed_line = '-' * 80
        head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

        print(f'{dashed_line}\n{head}\n{dashed_line}')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred_EOS = preds_str[0].find('[s]')
        pred = preds_str[0][:pred_EOS]  # prune after "end of sentence" token ([s])
        pred_max_prob = preds_max_prob[0][:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        print(f'{img_path:25s}\t{pred:25s}\t{confidence_score:0.4f}')


if __name__ == '__main__':
    opt = Configuration()
    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    img_path = './images/img.png'
    demo(opt, img_path)
