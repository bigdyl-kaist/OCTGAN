import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
import warnings
import argparse
from octgan.benchmark import run
from octgan.data import load_dataset


warnings.filterwarnings(action='ignore')

# HyperParameter
parser = argparse.ArgumentParser("ctgan with odes")
parser.add_argument('--dataset_name', type=str, default='adult')
parser.add_argument('--synthesizer', type=str, default='identity')
parser.add_argument('--gen_dim', nargs='+', type=int, default=(128, 128))
parser.add_argument('--dis_dim', nargs='+', type=int, default=(128, 128))
parser.add_argument('--num_split', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--random_dim', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--l2scale', type=float, default=1e-06)

parser.add_argument('--lr', type=float, default=2e-3)

parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--gpu', type=str, default="0")

config = parser.parse_args()    
os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu 

if config.synthesizer == 'identity':
    from octgan.synthesizers.identity import IdentitySynthesizer as Synthesizer
elif config.synthesizer == 'ctgan':
    from octgan.synthesizers.ctgan import CTGANSynthesizer as Synthesizer
elif config.synthesizer == 'tvae':
    from octgan.synthesizers.tvae import TVAESynthesizer as Synthesizer
elif config.synthesizer == 'octgan':
    from octgan.synthesizers.octgan import OCTGANSynthesizer as Synthesizer
elif config.synthesizer == 'tablegan':
    from octgan.synthesizers.tablegan import TableganSynthesizer as Synthesizer


train, test, meta, categoricals, ordinals = load_dataset(config.dataset_name, benchmark=True)

scores = run(Synthesizer, arguments=config, output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "/result"))
print(scores)