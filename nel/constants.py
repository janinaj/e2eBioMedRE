from os.path import dirname, join, realpath, isfile

TRAIN, DEV, TEST = 'train', 'dev', 'test'

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASE_RESOURCES_DIR = join(BASE_PATH, 'resources')
BASIC_CONF_PATH = join(BASE_PATH, 'configs/exp.conf')
# Set your own output path
BASE_OUTPUT_PATH = join(BASE_PATH, 'outputs/rescnn')
BASE_SAVE_PATH = join(BASE_OUTPUT_PATH, 'trained_models')
BASE_RESULT_PATH = join(BASE_OUTPUT_PATH, 'results')
BASE_SYNTHETIC_DATA_PATH = join(BASE_OUTPUT_PATH, 'synthetic_data')

SHOULD_SHUFFLE_DURING_INFERENCE = False
PRETRAINED_LIGHTWEIGHT_VDCNN_MODEL = \
    join(BASE_OUTPUT_PATH, 'pretrained_models/pretraining_lightweight_vdcnn/model.pt')
PRETRAINED_LIGHTWEIGHT_CNN_TEXT_MODEL = \
    join(BASE_OUTPUT_PATH, 'pretrained_models/pretraining_lightweight_cnn_text/model.pt')

# Datasets
BC5CDR_C = 'bc5cdr-chemical'
BC5CDR_D = 'bc5cdr-disease'
NCBI_D = 'ncbi-disease'
BIORED_C = 'biored-chemical'
BIORED_D = 'biored-disease'
BC8BIORED_C = 'bc8biored-chemical'
BC8BIORED_D = 'bc8biored-disease'
BC8BIORED_C_AIO = 'bc8biored-chemical-aio'
BC8BIORED_D_AIO = 'bc8biored-disease-aio'
DATASETS = [
    BC5CDR_C, BC5CDR_D, NCBI_D, BIORED_C, BIORED_D, 
    BC8BIORED_C, BC8BIORED_D, BC8BIORED_C_AIO, BC8BIORED_D_AIO
]
SEPARATE_ONTOLOGIES = [
    BC5CDR_C, BC5CDR_D, NCBI_D, BIORED_D, BIORED_C, 
    BC8BIORED_C, BC8BIORED_D, BC8BIORED_C_AIO, BC8BIORED_D_AIO
]
USE_TRAINDEV = False

# Nametypes
NAME_PRIMARY = 'primary'
NAME_SECONDARY = 'synonym/secondary'
NAMETYPES = [NAME_PRIMARY, NAME_SECONDARY]

# Ontologies File Path
BASE_ONTOLOGY_DIR = 'resources/ontologies'

# Model Types
DUMMY_MODEL = 'dummy'
CANDIDATES_GENERATOR = 'cg'
PRETRAINING_MODEL = 'pm'
RERANKER = 'rr'

