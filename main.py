from src.nba_detector.train_model import train
from src.nba_detector.create_model import get_model
from src.nba_detector.dataset import load_data
from src.nba_detector.transformations import get_transformation
import sys
import yaml


def main():
    #--------- Config -------------#
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = './config.yaml'
    with open(config_file) as cf_file:
        config = yaml.safe_load(cf_file.read())
        print(f"\nConfig file is:\n{config}\n")

    # Train configurations
    DATASET_PATH = config['train']['dataset_path']
    TRAINABLE_LAYERS = config['train']['trainable_layers']
    MODEL_NAME = config['train']['model_name']
    #-------------------------------#

    print("Loading dataset...")
    # transformations configurations
    transformation = None
    if config['train']['use_transformations']:
        transformation = get_transformation(config['transformations'])
    trainset, valset, testset = load_data(DATASET_PATH, train_transform=transformation)
    # Track dataset size
    if config['train']['use_wandb']:
        config['train']['train_size'] = len(trainset)
        config['train']['val_size'] = len(valset)

        # Initialize wandb
        import wandb
        wandb.init(
            project="nba-detector",
            entity=config['train']['wandb_entity'],
            config=config,
        )

    print("Building model...")
    model = get_model(MODEL_NAME, trainable_backbone_layers=TRAINABLE_LAYERS)

    print("Training model...")
    logs = train(model, trainset, valset, testset, config['train'])


if __name__ == '__main__':
    main()