import train_adni
from pathlib import Path
import pytorch_lightning as pl
import wandb
from tqdm import tqdm


train_adni.config['DEBUG'] = False
train_adni.config['WB_LOGGER'] = False
train_adni.config['GRADIENT_CLIP_VAL'] = 0
train_adni.config['freeze_backbone'] = False
train_adni.config['checkpoint_dir'] = ''
train_adni.config['batch_size'] = 40
train_adni.config['ssl_pretrained'] = False

def test(experiment_name):
    experiment_path = Path('/dhc/home/arkadiusz.kwasigroch/experiments/adni_classification') / experiment_name
    experiment_path = list(experiment_path.glob('*.ckpt'))[0]
    test_output_path = experiment_path.parent / 'test_output'
    test_output_path.mkdir(exist_ok=True)


    # TODO load model by test
    model = train_adni.AdniModule.load_from_checkpoint(experiment_path)
    model.test_output = test_output_path
    trainer = pl.Trainer(gpus=1, precision=32)
    trainer.test(model)

def test_sweep(sweep_id, run_id=True):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    train_adni.config['IMAGING_MODEL_NAME'] = None
    for i, run in enumerate(tqdm(sweep.runs)):
        train_adni.config['IMAGING_MODEL_NAME'] = run.config['IMAGING_MODEL_NAME']
        print(f"Testing {run.name} model {train_adni.config['IMAGING_MODEL_NAME']}")

        if run_id:
            test(run.id)
        else:
            test(run.name)


if __name__ == '__main__':
    test_sweep('akwasigroch/adni_classification/zmaijo0m') 