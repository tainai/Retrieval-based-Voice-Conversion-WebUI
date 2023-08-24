import importlib

model_name = 'test_model'
sample_rate = '40k'
pitch_guidance = True
trainset_dir = 'dataset'
speaker_id = 0
np = 8
f0method = 'rmvpe_gpu'
save_epoch = 50
total_epoch = 200
batch_size = 1
if_save_latest = 'Yes'
pretrained_G = 'pretrained_v2/f0G40k.pth'
pretrained_D = 'pretrained_v2/f0D40k.pth'
gpus = ''
if_cache_gpu = 'No'
if_save_every_weights = 'Yes'
version = 'v2'
gpus_rmvpe = '0'

inferWebModule = importlib.import_module('infer-web')

runner = inferWebModule.train1key(
    model_name,
    sample_rate,
    pitch_guidance,
    trainset_dir,
    speaker_id,
    np,
    f0method,
    save_epoch,
    total_epoch,
    batch_size,
    if_save_latest,
    pretrained_G,
    pretrained_D,
    gpus,
    if_cache_gpu,
    if_save_every_weights,
    version,
    gpus_rmvpe
)

for result in runner:
    print(result)



# torch_directml