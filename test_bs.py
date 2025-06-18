from options import test_options
from model import create_model
from util import visualizer
from itertools import islice
from configs import data_configs
from dataloader.images_dataset import ImagesDataset_psp
from torch.utils.data import DataLoader


if __name__=='__main__':
    # get testing options
    opts = test_options.TestOptions().parse()
    # creat a dataset
    assert not opts.channel_first
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    test_dataset = ImagesDataset_psp(source_root=dataset_args['test_source_root'],
                                    target_root=dataset_args['test_target_root'],
                                    source_transform=transforms_dict['transform_inference'],
                                    target_transform=transforms_dict['transform_test'],
                                    opts=opts, use_mask=True, return_name=True, hole_range=[0.3, 0.5],
                                    mask_root=dataset_args['test_mask_root'] if 'test_mask_root' in dataset_args else None,)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=opts.batchSize,
                                    shuffle=True,
                                    num_workers=int(opts.nThreads),
                                    drop_last=True)
    dataset_size = len(test_dataloader) * opts.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opts)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opts)

    how_many = opts.how_many if opts.how_many > 0 else dataset_size
    for i, data in enumerate(islice(test_dataloader, how_many)):
        model.set_input(data)
        model.test()
