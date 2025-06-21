from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'celebs_seg_to_face': {
		'transforms': transforms_config.SoftSegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation_label'],
		'train_target_root': dataset_paths['celeba_train_segmentation_img'],
		'test_source_root': dataset_paths['celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['celeba_test_mask'],
	},
	'ffhq_seg_to_face': {
		'transforms': transforms_config.SoftSegToImageTransforms,
		'train_source_root': dataset_paths['ffhq_train_segmentation_label'],
		'train_target_root': dataset_paths['ffhq_train_segmentation_img'],
		'test_source_root': dataset_paths['ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['ffhq_test_segmentation_img'],
	},
}
