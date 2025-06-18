from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'text_embed_celeba': {
		'transforms': transforms_config.TextTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
    'text_embed_coco': {
		'transforms': transforms_config.TextTransforms,
		'train_source_root': dataset_paths['coco_train'],
		'train_target_root': dataset_paths['coco_train'],
		'test_source_root': dataset_paths['coco_test'],
		'test_target_root': dataset_paths['coco_test'],
	},
	'celeba_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
    'coco_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['coco_train'],
		'train_target_root': dataset_paths['coco_train'],
		'test_source_root': dataset_paths['coco_test'],
		'test_target_root': dataset_paths['coco_test'],
	},
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'ffhq_frontalize': {
		'transforms': transforms_config.FrontalizationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_sketch_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_sketch'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_sketch'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_segmentation_label'],
		'train_target_root': dataset_paths['celeba_train_segmentation_img'],
		'test_source_root': dataset_paths['celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['celeba_test_mask'],
	},
	'ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['ffhq_train_segmentation_label'],
		'train_target_root': dataset_paths['ffhq_train_segmentation_img'],
		'test_source_root': dataset_paths['ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['ffhq_test_segmentation_img'],
	},
    'coco_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['coco_train_segmentation_label'],
		'train_target_root': dataset_paths['coco_train_segmentation_img'],
		'test_source_root': dataset_paths['coco_test_segmentation_label'],
		'test_target_root': dataset_paths['coco_test_segmentation_img'],
	},
	'city_seg_to_image': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['cityscpaes_train_segmentation_label'],
		'train_target_root': dataset_paths['cityscpaes_train_segmentation_img'],
		'test_source_root': dataset_paths['cityscpaes_test_segmentation_label'],
		'test_target_root': dataset_paths['cityscpaes_test_segmentation_img'],
	},
	'celebs_super_resolution': {
		'transforms': transforms_config.SuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
    
	'all_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['all_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['all_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['all_celeba_test_mask'],
	},
    
    'center_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['center_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['center_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['center_celeba_test_mask'],
    },
        
    'pc1_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc1_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['pc1_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc1_celeba_test_mask'],
    },
        
    'pc2_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc2_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['pc2_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc2_celeba_test_mask'],
    },
        
    'pc3_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc3_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['pc3_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc3_celeba_test_mask'],
    },
        
    'pc4_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc4_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['pc4_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc4_celeba_test_mask'],
    },
        
    'pc5_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc5_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['pc5_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc5_celeba_test_mask'],
    },
        
    'pc6_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc6_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['pc6_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc6_celeba_test_mask'],
	},
    'change_seg_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['seg_celeba_test_segmentation_change_label'],
		'test_target_root': dataset_paths['seg_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['seg_celeba_test_mask'],
	},
    'seg_celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['seg_celeba_test_segmentation_label'],
		'test_target_root': dataset_paths['seg_celeba_test_segmentation_img'],
        'test_mask_root': dataset_paths['seg_celeba_test_mask'],
	},

	# ffhq

	'all_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['all_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['all_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['all_ffhq_test_mask'],
	},
    
    'center_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['center_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['center_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['center_ffhq_test_mask'],
    },
        
    'pc1_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc1_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['pc1_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc1_ffhq_test_mask'],
    },
        
    'pc2_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc2_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['pc2_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc2_ffhq_test_mask'],
    },
        
    'pc3_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc3_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['pc3_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc3_ffhq_test_mask'],
    },
        
    'pc4_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc4_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['pc4_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc4_ffhq_test_mask'],
    },
        
    'pc5_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc5_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['pc5_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc5_ffhq_test_mask'],
    },
        
    'pc6_ffhq_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'test_source_root': dataset_paths['pc6_ffhq_test_segmentation_label'],
		'test_target_root': dataset_paths['pc6_ffhq_test_segmentation_img'],
        'test_mask_root': dataset_paths['pc6_ffhq_test_mask'],
	},


}
