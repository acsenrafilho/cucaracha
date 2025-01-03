VALID_MODALITIES = [
    'image_classification',
    'image_keypoint_detection',
    'image_object_detection',
    'image_segmentation',
]

# Pre-treined cucahacha models
CUCARACHA_PRESETS = {
    'image_classification': {
        'doc_is_signed': {
            'variation': 'cucaracha-project/cucaracha-imgclass-document-is-signed/tensorFlow2/cucaracha-imgclass-document_is_signed-v0.1.0',
            'dataset': 'cucaracha-project/cucaracha-mod-imgclass-constains-signature',
        }
    },
    'image_segmentation': {
        'TBD': {
            'variation': 'cucaracha-project/TBD',
            'dataset': 'cucaracha-project/TBD',
        }
    },
}
