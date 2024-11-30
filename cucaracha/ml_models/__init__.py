VALID_MODALITIES = [
    'image_classification',
    'image_keypoint_detection',
    'image_object_detection',
]

# Pre-treined cucahacha models
CUCARACHA_MODELS = {
    'image_classification': {
        'doc_is_signed': {
            'variation': 'acsenrafilho/cucaracha-imgclass-document-is-signed/tensorFlow2/cucaracha-imgclass-document_is_signed-v0.1.0',
            'dataset': 'acsenrafilho/cucaracha-mod-imgclass-constains-signature',
        }
    }
}
