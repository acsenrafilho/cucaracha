import os

VALID_MODALITIES = [
    'image_classification',
    'image_keypoint_detection',
    'image_object_detection',
    'image_segmentation',
]
DEFAULT_MODEL_LAKE = os.path.join(
    os.path.expanduser('~'), '.cache', 'kagglehub', 'models'
)


# Pre-treined cucahacha models
CUCARACHA_PRESETS = {
    'image_classification': {
        'doc_is_signed': {
            'variation': 'cucaracha-project/cucaracha-imgclass-document-is-signed/tensorFlow2/cucaracha-imgclass-document_is_signed-v0.1.0',
            'dataset': 'cucaracha-project/cucaracha-mod-imgclass-constains-signature',
            'labels': {0: 'unsigned', 1: 'signed'},
        },
        'cnh_cpf_rg': {
            'variation': 'cucaracha-project/cucaracha-imgclass-brazilian-personal-document/keras/cucaracha-imgclass-brazilian_personal_doc-v0.1.0',
            'dataset': 'cucaracha-project/cucaracha-mod-imgclass-brazilian-personal-doc',
            'labels': {0: 'rg', 1: 'others', 2: 'cpf', 3: 'cnh'},
        },
    },
    'image_segmentation': {
        'TBD': {
            'variation': 'cucaracha-project/TBD',
            'dataset': 'cucaracha-project/TBD',
        }
    },
}
