# import pytest

# import tests.sample_paths as sp
# from cucaracha.ml_trainer import KeypointDetectionModeling
# from cucaracha.ml_trainer.keypoint_detection_modeling import (
#     KeypointDataGenerator,
# )


# @pytest.mark.parametrize(
#     'path,',
#     [
#         ('tests/not-a-folder'),
#         ('wrong-path'),
#     ],
# )
# def test_keypoint_detection_modeling_raise_error_paths_dont_exist(path):
#     with pytest.raises(FileNotFoundError) as e:
#         obj = KeypointDetectionModeling(path, path)

#     assert e.value.args[0] == f'The path {path} does not exist.'


# @pytest.mark.parametrize(
#     'path',
#     [
#         (sp.ROOT_TEST_FOLDER),
#     ],
# )
# def test_load_dataset_raise_error_not_found_raw_data_folder(path):
#     obj = KeypointDetectionModeling(path, sp.ROOT_TEST_FOLDER)
#     with pytest.raises(FileNotFoundError) as e:
#         obj.load_dataset()

#     assert e.value.args[0] == f'The raw_data folder does not exist in {path}.'


# @pytest.mark.parametrize(
#     'path',
#     [
#         (sp.DOC_ML_DATASET),
#     ],
# )
# def test_load_dataset_sucess_dataset_folder(path):
#     obj = KeypointDetectionModeling(path, path)
#     dataset = obj.load_dataset()
#     assert obj.dataset_path == path
#     assert len(dataset) == 13
#     assert isinstance(dataset[0].get('img_filename'), str)
#     assert isinstance(dataset[0].get('annotations'), list)


# def test_create_augmented_datasets_sucess():
#     obj = KeypointDetectionModeling(sp.DOC_ML_DATASET, sp.DOC_ML_DATASET)
#     train_gen, valid_gen = obj.create_augmented_datasets()
#     assert isinstance(train_gen, KeypointDataGenerator)
#     assert isinstance(valid_gen, KeypointDataGenerator)
