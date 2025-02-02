import os

import numpy as np
import pytest

from cucaracha import Document
from cucaracha.tasks.aligment import inplane_deskew
from cucaracha.tasks.noise_removal import sparse_dots
from cucaracha.tasks.threshold import otsu
from tests import sample_paths


@pytest.mark.parametrize(
    'img_path',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG),
            (sample_paths.SAMPLE_TEXT_PNG),
            (sample_paths.SAMPLE_TEXT_TIF),
            (sample_paths.SAMPLE_TEXT_PDF),
        ]
    ),
)
def test_create_cucaracha_object_returns_numpy_array(img_path):
    obj = Document(img_path)
    arr = obj._doc_file[0]
    assert isinstance(arr, np.ndarray)


@pytest.mark.parametrize(
    'pdf_path,res_minor,res_major',
    (
        [
            (sample_paths.SAMPLE_TEXT_PDF, 96, 150),
            (sample_paths.SAMPLE_TEXT_PDF, 150, 300),
            (sample_paths.SAMPLE_TEXT_PDF, 300, 600),
            (sample_paths.SAMPLE_TEXT_PDF, 600, 1200),
        ]
    ),
)
def test_create_cucaracha_object_returns_numpy_array_change_dimensions_PDF_based_on_resolution(
    pdf_path, res_minor, res_major
):
    obj_pdf_minor = Document(pdf_path, resolution=res_minor)
    obj_pdf_major = Document(pdf_path, resolution=res_major)
    assert obj_pdf_minor._doc_file[0].size < obj_pdf_major._doc_file[0].size
    assert (
        obj_pdf_minor._doc_file[0].shape[0]
        < obj_pdf_major._doc_file[0].shape[0]
    )
    assert (
        obj_pdf_minor._doc_file[0].shape[1]
        < obj_pdf_major._doc_file[0].shape[1]
    )


@pytest.mark.parametrize(
    'img_path',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG),
            (sample_paths.SAMPLE_TEXT_PNG),
            (sample_paths.SAMPLE_TEXT_TIF),
            (sample_paths.SAMPLE_TEXT_PDF),
        ]
    ),
)
def test_load_document_update_class_doc_file(img_path):
    obj = Document()
    assert obj._doc_file == []

    obj.load_document(img_path)
    assert len(obj._doc_file) > 0


@pytest.mark.parametrize(
    'img_path,metadata',
    (
        [
            (
                sample_paths.SAMPLE_TEXT_JPG,
                {
                    'file_ext': '.jpg',
                    'file_name': 'sample-text-en',
                    'pages': 1,
                    'size': 0.03,
                },
            ),
            (
                sample_paths.SAMPLE_TEXT_PNG,
                {
                    'file_ext': '.png',
                    'file_name': 'sample-text-en',
                    'pages': 1,
                    'size': 0.06,
                },
            ),
            (
                sample_paths.SAMPLE_TEXT_TIF,
                {
                    'file_ext': '.tif',
                    'file_name': 'sample-text-en',
                    'pages': 1,
                    'size': 0.124,
                },
            ),
            (
                sample_paths.SAMPLE_TEXT_PDF,
                {
                    'file_ext': '.pdf',
                    'file_name': 'sample-text-en',
                    'pages': 1,
                    'size': 0.04,
                },
            ),
        ]
    ),
)
def test_create_cucaracha_object_update_file_metadata_with_file_inner_metadata(
    img_path, metadata
):
    obj = Document(img_path)
    inner_infos = ['file_ext', 'file_name', 'pages']
    for key, _ in obj._doc_metadata.items():
        if key in inner_infos:
            assert obj._doc_metadata.get(key) == metadata.get(key)

    assert (
        metadata['size'] - 5
        < obj._doc_metadata.get('size')
        < metadata['size'] + 5
    )


@pytest.mark.parametrize(
    'img_path',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG),
            (sample_paths.SAMPLE_TEXT_PNG),
            (sample_paths.SAMPLE_TEXT_TIF),
            (sample_paths.SAMPLE_TEXT_PDF),
        ]
    ),
)
def test_save_document_using_only_filname_keep_file_from_default_file_path_from_metadata(
    img_path, tmp_path
):
    obj = Document(img_path)
    obj.save_document(
        file_name=tmp_path.as_posix()
        + os.sep
        + 'test_img'
        + obj._doc_metadata.get('file_ext')
    )
    assert os.path.exists(
        tmp_path.as_posix()
        + os.sep
        + 'test_img'
        + obj._doc_metadata['file_ext']
    )


@pytest.mark.parametrize(
    'fullpath,filename',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG, 'sample-text-en_test.jpg'),
            (sample_paths.SAMPLE_TEXT_PNG, 'sample-text-en_test.png'),
            (sample_paths.SAMPLE_TEXT_TIF, 'sample-text-en_test.tif'),
            (sample_paths.SAMPLE_TEXT_PDF, 'sample-text-en_test.pdf'),
        ]
    ),
)
def test_save_document_with_filname_only_without_filepath_attached_can_save_using_default_file_path(
    fullpath, filename
):
    obj = Document(fullpath)
    obj.save_document(file_name=filename)
    out_fullpath = (
        obj._doc_metadata['file_path']
        + os.sep
        + 'sample-text-en_test'
        + obj._doc_metadata['file_ext']
    )

    assert os.path.exists(out_fullpath)
    os.remove(out_fullpath)


def test_save_document_raise_error_passing_only_filename_but_without_enough_metadata():
    obj = Document()
    with pytest.raises(Exception) as e:
        obj.save_document(file_name='test_img.png')

    assert (
        e.value.args[0] == 'Document metadata does not have a valid file path.'
    )


@pytest.mark.parametrize(
    'img_path',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG),
            (sample_paths.SAMPLE_TEXT_PNG),
            (sample_paths.SAMPLE_TEXT_TIF),
            (sample_paths.SAMPLE_TEXT_PDF),
        ]
    ),
)
def test_save_document_raise_error_when_file_name_does_not_has_file_format(
    img_path,
):
    obj = Document(img_path)
    with pytest.raises(Exception) as e:
        obj.save_document(file_name='test_img')

    assert (
        e.value.args[0]
        == 'File name must indicates the file format (ex: .pdf, .jpg, .png, etc)'
    )


@pytest.mark.parametrize(
    'img_path',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG),
            (sample_paths.SAMPLE_TEXT_PNG),
            (sample_paths.SAMPLE_TEXT_TIF),
            (sample_paths.SAMPLE_TEXT_PDF),
        ]
    ),
)
def test_get_page_returns_a_single_page_as_numpy_array(img_path):
    obj = Document(img_path)
    pages = []
    for page in range(obj._doc_metadata.get('pages')):
        pages.append(obj.get_page(page))

    for page in pages:
        assert isinstance(page, np.ndarray)


@pytest.mark.parametrize(
    'img_path,page',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG, -1),
            (sample_paths.SAMPLE_TEXT_PNG, 10),
            (sample_paths.SAMPLE_TEXT_TIF, 200),
            (sample_paths.SAMPLE_TEXT_PDF, -200),
        ]
    ),
)
def test_get_page_raise_error_with_page_not_in_document_pages_metadata(
    img_path, page
):
    obj = Document(img_path)
    with pytest.raises(Exception) as e:
        obj.get_page(page)
    assert e.value.args[0] == 'page number is not present at the document'


@pytest.mark.parametrize(
    'img_path,info,type_out',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG, 'file_ext', str),
            (sample_paths.SAMPLE_TEXT_PNG, 'size', float),
            (sample_paths.SAMPLE_TEXT_TIF, 'resolution', int),
            (sample_paths.SAMPLE_TEXT_PDF, 'file_path', str),
        ]
    ),
)
def test_get_metadata_returns_the_object_metadata(img_path, info, type_out):
    obj = Document(img_path)
    meta = obj.get_metadata(info)

    assert type(meta[info]) == type_out


@pytest.mark.parametrize(
    'img_path',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG),
            (sample_paths.SAMPLE_TEXT_PNG),
            (sample_paths.SAMPLE_TEXT_TIF),
            (sample_paths.SAMPLE_TEXT_PDF),
        ]
    ),
)
def test_get_metadata_returns_the_object_metadata_with_info_equal_to_None(
    img_path,
):
    obj = Document(img_path)
    meta = obj.get_metadata()

    assert type(meta) == dict
    assert type(meta.get('file_ext')) == str
    assert type(meta.get('file_name')) == str
    assert type(meta.get('file_path')) == str
    assert meta.get('size') > 0
    assert type(meta.get('resolution')) == int


@pytest.mark.parametrize(
    'img_path,info',
    (
        [
            (sample_paths.SAMPLE_TEXT_JPG, 'not_a_key'),
            (sample_paths.SAMPLE_TEXT_PNG, 'other_wrong_key'),
            (sample_paths.SAMPLE_TEXT_TIF, 'Size'),
            (sample_paths.SAMPLE_TEXT_PDF, 'filename'),
        ]
    ),
)
def test_get_metadata_with_wrong_info_raise_error(img_path, info):
    obj = Document(img_path)
    with pytest.raises(Exception) as e:
        obj.get_metadata(info)

    assert (
        e.value.args[0]
        == 'Info is not provided in the Document class metadata'
    )


def test_run_pipeline_raise_error_when_list_of_processors_has_invalid_method():
    def proc2(input):
        return sparse_dots(input, 3)

    def proc3(input):
        return inplane_deskew(input, 25)

    proc_list = [otsu, proc2, proc3, np.array]

    doc = Document(sample_paths.SAMPLE_CUCARACHA_GRAY_TIF)
    with pytest.raises(Exception) as e:
        doc.run_pipeline(proc_list)

    assert (
        e.value.args[0]
        == 'Processor: array is not valid. Unsure that the output processor is valid.'
    )


def test_run_pipeline_raise_error_when_processors_is_not_a_list():
    proc_list = otsu

    doc = Document(sample_paths.SAMPLE_CUCARACHA_GRAY_TIF)
    with pytest.raises(Exception) as e:
        doc.run_pipeline(proc_list)

    assert (
        e.value.args[0]
        == 'processors must be a list of valid cucaracha filter methods'
    )


def test_run_pipeline_with_valid_processors_methods():
    def proc2(input):
        return sparse_dots(input, 3)

    def proc3(input):
        return inplane_deskew(input, 25)

    proc_list = [otsu, proc2, proc3]

    doc = Document(sample_paths.SAMPLE_CUCARACHA_GRAY_TIF)
    doc.run_pipeline(proc_list)

    assert isinstance(doc.get_page(0), np.ndarray)


@pytest.mark.parametrize(
    'page,wrong_index',
    [
        (sample_paths.SAMPLE_CUCARACHA_GRAY_GAUSS_PNG, 2),
        (sample_paths.SAMPLE_CUCARACHA_GRAY_GAUSS_PNG, -1),
    ],
)
def test_set_page_raise_error_when_index_is_out_of_document_pages_range(
    page, wrong_index
):
    doc = Document(page)
    new_page = np.ones(doc.get_page(0).shape)

    with pytest.raises(Exception) as e:
        doc.set_page(new_page, wrong_index)

    assert (
        e.value.args[0]
        == 'Page index is out of range (total page is 1 and must be a positive integer)'
    )


@pytest.mark.parametrize(
    'page',
    [
        (sample_paths.SAMPLE_CUCARACHA_GRAY_GAUSS_PNG),
        (sample_paths.SAMPLE_TEXT_JPG),
        (sample_paths.SAMPLE_TEXT_PDF),
    ],
)
def test_set_page_raise_error_when_page_is_not_an_numpy_array(page):
    doc = Document(page)
    new_wrong_page = np.ones((1, 1, 1))

    with pytest.raises(Exception) as e:
        doc.set_page(new_wrong_page, 0)

    assert (
        e.value.args[0]
        == 'New page is not a numpy array or has different shape from previous pages'
    )


@pytest.mark.parametrize(
    'page,index',
    [
        (sample_paths.SAMPLE_CUCARACHA_GRAY_GAUSS_PNG, 0),
        (sample_paths.SAMPLE_CUCARACHA_GRAY_TIF, 0),
        (sample_paths.SAMPLE_TEXT_PDF, 0),
        (sample_paths.SAMPLE_TEXT_PNG, 0),
    ],
)
def test_set_page_change_correctly_document_page(page, index):
    doc = Document(page)
    new_page = np.ones(doc.get_page(0).shape)

    doc.set_page(new_page, index)

    assert np.max(doc.get_page(0)) == 1
