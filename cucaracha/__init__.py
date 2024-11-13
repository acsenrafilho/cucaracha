import os

import cv2 as cv
import numpy as np
import pymupdf
from pymupdf import Page


class Document:
    """The general concept of `Document`for the `cucaracha` library."""

    def __init__(self, doc_path: str = None, **kwargs):
        """Document class constructor

        This is the basic model to a document in the cucaracha library.
        It is important to notice that the basic data for processing and
        analysis is a Numpy array, which is automatically loaded using the
        input `doc_path`.

        The input data can be passed after the object creation. However, take
        care about the metadata created at the object instantiation. When there
        is no input path provided, the default information is used, being mostly
        `None` type.

        Note:
            It is used the PyMuPDF and OpenCV libraries to allow the loading
            data into cucaracha Document object. Both libraries have extensive
            documentation informating the image files formats avaliable. See
            more details at:

            - [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/index.html)

            - [OpenCV](https://opencv.org/)

        Args:
            doc_path (str, optional): Document path to be loaded. If None, a general object is created with `None` values in metadata information. Defaults to None.
        """
        self._doc_metadata = {
            'file_ext': None,
            'file_path': None,
            'file_name': None,
            'resolution': 96
            if kwargs.get('resolution') == None
            else int(kwargs.get('resolution')),
            'pages': None,
            'size': None,
        }

        self._doc_file = []
        if doc_path is not None:
            self._doc_file = self._read_by_ext(
                doc_path, dpi=self._doc_metadata['resolution']
            )

        self._collect_inner_metadata(doc_path)

    def load_document(self, path: str):
        """Load document using a full path.

        If the Document object was instantiated using a `None` value for path,
        it can be called this method to update the document data inside de object

        This method is called internally bu the `Document()` constructor.

        Note:
            It is used the PyMuPDF and OpenCV libraries to allow the loading
            data into cucaracha Document object. Both libraries have extensive
            documentation informating the image files formats avaliable. See
            more details at:

            - [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/index.html)

            - [OpenCV](https://opencv.org/)

        Args:
            path (str): Document full path to be loaded.
        """
        self._doc_file = self._read_by_ext(
            path, dpi=self._doc_metadata['resolution']
        )

    def save_document(self, file_name: str):
        """Saves the Document state as a file.

        The user can choose the file format by defining on it's naming

        The conversion if based on the file format. If the `.pdf` extension
        is passed, then the PyMuPDF constructor ir used. If an image file is
        passed, e.g. `.jpg`, `.png` and so on, the OpenCV constructor is used.

        Note:
            This method saves the actual state of the document object. Hence,
            after all the image processing being made, it is possible to save
            document status using this method

        Note:
            The file path can be seen by calling the `get_metadata('file_path')`
            command, where it recovery the original file path that was given at
            the moment the object was created.

        Args:
            file_name (str): File path where it should be save in the hard
            drive. If a single filename is passed, then the original image
            path is used from the constructor metadata.

        Raises:
            ValueError: Document metadata does not have a valid file path.
            TypeError: File name must indicates the file format (ex: .pdf, .jpg, .png, etc)
        """
        if self._doc_metadata.get('file_path') is None:
            raise ValueError(
                f'Document metadata does not have a valid file path.'
            )

        filename, file_ext = os.path.splitext(file_name)
        if file_ext == '':
            raise TypeError(
                'File name must indicates the file format (ex: .pdf, .jpg, .png, etc)'
            )

        if os.sep in filename:
            if file_ext != '.pdf':
                # Save using opencv
                for page in range(self._doc_metadata.get('pages')):
                    cv.imwrite(file_name, self._doc_file[page])
            else:
                # Save using PyMuPDF
                # Create a temporary file image
                for page in range(self._doc_metadata.get('pages')):
                    cv.imwrite(
                        self._doc_metadata.get('file_path')
                        + filename
                        + '_tmp.png',
                        self._doc_file[page],
                    )
                doc = pymupdf.open()                           # new PDF
                for page in range(self._doc_metadata.get('pages')):
                    tmp_img_path = filename + '_tmp_pg_' + str(page) + '.png'
                    cv.imwrite(tmp_img_path, self._doc_file[page])

                    # open image as a document
                    imgdoc = pymupdf.open(tmp_img_path)
                    # make a 1-page PDF of it
                    pdfbytes = imgdoc.convert_to_pdf()
                    imgdoc.close()
                    imgpdf = pymupdf.open('pdf', pdfbytes)
                    # insert the image PDF
                    doc.insert_pdf(imgpdf)

                    # Removing tmp file
                    os.remove(tmp_img_path)

                doc.save(file_name)
        else:
            if file_ext != '.pdf':
                # Save using opencv
                for page in range(self._doc_metadata.get('pages')):
                    cv.imwrite(
                        self._doc_metadata.get('file_path') + file_name,
                        self._doc_file[page],
                    )
            else:
                # Save using PyMuPDF
                # Create a temporary file image
                for page in range(self._doc_metadata.get('pages')):
                    cv.imwrite(
                        self._doc_metadata.get('file_path')
                        + filename
                        + '_tmp.png',
                        self._doc_file[page],
                    )

                doc = pymupdf.open()                           # new PDF
                for page in range(self._doc_metadata.get('pages')):
                    tmp_img_path = (
                        self._doc_metadata.get('file_path')
                        + filename
                        + '_tmp.png'
                    )
                    cv.imwrite(tmp_img_path, self._doc_file[page])

                    # open image as a document
                    imgdoc = pymupdf.open(tmp_img_path)
                    # make a 1-page PDF of it
                    pdfbytes = imgdoc.convert_to_pdf()
                    imgpdf = pymupdf.open('pdf', pdfbytes)
                    # insert the image PDF
                    doc.insert_pdf(imgpdf)

                    # Removing tmp file
                    os.remove(tmp_img_path)

                doc.save(self._doc_metadata.get('file_path') + file_name)

    def get_metadata(self, info: str = None):
        """Collect the document metadata that informs general information
        about the data construction and parameters.

        This method can be called setting the type of information that you want
        to retrieve. For instance, one can see the `resolution` of the data
        object, then:

        Examples:
            >>> doc = Document('.'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.pdf')
            >>> doc.get_metadata('resolution')
            {'resolution': 96}
            >>> doc.get_metadata('file_ext')
            {'file_ext': '.pdf'}

        If the method is called without providing a specific information, then
        all the metadata is shown

        Examples:
            >>> doc = Document('.'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.pdf')
            >>> meta = doc.get_metadata()
            >>> type(meta)
            <class 'dict'>
            >>> meta.keys()
            dict_keys(['file_ext', 'file_path', 'file_name', 'resolution', 'pages', 'size'])

        Args:
            info (str, optional): The kind of information that desired to obtain in the document metadata. Defaults to `None`, then all the metada is shown.

        Raises:
            KeyError: Info is not provided in the Document class metadata

        Returns:
            dict: _description_
        """
        if info in self._doc_metadata.keys():
            return {info: self._doc_metadata.get(info)}
        elif info is None:
            return self._doc_metadata
        else:
            raise KeyError(
                'Info is not provided in the Document class metadata'
            )

    def get_page(self, page: int):
        """Returns a determined page of the document defined by the `page`
        parameter

        The `page` value must be inside the range of possible pages that the
        document has. If not, an error is exposed.

        Info:
            The pages counting starts from zero (`0`)

        Examples:
            >>> doc = Document('./'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.pdf')
            >>> page = doc.get_page(0)
            >>> page.shape
            (103, 103, 3)


        Args:
            page (int): The page number that you want to collect

        Raises:
            ValueError: page number is not present at the document

        Returns:
            np.ndarray: The selected page extracted by Numpy array format
        """
        if page not in range(self._doc_metadata.get('pages')):
            raise ValueError('page number is not present at the document')

        return self._doc_file[page]

    def batch_processing(self, processors: list):
        # TODO Make a loop processor to make image processing to the doc_file
        pass

    def _read_by_ext(self, path, dpi):
        _, file_ext = os.path.splitext(path)

        out_file = []
        if file_ext != '.pdf':
            out_file = [cv.imread(path)]
        else:
            out_file = self._read_pdf(path, dpi)

        return out_file

    def _read_pdf(self, path, dpi):
        doc = pymupdf.open(path)  # open document
        out_file = []
        for page in doc:  # iterate through the pages
            pix = page.get_pixmap(dpi=dpi)
            im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )
            im = np.ascontiguousarray(im[..., [2, 1, 0]])
            out_file.append(im)

        return out_file

    def _collect_inner_metadata(self, doc_path):
        if doc_path is not None:
            # Set file_ext, file_path and file_name
            fullpath, file_ext = os.path.splitext(doc_path)
            self._doc_metadata['file_ext'] = file_ext

            lpath = fullpath.split(sep=os.sep)
            self._doc_metadata['file_path'] = os.sep.join(lpath[:-1]) + os.sep
            self._doc_metadata['file_name'] = lpath[-1]

            # Set file size
            self._doc_metadata['size'] = (
                os.path.getsize(doc_path) / 1024**2
            )   # informs size in Mb

            # Set file number of pages
            self._doc_metadata['pages'] = len(self._doc_file)
