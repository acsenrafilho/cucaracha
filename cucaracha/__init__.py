import os

import cv2 as cv
import numpy as np
import pymupdf
from pymupdf import Page


class Document:
    def __init__(self, doc_path: str = None, **kwargs):
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
        self._doc_file = self._read_by_ext(
            path, dpi=self._doc_metadata['resolution']
        )

    def save_document(self, file_name: str):
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
        if info in self._doc_metadata.keys():
            return self._doc_metadata.get(info)
        elif info is None:
            return self._doc_metadata
        else:
            raise KeyError(
                'Info is not provided in the Document class metadata'
            )

    def get_page(self, page: int):
        if page not in range(self._doc_metadata.get('pages')):
            raise ValueError('page number is not present at the document')

        return self._doc_file[page]

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
