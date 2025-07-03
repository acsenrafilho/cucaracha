import os

import cv2 as cv
import numpy as np
import pymupdf
from pymupdf import Page
from rich import print
from rich.progress import track

from cucaracha.tasks.aligment import inplane_deskew
from cucaracha.tasks.noise_removal import sparse_dots
from cucaracha.tasks.text_extraction import (
    extract_text,
    extract_text_easyocr,
    extract_text_tesseract,
)
from cucaracha.tasks.threshold import otsu


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
            resolution (int, optional): DPI resolution to use for document rendering. If None, defaults to 96. Defaults to None.
            estimate_dpi (bool, optional): Whether to estimate DPI when not provided in headers. Only applies when resolution is not explicitly set. Defaults to False.
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

        # Optionally estimate DPI if requested and resolution wasn't explicitly set
        if (
            doc_path is not None
            and kwargs.get('estimate_dpi', False)
            and kwargs.get('resolution') is None
        ):
            try:
                estimated_dpi = self.estimate_dpi()
                if estimated_dpi != self._doc_metadata['resolution']:
                    # Re-load document with estimated DPI if it's different
                    self._doc_metadata['resolution'] = estimated_dpi
                    self._doc_file = self._read_by_ext(
                        doc_path, dpi=estimated_dpi
                    )
            except (ValueError, NotImplementedError):
                # If estimation fails, keep the default DPI
                pass

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

    def set_page(self, page: np.ndarray, index: int):
        """Update a new page into the document file

        The page index must be passed considering the total range of pages
        in the document. See the metadata to get this information.

        Examples:
            >>> doc = Document('./'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.pdf')
            >>> doc.get_metadata('pages')
            {'pages': 1}

            The original information is loaded as usual
            >>> np.max(doc.get_page(0))
            255

            But a new page can be changed like this:
            >>> new_page = np.ones(doc.get_page(0).shape)
            >>> doc.set_page(new_page, 0)

            Then the new page is placed in the document object
            >>> np.max(doc.get_page(0))
            1.0

        Args:
            page (np.ndarray): A numpy array with the same shape of the other pages
            index (int): The index where the new page should be placed

        Raises:
            ValueError: Page index is out of range (total page is ... and must be a positive integer)
            ValueError: New page is not a numpy array or has different shape from previous pages
        """
        if index > len(self._doc_file) or index < 0:
            raise ValueError(
                f'Page index is out of range (total page is {len(self._doc_file)} and must be a positive integer)'
            )

        if (
            not isinstance(page, np.ndarray)
            or page.shape != self.get_page(index).shape
        ):
            raise ValueError(
                'New page is not a numpy array or has different shape from previous pages'
            )

        self._doc_file[index] = page

    def run_pipeline(self, processors: list):
        """Execute a list of image processing methods to the document file
        allocated in the `Document` object.

        The processing order is the same as indicated in the list of processors.

        Examples:
            One can define a processor as a function caller:
            >>> def proc2(input): return sparse_dots(input, 3)
            >>> def proc3(input): return inplane_deskew(input, 25)
            >>> proc_list = [otsu, proc2, proc3]

            After the `proc_list` being created, the proper execution can be
            called using:
            >>> doc = Document('.'+os.sep+'tests'+os.sep+'files'+os.sep+'sample-text-en.pdf')
            >>> doc.run_pipeline(proc_list) # doctest: +SKIP
            Applying processors... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

            Hence, the inner document file in the `doc` object is updated:
            >>> type(doc.get_page(0))
            <class 'numpy.ndarray'>

        Warning:
            All the processor in the list must be of `cucaracha` filter type.
            Hence, make sure that the processor instance accepts an numpy array
            as input and returns a tuple with numpy array and a dictionary of
            extra parameters (`(np.ndarray, dict)`).

        Note:
            All the pages presented in the document object is processed. If it
            is desired to apply only on specific pages, then it is need to
            process it individually and then update the page using the method
            `set_page`

        Args:
            processors (list): _description_
        """
        self._check_processor_list(processors)

        for proc in track(
            processors, description='[green]Applying processors...'
        ):
            for idx, page in enumerate(self._doc_file):
                self._doc_file[idx] = proc(page)[0]

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

    def estimate_dpi(self, method: str = 'auto', assume_a4: bool = True):
        """Estimate the DPI (dots per inch) of the document when not available in headers.

        This method provides different approaches to estimate DPI:
        1. 'auto': Automatically chooses the best method based on file type
        2. 'page_size': Estimates based on physical page dimensions (assumes A4 by default)
        3. 'pdf_dimensions': Uses PDF internal page dimensions (PDF files only)

        Args:
            method (str): Estimation method to use. Options: 'auto', 'page_size', 'pdf_dimensions'
            assume_a4 (bool): Whether to assume A4 page size (210 × 297 mm) for page_size method

        Returns:
            int: Estimated DPI value

        Raises:
            ValueError: If method is not supported or estimation cannot be performed

        Examples:
            >>> doc = Document('./tests/files/sample-text-en.pdf')
            >>> estimated_dpi = doc.estimate_dpi()
            >>> print(f"Estimated DPI: {estimated_dpi}")
            Estimated DPI: 96
        """
        if method == 'auto':
            # For PDF files, prefer using internal dimensions
            if self._doc_metadata.get('file_ext') == '.pdf':
                return self._estimate_dpi_from_pdf_page_size()
            else:
                # For image files, use A4 assumption as fallback
                return self._estimate_dpi_from_page_size(assume_a4)
        elif method == 'page_size':
            return self._estimate_dpi_from_page_size(assume_a4)
        elif method == 'pdf_dimensions':
            return self._estimate_dpi_from_pdf_page_size()
        else:
            raise ValueError(f'Unsupported estimation method: {method}')

    def _estimate_dpi_from_page_size(self, assume_a4: bool = True):
        """Estimate DPI based on assumed page size dimensions.

        Args:
            assume_a4 (bool): Whether to assume A4 page size (210 × 297 mm)

        Returns:
            int: Estimated DPI value

        Note:
            This method makes assumptions about the physical size of the document.
            For small document excerpts or non-standard sizes, the estimate may be inaccurate.
        """
        if len(self._doc_file) == 0:
            raise ValueError('No document loaded for DPI estimation')

        # Get the first page dimensions in pixels
        page = self._doc_file[0]
        height_px, width_px = page.shape[:2]

        if assume_a4:
            # A4 dimensions: 210 mm × 297 mm = 8.27 in × 11.69 in
            a4_width_inch = 210 / 25.4  # Convert mm to inches
            a4_height_inch = 297 / 25.4

            # Calculate DPI based on both dimensions and take average
            dpi_width = width_px / a4_width_inch
            dpi_height = height_px / a4_height_inch
            estimated_dpi = int((dpi_width + dpi_height) / 2)

            # Sanity check: if estimated DPI is too low/high, it's likely not A4
            if estimated_dpi < 50 or estimated_dpi > 600:
                # Document likely not A4 sized, return a reasonable default
                print(
                    f'Warning: Estimated DPI ({estimated_dpi}) seems unrealistic for A4. Document may not be A4 sized.'
                )
                return 96  # Return reasonable default

            return estimated_dpi
        else:
            raise ValueError(
                'Cannot estimate DPI without size assumptions for image files'
            )

    def _estimate_dpi_from_pdf_page_size(self):
        """Estimate DPI from PDF page size using PyMuPDF page dimensions.

        This method uses the actual page dimensions stored in the PDF file
        to calculate the DPI based on the pixel dimensions of the rendered page.

        Returns:
            int: Estimated DPI value
        """
        if self._doc_metadata.get('file_ext') != '.pdf':
            raise ValueError('This method only works with PDF files')

        # Re-open the PDF to get page dimensions
        doc_path = os.path.join(
            self._doc_metadata.get('file_path'),
            self._doc_metadata.get('file_name')
            + self._doc_metadata.get('file_ext'),
        )

        doc = pymupdf.open(doc_path)
        page = doc[0]
        page_rect = page.rect

        # Page rect is in points (1/72 inch)
        page_width_inch = page_rect.width / 72.0
        page_height_inch = page_rect.height / 72.0

        # Get pixel dimensions at current DPI
        page_array = self._doc_file[0]
        height_px, width_px = page_array.shape[:2]

        # Calculate actual DPI used during rendering
        if page_width_inch > 0 and page_height_inch > 0:
            dpi_width = width_px / page_width_inch
            dpi_height = height_px / page_height_inch
            estimated_dpi = int((dpi_width + dpi_height) / 2)
        else:
            estimated_dpi = 96  # Fallback to default

        doc.close()
        return estimated_dpi

    def _check_processor_list(self, processors):
        if type(processors) != list:
            raise ValueError(
                'processors must be a list of valid cucaracha filter methods'
            )

        for proc in processors:
            out_test = proc(self.get_page(0))   # Test the processor output
            if (
                type(out_test) != tuple
                or not isinstance(out_test[0], np.ndarray)
                or not isinstance(out_test[1], dict)
            ):
                raise TypeError(
                    f'Processor: {proc.__name__} is not valid. Unsure that the output processor is valid.'
                )
