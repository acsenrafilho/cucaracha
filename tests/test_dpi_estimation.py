import numpy as np
import pytest

from cucaracha import Document
from tests import sample_paths


class TestDPIEstimation:
    """Test suite for DPI estimation functionality."""

    @pytest.mark.parametrize(
        'img_path',
        [
            sample_paths.SAMPLE_TEXT_PDF,
            sample_paths.SAMPLE_TEXT_JPG,
            sample_paths.SAMPLE_TEXT_PNG,
            sample_paths.SAMPLE_TEXT_TIF,
        ],
    )
    def test_estimate_dpi_auto_method_returns_valid_dpi(self, img_path):
        """Test that auto DPI estimation returns a reasonable DPI value."""
        doc = Document(img_path)
        estimated_dpi = doc.estimate_dpi(method='auto')

        # DPI should be a positive integer in reasonable range
        assert isinstance(estimated_dpi, int)
        assert 50 <= estimated_dpi <= 600

    def test_estimate_dpi_pdf_dimensions_method_for_pdf(self):
        """Test PDF-specific DPI estimation method."""
        doc = Document(sample_paths.SAMPLE_TEXT_PDF)
        estimated_dpi = doc.estimate_dpi(method='pdf_dimensions')

        assert isinstance(estimated_dpi, int)
        assert estimated_dpi > 0

    def test_estimate_dpi_pdf_dimensions_method_raises_error_for_non_pdf(self):
        """Test that PDF dimensions method raises error for non-PDF files."""
        doc = Document(sample_paths.SAMPLE_TEXT_JPG)

        with pytest.raises(
            ValueError, match='This method only works with PDF files'
        ):
            doc.estimate_dpi(method='pdf_dimensions')

    @pytest.mark.parametrize(
        'img_path',
        [
            sample_paths.SAMPLE_TEXT_JPG,
            sample_paths.SAMPLE_TEXT_PNG,
            sample_paths.SAMPLE_TEXT_TIF,
        ],
    )
    def test_estimate_dpi_page_size_method_for_images(self, img_path):
        """Test page size estimation method for image files."""
        doc = Document(img_path)
        estimated_dpi = doc.estimate_dpi(method='page_size', assume_a4=True)

        assert isinstance(estimated_dpi, int)
        assert estimated_dpi > 0

    def test_estimate_dpi_page_size_method_without_a4_assumption_raises_error(
        self,
    ):
        """Test that page size method without A4 assumption raises error for images."""
        doc = Document(sample_paths.SAMPLE_TEXT_JPG)

        with pytest.raises(
            ValueError, match='Cannot estimate DPI without size assumptions'
        ):
            doc.estimate_dpi(method='page_size', assume_a4=False)

    def test_estimate_dpi_with_invalid_method_raises_error(self):
        """Test that invalid estimation method raises appropriate error."""
        doc = Document(sample_paths.SAMPLE_TEXT_PDF)

        with pytest.raises(ValueError, match='Unsupported estimation method'):
            doc.estimate_dpi(method='invalid_method')

    def test_estimate_dpi_with_no_document_loaded_raises_error(self):
        """Test that DPI estimation raises error when no document is loaded."""
        doc = Document()  # Create empty document

        with pytest.raises(
            ValueError, match='No document loaded for DPI estimation'
        ):
            doc.estimate_dpi()

    def test_estimate_dpi_consistency_between_methods_for_pdf(self):
        """Test that different methods give consistent results for PDF files."""
        doc = Document(sample_paths.SAMPLE_TEXT_PDF)

        auto_dpi = doc.estimate_dpi(method='auto')
        pdf_dpi = doc.estimate_dpi(method='pdf_dimensions')

        # For PDF files, auto method should use pdf_dimensions
        assert auto_dpi == pdf_dpi

    def test_estimate_dpi_creates_reasonable_dpi_for_current_samples(self):
        """Test that DPI estimation works appropriately for current sample files."""
        # These sample files are small excerpts, not full A4 pages
        # So the estimation should either return the actual DPI or a reasonable default

        doc_pdf = Document(sample_paths.SAMPLE_TEXT_PDF)
        estimated_pdf = doc_pdf.estimate_dpi()

        doc_jpg = Document(sample_paths.SAMPLE_TEXT_JPG)
        estimated_jpg = doc_jpg.estimate_dpi()

        # Both should return reasonable values
        assert 50 <= estimated_pdf <= 600
        assert 50 <= estimated_jpg <= 600

        # For our small sample files, the estimation should be sensible
        # PDF should match the current DPI used for rendering
        current_dpi = doc_pdf.get_metadata('resolution')['resolution']
        assert estimated_pdf == current_dpi

    def test_constructor_with_estimate_dpi_parameter(self):
        """Test that constructor respects estimate_dpi parameter."""
        # Normal construction without estimation
        doc1 = Document(sample_paths.SAMPLE_TEXT_PDF)
        normal_dpi = doc1.get_metadata('resolution')['resolution']

        # Construction with DPI estimation enabled
        doc2 = Document(sample_paths.SAMPLE_TEXT_PDF, estimate_dpi=True)
        estimated_dpi = doc2.get_metadata('resolution')['resolution']

        # For our sample PDF, estimation should give the same result as default
        assert estimated_dpi == normal_dpi

        # Construction with explicit resolution should ignore estimation
        doc3 = Document(
            sample_paths.SAMPLE_TEXT_PDF, resolution=150, estimate_dpi=True
        )
        explicit_dpi = doc3.get_metadata('resolution')['resolution']
        assert explicit_dpi == 150

    def test_constructor_estimate_dpi_does_not_break_on_estimation_failure(
        self,
    ):
        """Test that constructor gracefully handles DPI estimation failures."""
        # This should not raise an exception even if estimation fails
        doc = Document(sample_paths.SAMPLE_TEXT_JPG, estimate_dpi=True)

        # Should have a valid DPI value
        dpi = doc.get_metadata('resolution')['resolution']
        assert isinstance(dpi, int)
        assert dpi > 0
