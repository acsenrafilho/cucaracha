#!/usr/bin/env python3
"""
Demonstration script for DPI estimation functionality in cucaracha.

This script shows how to use the new DPI estimation methods to automatically
determine appropriate DPI values for documents when not available in headers.
"""

from cucaracha import Document
from tests import sample_paths


def demonstrate_dpi_estimation():
    """Demonstrate various DPI estimation scenarios."""
    
    print("=== Cucaracha DPI Estimation Demo ===\n")
    
    # Example 1: Basic usage
    print("1. Basic DPI Estimation:")
    doc = Document(sample_paths.SAMPLE_TEXT_PDF)
    estimated_dpi = doc.estimate_dpi()
    current_dpi = doc.get_metadata('resolution')['resolution']
    print(f"   Current DPI: {current_dpi}")
    print(f"   Estimated DPI: {estimated_dpi}")
    print(f"   Document shape: {doc.get_page(0).shape}")
    
    # Example 2: Different estimation methods
    print("\n2. Different Estimation Methods:")
    print(f"   Auto method: {doc.estimate_dpi(method='auto')}")
    print(f"   PDF dimensions method: {doc.estimate_dpi(method='pdf_dimensions')}")
    
    # Example 3: Constructor with DPI estimation
    print("\n3. Constructor with DPI Estimation:")
    doc_with_estimation = Document(sample_paths.SAMPLE_TEXT_PDF, estimate_dpi=True)
    print(f"   DPI with estimation enabled: {doc_with_estimation.get_metadata('resolution')['resolution']}")
    print(f"   Document shape: {doc_with_estimation.get_page(0).shape}")
    
    # Example 4: Image file estimation
    print("\n4. Image File Estimation:")
    doc_img = Document(sample_paths.SAMPLE_TEXT_JPG)
    estimated_img_dpi = doc_img.estimate_dpi()
    print(f"   Image DPI estimation: {estimated_img_dpi}")
    print(f"   Image shape: {doc_img.get_page(0).shape}")
    
    # Example 5: Comparison of different resolutions
    print("\n5. Resolution Comparison:")
    doc_96 = Document(sample_paths.SAMPLE_TEXT_PDF, resolution=96)
    doc_150 = Document(sample_paths.SAMPLE_TEXT_PDF, resolution=150)
    doc_300 = Document(sample_paths.SAMPLE_TEXT_PDF, resolution=300)
    
    print(f"   96 DPI:  {doc_96.get_page(0).shape}")
    print(f"   150 DPI: {doc_150.get_page(0).shape}")
    print(f"   300 DPI: {doc_300.get_page(0).shape}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_dpi_estimation()