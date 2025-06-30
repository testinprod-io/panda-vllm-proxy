from langchain_community.document_loaders.parsers import BaseImageBlobParser
from PIL.Image import Image
import numpy as np

from ..logger import log

class RapidOCRBlobParser(BaseImageBlobParser):
    """Parser for extracting text from images using the RapidOCR library.

    Attributes:
        ocr:
          The RapidOCR instance for performing OCR.
    """

    def __init__(
        self,
        **kwargs
    ) -> None:
        """
        Initializes the RapidOCRBlobParser.
        """
        super().__init__()
        self.ocr = None
        self.kwargs = kwargs

    def _analyze_image(self, img: "Image") -> str:
        """
        Analyzes an image and extracts text using RapidOCR.

        Args:
            img (Image):
              The image to be analyzed.

        Returns:
            str:
              The extracted text content.
        """
        if not self.ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self.ocr = RapidOCR(**self.kwargs)
            except ImportError:
                raise ImportError(
                    "`rapidocr-onnxruntime` package not found, please install it with "
                    "`pip install rapidocr-onnxruntime`"
                )
            except Exception as e:
                log.error(f"An unexpected error occurred during RapidOCR initialization: {e}", exc_info=True)
                raise

        try:
            img_array = np.array(img)
            
            # Skip OCR for very small images that are likely not text.
            MIN_IMAGE_HEIGHT = 30
            MIN_IMAGE_WIDTH = 30
            if img_array.shape[0] < MIN_IMAGE_HEIGHT or img_array.shape[1] < MIN_IMAGE_WIDTH:
                return ""
            
            ocr_result, _ = self.ocr(img_array)
        except Exception as e:
            log.error(f"CRITICAL: Error during ocr() call in _analyze_image: {e}", exc_info=True)
            raise ValueError(f"OCR engine failed internally. Original error: {e}") from e

        content = ""
        if ocr_result:
            content = ("\n".join([text[1] for text in ocr_result])).strip()
        return content
