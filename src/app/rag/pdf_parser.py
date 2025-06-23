from langchain_community.document_loaders.parsers import BaseImageBlobParser
from PIL.Image import Image
import numpy as np

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
        ocr_result, _ = self.ocr(np.array(img))  # type: ignore[misc]
        content = ""
        if ocr_result:
            content = ("\n".join([text[1] for text in ocr_result])).strip()
        return content
