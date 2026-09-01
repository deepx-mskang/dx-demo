"""
OCR Engine utilities and model management
Contains functions for creating and managing OCR engines
"""

from pathlib import Path

from dx_engine import InferenceEngine as IE
from dx_engine import InferenceOption as IO

# PP-OCRv6 assets live in the shared workspace tree, four levels above this
# module (scripts/ -> python/ -> paddle-ocr/ -> apps/ -> root). Resolved from
# __file__ rather than the CWD, matching how the C++ demo derives its assets
# directory from the binary location (apps/paddle-ocr/cpp/main.cpp:167).
V6_ASSETS_DIR = (
    Path(__file__).resolve().parent / ".." / ".." / ".." / ".." / "workspace" / "models" / "ocr" / "v6"
).resolve()


def make_det_engines(model_dirname):
    """
    Create detection engines for different aspect ratios and heights

    Args:
        model_dirname (str): Directory containing detection model files

    Returns:
        dict: Dictionary mapping aspect ratios to height-based model dictionaries

    Note:
        Creates models for aspect ratios from 5 to 25 in intervals of 10,
        and heights of 10, 20, and 30 pixels
    """

    io = IO().set_use_ort(True)
    det_model_map = {}

    for res in [640, 960]:
        model_path = f"{model_dirname}/det_v6_m_{res}.dxnn"
        det_model_map[res] = IE(model_path, io)

    return det_model_map


def make_rec_engines(model_dirname):
    """
    Create recognition engines for different aspect ratios and heights
    
    Args:
        model_dirname (str): Directory containing recognition model files
        
    Returns:
        dict: Dictionary mapping aspect ratios to height-based model dictionaries
        
    Note:
        Creates models for aspect ratios from 5 to 25 in intervals of 10,
        and heights of 10, 20, and 30 pixels
    """
    
    io = IO().set_use_ort(True)
    rec_model_map = {}

    # v6 drops ratio 35 and adds 1 and 40; keep this in step with the
    # rec_fixed_v6_ratio_*.dxnn files in workspace and with rec_router().
    for i in [1, 3, 5, 10, 15, 25, 40]:
        model_path = f"{model_dirname}/rec_fixed_v6_ratio_{i}.dxnn"
        rec_model_map[i] = IE(model_path, io)
    
    return rec_model_map


def create_ocr_models(use_doc_preprocessing=True, use_mobile=False):
    """
    Create detection and recognition models for PP-OCRv6

    v6 is detection + recognition only. The textline-orientation, document
    orientation and UVDoc unwarping models have no v6 counterpart in workspace,
    so they are returned as None - PaddleOcr/AsyncPipelineOCR skip those nodes
    when the model is None. This mirrors demo-ocr.py.

    Returns:
        tuple: (det_model, cls_model, rec_models, rec_dict_dir,
                doc_ori_model, doc_unwarping_model)
    """
    if use_mobile:
        # workspace ships rec_v6_m_ratio_{5,15,25} only, and those were compiled
        # against a different dictionary, so there is no usable mobile v6 set.
        print("Warning: use_mobile is ignored for PP-OCRv6; using the standard model set.")

    dir_name = str(V6_ASSETS_DIR)
    if not V6_ASSETS_DIR.is_dir():
        raise FileNotFoundError(
            f"PP-OCRv6 assets not found at {V6_ASSETS_DIR}. "
            "Run ./setup_assets.sh from the repository root."
        )

    rec_dict_dir = f"{dir_name}/ppocrv6_dict.txt"

    det_model = make_det_engines(dir_name)
    rec_models = make_rec_engines(dir_name)

    cls_model = None            # Textline orientation: no v6 model
    doc_ori_model = None        # Document orientation: no v6 model
    doc_unwarping_model = None  # UVDoc unwarping: no v6 model

    if use_doc_preprocessing:
        print("Note: document preprocessing is unavailable for PP-OCRv6; running detection + recognition only.")

    return det_model, cls_model, rec_models, rec_dict_dir, doc_ori_model, doc_unwarping_model


def create_ocr_workers(num_workers=3, use_doc_preprocessing=True, use_doc_orientation=True, use_mobile=False):
    """
    Create multiple OCR worker instances for parallel processing
    PP-OCRv5 구조: det + rec + doc_ori + UVDoc (cls 모델이 doc_ori 역할)
    
    Args:
        num_workers (int): Number of worker instances to create
        use_doc_preprocessing (bool): Document unwarping 사용 여부 (UVDoc)
        use_doc_orientation (bool): Document orientation 사용 여부 (doc_ori)
        
    Returns:
        list: List of PaddleOcr worker instances with document preprocessing
    """
    from engine.paddleocr import PaddleOcr
    print("Creating OCR models...", num_workers, "use_unwarping:", use_doc_preprocessing, "use_doc_ori:", use_doc_orientation, "use_mobile:", use_mobile)
    det_model, cls_model, rec_models, rec_dict_dir, doc_ori_model, doc_unwarping_model = create_ocr_models(
        use_doc_preprocessing=use_doc_preprocessing, use_mobile=use_mobile
    )
    
    ocr_workers = [
        PaddleOcr(
            det_model=det_model,
            cls_model=cls_model,
            rec_models=rec_models,
            rec_dict_dir=rec_dict_dir,
            doc_ori_model=doc_ori_model,
            doc_unwarping_model=doc_unwarping_model,
            use_doc_preprocessing=use_doc_preprocessing,
            use_doc_orientation=use_doc_orientation
        ) for _ in range(num_workers)
    ]
    
    return ocr_workers


def create_async_ocr_pipeline(use_doc_preprocessing=True, use_doc_orientation=True):
    """
    Create an async OCR pipeline instance
    
    Args:
        use_doc_preprocessing (bool): Document unwarping 사용 여부 (UVDoc)
        use_doc_orientation (bool): Document orientation 사용 여부 (doc_ori)
        
    Returns:
        AsyncPipelineOCR: Async OCR pipeline instance
    """
    from engine.paddleocr import AsyncPipelineOCR
    
    print("Creating async OCR pipeline...")
    print(f"  - Document unwarping: {use_doc_preprocessing}")
    print(f"  - Document orientation: {use_doc_orientation}")
    
    det_models, cls_model, rec_models, rec_dict_dir, doc_ori_model, doc_unwarping_model = create_ocr_models(
        use_doc_preprocessing=use_doc_preprocessing
    )
    
    async_pipeline = AsyncPipelineOCR(
        det_models=det_models,
        cls_model=cls_model,
        rec_models=rec_models,
        rec_dict_dir=rec_dict_dir,
        doc_ori_model=doc_ori_model,
        doc_unwarping_model=doc_unwarping_model,
        use_doc_preprocessing=use_doc_preprocessing,
        use_doc_orientation=use_doc_orientation
    )
    
    print("Async OCR pipeline created successfully!")
    return async_pipeline
