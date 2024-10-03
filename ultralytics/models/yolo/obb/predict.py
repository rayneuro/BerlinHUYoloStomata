# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.ops import xywhr2xyxyxyxy

class OBBPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import OBBPredictor

        args = dict(model="yolov8n-obb.pt", source=ASSETS)
        predictor = OBBPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            rotated=True,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, :4] = ops.scale_boxes(img.shape[2:], rboxes[:, :4], orig_img.shape, xywh=True)
            # xywh, r, conf, cls

            rboxes_predict = xywhr2xyxyxyxy(rboxes) # convert xywhr to xyxyxyxy

            orig_shape = orig_img.shape[:2] # get original image shape
            
            rboxes_predict[..., 0] /= orig_shape[1] # h
            rboxes_predict[..., 1] /= orig_shape[0] # w
            
            #print('rboxes_predict is',rboxes_predict)
            rboxes_predict = rboxes_predict.reshape(-1, 8)
            #print('rboxes_predict shape',rboxes_predict.shape)
            
           
            #result = model(rboxes_predict) # [ n, 2]
        
            for i in range(rboxes_predict.shape[0]):
                for j in range(8):
                    if (rboxes_predict[i][j] <= 0.001 or rboxes_predict[i][j] >= 0.999) :
                        if pred[i][5] == 0.0:
                            pred[i][5] = 1.0
                        elif pred[i][5] == 2.0:
                            pred[i][5] = 3.0

            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1) # ( x1, y1, x2, y2, rotation, confidence, cls)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results
