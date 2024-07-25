"""
python tests/test_datasets/test_sirst_det_voc.py
"""

from deepir.datasets import SIRSTVOCDetDataset



dataset = SIRSTVOCDetDataset(
    data_root='data/sirst/',
    ann_file='Splits/ImageSets/Main/trainval_v2.txt',
    pipeline=[],
    backend_args=None
)

