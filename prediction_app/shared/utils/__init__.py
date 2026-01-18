"""
共享工具函数
"""
from .data_loader import load_data
from .customer_segmentation import segment_customers_kmeans

__all__ = ["load_data", "segment_customers_kmeans"]
