"""
Startup Wellness Analysis Backend Package
"""

__version__ = '1.0.0'
__author__ = 'Startup Wellness Team'

import os
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 環境変数の設定
os.environ.setdefault('PYTHONPATH', os.path.dirname(os.path.dirname(__file__)))
