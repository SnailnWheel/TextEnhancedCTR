# pip install -U fuxictr
import sys
fp = '/mnt/ssd1/yangxinyao/FuxiCTR'  # absolute directory
sys.path.append(fp)
import fuxictr
assert fuxictr.__version__ >= "2.2.0"
