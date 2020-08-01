# Capstone-Tobacco-Width


# 密集型烟丝图宽度识别
## 主要步骤
### Superpixel Segmentation超像素分割
* 原理与Python技术文档：https://blog.csdn.net/haoji007/article/details/103433100，https://www.jianshu.com/p/d0ef931b3ddf
* 输入：密集型烟丝图像
### Piece Merging邻区域合并
* 原理：根据超像素分割结果，对分割后的区域块就行判断。若相邻区域块被判断为属于同一根烟丝，则进行合并
* 合并规则：根据相邻区域的平均RGB，平均HSV以及灰度直方图均值和方差，设置阈值，满足小于等于阈值的条件则合并
### Width Measurement宽度检测



## 代码
### Part I：超像素分割+邻区域合并
* final_params.py：参数合集
* final_region.py：Region类
* final_hsv.py：计算两点之间HSV值距离公式
* final_superpizel.py：运行代码，输入图像路径，输出超像素+合并结果对比，以及合并结果区域块单独图像文件

### Part II：宽度检测
