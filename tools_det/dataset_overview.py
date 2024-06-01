'''
Author: your name
Date: 2022-03-01 16:53:53
LastEditTime: 2022-03-01 16:53:55
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /research_workspace/tools_det/dataset_overview.py
'''

from pycocotools.coco import COCO
 
dataDir='data/ReCTS'
dataType='rects_train'
#dataType='train2017'
annFile='{}/annotations/{}.json'.format(dataDir, dataType)
 
# initialize COCO api for instance annotations
coco=COCO(annFile)
 
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
print('number of categories: ', len(cat_nms))
print('COCO categories: \n', cat_nms)
 
# 统计各类的图片数量和标注框数量
for cat_name in cat_nms:
    catId = coco.getCatIds(catNms=cat_name)     # 1~90
    imgId = coco.getImgIds(catIds=catId)        # 图片的id  
    annId = coco.getAnnIds(catIds=catId)        # 标注框的id
 
    print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))