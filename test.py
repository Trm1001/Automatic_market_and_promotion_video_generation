from PIL import Image

# 加载图像
image = Image.open('./sample_scripts/samples/segments/panda/mask_5_prev_ui.png')

# 转换为灰度图
gray_image = image.convert('L')

# 保存或显示图像
gray_image.save('./sample_scripts/samples/segments/panda/mask_5_prev_ui_gray_image.png')
gray_image.show()
