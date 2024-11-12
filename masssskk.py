import cv2
import base64
import numpy as np

bitmap_data = "eNoBOQLG/YlQTkcNChoKAAAADUlIRFIAAABuAAAANQgGAAAAoBLezwAAAgBJREFUeF7t3Nt2hCAMBdDx/z966lhp0VFJQrgkOT61a0HAbFDE2uWVHe/1yH9v8fOyHi3iRov5l8QeaFfJBaRsyG1wo9AAKUP71FpmQkungVlYBp0SrtztdcQFv1eahTvjRoN0A3c3S72CuocbsQDirhskg2u6VSXl/tayDCeJXCBJv+/6M/w5TnIyEeucAQ+7GD1GUMSka51zjve1/QQ8rTS3iZPwbvcNAdgm8RpRP3hPG76w08hygxhFuH0fs0HTCFmTgRLcFhvTribFbeqS4FLTAGyDII3KeqkJPGma9esBTj+nXSICrkua9RthwWGxog8gjciGA5401br1RHDA00WQRBPDAU+Sbr06VXDA04PgRqqGu8NL+6B49uOS0MqrwNGaOpYCqCRr/3WGweXdBiIfcQq4c7cBWYacEu6q28A8ZsUMHFawhuFwX/zNAOt9XPmq279E1Munebiol08XcBHx3MBFw3MFFwnPHVwUPMD1XwhXt1j8E/TqFgYF8PyI8PjRx6B8qzbrEe/xMyvV7A0M5g3u6hsPU3uV1LHgCa74RSo1KRbKeYF7+poKM27ikRgKLsJs294QTDzgxF2zjkf5zw8u4SzvoFDQ3M64NFUtzjzA7XpW8KhgaVC6vVTmN8jZ8bho7i+V4tUNoaLWYJCgAY4ARC1yBSlFobT5A/b23/ql+EuxAAAAAElFTkSuQmCCVaUQvA=="
bitmap_data_bytes = base64.b64decode(bitmap_data)

nparr = np.frombuffer(bitmap_data_bytes, np.uint8)
bitmap_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

if bitmap_image is None:
     raise ValueError("Не удалось декодировать изображение Thyroid tissue.")


# original_image = cv2.imread('screen foto/dataset 2024-04-21 14_33_36/img/4.jpg')

# gray_mask = cv2.cvtColor(bitmap_image, cv2.COLOR_BGR2GRAY)
# _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
# result = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

# cv2.imwrite('result_thyroid_tissue.png', result)
# cv2.imshow('Thyroid tissue Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
