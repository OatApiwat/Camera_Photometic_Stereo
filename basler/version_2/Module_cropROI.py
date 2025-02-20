import cv2
import numpy as np
import os

class CircleCropper:
    def __init__(self, input_folder, output_folder, circles):
        """
        Parameters:
        - input_folder: Path to the folder containing input images.
        - output_folder: Path to the folder where cropped images will be saved.
        - circles: List of tuples (x, y, r) representing circle positions and radii.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.circles = circles

        # ตรวจสอบว่า output_folder มีอยู่หรือยัง ถ้าไม่มีก็สร้าง
        os.makedirs(self.output_folder, exist_ok=True)

    def crop_circle(self, image, x, y, r):
        """Crop circular region from an image."""
        # สร้าง mask วงกลมสีขาวบนพื้นหลังสีดำ
        mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # ตัดเฉพาะบริเวณวงกลมจากภาพต้นฉบับ
        cropped_image = cv2.bitwise_and(image, image, mask=mask)

        # ครอบภาพเฉพาะส่วนที่เป็นวงกลม
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
        return cropped_image[y1:y2, x1:x2]


    def process_images(self):
        """Process all images in the input folder."""
        # อ่านไฟล์ภาพทั้งหมดจาก input_folder
        image_files = [f for f in os.listdir(self.input_folder) if f.endswith(('.jpg', '.png'))]

        for image_file in image_files:
            # โหลดภาพ
            image_path = os.path.join(self.input_folder, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Cannot read image: {image_file}")
                continue
            
            # ดำเนินการ crop และคำนวณมัธยฐานสำหรับวงกลมแต่ละตำแหน่ง
            crop_image = []
            for i, (x, y, r) in enumerate(self.circles, start=1):
                cropped_circle = self.crop_circle(image, x, y, r)


                # สร้าง mask ของวงกลมในภาพย่อยเพื่อให้ภาพมีขอบโปร่งใส
                circle_mask = np.zeros((2*r, 2*r), dtype=np.uint8)
                cv2.circle(circle_mask, (r, r), r, 255, -1)

                # ใช้ mask กับภาพที่ครอบ
                h, w = cropped_circle.shape[:2]
                final_mask = np.zeros((h, w), dtype=np.uint8)
                cx, cy = min(r, w // 2), min(r, h // 2)
                cv2.circle(final_mask, (cx, cy), r, 255, -1)

                circle_only = cv2.bitwise_and(cropped_circle, cropped_circle, mask=final_mask)

                # คำนวณสีเฉลี่ยจากบริเวณขอบของวงกลม
                border_mask = np.zeros_like(final_mask)
                cv2.circle(border_mask, (cx, cy), r, 255, thickness=1)
                border_pixels = cropped_circle[border_mask == 255]  # พิกเซลในบริเวณขอบ

                if len(border_pixels) > 0:  # ตรวจสอบว่าพบพิกเซล
                    average_color = border_pixels.mean(axis=0).astype(np.uint8)  # ค่าสีเฉลี่ย (BGR)
                    circle_only[final_mask == 0] = average_color  # เติมสีเฉลี่ยในบริเวณที่ไม่มีข้อมูล

                img_blurred = cv2.fastNlMeansDenoising(circle_only,None, 8 ,15, 15)
                
                # สร้างชื่อไฟล์ใหม่
                
                base_name, ext = os.path.splitext(image_file)
                output_cropped_filename = f"{base_name}_roi{i}{ext}"

                cropped_path = os.path.join(self.output_folder, output_cropped_filename)
                # บันทึกภาพที่ crop และภาพมัธยฐาน
                cv2.imwrite(cropped_path, circle_only)
                crop_image.append(circle_only)

                print(f"Saved cropped image: {cropped_path}")
            return crop_image


if __name__ == "__main__":
    # โฟลเดอร์ต้นทางและปลายทาง
    for index in range(1, 2):
        folder_number = index
        input_folder = f"/home/mic/Camera_project/basler/version_2/2_photometric_folder/photometric_picture_{folder_number}"
        output_folder = f"/home/mic/Camera_project/basler/version_2/3_cropROI_folder/cropROI_picture_{folder_number}"

        # ตำแหน่งวงกลมที่ต้องการ crop
        offset = 65
        circles = [
            (375-offset, 1430, 170),  # วงกลมที่ 1
            (870-offset, 545, 170),   # วงกลมที่ 2
        ]

        # สร้างออบเจ็กต์และรันโปรแกรม
        cropper = CircleCropper(input_folder, output_folder, circles)
        cropper.process_images()

