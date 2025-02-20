from Module_open_camera import OpenBaslerCamera
from Module_photometric_stereo import PhotometricStereo
from Module_cropROI import CircleCropper
from Module_yolo_detect import YOLOImageClassifier
import cv2
import os
import time


cv2.namedWindow('original_img',cv2.WINDOW_NORMAL)
window_width = 772  # ความกว้างของหน้าต่าง
window_height = 516  # ความสูงของหน้าต่าง
cv2.resizeWindow('original_img', window_width, window_height)  # ปรับขนาดหน้าต่าง

# กำหนดเส้นทางสำหรับบันทึกรูปภาพ
folder_path = r"/home/mic/Camera_project/basler/version_2"
raw_folder_path = r"/home/mic/Camera_project/basler/version_2/1_raw_folder"
photometric_folder_path = r"/home/mic/Camera_project/basler/version_2/2_photometric_folder"
cropROI_folder_path = r"/home/mic/Camera_project/basler/version_2/3_cropROI_folder"
result_folder_path = r"/home/mic/Camera_project/basler/version_2/4_rersult_folder"

if not os.path.exists(raw_folder_path):
    os.makedirs(raw_folder_path)
if not os.path.exists(photometric_folder_path):
    os.makedirs(photometric_folder_path)
if not os.path.exists(cropROI_folder_path):
    os.makedirs(cropROI_folder_path)


def process_2_photometric_stereo(input_folder,number_photometric_folder):
    photometric_path = os.path.join(photometric_folder_path, "photometric_picture_{}".format(number_photometric_folder))
    if os.path.exists(photometric_path):
        # ลบโฟลเดอร์หากมีอยู่แล้ว
        if os.path.isdir(photometric_path):
            shutil.rmtree(photometric_path)
        else:  # ถ้าเป็นไฟล์ ให้ลบด้วย os.remove
            os.remove(photometric_path)
    os.makedirs(photometric_path)
    photometric_image = PhotometricStereo(input_folder,photometric_path)
    normal_map_visual,B,G,R = photometric_image.ProcessImages()
    return photometric_path
    
def process_3_cropROI(input_folder,number_cropROI_folder,circles):
    cropROI_path = os.path.join(cropROI_folder_path, "cropROI_picture_{}".format(number_cropROI_folder))
    if os.path.exists(cropROI_path):
        # ลบโฟลเดอร์หากมีอยู่แล้ว
        if os.path.isdir(cropROI_path):
            shutil.rmtree(cropROI_path)
        else:  # ถ้าเป็นไฟล์ ให้ลบด้วย os.remove
            os.remove(cropROI_path)
    os.makedirs(cropROI_path)
    cropROI_image = CircleCropper(input_folder,cropROI_path,circles)
    cropROI_image.process_images()
    return cropROI_path
    
def process_4_yolo(input_folder,number_predict_folder,origin_image,circles):
    predict_path = os.path.join(result_folder_path, "result_picture_{}".format(number_predict_folder))
    if os.path.exists(predict_path):
        # ลบโฟลเดอร์หากมีอยู่แล้ว
        if os.path.isdir(predict_path):
            shutil.rmtree(predict_path)
        else:  # ถ้าเป็นไฟล์ ให้ลบด้วย os.remove
            os.remove(predict_path)
    os.makedirs(predict_path)
    result,result_image = yolo.classify_images(input_folder,predict_path,origin_image,circles)
    return result,result_image
    
number_raw_file = 0
if __name__ == "__main__":
###############-----------------------------------------------get picture from camera and save to raw_folder_path--------------------------------------------------------#####################
    basler_camera = OpenBaslerCamera()
    basler_camera.start_grabbing()
    model_path = r"/home/mic/Camera_project/basler/version_2/weight_02.pt"
    yolo = YOLOImageClassifier(model_path)
    print("Program is started")
    print("press 't' to save or 'q' for exit")
    while True:
        img = basler_camera.get_image()
        if img is None:
            print("cannot get picture from camera")
            break
        # Display the image
        cv2.imshow('original_img', img)
        window_width = 1544  # ความกว้างของหน้าต่าง
        window_height = 1032  # ความสูงของหน้าต่าง
        cv2.namedWindow('result_img',cv2.WINDOW_NORMAL)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            #print("number_raw_file: ",len(os.listdir(raw_folder_path)))
            number_raw_folder = len([item for item in os.listdir(raw_folder_path) if os.path.isdir(os.path.join(raw_folder_path, item))])
            if number_raw_file % 4 == 0:
                raw_path = os.path.join(raw_folder_path, "raw_picture_{}".format(len(os.listdir(raw_folder_path)) + 1))
            else:
                raw_path = os.path.join(raw_folder_path, "raw_picture_{}".format(len(os.listdir(raw_folder_path))))
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
            number_raw_file = len(os.listdir(raw_path)) + 1 
            filename = os.path.join(raw_path, "image_{}.jpg".format(number_raw_file))
            cv2.imwrite(filename, img)
            print(f"save at: {filename}")
            if number_raw_file % 4 == 0: #got image
                start_time = time.time()
                print("next to process 2")
                start_time_2 = time.time()
                photometric_path = process_2_photometric_stereo(raw_path,number_raw_folder)
                print("end process_2")
                print("used time 2: ",time.time()-start_time_2)
                print("next to process 3 ROI")
                start_time_3 = time.time()
                offset = 65
                circles = [
                    (375-offset, 1430, 170),  # วงกลมที่ 1
                    (870-offset, 545, 170),   # วงกลมที่ 2
                ]
                cropROI_path = process_3_cropROI(photometric_path,number_raw_folder,circles)
                print("end process_3")
                print("used time 3: ",time.time()-start_time_3)
                start_time_4 = time.time()
                origin_image = os.path.join(photometric_path, "B.png")
                result,result_image = process_4_yolo(cropROI_path,number_raw_folder,origin_image,circles)
                
                cv2.imshow('result_img', result_image)
                print("end process_4")
                print("used time 4: ",time.time()-start_time_4)
                print("usage time: ",time.time() - start_time," sec")
                print("press 't' to save or 'q' for exit")
                #break
        elif key == 27 or key == ord('q'):  # Exit if 'Esc' key is pressed
            break
        

    # Release resources
    basler_camera.release()
    cv2.destroyAllWindows()
