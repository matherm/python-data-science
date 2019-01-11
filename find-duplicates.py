import cv2
import time
import numpy as np
from os import listdir
from os.path import isfile, join

class Duplicate_Finder:

    SHOW_IMAGES = True
    IM_SIZE = (1024,1024)
    GUI_SIZE = (500,500)
    THRESHOLD = 100
  
    def __init__(self, ROOT="C:\\_SSDWORKSPACE\\rootrepo\\im_duplicate_finder\\test_images", VALID_TYPES=["jpg","png","bmp"]):
        files_to_compare = self.get_files(ROOT,VALID_TYPES)
        log_file = open('result.txt', 'w')
        self.start_comparing(files_to_compare,log_file)
        log_file.close()

    def get_files(self,ROOT,VALID_TYPES):
        candidate_files = [f for f in listdir(ROOT) if isfile(join(ROOT, f))]
        filtered_files = []
        for filename in candidate_files:
            valid = False
            for ttype in VALID_TYPES:
                if ttype in filename:
                    valid = True
            if valid == True:
                filtered_files.append([ROOT + "\\" + filename, filename])
        return filtered_files


    def start_comparing(self,files_to_compare,log_file):
        total_length = str(len(files_to_compare))
        similar_files = []
        counter = 0
        for f1 in files_to_compare:
            print("Image " + str(counter) +" of " + total_length)#, end=""))
            IM_1 = cv2.imread(f1[0])
            for f2 in files_to_compare:
                # Skip diagnonal elements
                if f1[0] == f2[0]:
                    continue
                # skip found similar_files
                if f2[0] in similar_files:
                    continue
                IM_2 = cv2.imread(f2[0])
                if self.are_images_equal(IM_1,IM_2) == True:
                    similar_files.append(f1[0])
                    if self.SHOW_IMAGES == True:
                        cv2.imshow("IM_1", cv2.resize(IM_1,(self.GUI_SIZE[0],self.GUI_SIZE[1])))
                        cv2.imshow("IM_2", cv2.resize(IM_2,(self.GUI_SIZE[0],self.GUI_SIZE[1])))
                        cv2.waitKey(25)
                    print(f1[0] + " == " + f2[0])
                    log_file.write(f1[0] + "\n")
                    log_file.write(f2[0] + "\n")
                    log_file.write("####" + "\n")
            counter = counter + 1


    def are_images_equal(self,IM_1,IM_2):
        IM_1 = self.normalize_size(IM_1,self.IM_SIZE)
        IM_2 = self.normalize_size(IM_2,self.IM_SIZE)
        #Difference image_tools
        DIFF = cv2.subtract(IM_1,IM_2) + cv2.subtract(IM_2,IM_1)
        if np.sum(DIFF) < self.THRESHOLD:
            return True
        else:
            return False


    def normalize_size(self,IM,IM_SIZE):
        if(IM.ndim == 3):
            IM_normal = np.zeros((IM_SIZE[0],IM_SIZE[1],IM.shape[2]),"uint8")
        else:
            IM_normal = np.zeros((IM_SIZE[0],IM_SIZE[1],IM.shape[2]),"uint8")
        scale = 1
        if IM.shape[0] > IM.shape[1]:
            #higher than width
            scale = IM_normal.shape[0] / IM.shape[0]
        else:
            #widther than high
            scale = IM_normal.shape[1] / IM.shape[1]
        new_y =  int(IM.shape[0] * scale)
        new_x =  int(IM.shape[1] * scale)
        offset_y = int((IM_normal.shape[0] - new_y)/2) 
        offset_x = int((IM_normal.shape[1] - new_x)/2)
        IM_resized = cv2.resize(IM, (new_x,new_y),cv2.INTER_AREA)
        if(IM.ndim == 3):
            IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x,:] = IM_resized
        else:
            IM_normal[offset_y:offset_y+new_y,offset_x:offset_x+new_x] = IM_resized
        return IM_normal


if __name__ == "__main__":
    finder = Duplicate_Finder(ROOT="E:\Dropbox (Personal)\Kamera-Uploads")
    #finder = Duplicate_Finder()