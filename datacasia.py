import glob
import os
import cv2
import h5py as h5
import utils.coalbp
import numpy as np
import utils.preprocess as prep

SEP = os.path.sep

def get_path(is_training=True):
    if is_training:
        dataset = "train_release"
    else:
        dataset = "test_release"

    root_path =  os.path.join("D:\Database", "CASIA-CBSR",dataset)
    path_real = []
    path_fake = []
    for dirpath_, dirname_, filename_ in os.walk(root_path):
        for dir_ in dirname_:
            for _, _, aviname_ in os.walk( os.path.join(root_path, dir_)):
                for avi in aviname_:
                    avi_path = os.path.join(root_path, dir_, avi )
                    print(avi_path)
                    if avi == "1.avi"  or avi == "2.avi" or avi == '3.avi':   
                        path_real.append(avi_path)
                    else:
                        path_fake.append(avi_path)

    return path_real, path_fake


def get_face(path_real, path_fake, shape=(128,128,3)):
    face_real = np.empty((0,128,128,3),np.uint8)
    face_fake = np.empty((0,128,128,3),np.uint8)
    for path in path_real:
        frames = prep.parse_video(path)
        i_ = 0
        for i in range(len(frames)):  
            frame = frames[i] 
            i_ = i_+1   
            _, face = prep.face_detection(frame, 1.4)
           
            if face is None:
                continue 
            print("Processing {} {}/{}".format(path, i_, len(frames))) 
            face = cv2.resize(face, shape[0:2])
            face = face[np.newaxis,:,:,:]
            face_real = np.concatenate((face_real,face), axis=0)
            print("shape: ",face_real.shape)

    for path in path_fake:
        frames = prep.parse_video(path)
        i_ = 0  
        for frame in frames:
            i_ = i_+1      
            _, face = prep.face_detection(frame, 1.4)
            if face is None:
                continue
            face = cv2.resize(face, shape)
            print("Processing {} {}/{}".format(path, i_, len(frames))) 
            face = face[np.newaxis,:,:,:]
            face_fake = np.concatenate((face_fake,face), axis=0)
            print("shape: ",face_fake.shape)

    return face_real, face_fake

def creat_database(is_training):
    path_real, path_fake = get_path(is_training)
    face_real, face_fake = get_face(path_real, path_fake)
    db_name = "CASIA_FACE_128x128_s1_BGR.h5df"
    db = h5.File(db_name, "a")
    y_real = np.ones(face_real.shape[0], np.int8)
    y_fake = np.ones(face_fake.shape[0], np.int8)

    db.create_dataset("X_real", data=face_real)
    db.create_dataset("y_real", data=face_real)

    db.create_dataset("X_fake", data=face_fake)
    db.create_dataset("y_fake", data=y_fake)
    db.close()

def load_database(is_training=True, stride=1, db_file=r"D:\Database\CASIA-CBSR\CASIA-FASD_128.mat"):
    """Load database created by Dr.Sun
    """
    mat = h5.File(db_file,'r')
    if is_training:
        X1 = mat["TRAIN_X"][::stride]
        X2 = mat["VAL_X"][::stride]
        X = np.concatenate((X1,X2),axis=0)
        print("Concate:" ,X.shape)
        y1 = mat["TRAIN_LBL"][::stride]
        y2 = mat["VAL_LBL"][::stride]
        y = np.concatenate((y1,y2),axis=0)
    else:
        X = mat["TEST_X"][::stride]
        y = mat["TEST_LBL"][::stride]
    return X, np.int8(y)
        
def main():
    pass

if __name__ == '__main__':
    is_training = True
    creat_database(is_training)
    

        

    



        

def main():
    get_path()

if __name__ == '__main__':
    main()