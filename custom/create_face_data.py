# By reading files in directoris, it generates source training data list into csv
# the file format will be
#  {file name},{text label},{label number}
# this program is inteneded to be ran in local environment 
# before run this program plz install dependencies
# google cloud vision api : pip install --upgrade google-cloud-vision
import os
import sys
from google.cloud import vision
from PIL import Image,ImageDraw,ImageFont
import tensorflow as tf
from object_detection.utils import dataset_util

MAX_ROLL = 20
MAX_TILT = 20
MAX_PAN = 20

label_index = 0
labels = {}
label_count = {}

RESULT_FILES = 'converted_result_files.csv'
ALL_FILES = 'all_files.csv'
FILTERED_FILES = 'filtered_files.csv'
SUMMARY_FILES = 'summary_files.csv'
LABEL_MAP_FILE = 'face_label_map.pbtxt'
TF_RECORD_TRAINING = 'face_training.record'
TF_RECORD_EVALUATION = 'face_evaluation.record'
DEBUG = False
DEBUG_COUNT=5

'''
    Scan base directory and generate file list csv file,
     which as file_name, label_text, label (number)
'''
def get_dirlist(base_dir,destination_dir):
    global label_index
    global DEBUG
    
    destination_file = destination_dir + '/' + ALL_FILES
    des = open(destination_file,'w')
    
    label_map_file_path = destination_dir + '/' + LABEL_MAP_FILE
    label_map_file = open(label_map_file_path,'w')
    
    count = 0
    label_index = 1
    
    for d in os.listdir(base_dir):
        dir_name = str(d)
        current_dir = base_dir+'/'+dir_name
        if (os.path.isdir(current_dir)):
            # if the file is directory
            
            # generate number label
            # in object detection API, the label should be started from 1
            index = 0
            if dir_name not in labels:
                labels[dir_name] = label_index
                index = label_index
                label_index = label_index + 1
                
                # write label map
                label_map = 'item {\n'
                label_map = label_map + ' id:' + str(index) + '\n'
                label_map = label_map + ' name:\'' + dir_name + '\'\n'
                label_map = label_map + '}\n\n'
                label_map_file.write(label_map)
                
            else:
                index = labels[dir_name]
                
            for f in os.listdir(current_dir):
                if( not os.path.isdir(f)):
                    file_name = str(f)
                    
                    # write training data information into csv file
                    # label_index, label_text,label_count,file_name    
                    buf = ( str(index) + ',' + dir_name + ','   +file_name+'\n')
                    count = count + 1
                    if (DEBUG and count > DEBUG_COUNT):
                        break
                    des.write(buf)
                    
        if (DEBUG and count > DEBUG_COUNT):
            break
        
    des.close()
    label_map_file.close()
'''
    Extract face location from image file
    Filter inappropriate image
    it uses Google cloud vision API, so the API needs to be enabled by GCP console first    
'''    
def get_imageinfo(image_file):
    
    name,ext = os.path.splitext(image_file)
    ext = str(ext.lower()).rstrip()
    if not (ext == '.jpeg' or ext == '.jpg' or ext =='.png' ):
        result='[Skipped] %s It is not image file'%image_file
        return False,result
    
    vision_client = vision.ImageAnnotatorClient()
    with open(image_file,'rb') as image:
        content = image.read()
    response = vision_client.face_detection({'content':content})
    
    faces = response.face_annotations
    if(len(faces)<1):
        result='[Skipped] %s has no faces'%image_file
        return False,result
        
    if(len(faces)>1):
        result='[Skipped] %s has more than 2 faces'%image_file
        return False,result
    face = faces[0]
    
    # extract face angle
    roll_angle = face.roll_angle
    pan_angle = face.pan_angle
    tilt_angle = face.tilt_angle
    angle = [roll_angle,pan_angle,tilt_angle]
    
    # filter out based on angle
    if abs(roll_angle) > MAX_ROLL or abs(pan_angle) > MAX_PAN or abs(tilt_angle) > MAX_TILT:
        result = '[Skipped] %s face skew angle is big'%image_file
        return False,result
    
    # extract face boundary
    xmin = face.bounding_poly.vertices[0].x
    ymin = face.bounding_poly.vertices[0].y
    xmax = face.bounding_poly.vertices[2].x
    ymax = face.bounding_poly.vertices[2].y
    rect = [xmin,ymin,xmax,ymax]
    
    print('[Info] %s Sucess found face in (%d,%d) (%d,%d)'%(image_file,
        xmin,ymin,xmax,ymax))
    return True,rect

'''
    filter training data imageS and generate list of filtered file
    it filters data by calling get_imageinfo
'''
def filter_images(base_dir,destination_dir):
    global label_count
    global ALL_FIELS
    global FILTERED_FILES
    
    all_files_path = (destination_dir + '/'+ALL_FILES).rstrip()
    all_files = open(all_files_path,'r')

    filtered_files_path = (destination_dir + '/'+FILTERED_FILES).rstrip()
    
    with open(filtered_files_path,'w') as filtered_files:
        for f in all_files:
            buf = f.split(',')
            label = int(buf[0])
            label_text = str(buf[1])
            file_name = str(buf[2]).rstrip()
            file_path = base_dir+'/'+label_text+'/'+file_name
            file_path = file_path.rstrip()
        
            success,rect = get_imageinfo(file_path)
            print('[Info] filtering %s %s'%(file_path,success))
            
            if label_text in label_count:
                count = label_count[label_text]
            else:
                count = 0
                
            if(success):
                with Image.open(file_path) as image:
                    width,height = image.size
                    filtered_files.write('%s,%s,%d,%d,%d,%d,%d,%d,%d,%d\n'%(file_name,label_text,label,count,rect[0],rect[1],rect[2],rect[3],width,height))
                label_count[label_text] = count + 1
            
    
    summary_file_path = (destination_dir + '/'+SUMMARY_FILES).rstrip()
    
    with open(summary_file_path,'w') as summary_files:
        for key in label_count.keys():
            value = int(label_count[key])
            summary_files.write('%s,%d\n'%(key,value))
            

'''
    draw box in image and write it into destination directory
    it also convert all of file fortmat into jpeg format and also change file ext to '.jpeg'
'''
def draw_box(base_dir,label_text,label,file_name,rect,destination_dir):
    global label_count
    global RESULT_FILES
    
    image_file = base_dir+'/'+label_text+'/'+file_name
    image_file = image_file.rstrip()
    
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)
    #draw.rectangle( ((rect[0],rect[1]),(rect[2],rect[3])),outline='red' )
    
    label_x = int(rect[0])
    
    if ( int(rect[1]) > 20):
        label_y = int(rect[1]-15)
    else:
        label_y = int(rect[3])
    
    #draw.rectangle( ((label_x,label_y),(label_x+50,label_y+15)),fill='red' )
    #draw.text((label_x+3,label_y+2), label_text, font=ImageFont.load_default().font )
    
    name,ext = os.path.splitext(file_name)
    
    destination_file_name = (name+'.jpeg').rstrip()
    destination_file = (destination_dir+'/'+destination_file_name).rstrip()
    image.save(destination_file,"JPEG")
    
    result_file = (destination_dir+'/'+RESULT_FILES).rstrip()

    with open(result_file,'a') as rf:
        rf.write('%s,%s,%d,%d,%d,%d,%d\n'%(destination_file_name,label_text,label,rect[0],rect[1],rect[2],rect[3]))
        
    return destination_file_name
'''
    Save image and meta data into tfrecord file
    it writes data into single file.
    But for huge number of data training, it needs to be enhanced to store the data
    to multiple files.
'''
def write_tfrecord(destination_dir,label_text,label,file_name,rect,width,height,tfwriter):
    
    filename = file_name
    image_format = 'jpg'
    
    xmins = [float(rect[0]) / float(width)]
    ymins = [float(rect[1]) / float(height)]
    xmaxs = [float(rect[2]) / float(width)]
    ymaxs = [float(rect[3]) / float(height)]
    classes_text = [label_text]
    classes = [int(label)]
    
    image_file_path = destination_dir+'/'+file_name
    image_file_path = image_file_path.rstrip()
    image_file = open(image_file_path,'rb')
    encoded_image_data = image_file.read()
    
    print('tf record :',xmins,ymins,xmaxs,ymaxs,width,height,classes_text,classes,len(encoded_image_data))
    if (xmins[0] > 1.0 or ymins[0] > 1.0 or xmaxs[0] > 1.0 or ymaxs[0]> 1.0):
        print ('WARNING : box value is bigger than 1.0',xmins,ymins,xmaxs,ymaxs,width,height,rect,classes_text,classes)
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    tfwriter.write(tf_example.SerializeToString())
    image_file.close()
     

'''
    Read csv file and draw a box for face in the image
    If the image is not appropriate for face recognition training, it will
    automatically filter out the image
     - too much rotated image
     - image which has more than two face
     - image that has sunglass

'''
def convert_images(base_dir,destination_dir,m):
    
    global label_count
    global RESULT_FILES
    global FILTERED_FILES
    global TF_RECORD_TRAINING
    
    filtered_files_path = (destination_dir + '/'+FILTERED_FILES).rstrip()
    result_file_path = (destination_dir + '/'+RESULT_FILES).rstrip()
    
    # find min number of labels for each label
    # for example # of data in label "a" is 5, b is 4, c is 8
    # the return will be 8
    # the reason to find this is, it will generate trainining data with same number across lables
    
    if (m is  None):
        min_num = min(list(label_count.values()))
    else:
        min_num = m
    num_of_training = int(min_num*0.75)
    num_of_evaluation = int(min_num*0.25)
    gen_count = {}
    
    training_record_file_path = (destination_dir + '/'+TF_RECORD_TRAINING).rstrip()
    evaluation_record_file_path = (destination_dir + '/'+TF_RECORD_EVALUATION).rstrip()
    
    training_writer = tf.python_io.TFRecordWriter(training_record_file_path)
    evaluation_writer = tf.python_io.TFRecordWriter(evaluation_record_file_path)
    
    with open(result_file_path,'w') as result_files:
        result_files.write('file_name,label text,label,xmin,ymin,xmax,ymax')
        
    with open(filtered_files_path,'r') as filtered_files:
        for f in filtered_files:
            buf = f.split(',')
            file_name = str(buf[0]).rstrip()
            label_text = str(buf[1])
            label = int(buf[2])
            count = int(buf[3])
            xmin = int(buf[4])
            ymin = int(buf[5])
            xmax = int(buf[6])
            ymax = int(buf[7])
            width = int(buf[8])
            height = int(buf[9])
            rect = [xmin,ymin,xmax,ymax]
            
            if label not in gen_count:
                gen_count[label] = 0
                
            if( gen_count[label] < min_num ):
                new_file_name = draw_box(base_dir,label_text,label,file_name,rect,destination_dir)
                if( gen_count[label] < num_of_training):
                    # write to training file
                    writer = training_writer
                else:
                    # write to evaluation file
                    writer = evaluation_writer
                print('[INFO] adding ',destination_dir,label_text,label,new_file_name,rect,width,height)
                write_tfrecord(destination_dir,label_text,label,new_file_name,rect,width,height,writer)
                gen_count[label] = gen_count[label] + 1
        
            

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/terrycho/keys/terrycho-ml.json"

def main():
    if( len(sys.argv) < 4):
        print('Usage : python convert.py source_directory destination_directoy ')
        return
    base_dir = str(sys.argv[1])
    destination_dir = str(sys.argv[2])
    csv_file = ALL_FILES
    filtered_file = FILTERED_FILES
    tfrecord_file = str(sys.argv[3])
    
    print ('[Info] Scan directory %s'%base_dir)
    get_dirlist(base_dir,destination_dir)
    filter_images(base_dir,destination_dir)
    convert_images(base_dir,destination_dir,m=54)

main()
