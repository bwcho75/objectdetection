# reference /Users/terrycho/dev/workspace/objectdetection/models/object_detection/data_decoders
# reference http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
import tensorflow as tf
from PIL import Image
import io

tfrecord_filename = 'example_cat.tfrecord'

def readRecord(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    #'''
    keys_to_features = {
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        #'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        # Object boxes and classes.
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/object/area': tf.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.VarLenFeature(tf.int64),
        'image/object/difficult': tf.VarLenFeature(tf.int64),
        # Instance masks and classes.
        #'image/segmentation/object': tf.VarLenFeature(tf.int64),
        #'image/segmentation/object/class': tf.VarLenFeature(tf.int64)
    }
    
    features = tf.parse_single_example(serialized_example,features= keys_to_features)
    
    height = tf.cast(features['image/height'],tf.int64)
    width = tf.cast(features['image/width'],tf.int64)
    filename = tf.cast(features['image/filename'],tf.string)
    source_id = tf.cast(features['image/source_id'],tf.string)
    encoded = tf.cast(features['image/encoded'],tf.string)
    image_format = tf.cast(features['image/format'],tf.string)
    xmin = tf.cast(features['image/object/bbox/xmin'],tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'],tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'],tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'],tf.float32)
    #'''
    #height = tf.Variable(1.0)
    return height,width,filename,source_id,encoded,image_format
    
def main():
     filename_queue = tf.train.string_input_producer([tfrecord_filename])
     height,width,filename,source_id,encoded,image_format = readRecord(filename_queue)
     
     init_op = tf.global_variables_initializer()

     with tf.Session() as sess:
         sess.run(init_op)
    
         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(coord=coord)
         
         vheight,vwidth,vfilename,vsource_id,vencoded,vimage_format = sess.run([height,width,filename,source_id,encoded,image_format])
         print vheight,vwidth,vfilename,vsource_id,vimage_format
         image = Image.open(io.BytesIO(vencoded))
         image.show()
         
         
         coord.request_stop()
         coord.join(threads)
         
main()