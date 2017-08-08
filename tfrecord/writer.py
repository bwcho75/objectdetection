import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

def create_cat_tf_example(encoded_image_data):

  height = 1032
  width = 1200
  filename = 'example_cat.jpg'
  image_format = 'jpg'

  xmins = [322.0 / 1200.0]
  xmaxs = [1062.0 / 1200.0]
  ymins = [174.0 / 1032.0]
  ymaxs = [761.0 / 1032.0]
  classes_text = ['Cat']
  classes = [1]

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
  return tf_example

def read_imagebytes(imagefile):
    file = open(imagefile,'rb')
    bytes = file.read()

    return bytes
    
def main():
    print ('Converting example_cat.jpg to example_cat.tfrecord')
    tfrecord_filename = 'example_cat.tfrecord'
    bytes = read_imagebytes('example_cat.jpg')
    tf_example = create_cat_tf_example(bytes)

    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    writer.write(tf_example.SerializeToString())

main()
