import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
tf.compat.v1.enable_eager_execution()

from PIL import Image
import io
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------------------------------------------------
# Creating Record Files
# ---------------------------------------------------------------------------------------------------------------------

def _create_record_file_path(path):
    # clear the otuput path flag and set
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    try:
        FLAGS.__delattr__('output_path')
        FLAGS.__delattr__('f')
    except:
        pass

    tf.app.flags.DEFINE_string('f', '', 'kernel') 
    flags.DEFINE_string('output_path', path, '')
    FLAGS = flags.FLAGS

    print("New record file : {}".format(flags.FLAGS.output_path))  
    return tf.app.flags.FLAGS.output_path

def create_record_file(image_path, output_path, examples_dict, class_to_index):
    output_path = _create_record_file_path(output_path)
    
    writer = tf.python_io.TFRecordWriter(output_path)
    for key, val in examples_dict.items():
        example = val
        example["filename"] = key
        tf_example = create_tf_example(example, image_path, class_to_index)
        writer.write(tf_example.SerializeToString())
    writer.close()
    
    print("Wrote {} examples".format(len(examples_dict)))


def create_tf_example(example, path, class_mapping):
    """
    Create a single Tensorflow Example object to be used in creating record
    
    Parameters
    ----------
    
        example : dict
            A single object; the dictionary should contains the keys "filename" referring to the jpg containing
            the object, and "box_coords" which gives the location of the object, and "class" the name of the object
            
        path : str
            The path to the image files.
    
    Returns
    -------
    
        The tf Example object
    
    """
    path = (path + os.sep).encode('ascii')
    filename = example['filename'].encode('ascii')
    image_format = b'jpg'
    
    image = plt.imread(path +filename, "jpg") 
    height, width = image.shape[:2]
    
    # Encode the jpg to byte form
    with tf.gfile.GFile(path+filename, 'rb') as fid:
        encoded_jpg = bytes(fid.read())

    # normalize the box coordinates
    xmins = [box[0]/width  for box in example['box_coords']] 
    ymins = [box[1]/height for box in example['box_coords']] 
    xmaxs = [box[2]/width  for box in example['box_coords']]
    ymaxs = [box[3]/height for box in example['box_coords']]

    classes_text = [cls.encode('ascii') for cls in example["class"]]
    classes      = [class_mapping[cls]  for cls in example["class"]]

    # create the example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height'             : dataset_util.int64_feature(height),
      'image/width'              : dataset_util.int64_feature(width),
      'image/filename'           : dataset_util.bytes_feature(filename),
      'image/source_id'          : dataset_util.bytes_feature(filename),
      'image/encoded'            : dataset_util.bytes_feature(encoded_jpg),
      'image/format'             : dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin'   : dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax'   : dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin'   : dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax'   : dataset_util.float_list_feature(ymaxs),
      'image/object/class/text'  : dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label' : dataset_util.int64_list_feature(classes),
      }))
    return tf_example

# ---------------------------------------------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------------------------------------------

def peek_in_record(path, plot=True):

    objects = dict()
    obj_per_img = []

    obj_shapes = []
    img_shapes = []

    total_images = 0
    for result in load_tf_record_file(path):
        total_images += 1

        names = result.context.feature['image/object/class/text'].bytes_list.value
        for name in names:
            if name not in objects:
                objects[name] = 1
            else:
                objects[name] += 1

        obj_per_img += [len(names)]

        width  = result.context.feature['image/width'].int64_list.value[0]
        height = result.context.feature['image/height'].int64_list.value[0]
        xmins = np.array(result.context.feature['image/object/bbox/xmin'].float_list.value)
        ymins = np.array(result.context.feature['image/object/bbox/ymin'].float_list.value)
        xmaxs = np.array(result.context.feature['image/object/bbox/xmax'].float_list.value)
        ymaxs = np.array(result.context.feature['image/object/bbox/ymax'].float_list.value)
        xmins *= width
        xmaxs *= width
        ymins *= height
        ymaxs *= height

        img_shapes += [[height, width]]

        for xmin, ymin, xmax, ymax, name in zip(xmins, ymins, xmaxs, ymaxs, names):
            obj_shapes += [[ymax-ymin, xmax-xmin]]

            if (ymin < 0) or (ymax > height) or (xmin < 0) or (xmax > width):
                print("WARNING : Object {} outisde of image region".format(name))

        total_objects = sum(objects.values())

    obj_shapes = np.array(obj_shapes)
    img_shapes = np.array(img_shapes)

    print("="*100)
    print("Total Images            : {0}".format(total_images))
    print("Total Objects           : {0}".format(total_objects))
    print("Ave. Objects per Image  : {0}".format(total_objects/total_images))
    print("Classes                 : {0}".format(len(objects)))
    print("="*100)
    
    if plot:
        fig, axes  = plt.subplots(2,2,figsize=(12,12))

        ax = axes[1,0]

        ax.scatter(obj_shapes[:,1], obj_shapes[:,0])
        ax.set_ylim([0.9*obj_shapes[:,0].min(), 1.1*obj_shapes[:,0].max()])
        ax.set_xlim([0.9*obj_shapes[:,1].min(), 1.1*obj_shapes[:,1].max()])
        ax.set_xlabel("Width", fontsize=18)
        ax.set_ylabel("Height", fontsize=18)
        ax.set_title("Object Shapes", fontsize=18)

        ax = axes[1,1]

        ax.scatter(img_shapes[:,1], img_shapes[:,0])
        ax.set_ylim([0.9*img_shapes[:,0].min(), 1.1*img_shapes[:,0].max()])
        ax.set_xlim([0.9*img_shapes[:,1].min(), 1.1*img_shapes[:,1].max()])
        ax.set_xlabel("Width", fontsize=18)
        ax.set_ylabel("Height", fontsize=18)
        ax.set_title("Image Shapes", fontsize=18)

        ax = axes[0,1]

        ax.hist(obj_per_img, bins = np.arange(0.5,max(obj_per_img)+1.5,1),density=1.0)
        ax.set_xlabel("Objects per image", fontsize=18)

        ax = axes[0,0]

        labels = [x.decode() for x in objects.keys()]
        sizes  = list(objects.values())

        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.show()


# ---------------------------------------------------------------------------------------------------------------------
# Reading Record Files
# ---------------------------------------------------------------------------------------------------------------------

def load_tf_record_file(path):
    for example in tf.python_io.tf_record_iterator(path):
        yield tf.train.SequenceExample.FromString(example)

def read_record_file(path, index_to_class, return_dict=True, plot=True, **plot_kwargs):
    
    record = dict()

    for result in load_tf_record_file(path):
        
        fname = str(result.context.feature['image/filename'].bytes_list.value[0], "utf-8") 
        
        width  = result.context.feature['image/width'].int64_list.value[0]
        height = result.context.feature['image/height'].int64_list.value[0]
        data   = result.context.feature['image/encoded'].bytes_list.value[0]
        
        img = Image.open(io.BytesIO(data), mode="r")
        img = np.asarray(img)

        xmins = np.array(result.context.feature['image/object/bbox/xmin'].float_list.value)
        ymins = np.array(result.context.feature['image/object/bbox/ymin'].float_list.value)
        xmaxs = np.array(result.context.feature['image/object/bbox/xmax'].float_list.value)
        ymaxs = np.array(result.context.feature['image/object/bbox/ymax'].float_list.value)

        xmins *= width
        xmaxs *= width
        ymins *= height
        ymaxs *= height

        labels = np.array(result.context.feature['image/object/class/label'].int64_list.value, dtype=int)

        if return_dict:
            
            record[fname] = dict()
            record[fname]["width"]  = width
            record[fname]["height"] = height
            record[fname]["image"]  = img
            record[fname]["xmins"]  = xmins
            record[fname]["xmaxs"]  = xmaxs
            record[fname]["ymins"]  = ymins
            record[fname]["ymaxs"]  = ymaxs
        
        if plot:
            fig, ax  = plt.subplots(1,1,figsize=plot_kwargs.get("figsize",(8,8)))
            ax.imshow(np.asarray(img))

            coords = []
            for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
                coord = [xmin, ymin, xmax, ymax]
                coords +=[coord]
                ax.add_patch(Rectangle(xy=(coord[0], coord[1]), 
                                       width=coord[2]-coord[0], 
                                       height=coord[3]-coord[1], 
                                       fill=None, color="r"))

            num = len(labels)
            x_coords = [1.1]*num
            y_coords = [i/num for i in range(num)]
            for (x,y,s, coord) in zip(x_coords, y_coords, labels, coords):
                text = index_to_class[s]
                ax.text(x*width,y*height,text, fontsize=16)
                ax.plot([coord[2],x*width*0.99],[coord[1],y*height], linestyle=":", color="r")

            ax.set_title(fname.replace("_", "-"), fontsize=18)
            ax.axis('off')
            
            plt.show()
            
    return record    