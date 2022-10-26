#  Face Detection using SSD+MobileNet-v2 Trained on WIDER FACE Dataset

<img src="images/SSD-MobileNet.webp" width="1000"/>

## 1. Objective

In this project, we shall demonstrate the Deep Learning (DL) inference using a DL object detection model, SSD+MobileNet-v2, which has already been trained on the WIDER FACE dataset, to detect human faces.

## 2. SSD+MobileNet-v2

* SSD+MobileNet-v2 is a state-of-the-art, real-time object detection model. It is not straight forward to reasonably train this network from scratch, due to several reasons including: 

  * Lack of large volume of annotated data 
  * Lack of sufficiently powerful computing resources.

* Instead of exploring the training of SSD+MobileNet from scratch we use an already trained model retrieved from the following source: 
    * Trained SSD-MobileNet-v2 model source: https://github.com/Fszta/Tensorflow-face-detection
    * This model has been trained on the WIDER FACE dataset:
      * Source: http://shuoyang1213.me/WIDERFACE/
      * WIDER FACE dataset is a face detection benchmark dataset, of which images are selected from the publicly available WIDER dataset.
      * It has 32,203 images and label 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images.
      * WIDER FACE dataset is organized based on 61 event classes.
      * For each event class, we randomly select 40%/10%/50% data as training, validation and testing sets

In this work, we shall demonstrate how to deploy the trained model to detect objects of interest:

## 3. Development

* Project:  Face Detection using SSD+MobileNet-v2 Trained on WIDER FACE Dataset:
  * The objective of this project is to demonstrate how to use the state of the art in object detection. 

* Author: Mohsen Ghazel (mghazel)
* Date: April 15th, 2021

### 3.1. Step 1: Imports and global variables

#### 3.1.1. Python import:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># Python imports and environment setup</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># opencv</span>
<span style="color:#200080; font-weight:bold; ">import</span> cv2
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>image <span style="color:#200080; font-weight:bold; ">as</span> mpimg

<span style="color:#595979; "># tensorflow</span>
<span style="color:#200080; font-weight:bold; ">import</span> tensorflow <span style="color:#200080; font-weight:bold; ">as</span> tf

<span style="color:#595979; "># utilities functionalities</span>
<span style="color:#200080; font-weight:bold; ">from</span> utils <span style="color:#200080; font-weight:bold; ">import</span> label_map_util
<span style="color:#595979; "># visualization utilities functionalities</span>
<span style="color:#200080; font-weight:bold; ">from</span> utils <span style="color:#200080; font-weight:bold; ">import</span> visualization_utils_color <span style="color:#200080; font-weight:bold; ">as</span> vis_util

<span style="color:#595979; "># input/output OS</span>
<span style="color:#200080; font-weight:bold; ">import</span> os 

<span style="color:#595979; "># date-time to show date and time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># import time</span>
<span style="color:#200080; font-weight:bold; ">import</span> time

<span style="color:#595979; "># to display the figures in the notebook</span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Test imports and display package versions</span>
<span style="color:#595979; ">#------------------------------------------</span>
<span style="color:#595979; "># Testing the OpenCV version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"OpenCV : "</span><span style="color:#308080; ">,</span>cv2<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>
<span style="color:#595979; "># Testing the numpy version</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy : "</span><span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span>

OpenCV <span style="color:#308080; ">:</span>  <span style="color:#008000; ">3.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">8</span>
Numpy <span style="color:#308080; ">:</span>  <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
</pre>

### 1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Path to the trained SSD-MobileNet-v2 frozen detection graph:</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># This is the actual model that is used for the object detection.</span>
PATH_TO_CKPT <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'./model/frozen_inference_graph.pb'</span>

<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># List of the strings that is used to add correct label for each </span>
<span style="color:#595979; "># box.</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
PATH_TO_LABELS <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'./proto/label_map.pbtxt'</span>

<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Since the model was trained to detected human faces: </span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Number of classes = 1: FACE</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
NUM_CLASSES <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>

<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># The detection confidence threshold: </span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># - Only detections with confidence higher than this threshold </span>
<span style="color:#595979; ">#   are kept.</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
DETECTION_CONFIDENCE_THRESHOLD <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.50</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Set test images folder name: </span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Selected 10 test images from the WIDER-FACE TEST </span>
<span style="color:#595979; "># data subset</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
test_images_folder <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'images/wider-face--10-test-images/'</span>
</pre>

### 4.2. Step 2: Implement the DL inference of the trained SSD-MobilenNet-v2 model:
* We now run deploy the trained SSD-MobileNet-v2 model to detect faces from test images
* Selected 10 test images from the WIDER-FACE TEST data subset

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Load label map</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># - This allows us to map the class ID to class name</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># get the label-map</span>
label_map <span style="color:#308080; ">=</span> label_map_util<span style="color:#308080; ">.</span>load_labelmap<span style="color:#308080; ">(</span>PATH_TO_LABELS<span style="color:#308080; ">)</span>
<span style="color:#595979; "># get the class categories/names</span>
categories <span style="color:#308080; ">=</span> label_map_util<span style="color:#308080; ">.</span>convert_label_map_to_categories<span style="color:#308080; ">(</span>label_map<span style="color:#308080; ">,</span> max_num_classes<span style="color:#308080; ">=</span>NUM_CLASSES<span style="color:#308080; ">,</span> use_display_name<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># get the class categories indices</span>
category_index <span style="color:#308080; ">=</span> label_map_util<span style="color:#308080; ">.</span>create_category_index<span style="color:#308080; ">(</span>categories<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#595979; "># Start the face detection inference:</span>
<span style="color:#595979; ">#----------------------------------------------------------------</span>
<span style="color:#200080; font-weight:bold; ">def</span> face_detection<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>

    <span style="color:#595979; "># Load Tensorflow model</span>
    detection_graph <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>Graph<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">with</span> detection_graph<span style="color:#308080; ">.</span>as_default<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># </span><span style="color:#ffffff; background:#808000; ">FIXED ERROR: AttributeError: module 'tensorflow' has no attribute 'GraphDef'</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># tf.compat.v1.GraphDef()   # -&gt; instead of tf.GraphDef()</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># od_graph_def = tf.GraphDef()</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        od_graph_def <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>compat<span style="color:#308080; ">.</span>v1<span style="color:#308080; ">.</span>GraphDef<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># </span><span style="color:#ffffff; background:#808000; ">FIXED THE ERROR: AttributeError: module 'tensorflow' has no attribute 'gfile'</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># 1.Find label_map_util.py line 137.</span>
        <span style="color:#595979; "># 2.Replace tf.gfile.GFile to tf.io.gfile.GFile</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">with</span> tf<span style="color:#308080; ">.</span>io<span style="color:#308080; ">.</span>gfile<span style="color:#308080; ">.</span>GFile<span style="color:#308080; ">(</span>PATH_TO_CKPT<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'rb'</span><span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">as</span> fid<span style="color:#308080; ">:</span>
            serialized_graph <span style="color:#308080; ">=</span> fid<span style="color:#308080; ">.</span>read<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
            od_graph_def<span style="color:#308080; ">.</span>ParseFromString<span style="color:#308080; ">(</span>serialized_graph<span style="color:#308080; ">)</span>
            tf<span style="color:#308080; ">.</span>import_graph_def<span style="color:#308080; ">(</span>od_graph_def<span style="color:#308080; ">,</span> name<span style="color:#308080; ">=</span><span style="color:#1060b6; ">''</span><span style="color:#308080; ">)</span>
        
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># </span><span style="color:#ffffff; background:#808000; ">FIXED ERROR: AttributeError: module 'tensorflow' has no attribute 'Session'</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># According to TF 1:1 Symbols Map, in TF 2.0 you should use tf.compat.v1.Session() instead of tf.Session()</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># sess = tf.Session(graph=detection_graph)</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        sess <span style="color:#308080; ">=</span> tf<span style="color:#308080; ">.</span>compat<span style="color:#308080; ">.</span>v1<span style="color:#308080; ">.</span>Session<span style="color:#308080; ">(</span>graph<span style="color:#308080; ">=</span>detection_graph<span style="color:#308080; ">)</span>
        
    image_tensor <span style="color:#308080; ">=</span> detection_graph<span style="color:#308080; ">.</span>get_tensor_by_name<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'image_tensor:0'</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Each box represents a part of the image where a particular object was detected.</span>
    detection_boxes <span style="color:#308080; ">=</span> detection_graph<span style="color:#308080; ">.</span>get_tensor_by_name<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'detection_boxes:0'</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Each score represent how level of confidence for each of the objects.</span>
    <span style="color:#595979; "># Score is shown on the result image, together with the class label.</span>
    detection_scores <span style="color:#308080; ">=</span> detection_graph<span style="color:#308080; ">.</span>get_tensor_by_name<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'detection_scores:0'</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># Actual detection.</span>
    detection_classes <span style="color:#308080; ">=</span> detection_graph<span style="color:#308080; ">.</span>get_tensor_by_name<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'detection_classes:0'</span><span style="color:#308080; ">)</span>
    num_detections <span style="color:#308080; ">=</span> detection_graph<span style="color:#308080; ">.</span>get_tensor_by_name<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'num_detections:0'</span><span style="color:#308080; ">)</span>


    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># Itetate over all the images in the test images </span>
    <span style="color:#595979; "># folder, and detect human faces using the trained </span>
    <span style="color:#595979; "># SSD-MobbileNET-ve model</span>
    <span style="color:#595979; ">#------------------------------------------------------</span>
    <span style="color:#595979; "># initialize the image counter</span>
    image_counter <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
    <span style="color:#200080; font-weight:bold; ">for</span> filename <span style="color:#200080; font-weight:bold; ">in</span> os<span style="color:#308080; ">.</span>listdir<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        <span style="color:#595979; "># read the test image</span>
        <span style="color:#595979; ">#------------------------------------------------------</span>
        frame <span style="color:#308080; ">=</span> cv2<span style="color:#308080; ">.</span>imread<span style="color:#308080; ">(</span>os<span style="color:#308080; ">.</span>path<span style="color:#308080; ">.</span>join<span style="color:#308080; ">(</span>test_images_folder<span style="color:#308080; ">,</span>filename<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># check if image was read properly</span>
        <span style="color:#200080; font-weight:bold; ">if</span> frame <span style="color:#200080; font-weight:bold; ">is</span> <span style="color:#200080; font-weight:bold; ">not</span> <span style="color:#074726; ">None</span><span style="color:#308080; ">:</span>
            <span style="color:#595979; "># increment the image counter</span>
            image_counter <span style="color:#308080; ">=</span> image_counter <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span>
            <span style="color:#595979; ">#------------------------------------------------------</span>
            <span style="color:#595979; "># Deploy the YOLO model to conduct inference in </span>
            <span style="color:#595979; "># the image</span>
            <span style="color:#595979; ">#------------------------------------------------------</span>
            <span style="color:#595979; "># Expand dimensions since the model expects images to have shape: [1, None, None, 3]</span>
            expanded_frame <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>expand_dims<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> axis<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
            <span style="color:#308080; ">(</span>boxes<span style="color:#308080; ">,</span> scores<span style="color:#308080; ">,</span> classes<span style="color:#308080; ">,</span> num_c<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> sess<span style="color:#308080; ">.</span>run<span style="color:#308080; ">(</span>
                <span style="color:#308080; ">[</span>detection_boxes<span style="color:#308080; ">,</span> detection_scores<span style="color:#308080; ">,</span> detection_classes<span style="color:#308080; ">,</span> num_detections<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span>
                feed_dict<span style="color:#308080; ">=</span><span style="color:#406080; ">{</span>image_tensor<span style="color:#308080; ">:</span> expanded_frame<span style="color:#406080; ">}</span><span style="color:#308080; ">)</span>

            <span style="color:#595979; ">#------------------------------------------------------</span>
            <span style="color:#595979; "># Visualization of the face detection results</span>
            <span style="color:#595979; ">#------------------------------------------------------</span>
            vis_util<span style="color:#308080; ">.</span>visualize_boxes_and_labels_on_image_array<span style="color:#308080; ">(</span>
                frame<span style="color:#308080; ">,</span>
                np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>boxes<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>classes<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>astype<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>int32<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>scores<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span>
                category_index<span style="color:#308080; ">,</span>
                use_normalized_coordinates<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">,</span>
                line_thickness<span style="color:#308080; ">=</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">,</span>
                min_score_thresh<span style="color:#308080; ">=</span>DETECTION_CONFIDENCE_THRESHOLD<span style="color:#308080; ">)</span>

            <span style="color:#595979; "># display the frame with overlaid face detections results</span>
            <span style="color:#595979; "># create a figure</span>
            plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>uint8<span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span> <span style="color:#44aadd; ">*</span> frame<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span>frame<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
            <span style="color:#595979; "># visualize detection results</span>
            plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">111</span><span style="color:#308080; ">)</span>
            plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"SSD-MobileNet-v2: Face detection results"</span><span style="color:#308080; ">,</span> fontsize<span style="color:#308080; ">=</span><span style="color:#008c00; ">12</span><span style="color:#308080; ">)</span>
            plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> plt<span style="color:#308080; ">.</span>yticks<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
            plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>cv2<span style="color:#308080; ">.</span>cvtColor<span style="color:#308080; ">(</span>frame<span style="color:#308080; ">,</span> cv2<span style="color:#308080; ">.</span>COLOR_BGR2RGB<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
            <span style="color:#595979; "># save the image</span>
            cv2<span style="color:#308080; ">.</span>imwrite<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'images/face-detection-results/'</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'test-image-0'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>image_counter<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">'.jpg'</span><span style="color:#308080; ">,</span> frame<span style="color:#308080; ">)</span> 
            
            

<span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#074726; ">__name__</span> <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'__main__'</span><span style="color:#308080; ">:</span>
    face_detection<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

### 4.5. Step 5: Display a successful execution message:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">04</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">15</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">36</span><span style="color:#308080; ">:</span><span style="color:#008000; ">19.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>

## 4. Sample Face Detection Results

* In this section, we illustrate sample face and eyes detection results from the same 10 test images used for testing the Haar Cascades face detector. These images were selected from the WIDER FACE Test data subset.

<table>
  <tr>
    <td> <img src="images/test-image-01.jpg"  width = "500" align = "center" ></td>
    <td> <img src="images/test-image-02.jpg"  width = "500" align = "center" ></td>
   </tr> 
   <tr>
      <td> <img src="images/test-image-03.jpg"  width = "500" align = "center" ></td>
      <td> <img src="images/test-image-04.jpg"  width = "500" align = "center" ></td>
  </td>
  <tr>
      <td> <img src="images/test-image-05.jpg"  width = "500" align = "center" ></td>
      <td> <img src="images/test-image-06.jpg"  width = "500" align = "center" ></td>
  </td>
  <tr>
      <td> <img src="images/test-image-07.jpg"  width = "500" align = "center" ></td>
      <td> <img src="images/test-image-08.jpg"  width = "500" align = "center" ></td>
  </td>
  <tr>
      <td> <img src="images/test-image-09.jpg"  width = "500" align = "center" ></td>
      <td> <img src="images/test-image-010.jpg"  width = "500" align = "center" ></td>
  </td>
  </tr>
</table>


## 5. Analysis

* In view of the presented results, we make the following observations:

  * The face detection results generated by the SSD+MobileNet-v2 DL object detection model, which was trained on the WIDER FACE dataset, are superior to those generated by the Haar Cascades face detector, presented in Section 5.1:
  * We used the  same  10 test images used for testing the Haar Cascades face detector. 
  * These images were selected from the WIDER FACE test data subset.
  * Unlike the Haar Cascaades face detector, we observe the following about the trained SSD+MobileNet-v2 face detector:
    * Very few false-positives are observed
    * Some of the missed detections for clearly visible faces are just puzzling
    * It is not so sensitive to the face pose: It is able to detect most of the turned face
    * It is not so sensitive face scale: It is  able to detect most of the far and small faces
    * It is not so sensitive skin tone: It is able to detect some of the faces with darker skin tones.
    * It is not so sensitive obstruction: It is able to detect some of the partially obscured faces.
    * In spite of its superior performance, the trained SSD+MobileNet-v2 face detector is not without limitations:
      * There are missed detections that are clear faces
      * It missed most of the small or rotated faces.

## 7. Future Work

* We plan to investigate some of the related issues:

* Get a better understanding of the SSD+MobileNet-v2 model structures details 
* Test the trained SSD+MobileNet-v2 with more diverse data and identity their limitations
* Search for and test other pre-trained SSD+MobileNet-v2 face  detectors.

## 7. References

1. Fszta. Tensorflow-face-detection. https://github.com/Fszta/Tensorflow-face-detection 
2. WIDER FACE. WIDER FACE: A Face Detection Benchmark. http://shuoyang1213.me/WIDERFACE/ 
3. Mayank Singhal. Object Detection using SSD Mobilenet and Tensorflow Object Detection API: Can detect any single class from coco dataset. https://medium.com/@techmayank2000/object-detection-using-ssd-mobilenetv2-using-tensorflow-api-can-detect-any-single-class-from-31a31bbd0691 
4. Awesome Open Source. Face Detection With SSD+Mobilenet. https://awesomeopensource.com/project/bruceyang2012/Face-detection-with-mobilenet-ssd 
5. Horned Sungem. SSD+MobileNet Face Detector. https://hornedsungem.github.io/Docs/en/model/graph_face_ssd/ 
6. Analytics Vidhya. How to Build A Real-Time Face Mask Detector. https://medium.com/analytics-vidhya/real-time-face-mask-detector-8b484a5a6de0 
7. Pranilshinde. Face-Mask Detector Using TensorFlow-Object Detection(SSD+MobileNet). https://medium.com/pranil-shinde/face-mask-detector-using-tensorflow-object-detection-ssd-mobilenet-37f233202c67
 
